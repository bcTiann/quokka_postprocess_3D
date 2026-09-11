"""Independent reuse, holdout, and runtime-lookup checks for length refinement."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quokka2s.cloudy_sixline_lookup import CloudySixLineLookup, DEPTH_AXIS_ORDER
from quokka2s.tables.abundances import abundance_metadata
from scripts import refine_cloudy_model_depth_pilot as refinement
from scripts.cloudy_model_depth_common import direct_input, sha256


def synthetic_records():
    """Curved positive lines and true-zero corners at known physical depths."""
    records = []
    for track in range(7):
        for index, depth in enumerate(refinement.REFINED_LOG_L_PC):
            values = [10**(-20 + .2 * track + .3 * depth + .4 * depth**2),
                      0. if index in (0, 4, 12) else 10**(-21 + .2 * depth),
                      0., 10**(-250 + depth), 10**(-23 - depth**2),
                      10**(-22 + np.sin(depth)), 1e-24, 2e-24]
            records.append(dict(track=track, depth_index=index,
                                log_L_model_pc=float(depth),
                                checks=dict(valid=True, emissivity_per_nH2=values)))
    return records


class RefinementAxisTests(unittest.TestCase):
    def test_new_states_are_only_disjoint_quarter_points(self):
        cases = refinement.new_cases()
        self.assertEqual(len(cases), 126)
        self.assertEqual(len({case['id'] for case in cases}), 126)
        for track in range(7):
            selected = [case for case in cases if case['track'] == track]
            self.assertEqual([case['depth_index'] for case in selected], list(range(1, 36, 2)))
            for case in selected:
                self.assertEqual(case['log_L_model_pc'], -.25 + case['depth_index'] / 16)
                self.assertEqual(case['role'], 'independent_holdout')
                text = direct_input(case['id'], log_nH=case['log_nH'], log_T=case['log_T'],
                                    log_NH=case['log_NH'], log_L_pc=case['log_L_model_pc'])
                self.assertNotIn('drmax', text.lower())
                self.assertNotIn('set dr ', text.lower())
        np.testing.assert_array_equal(refinement.REFINED_LOG_L_PC[::2], np.linspace(-.25, 2, 19))
        np.testing.assert_array_equal(refinement.REFINED_LOG_L_PC[::4], np.linspace(-.25, 2, 10))

    def test_same_126_independent_holds_are_used_for_both_interpolants(self):
        records = synthetic_records()
        ten = refinement.compare_grid(records, 4, list(range(1, 37, 2)))
        nineteen = refinement.compare_grid(records, 2, list(range(1, 37, 2)))
        self.assertEqual(len(ten), 126)
        self.assertEqual([(r['track'], r['depth_index']) for r in ten],
                         [(r['track'], r['depth_index']) for r in nineteen])
        self.assertEqual({r['fraction'] for r in ten}, {.25, .75})
        self.assertEqual({r['fraction'] for r in nineteen}, {.5})
        for coarse, refined in zip(ten, nineteen):
            self.assertEqual(coarse['direct'], refined['direct'])
            self.assertLess(refined['absolute_error_dex'][0], coarse['absolute_error_dex'][0])
        original = refinement.compare_grid(records, 4, list(range(2, 37, 4)))
        self.assertEqual(len(original), 63)
        self.assertTrue(set(r['depth_index'] for r in original).isdisjoint(r['depth_index'] for r in ten))
        with self.assertRaisesRegex(ValueError, 'independent'):
            refinement.compare_grid(records, 2, list(range(2, 37, 4)))

    def test_missing_or_invalid_endpoints_never_become_filled_comparisons(self):
        records = synthetic_records()
        records[0]['checks']['valid'] = False
        result = refinement.compare_grid(records, 2, [1])
        self.assertFalse(result[0]['valid'])
        self.assertTrue(all(row['valid'] for row in result[1:]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            refinement.compare_grid(records + [records[0]], 2, [1])

    def test_zero_mismatch_is_retained_when_dex_is_undefined(self):
        records = synthetic_records()
        # The original ten-node grid misses emission between two true-zero corners.
        result = refinement.compare_grid(records, 4, [1])[0]
        self.assertTrue(result['valid'])
        self.assertEqual(result['interpolated'][1], 0.)
        self.assertGreater(result['direct'][1], 0.)
        self.assertTrue(result['zero_mismatch'][1])
        self.assertIsNone(result['absolute_error_dex'][1])
        self.assertEqual(result['relative_error'][1], 1.)


class RuntimeOracleTests(unittest.TestCase):
    def test_both_interpolants_match_actual_four_dimensional_lookup(self):
        records = synthetic_records()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'oracle.npz'
            for stride in (4, 2):
                node_indices = list(range(0, 37, stride))
                values = np.asarray([records[i]['checks']['emissivity_per_nH2']
                                     for i in node_indices]).T
                zeros = values == 0
                log_values = np.full_like(values, -99.)
                log_values[~zeros] = np.round(np.log10(values[~zeros]), 4)
                rounded = np.zeros_like(values)
                rounded[~zeros] = 10**log_values[~zeros]
                shape = (8, 2, 2, 2, len(node_indices))
                expand = lambda a: np.broadcast_to(a[:, None, None, None, :], shape)
                np.savez(path, axis_order=np.array(DEPTH_AXIS_ORDER),
                         line_keys=np.array([f'line{i}' for i in range(8)]),
                         log_NH_attenuation=np.array([18., 21.]),
                         log_nH=np.array([-1., 1.]), log_T=np.array([3., 5.]),
                         log_L_model_pc=refinement.REFINED_LOG_L_PC[node_indices],
                         log_emissivity_per_nH2=expand(log_values),
                         emissivity_per_nH2=expand(rounded), zero_mask=expand(zeros),
                         failure_mask=np.zeros(shape, dtype=bool))
                actual = CloudySixLineLookup(path).sample(
                    1e4, 1., 1e20, model_depth_pc=10**refinement.REFINED_LOG_L_PC[1::2]
                ).emissivity_per_nH2
                rows = refinement.compare_grid(records, stride, list(range(1, 37, 2)))[:18]
                measured = np.asarray([r['interpolated'] for r in rows]).T
                np.testing.assert_allclose(measured, actual, rtol=2e-13, atol=0)
                self.assertGreater(actual[3].min(), 0.)


class ParentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.exe = self.base / 'cloudy.exe'; self.exe.write_text('synthetic executable fixture')
        self.abn = self.base / 'default.abn'; self.abn.write_text('synthetic abundance fixture')
        sed = self.base / refinement.SED_DIRECTORY_NAME; sed.mkdir()
        sed_names = {'build_report.json'} | {
            f'logNH{column:g}.{extension}' for _, _, column in refinement.TRACKS
            for extension in ('sed', 'out')}
        for name in sed_names: (sed / name).write_text('synthetic spectral evidence: ' + name)
        scripts = Path(refinement.__file__).parent
        provenance = dict(cloudy_executable_sha256=sha256(self.exe), default_abn_sha256=sha256(self.abn),
                          common_code_sha256=sha256(scripts / 'cloudy_model_depth_common.py'),
                          pilot_code_sha256=sha256(scripts / 'validate_cloudy_model_depth_pilot.py'),
                          abundance=abundance_metadata(),
                          sed_sha256={name: sha256(sed / name) for name in sed_names})
        records = []
        for track, (density, temperature, column) in enumerate(refinement.TRACKS):
            for old_index in range(19):
                name = f'track{track:02d}_depth{old_index:02d}'
                case = dict(id=name, track=track, depth_index=old_index,
                            role='node' if old_index % 2 == 0 else 'midpoint',
                            log_nH=refinement.LOG_NH_DENSITY[density], log_T=float(refinement.LOG_T[temperature]),
                            log_NH=column, log_L_model_pc=float(-.25 + old_index / 8),
                            provenance=provenance, checks=dict(valid=True, emissivity_per_nH2=[1e-20] * 8))
                root = self.base / name
                root.with_suffix('.in').write_text(direct_input(
                    name, log_nH=case['log_nH'], log_T=case['log_T'],
                    log_NH=case['log_NH'], log_L_pc=case['log_L_model_pc']))
                for suffix in ('.out', '.radius', '.physical', '.lines'):
                    root.with_suffix(suffix).write_text('synthetic frozen checkpoint evidence: ' + suffix)
                case['output_sha256'] = {p.name: sha256(p) for p in self.base.glob(name + '.*')}
                root.with_suffix('.json').write_text(json.dumps(case))
                records.append(case)
        self.summary = dict(status='completed', total=133, provenance=provenance, records=records)
        self.save_summary()

    def save_summary(self):
        (self.base / 'summary.json').write_text(json.dumps(self.summary))

    def test_parent_indices_and_values_are_reused_without_mutation(self):
        original = copy.deepcopy(self.summary)
        _, reused = refinement.verify_parent(self.base, self.exe, self.abn)
        self.assertEqual(len(reused), 133)
        for old, new in zip(original['records'], reused):
            self.assertEqual(new['depth_index'], 2 * old['depth_index'])
            self.assertEqual(new['original_depth_index'], old['depth_index'])
            self.assertEqual(new['log_L_model_pc'], old['log_L_model_pc'])
            self.assertEqual(new['checks'], old['checks'])
            self.assertEqual(Path(new['source_record_path']), (self.base / (old['id'] + '.json')).resolve())
        self.assertEqual(json.loads((self.base / 'summary.json').read_text()), original)

    def test_changed_raw_output_is_rejected(self):
        (self.base / 'track00_depth00.lines').write_text('changed after checkpoint')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            refinement.verify_parent(self.base, self.exe, self.abn)

    def test_changed_executable_is_rejected(self):
        self.exe.write_text('different executable')
        with self.assertRaisesRegex(ValueError, 'Executable'):
            refinement.verify_parent(self.base, self.exe, self.abn)

    def test_summary_cannot_override_unchanged_case_checkpoint(self):
        self.summary['records'][0]['checks']['emissivity_per_nH2'][0] *= 100
        self.save_summary()
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            refinement.verify_parent(self.base, self.exe, self.abn)

    def test_missing_source_evidence_is_rejected(self):
        first = self.summary['records'][0]
        del first['output_sha256'][first['id'] + '.radius']
        (self.base / (first['id'] + '.json')).write_text(json.dumps(first)); self.save_summary()
        with self.assertRaisesRegex(ValueError, 'evidence is incomplete'):
            refinement.verify_parent(self.base, self.exe, self.abn)


if __name__ == '__main__':
    unittest.main()
