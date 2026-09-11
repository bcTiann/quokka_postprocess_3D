"""Coverage counts use physical query weights and whole-group mass totals."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.cloudy_cell_coverage import validated_coverage_lookup, classify_cell_queries, CellCoverageTotals
from quokka2s.cloudy_cell_queries import prepare_cloudy_cell_queries
from quokka2s.cloudy_sixline_lookup import DEPTH_AXIS_ORDER
from quokka2s.tables.abundances import QUOKKA_MASS_FRACTIONS

CONSTANTS = dict(hydrogen_mass_g=1.6735575e-24, boltzmann_erg_K=1.380649e-16,
                 gravitational_cm3_g_s2=6.67430e-8, parsec_cm=3.0856775814913673e18)


class CloudyCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.table = Path(self.temp.name)/'table.npz'
        self.validation = Path(self.temp.name)/'validation.json'
        axes = dict(log_NH_attenuation=[18.,21.], log_nH=[0.,1.], log_T=[2.,4.], log_L_model_pc=[1.,2.])
        shape = (1,2,2,2,2)
        failure = np.zeros(shape,dtype=bool)
        failure[0,0,0,0,1] = True
        diagnostic = failure[0].copy()
        diagnostic[1,0,0,1] = True
        coefficient = np.where(failure,0.,1e-30)
        logs = np.where(failure,np.nan,-30.)
        packed_maps=[dict(path=str(Path(self.temp.name)/f'map{i}.dat'),sha256=f'hash{i}') for i in range(8)]
        np.savez(self.table, **axes, axis_order=DEPTH_AXIS_ORDER,line_keys=['hi21'],
                 emissivity_per_nH2=coefficient,log_emissivity_per_nH2=logs,
                 failure_mask=failure,zero_mask=np.zeros(shape,dtype=bool),manifest_sha256='build-hash',
                 provenance_json=json.dumps(dict(maps=packed_maps)))
        self.report = dict(axes=axes,axis_order=','.join(axes),execution_status='completed',global_issues=[],
                           manifest_sha256='build-hash',raw_map_failure_mask=failure.tolist(),
                           diagnostic_failure_mask=diagnostic.tolist(),state_count=16,
                           invalid_state_count=2,valid_state_count=14,
                           maps=[dict(path=p['path'],hashes={'.dat':p['sha256']}) for p in packed_maps])
        self.validation.write_text(json.dumps(self.report))

    def queries(self):
        n_h = np.array([1.,1.,100.,1.,1.])
        rho = n_h*CONSTANTS['hydrogen_mass_g']/QUOKKA_MASS_FRACTIONS['X']
        u = np.full(5,np.nan)
        u[-1] = rho[-1]*CONSTANTS['boltzmann_erg_K']*1e4/((5/3-1)*CONSTANTS['hydrogen_mass_g']*.62)
        return prepare_cloudy_cell_queries(rho,[1e18,1e21,1e19,1e18,1e18],
            [100.,100.,100.,100.,1e4],u,[100.,100.,100.,np.nan,np.nan],
            [1.3,1.3,1.3,np.nan,np.nan],authorized_excluded=[False,False,False,True,False],**CONSTANTS)

    def test_failure_sources_zero_weight_and_domain_are_separate(self):
        raw, checked = validated_coverage_lookup(self.table,self.validation)
        flags = classify_cell_queries(self.queries(),raw,checked)
        np.testing.assert_array_equal(flags['raw_failure:hi21'],[True,False,False,False,False])
        np.testing.assert_array_equal(flags['unavailable:hi21'],[True,True,False,False,False])
        np.testing.assert_array_equal(flags['outside_density'],[False,False,True,False,False])
        np.testing.assert_array_equal(flags['query_available_all_lines'],[False,False,False,False,True])

    def test_mass_fractions_include_excluded_cells_in_denominator(self):
        raw,checked = validated_coverage_lookup(self.table,self.validation)
        queries = self.queries()
        flags = classify_cell_queries(queries,raw,checked)
        totals = CellCoverageTotals()
        for sl in (slice(0,2),slice(2,5)):
            totals.add({k:v[sl] for k,v in flags.items()},queries.state.cold_mask[sl],
                       np.array([1.,10.,100.,1000.,10000.])[sl])
        result = totals.result()
        self.assertEqual(result['all']['cells'],5)
        self.assertEqual(result['all']['mass_g'],11111)
        self.assertEqual(result['cold']['mass_g'],1111)
        self.assertAlmostEqual(result['all']['flags']['unavailable_any_line']['mass_fraction'],11/11111)
        self.assertAlmostEqual(result['cold']['flags']['authorized_excluded']['mass_fraction'],1000/1111)
        self.assertEqual(result['hot']['flags']['query_available_all_lines']['cells'],1)
        self.assertEqual(totals.result(),result)

    def test_checked_view_does_not_change_source_or_raw_mask(self):
        original = self.table.read_bytes()
        raw,checked = validated_coverage_lookup(self.table,self.validation)
        self.assertEqual(int(raw.failure_mask.sum()),1)
        self.assertEqual(int(checked.failure_mask.sum()),2)
        self.assertGreater(raw.emissivity_per_nH2[0,1,0,0,1],0)
        self.assertEqual(checked.emissivity_per_nH2[0,1,0,0,1],0)
        self.assertEqual(self.table.read_bytes(),original)

    def test_validator_provenance_and_totals_cannot_be_ignored(self):
        for key,value in (('manifest_sha256','wrong'),('state_count',15),
                          ('global_issues',['changed SED']),('execution_status','running')):
            with self.subTest(key=key):
                report = dict(self.report,**{key:value})
                self.validation.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    validated_coverage_lookup(self.table,self.validation)

    def test_changed_map_bytes_are_rejected_even_with_same_manifest(self):
        self.report['maps'][0]['hashes']['.dat']='changed-map'
        self.validation.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError,'raw-map contents'):
            validated_coverage_lookup(self.table,self.validation)

    def test_invalid_mass_and_changed_flags_raise(self):
        totals=CellCoverageTotals()
        with self.assertRaises(ValueError):totals.add({'x':np.array([True])},[True],[np.nan])
        totals.add({'x':np.array([True])},[True],[1.])
        with self.assertRaises(ValueError):totals.add({'y':np.array([True])},[True],[1.])

    def test_cli_rejects_incomplete_validation_before_snapshot_or_output(self):
        from scripts import check_cloudy_model_depth_coverage as scanner
        self.report['execution_status']='running'
        self.validation.write_text(json.dumps(self.report))
        out=Path(self.temp.name)/'new_coverage'
        argv=['coverage','--cloudy-table',str(self.table),'--cloudy-validation',str(self.validation),
              '--checkpoint-dir',str(Path(self.temp.name)/'missing_checkpoint'),
              '--checkpoint-validation',str(Path(self.temp.name)/'missing_checkpoint_validation.json'),
              '--despotic-table',str(Path(self.temp.name)/'missing_despotic.npz'),
              '--dataset',str(Path(self.temp.name)/'plt0655228'),'--output-dir',str(out)]
        with patch.object(scanner.sys,'argv',argv), patch.object(scanner,'load_table') as load:
            with self.assertRaisesRegex(ValueError,'incomplete'):
                scanner.main()
        load.assert_not_called()
        self.assertFalse(out.exists())


if __name__ == '__main__':unittest.main()
