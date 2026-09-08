"""Install checked fixed-temperature GOW integration for DESPOTIC tables."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distribution
from pathlib import Path


# Official DESPOTIC ed18e5669adb7306f795a3d30d8919995793bc61.
ORIGINAL_SHA256 = "dd8b35a1bd8ea01f08a6884b2c34dd879d7c3deaa8bffcff6b320a994e9ae479"
PATCHED_SHA256 = "c1376df46886bd8f7c2638df7323125a110631c17b3b9ccf8dc74d9aea9d7b09"
BEFORE = b"""        xOut = odeint(cloud.chemnetwork.dxdt, cloud.chemnetwork.x,
                      tOut1)
"""
AFTER = b"""        from .GOW import GOW
        if not isinstance(cloud.chemnetwork, GOW):
            xOut = odeint(cloud.chemnetwork.dxdt, cloud.chemnetwork.x,
                          tOut1)
        else:
            import warnings
            from scipy.integrate import ODEintWarning

            # Keep successful integrations at their original physical times.
            # On failure, discard all output and restart the same GOW state
            # in local time. Restore physical time on every RHS evaluation.
            initial = np.array(cloud.chemnetwork.x, dtype=float, copy=True)
            times = np.array(tOut1, dtype=float, copy=True)
            offsets = [0.0]
            if times[0] != 0.0:
                offsets.append(float(times[0]))
            if not hasattr(cloud, '_fixed_chemistry_attempts'):
                cloud._fixed_chemistry_attempts = []
            for offset in offsets:
                if offset == 0.0:
                    rhs = cloud.chemnetwork.dxdt
                else:
                    def rhs(state, local_time):
                        return cloud.chemnetwork.dxdt(state, local_time + offset)
                with warnings.catch_warnings():
                    # Failure is handled explicitly below, including retries.
                    warnings.simplefilter('ignore', ODEintWarning)
                    xOut, status = odeint(
                        rhs, initial.copy(), times - offset,
                        rtol=1e-8, atol=1e-12, mxstep=10000, full_output=True)
                valid = (
                    status['message'] == 'Integration successful.'
                    and xOut.shape == (len(times), len(initial))
                    and np.isfinite(xOut).all())
                cloud._fixed_chemistry_attempts.append({
                    'offset_s': offset, 'success': bool(valid),
                    'message': status['message']})
                if valid:
                    break
            else:
                raise despoticError(
                    'Incomplete GOW chemical integration; all same-state '
                    'attempts rejected: ' + status['message'])
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without changing the package.")
    args = parser.parse_args()
    path = Path(distribution("despotic").locate_file("despotic/chemistry/chemEvol.py"))
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest == PATCHED_SHA256:
        print(f"Checked GOW integration is installed: {path}")
        return
    if digest != ORIGINAL_SHA256:
        raise SystemExit(f"Unrecognized chemEvol.py ({digest}); review before patching: {path}")
    if args.check:
        raise SystemExit("Checked GOW integration is not installed.")
    if original.count(BEFORE) != 1:
        raise SystemExit("Expected one fixed-temperature integration call.")
    updated = original.replace(BEFORE, AFTER)
    if hashlib.sha256(updated).hexdigest() != PATCHED_SHA256:
        raise SystemExit("Patched source does not match the reviewed change.")
    path.write_bytes(updated)
    print(f"Installed checked GOW integration: {path}")


if __name__ == "__main__":
    main()
