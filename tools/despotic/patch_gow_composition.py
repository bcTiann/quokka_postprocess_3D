"""Apply the composition refresh fix to the pinned DESPOTIC installation."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distribution
from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from quokka2s.despotic.solver_settings import (
    GOW_ORIGINAL_SOURCE_SHA256,
    GOW_REFRESH_SOURCE_SHA256,
)

ANCHOR = b"\n        # Make a case-insensitive version of the emitter list for\n"
INSERTION = (
    b"\n        # Refresh derived quantities after updating the chemical composition.\n"
    b"        self.cloud.comp.computeDerived(self.cloud.nH)\n"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without modifying the package.")
    args = parser.parse_args()
    path = Path(distribution("despotic").locate_file("despotic/chemistry/GOW.py"))
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest == GOW_REFRESH_SOURCE_SHA256:
        print(f"DESPOTIC GOW derived-quantity refresh is installed: {path}")
        return
    if digest != GOW_ORIGINAL_SOURCE_SHA256:
        raise SystemExit(f"Unrecognized GOW.py ({digest}); review this version before patching: {path}")
    if args.check:
        raise SystemExit("DESPOTIC GOW derived-quantity refresh is not installed.")
    if original.count(ANCHOR) != 1:
        raise SystemExit("Expected exactly one insertion point in GOW.applyAbundances.")
    updated = original.replace(ANCHOR, INSERTION + ANCHOR)
    if hashlib.sha256(updated).hexdigest() != GOW_REFRESH_SOURCE_SHA256:
        raise SystemExit("Patched source does not match the reviewed change.")
    path.write_bytes(updated)
    print(f"Installed DESPOTIC GOW derived-quantity refresh: {path}")


if __name__ == "__main__":
    main()
