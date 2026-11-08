"""Generate portable C locally or build all package extensions offline."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import sysconfig


from build_support import MODULES, OPENMP_MODULES, compiler_command, extension_flags, probe_openmp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cython-root', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--from-c', action='store_true', help='Compile supplied C; never import Cython')
    modes.add_argument('--generate-c-only', action='store_true')
    args = parser.parse_args()
    if not args.from_c:
        if args.cython_root is not None:
            sys.path.insert(0, str(args.cython_root.resolve(strict=True)))
        from Cython.Compiler.Main import compile as compile_cython
    import numpy as np
    root = Path(__file__).resolve().parent
    os.chdir(root)
    receipts = []
    openmp = probe_openmp() if not args.generate_c_only else None
    for module in MODULES:
        source = root / 'src' / (module.replace('.', '/') + '.pyx')
        c_file = source.with_suffix('.c')
        if not args.from_c:
            generated = compile_cython(str(source.relative_to(root)), full_module_name=module, language_level=3)
            if generated.num_errors:
                raise RuntimeError('Could not generate ' + module)
        if not c_file.is_file():
            raise FileNotFoundError('Generate portable C before offline compilation: ' + str(c_file))
        receipt = {
            'module': module,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'generated_c_sha256': hashlib.sha256(c_file.read_bytes()).hexdigest(),
        }
        if not args.generate_c_only:
            binary = source.parent / (source.stem + sysconfig.get_config_var('EXT_SUFFIX'))
            temporary = binary.with_name(binary.name + '.new')
            link = ['-bundle', '-undefined', 'dynamic_lookup'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
            command = compiler_command() + link
            command += shlex.split(sysconfig.get_config_var('CFLAGS') or '')
            command += shlex.split(sysconfig.get_config_var('CCSHARED') or '')
            compile_flags, link_flags = extension_flags(
                module, openmp_supported=openmp['supported'],
            )
            command += compile_flags
            command += [
                '-I' + sysconfig.get_path('include'),
                '-I' + np.get_include(),
                str(c_file),
                '-o', str(temporary),
            ]
            command += link_flags
            subprocess.run(command, check=True)
            temporary.replace(binary)
            receipt['command'] = command
            receipt['openmp_probe'] = openmp
            receipt['openmp_enabled'] = module in OPENMP_MODULES and openmp['supported']
            receipt['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
        receipts.append(receipt)
    name = 'portable_c_manifest.json' if args.generate_c_only else 'build_receipt.json'
    (root / name).write_text(json.dumps(receipts, indent=2) + '\n')
    print(json.dumps({'complete': True, 'extensions': len(receipts),
                      'generated_c_only': args.generate_c_only, 'from_c': args.from_c}))


if __name__ == '__main__':
    main()
