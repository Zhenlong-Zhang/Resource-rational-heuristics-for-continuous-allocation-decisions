"""Load shared geometry under either installed or repository package names.

The installed entry imports design_validation directly. Its sibling analysis
kernels use parent-relative imports, so give them a small common parent without
executing the broad repository src initializer or duplicating numerical caches.
"""
import importlib
from pathlib import Path
import sys
import types

from . import personal_time_inputs  # Load the shared finite kernels before aliasing.


def _analysis(name):
    package = __package__
    if '.' in package:
        parent = package.rsplit('.', 1)[0]
    else:
        parent = '_personal_time_sensitivity_kernels'
        if parent not in sys.modules:
            namespace = types.ModuleType(parent)
            namespace.__path__ = [str(Path(__file__).resolve().parents[1])]
            sys.modules[parent] = namespace
        for key, module in list(sys.modules.items()):
            if key == package or key.startswith(package + '.'):
                alias = parent + '.' + key
                if alias in sys.modules and sys.modules[alias] is not module:
                    raise ValueError('Conflicting shared numerical module: ' + alias)
                sys.modules[alias] = module
    return importlib.import_module(parent + '.tutoring_analysis.' + name)


geometry = _analysis('expected_outcomes')
nominal = _analysis('nominal_inference')
