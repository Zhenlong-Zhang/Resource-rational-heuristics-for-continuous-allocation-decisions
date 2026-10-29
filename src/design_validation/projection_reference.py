"""Isolated numerical reference for response-independent initial-slope plans.

Only projection preparation uses AVX dispatch. The parent keeps its own runtime
for policy decisions and outcome calculations.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys


@lru_cache(maxsize=1)
def frozen_plans(predictors):
    """Return plans and numerical identity without changing the parent runtime."""
    environment = {**os.environ, "MKL_CBWR": "AVX"}
    for name in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
        input=json.dumps(predictors, allow_nan=False), capture_output=True,
        text=True, env=environment, check=True)
    value = json.loads(result.stdout)
    if value["runtime"]["mklRequested"] != "AVX" or value["runtime"]["mklEffective"] != "avx":
        raise ValueError("The reference projection requires effective MKL AVX dispatch")
    return value


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import mkl
    import numpy as np
    import scipy
    from src.design_validation import cost_aware_aggregation as A
    from src.design_validation.cost_aware_campaign import numerical_runtime
    from src.tutoring_analysis.nominal_inference import _projection_plan

    if (np.__version__, scipy.__version__, mkl.__version__) != ("1.23.5", "1.10.0", "2.4.0"):
        raise ValueError("Use configs/native_reference_environment.yml for frozen projection preparation")
    runtime = numerical_runtime()
    if not any(r["api"] == "mkl" and r["version"].startswith("2021.4")
               for r in runtime["loadedLibraries"]):
        raise ValueError("The frozen projection reference requires MKL 2021.4")
    runtime.update(mklRequested=os.environ.get("MKL_CBWR"), mklEffective=mkl.cbwr_get(),
                   mklService=mkl.__version__)
    if runtime["mklRequested"] != "AVX" or runtime["mklEffective"] != "avx":
        raise ValueError("MKL AVX is not effective on this reference platform")
    plans = []
    for values in json.load(sys.stdin):
        if len(values) != 10:
            raise ValueError("Ten ordered initial predictors required")
        original = _projection_plan([dict(x0=x, share=.5) for x in values], "x0")
        frozen = None
        if original["status"] == "available":
            payload = {"schema": "r9-frozen-projection-plan-v1",
                "predictorBinary64": [A.bits(x) for x in values],
                "weightBinary64": [A.bits(float(w)) for w in original["weights"]],
                "operationOrder": "ordered_float64_multiply_then_add_mainIndex_1_to_10",
                "diagnostics": {k: v for k, v in original.items() if k != "weights"}}
            frozen = {**payload, "planHash": A.digest(payload)}
        plans.append(frozen)
    print(json.dumps(dict(plans=plans, runtime=runtime), allow_nan=False))


if __name__ == "__main__":
    main()
