"""Reproduce operation-cost, five-rule and fixed ask-once result families."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','produce','check','tables','all'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/operation_costs.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditions',default='C1,C2,C3,C4,C5',help='Original variant0/main1 roots')
    p.add_argument('--families',default='noask,literal,ask_once')
    p.add_argument('--directions',default='A,B',help='Fixed physical directions; never an adaptive selector')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--node',default='node')
    a=p.parse_args()
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
    sys.path.insert(0,str(ROOT))
    from threadpoolctl import threadpool_limits
    from src.design_validation.operation_costs import inputs as I, reproduction as R
    families=tuple(a.families.split(','));directions=tuple(a.directions.split(','))
    phases=('prepare','produce','check','tables') if a.phase=='all' else (a.phase,)
    answers=[]
    with threadpool_limits(limits=1):
        for phase in phases:
            if phase=='prepare':
                I.prepare(a.output,a.config,a.conditions.split(','),node=a.node)
                result=dict(phase=phase,prepared=True)
            elif phase=='tables':result=R.tables(a.output,a.config,families,directions)
            else:result=R.process(a.output,a.config,phase,families,directions,a.workers,
                                  node=a.node if phase=='check' else None)
            answers.append(result);print(json.dumps(result,sort_keys=True),flush=True)
    return answers


if __name__=='__main__':main()
