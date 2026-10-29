"""Known-covariance diagnostic invariants with fixed nonrandom small designs."""
import copy
import math
import unittest
from src.tutoring_statistics import known_covariance as p
from src.tutoring_statistics import independent_known_covariance as r


def fixture(label='s5_l2_rep1'):
    slopes={'C1':dict(A=.25,B=-.25),'C2':dict(A=.55,B=-.55),'C3':dict(A=.55,B=-.55)}
    if label=='s1_rep1': slopes={a:dict(A=.25,B=-.25) for a in p.ARMS}
    case=dict(parameters=dict(alpha=60.,sdU=8.,sigmaByArm={a:12. for a in p.ARMS},slopes=slopes))
    rows=[]
    for ai,arm in enumerate(p.ARMS):
        for pi in range(2):
            for ti in range(10):
                a=float(ti-4+pi);b=float((3*ti+pi)%7-3)
                rows.append(dict(participant=arm+str(pi),condition=arm,need_A=a,need_B=b,
                    y=60.+slopes[arm]['A']*a+slopes[arm]['B']*b+(ti%3-1)*.2+ai*.1))
    return dict(model='H10',reference='C2',rows=rows),case


def close_tree(left,right):
    if isinstance(left,dict):
        for key in left: close_tree(left[key],right[key])
    elif isinstance(left,list):
        if len(left)!=len(right): raise AssertionError('Array mismatch')
        for a,b in zip(left,right):close_tree(a,b)
    elif type(left) in (int,float) and type(left) is not bool:
        if not math.isclose(left,right,abs_tol=2e-9,rel_tol=2e-8): raise AssertionError((left,right))
    elif left!=right:raise AssertionError((left,right))


class KnownOmegaTests(unittest.TestCase):
    def test_two_fixed_targets_information_matches_independent_qr(self):
        for label in p.REFERENCES:
            actual,case=fixture(label)
            first=p.known_gls(label,actual,case);second=r.reconstruct(label,actual,case)
            for key in ('coefficientNames','beta','covariance','actualK','rowCount','clusterCount'):
                close_tree(first[key],second[key])
            for key,row in first['references'].items():
                for field in ('L','truth','estimate','SE','z','CI','covered'):
                    close_tree(row[field],second['references'][key][field])
            self.assertEqual(first['rowCount'],60)
            self.assertEqual(first['actualK'],dict(C1=2,C2=2,C3=2))

    def test_invalid_population_rows_target_and_nonfinite_are_rejected(self):
        actual,case=fixture()
        for function in (p.known_gls,r.reconstruct):
            with self.assertRaises(ValueError):function('not_registered',actual,case)
            bad=copy.deepcopy(actual);bad['rows'].pop()
            with self.assertRaises(ValueError):function('s5_l2_rep1',bad,case)
            for value in (float('nan'),float('inf'),True):
                bad=copy.deepcopy(actual);bad['rows'][0]['y']=value
                with self.assertRaises(ValueError):function('s5_l2_rep1',bad,case)
            bad=copy.deepcopy(actual);bad['rows'][0]['condition']='C2'
            with self.assertRaises(ValueError):function('s5_l2_rep1',bad,case)

    def test_rank_and_known_covariance_have_no_silent_fallback(self):
        actual,case=fixture()
        for row in actual['rows']:row['need_A']=row['need_B']=0.
        for function in (p.known_gls,r.reconstruct):
            with self.assertRaises(ValueError):function('s5_l2_rep1',actual,case)
            valid,changed=fixture();changed['parameters']['sigmaByArm']['C1']=13.
            with self.assertRaises(ValueError):function('s5_l2_rep1',valid,changed)



if __name__ == '__main__': unittest.main()
