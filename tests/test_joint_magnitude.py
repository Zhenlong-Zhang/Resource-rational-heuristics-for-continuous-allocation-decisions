"""Joint-family math and guard fixtures, not fresh calibration evidence."""
import math
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tutoring_statistics.joint_magnitude import compose_saved_method, METHOD_ID, COMPARISON_IDS


def fixture():
    matrix=[[float(i==j) for j in range(8)] for i in range(8)]
    refs={}
    for j,key in enumerate(COMPARISON_IDS):
        matrix[2*j][2*j+1]=matrix[2*j+1][2*j]=.5
        for kind,estimate in (('D',-3.-j/10),('S',3.+j/10)):
            label=key+':'+kind
            refs[label]=dict(linearID=label,status='available',reason=None,
                            estimate=estimate,SE=1.,df=160.)
    return dict(status='complete',contrastCovariance=matrix,references=refs)


def make_unavailable(value,key,status):
    for kind in ('D','S'):
        value['references'][key+':'+kind].update(status=status,reason='fixed-test-reason',estimate=None,SE=None,df=None)


class JointAdapterTests(unittest.TestCase):
    def test_direct_normal_and_strict_holm(self):
        result=compose_saved_method(fixture())
        raw=[math.erfc((3.+j/10)/math.sqrt(2))/2 for j in range(4)]
        order=sorted(range(4),key=raw.__getitem__)
        expected={order[k]:max((4-i)*raw[order[i]] for i in range(k+1)) for k in range(4)}
        for j,key in enumerate(COMPARISON_IDS):
            self.assertAlmostEqual(result['tests'][key]['pJoint'],raw[j],places=14)
            self.assertAlmostEqual(result['tests'][key]['holmP'],expected[j],places=14)
        self.assertTrue(result['anyRejects'])
        self.assertEqual(result['methodID'],METHOD_ID)
        # Exact probability fixture tests the decision boundary; no approximate
        # inverse normal quantile is represented as an exact alpha boundary.
        with patch('src.tutoring_statistics.joint_magnitude.gaussian_comparison',
                   return_value={'pGaussianReference':.0125}):
            boundary=compose_saved_method(fixture())
        self.assertFalse(boundary['anyRejects'])
        self.assertTrue(all(t['holmP']==.05 for t in boundary['tests'].values()))

    def test_partial_availability_and_original_guard(self):
        value=fixture()
        for key in COMPARISON_IDS[1:]: make_unavailable(value,key,'statistical_unavailable')
        result=compose_saved_method(value)
        self.assertEqual(result['availableCount'],1)
        self.assertTrue(result['technicalCompleted'] and result['anyRejects'])
        self.assertTrue(all(result['tests'][k]['rejected'] is False for k in COMPARISON_IDS[1:]))
        make_unavailable(value,COMPARISON_IDS[0],'statistical_unavailable')
        value.update(status='statistical_unavailable',contrastCovariance=None)
        result=compose_saved_method(value)
        self.assertTrue(result['technicalCompleted'])
        self.assertFalse(result['anyRejects'])
        self.assertEqual(result['availableCount'],0)

    def test_technical_unknown_and_model_based_label(self):
        value=fixture();make_unavailable(value,COMPARISON_IDS[2],'technical_unresolved')
        result=compose_saved_method(value)
        self.assertFalse(result['technicalCompleted'])
        self.assertIsNone(result['anyRejects'])
        self.assertTrue(all(t['rejected'] is None for t in result['tests'].values()))
        base=compose_saved_method(fixture(),covariance_source='model_based_diagnostic')
        self.assertNotEqual(base['methodID'],METHOD_ID)
        self.assertFalse(base['scientificAdoption'] or base['originalConfidenceIntervalsChanged'])

    def test_changed_labels_variance_and_invalid_scalars(self):
        cases=[]
        value=fixture();value['references']['C1-C2:A:D']['linearID']='wrong';cases.append(value)
        value=fixture();value['references']['C1-C2:A:D']['SE']=2.;cases.append(value)
        value=fixture();value['contrastCovariance'][0][1]=value['contrastCovariance'][1][0]=1.1;cases.append(value)
        value=fixture();value['contrastCovariance'][0][0]=float('nan');cases.append(value)
        value=fixture();value['references']['C1-C2:A:D']['estimate']=True;cases.append(value)
        for value in cases:
            with self.assertRaises(ValueError): compose_saved_method(value)


if __name__ == '__main__': unittest.main()
