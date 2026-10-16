"""Analytic planning identities; no Monte Carlo or human power estimate."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutoring_analysis.inference import CORE
from tutoring_analysis.planning import (covariance_matrix,joint_bounds,moment_check,one_sample_power,
                                       planning_report,summary_geometry,usable_people,weighted_contrast_power)


class PlanningTests(unittest.TestCase):
    def test_t_nct_oracles_and_null_opposite_sides(self):
        from scipy.stats import t,nct
        self.assertAlmostEqual(t.sf(2.5,7),.020496109292876437,15)
        self.assertAlmostEqual(nct.sf(2.5,7,2),.36771597992466903,15)
        zero=one_sample_power(0.,1.,20,.05)
        self.assertAlmostEqual(zero["power"],.05,12)
        positive=one_sample_power(.3,1.,40,.05);negative=one_sample_power(-.3,1.,40,.05)
        self.assertGreater(positive["power"],zero["power"]);self.assertLess(negative["power"],zero["power"])
        self.assertAlmostEqual(negative["power"],one_sample_power(.3,1.,40,.05,"less")["power"])
        self.assertIsNone(one_sample_power(.3,0.,40)["power"])
        self.assertEqual(one_sample_power(.3,1.,1)["status"],"unavailable")

    def test_weighted_welch_power_arithmetic(self):
        groups={"C2":{"mean":.2,"variance":.04,"n":20,"bounds":[0,1]},"C3":{"mean":.4,"variance":.06,"n":40,"bounds":[0,1]}}
        r=weighted_contrast_power(groups,{"C3":1.,"C2":-1.})
        self.assertAlmostEqual(r["effect"],.2);self.assertAlmostEqual(r["standardError"]**2,.04/20+.06/40)
        self.assertAlmostEqual(r["df"],(.04/20+.06/40)**2/((.04/20)**2/19+(.06/40)**2/39))
        self.assertTrue(r["approximate"]);self.assertFalse(r["exactJointPower"])
        self.assertEqual(weighted_contrast_power(groups,{"C4":1.})["status"],"unavailable")

    def test_mean_icc_identity_and_dose(self):
        for dose in (10,30):
            cov=covariance_matrix(dose,.04,.3)
            g=summary_geometry(np.full(dose,.3),cov,eligibility="complete_ten" if dose==10 else "hypothetical_future_profile",endpoint_bounds=[0,1])
            self.assertAlmostEqual(g["variance"],.04*(.3+.7/dose),14)
        self.assertLess(summary_geometry(np.full(30,.3),covariance_matrix(30,.04,.3))["variance"],summary_geometry(np.full(10,.3),covariance_matrix(10,.04,.3))["variance"])

    def test_slope_sxx_random_slope_intercept_cancellation(self):
        x=np.linspace(-.2,.2,10);y=.5+.3*x
        for rho in (0.,.3,.6):
            cov=covariance_matrix(10,.01,rho);g=summary_geometry(y,cov,x,random_slope_variance=.02,eligibility="complete_ten",endpoint_bounds=[0,1])
            sxx=np.sum((x-x.mean())**2)
            self.assertAlmostEqual(g["effect"],.3,13);self.assertAlmostEqual(g["variance"],.02+.01*(1-rho)/sxx,13)
            self.assertAlmostEqual(g["allocationChangePPOverSpan"],12.,12)
        pure=np.ones((10,10))*.01
        g=summary_geometry(y,pure,x,random_slope_variance=.02)
        self.assertAlmostEqual(g["variance"],.02,13)
        self.assertEqual(summary_geometry(y,np.eye(10),np.zeros(10))["status"],"unavailable")

    def test_ar1_learning_retained_projection(self):
        x=np.linspace(-.2,.2,10);time=np.linspace(-1,1,10)
        y=.5+(.2+.1*time)*x+.03*time
        cov=covariance_matrix(10,.01,.6,"ar1")
        g=summary_geometry(y,cov,x,retained=list(range(8)))
        a=x[:8]-x[:8].mean();weights=a/(a@a)
        self.assertAlmostEqual(g["effect"],weights@y[:8],14)
        self.assertAlmostEqual(g["variance"],weights@cov[:8,:8]@weights,14)
        self.assertNotAlmostEqual(g["effect"],.2)
        self.assertEqual(summary_geometry(y,cov,x,retained=list(range(8)),eligibility="complete_ten")["status"],"unavailable")
        self.assertEqual(summary_geometry(np.zeros(30),np.eye(30),eligibility="complete_ten")["status"],"unavailable")

    def test_covariance_and_retention_domains(self):
        y=[.4,.5,.6]
        for cov in ([[1,1],[1,1]],[[1,2,0],[0,1,0],[0,0,1]],[[1,2,0],[2,1,0],[0,0,1]],[[float("nan"),0,0],[0,1,0],[0,0,1]]):
            self.assertEqual(summary_geometry(y,cov)["status"],"unavailable")
        for retained in ([],[0,0],[3],[.5]):self.assertEqual(summary_geometry(y,np.eye(3),retained=retained)["status"],"unavailable")
        x=np.asarray([-1e-5,0,1e-5]);u=x/np.linalg.norm(x)
        c=1e-6*(np.eye(3)-np.outer(u,u))-5e-11*np.outer(u,u)
        self.assertEqual(summary_geometry(.5+.2*x,c,x,endpoint_bounds=[0,1])["status"],"unavailable")
        self.assertEqual(summary_geometry(.5+.2*x,c,x,random_slope_variance=1.,endpoint_bounds=[0,1])["status"],"unavailable")
        for rho in (-.9,1.1):
            with self.assertRaises(ValueError):covariance_matrix(10,.01,rho)
        with self.assertRaises(ValueError):covariance_matrix(10,.01,1.,"ar1")

    def test_bounded_moments_fixed_spacing_guards(self):
        for mean,var,bounds in ((1.1,.01,[0,1]),(.4,.25,[0,1]),(.5,.001,[-.5,.5]),(2.,9.,[-1.7,1.7]),(1.6,1.,[-1.7,1.7])):
            with self.assertRaises(ValueError):moment_check(mean,var,bounds)
        self.assertFalse(moment_check(.3,.04,[0,1])["jointBoundedGenerativeFeasibilityProven"])
        self.assertFalse(moment_check(2.,9.,[-20.,20.])["jointBoundedGenerativeFeasibilityProven"])
        self.assertEqual(summary_geometry([.1,.2,1.1],np.eye(3)*.001,endpoint_bounds=[0,1])["status"],"unavailable")

    def test_primary_missingness_and_available_history_not_full(self):
        r=usable_people(40,.8,.98,10,"complete_ten")
        self.assertAlmostEqual(r["expectedUsablePeople"],40*.8*.98**10)
        self.assertEqual(r["planningCountFloor"],math.floor(40*.8*.98**10))
        self.assertEqual(usable_people(20,trial_retention=0.,eligibility="available_history")["planningCountFloor"],0)
        with self.assertRaises(ValueError):usable_people(40,dose=30,eligibility="complete_ten")

    def test_joint_bounds_all_upper_not_conservative_min(self):
        slots={d[0]:{"status":"available","firstThresholdPower":.2,"rawAlphaPower":.8} for d in CORE}
        r=joint_bounds(slots)
        self.assertEqual(r["anyCoreLower"],.2);self.assertEqual(r["anyCoreUpper"],1.)
        self.assertEqual(r["allCoreLower"],0.);self.assertEqual(r["allCoreUpper"],.8)
        slots[CORE[0][0]]={"status":"unavailable","firstThresholdPower":None,"rawAlphaPower":None}
        self.assertEqual(joint_bounds(slots)["allCoreUpper"],0.)
        with self.assertRaises(ValueError):joint_bounds({})

    def test_full_config_routes_and_labels(self):
        config=json.loads((Path(__file__).resolve().parents[1]/"configs/tutoring_power_sensitivity_v01.json").read_text())
        report=planning_report(config);s={r["id"]:r for r in report["scenarios"]}
        self.assertFalse(report["humanVarianceEstimated"]);self.assertFalse(report["statisticalCalibrationPerformed"])
        self.assertEqual(len(s),28)
        self.assertLess(s["trial_loss_complete10"]["retention"]["C2"]["planningCountFloor"],40)
        self.assertEqual(s["constant_unavailable"]["slopeGeometry"]["C2"]["status"],"unavailable")
        self.assertEqual(s["impossible_old_fixed_spacing_moments"]["coreMarginals"][5]["first"]["status"],"unavailable")
        self.assertLess(s["trial_covariance_dose30"]["meanGeometry"]["lower"]["variance"],s["trial_covariance_means"]["meanGeometry"]["lower"]["variance"])
        self.assertNotEqual(s["learning_early8"]["meanGeometry"]["lower"]["effect"],s["learning"]["meanGeometry"]["lower"]["effect"])


if __name__=="__main__":unittest.main()
