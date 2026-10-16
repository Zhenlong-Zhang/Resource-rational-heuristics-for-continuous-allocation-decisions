"""Portable prospective-method functionality, NOT statistical calibration."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutoring_analysis.inference import (CORE, contrast, fit, multiplicity, participant_slope,
                                        participant_values, synthetic_example_rows, validate_rows)


class InferenceTests(unittest.TestCase):
    def setUp(self):self.rows=synthetic_example_rows(6)

    def test_known_one_sample_arithmetic(self):
        rows=[{"conditionId":"C2","status":"available","value":v} for v in [1.,2.,3.,4.]]
        r=contrast(rows,{"C2":1.})
        self.assertAlmostEqual(r["estimate"],2.5);self.assertAlmostEqual(r["standardError"],math.sqrt(5/12))
        self.assertEqual(r["df"],3)
        from scipy.stats import ttest_1samp
        oracle=ttest_1samp([1.,2.,3.,4.],0.,alternative="greater")
        self.assertAlmostEqual(r["pValue"],oracle.pvalue,14)
        self.assertEqual(contrast(rows,{"C2":1.},"less")["pValue"],1-r["pValue"])

    def test_weighted_independent_contrast(self):
        rows=[{"conditionId":c,"status":"available","value":v} for c,values in
              (("C2",[1.,2.,4.]),("C3",[5.,6.,7.,8.]),("C4",[2.,3.,4.])) for v in values]
        weights={"C2":-.5,"C3":1.,"C4":-.5};r=contrast(rows,weights)
        means={c:np.mean([q["value"] for q in rows if q["conditionId"]==c]) for c in weights}
        pieces=[weights[c]**2*np.var([q["value"] for q in rows if q["conditionId"]==c],ddof=1)/sum(q["conditionId"]==c for q in rows) for c in weights]
        self.assertAlmostEqual(r["estimate"],sum(weights[c]*means[c] for c in weights))
        self.assertAlmostEqual(r["standardError"]**2,sum(pieces))
        self.assertAlmostEqual(r["df"],sum(pieces)**2/sum(p**2/(sum(q["conditionId"]==c for q in rows)-1) for c,p in zip(weights,pieces)))
        from scipy.stats import ttest_ind
        pair=contrast(rows,{"C3":1.,"C2":-1.})
        self.assertAlmostEqual(pair["pValue"],ttest_ind([5.,6.,7.,8.],[1.,2.,4.],equal_var=False,alternative="greater").pvalue)

    def test_exact_slope_raw_units_and_adjustment(self):
        own=copy.deepcopy([r for r in self.rows if r["participantId"]=="synthetic-C2-00" and r["stage"]=="main"])
        for r in own:r["shareA"]=.45+.3*r["xInitial"]
        initial=participant_slope(own,"xInitial");raw=participant_slope(own,"initialNeedGapRaw")
        self.assertAlmostEqual(initial["value"],.3,13);self.assertAlmostEqual(raw["value"],.3/120,13)
        self.assertAlmostEqual(initial["allocationChangeOverSpan"],.3*.36,13)
        self.assertEqual(participant_slope(own,"xInitial","order_only")["status"],"unavailable") # linear schedule/order collinearity
        c3=[r for r in self.rows if r["conditionId"]=="C3" and r["participantId"]=="synthetic-C3-00" and r["stage"]=="main"]
        self.assertEqual(participant_slope(c3,"xInitial","order_recipe")["reason"],"rank_or_residual_df_failure_no_fallback")
        # Independent rank-valid fixed sensitivity recovers its known x coefficient.
        for i,r in enumerate(own):
            r["xInitial"]=(-.2,.1,.15,-.1,.05)[i%5]
            r["scale"]=(.8,1.,1.2)[i%3];r["gapOffset"]=(-12,0,12)[(i//2)%3]
            r["shareA"]=.5+.2*r["xInitial"]+.001*r["stageIndex"]+.01*r["scale"]+.0001*r["gapOffset"]
        self.assertAlmostEqual(participant_slope(own,"xInitial","order_recipe")["value"],.2,12)

    def test_equal_person_not_trial_weight_and_complete_indices(self):
        own=[r for r in self.rows if r["conditionId"]=="C2" and r["participantId"] in ("synthetic-C2-00","synthetic-C2-01") and r["stage"]=="main"]
        own=copy.deepcopy([r for r in own if r["participantId"].endswith("00") or r["stageIndex"]==1])
        for r in own:r["EOResemblance"]=0. if r["participantId"].endswith("00") else 10.
        values=participant_values(own,"EOResemblance","available_history")
        self.assertEqual(contrast(values,{"C2":1.})["estimate"],5.)
        self.assertNotEqual(contrast(values,{"C2":1.})["estimate"],10/11)
        primary=participant_values(own,"EOResemblance")
        self.assertEqual(sum(r["status"]=="available" for r in primary),1)
        fake=copy.deepcopy([r for r in own if r["participantId"].endswith("00")])
        fake[-1]["stageIndex"]=9
        self.assertFalse(participant_values(fake,"EOResemblance")[0]["completeTen"])

    def test_provenance_identity_and_corrupt_fields(self):
        validate_rows(self.rows,"synthetic_design","synthetic_validation")
        for edit in (lambda r:r.update(dataRole="software_check"),lambda r:r.update(dataRole="unclassified"),
                     lambda r:r.update(participantId=" "),lambda r:r.update(trialId="main-02"),
                     lambda r:r.update(stageIndex=1.1),lambda r:r.update(xInitial=float("nan")),
                     lambda r:r.update(anyQuestion=.5),lambda r:r.update(questions=21),
                     lambda r:r.update(terminalTie=True,lowerNeedDirectional=.01),
                     lambda r:r.update(distancePP_equal_split=None),
                     lambda r:r.update(EOResemblance=None)):
            bad=copy.deepcopy(self.rows);edit(bad[0])
            with self.assertRaises(ValueError):validate_rows(bad,"synthetic_design","synthetic_validation")
        for origin,purpose in (("software_check","synthetic_validation"),("synthetic_design","human_analysis")):
            with self.assertRaises(ValueError):validate_rows(self.rows,origin,purpose)
        with self.assertRaises(ValueError):validate_rows(self.rows+[self.rows[0]],"synthetic_design","synthetic_validation")
        bad=copy.deepcopy(self.rows);bad[11]["participantId"]=bad[0]["participantId"]
        with self.assertRaises(ValueError):validate_rows(bad,"synthetic_design","synthetic_validation")
        bad=copy.deepcopy(self.rows);bad[11]["sessionId"]=bad[0]["sessionId"]
        with self.assertRaises(ValueError):validate_rows(bad,"synthetic_design","synthetic_validation")

    def test_zero_variance_insufficient_support_and_no_fallback(self):
        same=[{"conditionId":"C2","status":"available","value":.3}]*3
        self.assertEqual(contrast(same,{"C2":1.})["reason"],"zero_or_nonfinite_standard_error")
        self.assertIsNone(contrast(same[:1],{"C2":1.})["pValue"])
        own=[r for r in self.rows if r["participantId"]=="synthetic-C1-00" and r["stage"]=="main"]
        self.assertEqual(participant_slope(own,"xInitial")["status"],"unavailable")

    def test_historical_inference_rejects_budget_only_profiles(self):
        for changes in ({"questionLimitRule": "budget_only"},
                        {"profileId": "pilot-engagement-budget-v1"}):
            bad = copy.deepcopy(self.rows)
            bad[0].update(changes, questions=0, anyQuestion=0)
            with self.assertRaisesRegex(ValueError, "historical capped"):
                validate_rows(bad, "synthetic_design", "synthetic_validation")
        explicit = copy.deepcopy(self.rows)
        for row in explicit:
            row.update(profileId="pilot-4p10m-v1", questionLimitRule="count_cap_20")
        original = validate_rows(self.rows, "synthetic_design", "synthetic_validation")
        checked = validate_rows(explicit, "synthetic_design", "synthetic_validation")
        for row in checked:
            row.pop("profileId")
            row.pop("questionLimitRule")
        self.assertEqual(checked, original)

    def test_required_roster_and_timing(self):
        result=fit(self.rows);index={r["id"]:r for r in result["tests"]}
        self.assertEqual(sum(r["role"]=="core" for r in result["tests"]),9)
        self.assertEqual(result["practiceRowsExcluded"],30)
        for name in ("distance_ES_C2_C1","distance_EO_C1_C3","lower_pooled","terminal_C4C5_C2C3","raw_initial_C2","terminal_C5"):
            self.assertEqual(index[name]["status"],"available");self.assertIsNone(index[name]["reject"])
        for name in ("initial_C1","initial_C5","initial_C2_C1","initial_C3_C1","initial_C2C3_C1","initial_C4C5_C2C3","raw_initial_C1","raw_initial_C5","C1_reverse_no_search_resemblance"):
            self.assertEqual(index[name]["status"],"unavailable");self.assertIsNone(index[name]["pValue"])
        self.assertEqual(index["C1_reverse_no_search_resemblance"]["side"],"less")
        self.assertEqual(index["terminal_C5"]["timing"],"endogenous_terminal_association")

    def test_ties_and_windows_are_not_imputed(self):
        result=fit(self.rows);index={r["id"]:r for r in result["tests"]}
        self.assertEqual(index["lower_C1_tie_free"]["status"],"unavailable")
        self.assertEqual(index["lower_C1_tie_free"]["groups"]["C1"]["participants"],0)
        early=[r for r in result["participantEstimates"] if r["window"]=="early"]
        self.assertTrue(all(r["observations"]==5 for r in early))
        c5=copy.deepcopy([r for r in self.rows if r["conditionId"]=="C5"])
        for r in c5:r["questions"]=0;r["anyQuestion"]=0
        values=participant_values(c5,"slope_terminal")
        self.assertTrue(all(v["status"]=="unavailable" for v in values))

    def test_complete_record_history_is_not_complete_endpoint_history(self):
        own=copy.deepcopy([r for r in self.rows if r["participantId"]=="synthetic-C2-00"])
        own[0]["EOResemblance"]=None;own[0]["distancePP_equal_split"]=None
        checked=validate_rows(own,"synthetic_design","synthetic_validation")
        value=participant_values(checked,"EOResemblance")[0]
        self.assertTrue(value["completeTen"]);self.assertEqual(value["mainTrials"],10)
        self.assertEqual(value["observations"],9);self.assertTrue(value["endpointMissingWithinCompleteHistory"])
        self.assertEqual(value["endpointMissingWithinSelectedRecords"],1)

    def test_holm_exact_slots_unavailable_and_families(self):
        p=[.001,.002,.004,.008,.011,.02,.03,.04,.2]
        rows=[{"id":d[0],"family":d[1],"status":"available","pValue":v} for d,v in zip(CORE,p)]
        result=multiplicity(rows);ordered=result["tests"]
        expected=[.009,.016,.028,.048,.055,.08,.09,.09,.2]
        for row,value in zip(ordered,expected):self.assertAlmostEqual(row["adjustedP"],value)
        self.assertEqual(sum(r["reject"] for r in ordered),4)
        self.assertEqual(multiplicity(list(reversed(rows)))["anyCoreSuccess"],result["anyCoreSuccess"])
        bad=copy.deepcopy(rows);bad[0].update(status="unavailable",pValue=None)
        absent=multiplicity(bad)
        self.assertFalse(absent["allCoreSuccess"]);self.assertIsNone(absent["tests"][0]["adjustedP"])
        self.assertEqual(absent["tests"][1]["plannedSlots"],9)
        boundary=[{**r,"pValue":.05/9} for r in rows]
        self.assertTrue(multiplicity(boundary)["allCoreSuccess"])
        self.assertFalse(multiplicity([{**r,"pValue":.05/9+1e-10} for r in rows])["anyCoreSuccess"])
        family=multiplicity(rows,"within_family_holm");budget=multiplicity(rows,"four_family_alpha_budget")
        self.assertFalse(family["globalAnyGuarantee"]);self.assertTrue(budget["globalAnyGuarantee"])
        self.assertEqual(family["tests"][0]["plannedSlots"],2)
        with self.assertRaises(ValueError):multiplicity(rows[:-1])


if __name__=="__main__":unittest.main()
