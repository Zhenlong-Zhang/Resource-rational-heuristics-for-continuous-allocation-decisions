"""Portable deterministic calibration contracts; no human or finite-task evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutoring_analysis import calibration as C
from tutoring_analysis.inference import CORE,participant_slope
from tutoring_analysis.planning import covariance_matrix

CONFIG=json.loads((ROOT/"configs/tutoring_calibration_v01.json").read_text())
CELLS=C.resolved_cells(CONFIG)


class CalibrationTests(unittest.TestCase):
    def test_frozen_roster_and_hash(self):
        self.assertEqual(hashlib.sha256((ROOT/"configs/tutoring_calibration_v01.json").read_bytes()).hexdigest(),"cc64f48180decc42092ed384470700949e3186008178736607124918fe562871")
        self.assertEqual(len(CELLS),30);self.assertEqual(sum(c["replications"] for c in CELLS),390000)
        self.assertEqual([c["seed"] for c in CELLS],list(range(2026100601,2026100631)))

    def test_scalar_parity_all_cells_populations_policies(self):
        for cell in CELLS:
            batch=C.generate_batch(cell,0,2)
            for population,groups in batch["populations"].items():
                stats=C.core_statistics(groups)
                for policy in CONFIG["policies"]:C.scalar_parity(cell,batch,stats,policy,0,population=population)

    def test_invalid_estimate_retained_without_inference(self):
        b=C.generate_batch(CELLS[28],0,2);s=C.core_statistics(b["populations"]["primary"])
        self.assertTrue(np.isfinite(s["estimate"][0,0]));self.assertFalse(s["available"].any())
        self.assertFalse(C.vector_holm(s["p"],s["available"],"global_holm")[1].any())

    def test_actual_projection_learning_and_adjustment(self):
        b=C.generate_batch(CELLS[12],0,2)
        for window,indices in (("early",range(5)),("late",range(5,10)),("primary",range(10))):
            raw=b["raw"]["C2"];x=raw["x"][0,list(indices)];y=raw["expectedY"][0,list(indices)]
            w=(x-x.mean())/np.sum((x-x.mean())**2)
            self.assertAlmostEqual(b["populations"][window]["C2"]["initial"]["truth"][0,0],float(w@y))
        cell=CELLS[15];b=C.generate_batch(cell,0,2)
        x=np.asarray(cell["recipeX"]);stage=np.arange(1,11);scale=np.asarray(cell["recipeScale"]);gap=np.asarray(cell["recipeGap"])
        X=np.column_stack([np.ones(10),x-x.mean(),stage-stage.mean(),scale-scale.mean(),gap-gap.mean()]);self.assertEqual(np.linalg.matrix_rank(X),5)
        w=np.linalg.pinv(X)[1];v=float(w@covariance_matrix(10,.01,.3)@w)+.01
        self.assertAlmostEqual(v,.06522269795373666)
        self.assertAlmostEqual(b["populations"]["order_recipe"]["C2"]["initial"]["variance"][0,0],v)
        self.assertFalse(b["populations"]["order_recipe"]["C3"]["initial"]["available"].any())

    def test_clip_truth_support_and_binary(self):
        self.assertAlmostEqual(float(C.clip_normal_mean(.5,.2,0,1)),.5)
        self.assertAlmostEqual(float(C.clip_normal_mean(0,1,0,1)),.3156268098137464)
        b=C.generate_batch(CELLS[23],0,3)
        for g in b["populations"]["primary"].values():
            self.assertTrue(np.all((g["q"]["values"]>=0)&(g["q"]["values"]<=1)))
            self.assertTrue(np.all(np.abs(g["L"]["values"])<=.5));self.assertTrue(np.all(np.abs(g["R"]["values"])<=20))
        for raw in b["raw"].values():self.assertTrue(np.all((raw["y"]>=0)&(raw["y"]<=1)))

    def test_missing_record_vs_endpoint_population(self):
        b=C.generate_batch(CELLS[18],0,3);raw=b["raw"]["C2"]
        complete=b["populations"]["primary"]["C2"]["q"]["available"]
        self.assertTrue(np.array_equal(complete,np.all(raw["records"],axis=-1)))
        self.assertGreater(b["populations"]["available_history"]["C2"]["q"]["available"].sum(),complete.sum())
        e=C.generate_batch(CELLS[21],0,3)
        self.assertTrue(e["raw"]["C2"]["records"].all());self.assertTrue((e["populations"]["primary"]["C2"]["q"]["observations"]<10).any())

    def test_mixed_null_and_MNAR_truth(self):
        b=C.generate_batch(CELLS[3],0,3);s=C.core_statistics(b["populations"]["primary"])
        self.assertEqual(np.flatnonzero(~s["null"][0]).tolist(),[0,2,5])
        b=C.generate_batch(CELLS[22],0,3);g=b["populations"]["primary"]["C2"]["initial"]
        self.assertAlmostEqual(float(g["selectedTruth"][0,0]),np.sqrt(.01*.5)*np.sqrt(2/np.pi))
        s=C.core_statistics(b["populations"]["primary"]);self.assertTrue(s["null"][0,2]);self.assertFalse(s["selectedNull"][0,2])

    def test_exact_intervals_merge_and_denominators(self):
        ci=C.binomial_interval(0,10);self.assertAlmostEqual(ci["interval"][1],1-.025**.1)
        self.assertEqual(C.binomial_interval(0,0)["interval"],None)
        b=C.evaluate_batch(CELLS[0],CONFIG,0,2);s=C.summarize_cell(CELLS[0],CONFIG,[b])
        p=s["populations"]["primary"];self.assertEqual(p["attempts"],2)
        self.assertEqual(p["tests"][0]["rawFalseReject"]["denominator"],2)
        self.assertEqual(p["policies"]["global_holm"]["alternativeDiscoveryPowerStatus"],"unavailable_empty_alternative_subset")
        self.assertEqual(set(s["powerComparison"]["policyJointBounds"]),set(CONFIG["policies"]))
        self.assertEqual(len(s["powerComparison"]["comparisons"]),36)
        x={"counts":[1,2],"knownTruthMin":0.,"knownTruthMax":3.};z=C.merge_counts([x,{"counts":[2,4],"knownTruthMin":-1.,"knownTruthMax":2.}])
        self.assertEqual(z,{"counts":[3,6],"knownTruthMin":-1.,"knownTruthMax":3.})

    def test_full84_bounded_parity(self):
        out=C.full_row_parity(CONFIG);self.assertEqual(len(out["cases"]),54)
        for item in out["cases"]:self.assertEqual(item["routes"],84);self.assertEqual(item["status"],"PASS")

    def test_seed_and_invalid_parameters(self):
        x=C.evaluate_batch(CELLS[0],CONFIG,0,2);self.assertEqual(x,C.evaluate_batch(CELLS[0],CONFIG,0,2))
        self.assertNotEqual(x,C.evaluate_batch(CELLS[0],CONFIG,1,2))
        bad=copy.deepcopy(CONFIG);bad["cells"][1]["seed"]=bad["cells"][0]["seed"]
        with self.assertRaises(ValueError):C.resolved_cells(bad)
        bad=copy.deepcopy(CONFIG);bad["defaults"]["correlation"]=2
        with self.assertRaises(ValueError):C.resolved_cells(bad)

    def test_cli_checkpoint_resume_no_overwrite(self):
        spec=importlib.util.spec_from_file_location("calibration_cli",ROOT/"scripts/calibrate_tutoring_methods.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);config=copy.deepcopy(CONFIG);config["batchSize"]=2;config["cells"]=config["cells"][:1];config["cells"][0]["replications"]=4
            (d/"config.json").write_text(json.dumps(config))
            args=SimpleNamespace(config=str(d/"config.json"),output=str(d/"out"),mode="full",resume=False,stop_after_batches=1)
            module.run(args);self.assertFalse((d/"out/completion.json").exists())
            with self.assertRaises(ValueError):module.run(args)
            args.resume=True;args.stop_after_batches=None;module.run(args)
            self.assertEqual(json.loads((d/"out/completion.json").read_text())["attempts"],4)
            module.run(args)
            ledger=json.loads((d/"out/checkpoint.json").read_text());bad=copy.deepcopy(ledger);bad["batches"].append(bad["batches"][0])
            with self.assertRaises(ValueError):module.validate_ledger(bad,d/"out",C.resolved_cells(config),{"gaussian_null":4},2)
            bad=copy.deepcopy(ledger);bad["batches"][0]["file"]="../other.json"
            with self.assertRaises(ValueError):module.validate_ledger(bad,d/"out",C.resolved_cells(config),{"gaussian_null":4},2)
            (d/"config.json").write_text(json.dumps({**config,"alpha":.04}))
            with self.assertRaises(ValueError):module.run(args)
            (d/"config.json").write_text(json.dumps(config))
            path=d/"out/cells/gaussian_null/batch_0000.json";path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaises(ValueError):module.run(args)


if __name__=="__main__":unittest.main()
