"""Public orchestration admission checks; no random data, model fits or R runs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.tutoring_statistics import magnitude_reproduction as runner
from src.tutoring_statistics import magnitude_reduction as producer
from src.tutoring_statistics import independent_magnitude_reduction as reader
from test_magnitude_reduction import record
from test_saved_fit_magnitude import population_fixture


class MagnitudeReproductionTests(unittest.TestCase):
    def test_fixed_selectors(self):
        manifest = runner.plan("saved_power_core")
        self.assertEqual(runner.select_ids(manifest, start=0, count=3), [1, 3, 5])
        self.assertEqual(len(runner.select_ids(manifest, all_ids=True)), 16000)
        for kwargs in ({}, {"ids": [1, 1]}, {"ids": [2]}, {"ids": [True]},
                       {"ids": [3, 1]}, {"ids": [1], "count": 2},
                       {"start": 15999, "count": 2}, {"all_ids": True, "count": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                runner.select_ids(manifest, **kwargs)

    def test_default_cli_is_plan_only(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / "scripts/reproduce_magnitude_study.py"),
                                 "--mode", "fresh_validation"], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["expectedTotal"], 10645)
        self.assertFalse(plan["methodAdoptedForHumanData"])

    def test_invalid_selector_refused_before_initialization(self):
        with patch.object(sys, "argv", ["runner", "--mode", "fresh_validation", "--execute", "--output", "unused"]), \
                patch.object(runner, "initialize") as initialize, self.assertRaises(ValueError):
            runner.main()
        initialize.assert_not_called()

    def test_wrong_mode_input_refused_before_writing(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not_created"
            with self.assertRaises(ValueError):
                runner.initialize(target, "saved_power_core", SimpleNamespace(power_directory=None))
            self.assertFalse(target.exists())

    def test_compact_coverage_or_discovery_tamper_rejected(self):
        manifest = runner.plan("fresh_validation")
        rec = record(manifest, 4)
        event = producer.compact_event(rec)
        checked = reader.compact_event(rec,
            {name: rec["methods"][name]["references"] for name in ("baseline", "candidate")},
            {name: rec["methods"][name]["composition"] for name in ("baseline", "candidate")})
        reader.compare_report(event, checked)
        for field in ("F", "D"):
            changed = copy.deepcopy(event)
            changed["methods"]["candidate"][field] = True
            with self.assertRaises(ValueError):
                reader.compare_report(changed, checked)
        changed = copy.deepcopy(event)
        changed["methods"]["candidate"]["references"]["C1-C2:A:D"]["covered"] = False
        with self.assertRaises(ValueError):
            reader.compare_report(changed, checked)

    def test_commit_cannot_be_moved_to_another_unit(self):
        manifest_pin = dict(path="/public/output/manifest.json", sha256="manifest")
        commitment = dict(schema=runner.SCHEMA, manifest=manifest_pin, sourceClosure=True, ordinal=1)
        with patch.object(runner, "observe_manifest", return_value={}), \
                patch.object(runner.Observations, "read", return_value=commitment):
            for path, requested in (("/public/output/units/00000/commit.json", 0),
                                    ("/public/output/units/00001/commit.json", 0)):
                with self.assertRaises(ValueError):
                    runner.admit_commit(manifest_pin, dict(path=path, sha256="commit"), expected_ordinal=requested)

    def test_resume_joins_fit_and_population_without_work(self):
        from src.tutoring_statistics import magnitude_design as design, saved_fit_magnitude as core
        from src.tutoring_statistics import independent_saved_fit as numerical
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(schema=runner.SCHEMA, plan=runner.plan("fresh_validation"))
            mp = runner.io.write(root / "manifest.json", manifest)
            directory = root / "units/00000"
            rec = record(manifest["plan"], 0)
            case = design.case_for(manifest["plan"], 0)
            original, spec, generation, _ = population_fixture()
            generation.update(case=case, retentionMask=spec["retentionMask"], recruitedCounts=spec["recruitedCounts"])
            request_pin = runner.io.write(directory / "request.json", original)
            gen_pin = runner.io.write(directory / "generation.json", generation)
            key = case["caseID"] + "__r0001"
            spec.update(case=case, ordinal=0, caseID=case["caseID"], repID=1, fitKey=key,
                        request=request_pin, generation=gen_pin)
            sp = runner.io.write(directory / "spec.json", spec)
            fit = runner.io.write(directory / "fit_fixture.json", {})
            raw = dict(status="fitted", coefficientNames=list(core.NAMES), coefficients={k: 0. for k in core.NAMES},
                vcov=[[float(i == j) for j in range(9)] for i in range(9)], diagnostics=dict(
                converged=True, singular=False, gradient=[], hessianEigenvalues=[1.], usableCandidateInference=True))
            base = dict(fitKey=key, request=request_pin, generation=gen_pin, fitArtifact=fit, raw=raw)
            bp = runner.io.write(directory / "baseline.json", base)
            cp = runner.io.write(directory / "candidate.json", dict(base, baseline=bp))
            origin = runner.io.write(directory / "receipt.json", dict(manifest=mp,
                       records=[dict(fitKey=key, baseline=bp, candidate=cp)]))
            proof = runner.io.write(directory / "proof.json", dict(spec=sp, manifest=mp,
                        request=request_pin, generation=gen_pin, independentReplay=True))
            binding = runner.io.write(directory / "input.json", dict(schema=runner.INPUT_SCHEMA,
                        mode="fresh_validation", ordinal=0, spec=sp, baseline=bp, candidate=cp))
            rec["populationCounts"] = spec["populationCounts"]
            ind = dict(schema="h10-magnitude-independent-r-v1", target={k: rec[k] for k in ("ordinal", "caseID", "repID")},
                request=request_pin, fitArtifact=fit, actualK=spec["populationCounts"], sourceClosure=True,
                frameVerified=True, exactSourceDesignInR=True, newFits=0, newDraws=0,
                guard=dict(usableCandidateInference=True), methods=rec["methods"])
            event = producer.compact_event(rec)
            commit_path = directory / "commit.json"
            def store(independent):
                # Update all outer hashes to test semantic joins, not stale hashes.
                (directory / "independent.json").write_text(json.dumps(independent))
                result = dict(schema=runner.SCHEMA, mode="fresh_validation", sourceClosure=True,
                    independentEightLVerified=True, input=binding, record=rec, event=event, independentEvent=event,
                    observedInputs={}, artifacts=dict(independent=dict(output=runner.io.pin(directory / "independent.json"))))
                (directory / "result.json").write_text(json.dumps(result))
                commit = dict(schema=runner.SCHEMA, manifest=mp, ordinal=0, sourceClosure=True,
                    result=runner.io.pin(directory / "result.json"), origin=origin, generationProof=proof, input=binding, event=event)
                commit_path.write_text(json.dumps(commit))
                return runner.io.pin(commit_path)
            with patch.object(runner, "observe_manifest", return_value=manifest), \
                    patch.object(numerical, "compare_method", side_effect=lambda actual, other, name: actual["composition"]), \
                    patch.object(runner, "analyze_input") as analyze:
                self.assertEqual(runner.admit_commit(mp, store(ind), expected_ordinal=0)["event"], event)
                for changed in (dict(ind, fitArtifact=request_pin), dict(ind, target=dict(ind["target"], repID=2)),
                                dict(ind, actualK=dict(C1=1, C2=1, C3=1))):
                    with self.assertRaises(ValueError):
                        runner.admit_commit(mp, store(changed), expected_ordinal=0)
                analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
