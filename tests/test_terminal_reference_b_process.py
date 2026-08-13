from __future__ import annotations

from dataclasses import fields, replace
import json
import os
import struct
import unittest
from unittest.mock import Mock, patch

import src.experiments.terminal_evidence_rows as evidence_module
import src.experiments.terminal_reference_b_process as process_module
from src.experiments.terminal_evidence_rows import evaluate_terminal_evidence_descriptor
from src.experiments.terminal_reference_b_process import (
    canonical_ipc_bytes,
    solve_terminal_reference_b_concurrently,
)
from src.mdp.finite_support import FiniteSupportMetaMDP
from src.solvers.terminal import optimize_terminal_allocation
from src.solvers.terminal_reference import (
    TerminalReferenceCrossProcessValidationProof,
    terminal_reference_cross_process_proof_matches,
    terminal_scientific_spec_hash,
)
from src.solvers.terminal_reference_b import (
    solve_terminal_reference_b_with_trace,
    terminal_reference_b_numerical_method_config_hash,
)
from tests.test_terminal_evidence_rows import descriptor_for
from tests.test_terminal_optimizer import swapped_two_atom_prior, terminal_config


class TerminalReferenceBProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mdp = FiniteSupportMetaMDP(
            terminal_config(), swapped_two_atom_prior(total_need=40.0, gap_fraction=0.25)
        )
        cls.belief = cls.mdp.initial_belief()
        cls.descriptor = descriptor_for(cls.mdp, cls.belief, prespecified=True)
        cls.production = optimize_terminal_allocation(cls.mdp, cls.belief)
        cls.concurrent = solve_terminal_reference_b_concurrently(
            cls.descriptor,
            cls.mdp,
            cls.belief,
            cls.production,
            timeout_seconds=120.0,
        )

    def proof_matches(self, proof, record=None, mdp=None, belief=None):
        record = self.concurrent.record if record is None else record
        mdp = self.mdp if mdp is None else mdp
        belief = self.belief if belief is None else belief
        return terminal_reference_cross_process_proof_matches(
            proof,
            record,
            mdp,
            belief,
            scientific_spec_hash=terminal_scientific_spec_hash(mdp),
            numerical_method_config_hash=(
                terminal_reference_b_numerical_method_config_hash(record.evaluation_cap)
            ),
            source_identity_hash=self.concurrent.traced_worker.source_identity_hash,
            interpreter_identity_hash=(
                self.concurrent.traced_worker.interpreter_identity_hash
            ),
            production_allocation=self.production.allocation,
        )

    def test_concurrent_workers_equal_unchanged_serial_record_and_trace(self):
        serial_record, serial_trace = solve_terminal_reference_b_with_trace(
            self.mdp, self.belief, self.production.allocation
        )
        self.assertEqual(self.concurrent.record, serial_record)
        self.assertEqual(
            canonical_ipc_bytes({"trace": self.concurrent.complete_trace}),
            canonical_ipc_bytes({"trace": serial_trace}),
        )
        self.assertEqual(
            self.concurrent.traced_worker.record_bytes_hash,
            self.concurrent.source_worker.record_bytes_hash,
        )
        self.assertTrue(
            self.proof_matches(self.concurrent.source_validation_proof)
        )

    def test_isolated_workers_do_not_overlap(self):
        launched = []
        original = process_module._launch_worker

        def launch(command, input_fd, output_fd):
            if launched:
                self.assertIsNotNone(launched[-1].poll())
            process = original(command, input_fd, output_fd)
            launched.append(process)
            return process

        with patch.object(process_module, "_launch_worker", side_effect=launch):
            result = solve_terminal_reference_b_concurrently(
                self.descriptor,
                self.mdp,
                self.belief,
                self.production,
                timeout_seconds=120.0,
            )
        self.assertEqual(len(launched), 2)
        self.assertEqual(result.record, self.concurrent.record)

    def test_concurrent_and_serial_evaluators_produce_identical_bundle_bytes(self):
        serial = evaluate_terminal_evidence_descriptor(
            self.descriptor, self.mdp, self.belief
        )
        runtime = {}
        concurrent = evaluate_terminal_evidence_descriptor(
            self.descriptor,
            self.mdp,
            self.belief,
            concurrent_reference_b=True,
            reference_b_runtime_evidence=runtime,
        )
        self.assertEqual(serial, concurrent)
        self.assertEqual(tuple(serial.sidecars), tuple(concurrent.sidecars))
        self.assertEqual(
            set(runtime),
            {"traced_worker", "source_worker", "coordinator_peak_rss_bytes"},
        )

    def test_proof_rejects_copy_forgery_and_changed_scope(self):
        proof = self.concurrent.source_validation_proof
        copied_record = replace(self.concurrent.record)
        self.assertFalse(self.proof_matches(proof, record=copied_record))
        forged = replace(proof, _seal=object())
        self.assertFalse(self.proof_matches(forged))
        self.assertFalse(
            terminal_reference_cross_process_proof_matches(
                proof,
                self.concurrent.record,
                self.mdp,
                self.belief,
                scientific_spec_hash=terminal_scientific_spec_hash(self.mdp),
                numerical_method_config_hash=(
                    terminal_reference_b_numerical_method_config_hash(
                        self.concurrent.record.evaluation_cap
                    )
                ),
                source_identity_hash="0" * 64,
                interpreter_identity_hash=(
                    self.concurrent.traced_worker.interpreter_identity_hash
                ),
                production_allocation=self.production.allocation,
            )
        )
        self.assertFalse(
            terminal_reference_cross_process_proof_matches(
                proof,
                self.concurrent.record,
                self.mdp,
                self.belief,
                scientific_spec_hash=terminal_scientific_spec_hash(self.mdp),
                numerical_method_config_hash=(
                    terminal_reference_b_numerical_method_config_hash(
                        self.concurrent.record.evaluation_cap
                    )
                ),
                source_identity_hash=self.concurrent.traced_worker.source_identity_hash,
                interpreter_identity_hash=(
                    self.concurrent.traced_worker.interpreter_identity_hash
                ),
                production_allocation=float.fromhex("0x1.0000000000001p-1"),
            )
        )

    def test_proof_rejects_in_place_belief_mutation(self):
        original = self.belief.weights
        try:
            self.belief.weights = (0.75, 0.25)
            self.assertFalse(
                self.proof_matches(self.concurrent.source_validation_proof)
            )
        finally:
            self.belief.weights = original

    def test_public_proof_type_cannot_be_constructed_into_acceptance(self):
        values = {
            field.name: getattr(self.concurrent.source_validation_proof, field.name)
            for field in fields(TerminalReferenceCrossProcessValidationProof)
        }
        values["_seal"] = object()
        self.assertFalse(
            self.proof_matches(TerminalReferenceCrossProcessValidationProof(**values))
        )

    def test_launch_uses_only_role_specific_pipe_descriptors_and_allowlisted_env(self):
        sentinel = Mock()
        command = process_module._worker_command("traced", 17, 19)
        with patch.object(process_module.subprocess, "Popen", return_value=sentinel) as popen:
            self.assertIs(process_module._launch_worker(command, 17, 19), sentinel)
        args, kwargs = popen.call_args
        self.assertEqual(args, (command,))
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(kwargs["pass_fds"], (17, 19))
        self.assertEqual(kwargs["env"], process_module._worker_environment())
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertIn("-B", command)
        self.assertTrue(all(kwargs["env"][name] == "1" for name in process_module._THREAD_ENVIRONMENT))

    def test_canonical_pipe_rejects_truncation_and_duplicate_payloads(self):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, struct.pack(">Q", 8) + b"{}")
        os.close(write_fd)
        with self.assertRaises(EOFError):
            process_module._read_frame(read_fd)
        os.close(read_fd)

        read_fd, write_fd = os.pipe()
        payload = b"{}"
        os.write(write_fd, struct.pack(">Q", len(payload)) + payload + b"x")
        os.close(write_fd)
        with self.assertRaises(ValueError):
            process_module._read_frame(read_fd)
        os.close(read_fd)

    def test_canonical_json_rejects_duplicate_keys_and_raw_float(self):
        with self.assertRaises(ValueError):
            process_module._parse_canonical_json(b'{"a":1,"a":1}')
        raw_float = b'{"value":0.5}'
        parsed = process_module._parse_canonical_json(raw_float)
        with self.assertRaises(ValueError):
            process_module._decode_canonical(parsed)

    def test_swapped_role_and_self_rehashed_tampering_are_rejected(self):
        source_identity = process_module.capture_reference_b_source_identity()
        interpreter_identity = process_module.capture_reference_b_interpreter_identity()
        started = process_module.time.perf_counter()
        authentication_key = b"k" * 32
        payload_bytes = process_module._worker_output(
            role="traced",
            input_hash="a" * 64,
            record=self.concurrent.record,
            trace=self.concurrent.complete_trace,
            source_identity=source_identity,
            interpreter_identity=interpreter_identity,
            authentication_key=authentication_key,
            started=started,
        )
        with self.assertRaises(ValueError):
            process_module._decode_worker_output(
                payload_bytes,
                role="source_validation",
                expected_input_hash="a" * 64,
                expected_source_identity_hash=source_identity["identity_hash"],
                expected_interpreter_identity_hash=interpreter_identity["identity_hash"],
                authentication_key=authentication_key,
                command=process_module._worker_command("source_validation", 17, 19),
            )

        payload = json.loads(payload_bytes.decode("utf-8"))
        payload["record_bytes_hash"] = "0" * 64
        payload["output_hash"] = process_module._payload_hash(payload, "output_hash")
        tampered = process_module._encoded_bytes(payload)
        with self.assertRaises(ValueError):
            process_module._decode_worker_output(
                tampered,
                role="traced",
                expected_input_hash="a" * 64,
                expected_source_identity_hash=source_identity["identity_hash"],
                expected_interpreter_identity_hash=interpreter_identity["identity_hash"],
                authentication_key=authentication_key,
                command=process_module._worker_command("traced", 17, 19),
            )

        trace_tampered = json.loads(payload_bytes.decode("utf-8"))
        trace_tampered["complete_trace"]["objective_cache"] = []
        trace_tampered["output_hash"] = process_module._payload_hash(
            trace_tampered, "output_hash"
        )
        with self.assertRaises(ValueError):
            process_module._decode_worker_output(
                process_module._encoded_bytes(trace_tampered),
                role="traced",
                expected_input_hash="a" * 64,
                expected_source_identity_hash=source_identity["identity_hash"],
                expected_interpreter_identity_hash=interpreter_identity["identity_hash"],
                authentication_key=authentication_key,
                command=process_module._worker_command("traced", 17, 19),
            )

    def test_worker_rejects_descriptor_source_mismatch(self):
        source_identity = process_module.capture_reference_b_source_identity()
        interpreter_identity = process_module.capture_reference_b_interpreter_identity()
        command = tuple(process_module.sys.orig_argv)
        payload_bytes, _ = process_module._build_input(
            role="traced",
            command=command,
            descriptor=self.descriptor,
            mdp=self.mdp,
            belief=self.belief,
            production=self.production,
            evaluation_cap=self.concurrent.record.evaluation_cap,
            source_identity=source_identity,
            interpreter_identity=interpreter_identity,
        )
        payload = json.loads(payload_bytes.decode("utf-8"))
        # A self-consistent forged descriptor still must not bind to the source.
        forged_descriptor = replace(
            self.descriptor,
            posterior_weight_hash="0" * 64,
            descriptor_hash="",
        )
        forged_descriptor = replace(
            forged_descriptor,
            descriptor_hash=process_module.terminal_validation_descriptor_hash(
                forged_descriptor
            ),
        )
        payload["descriptor"] = process_module._canonical_value(forged_descriptor)
        payload["input_hash"] = process_module._payload_hash(payload, "input_hash")
        parsed = process_module._parse_canonical_json(
            process_module._encoded_bytes(payload)
        )
        with self.assertRaisesRegex(ValueError, "descriptor/source mismatch"):
            process_module._validate_input(parsed, "traced")

    def test_worker_failure_and_timeout_helpers_fail_closed(self):
        crashed = Mock()
        crashed.poll.return_value = 70
        crashed.returncode = 70
        with self.assertRaises(RuntimeError):
            process_module._wait_for_workers((crashed,), 0.01)

        hanging = Mock()
        hanging.poll.return_value = None
        with self.assertRaises(TimeoutError):
            process_module._wait_for_workers((hanging,), 0.001)

    def test_evaluator_does_not_fall_back_after_concurrent_failure(self):
        failure = RuntimeError("injected concurrent worker failure")
        with (
            patch.object(
                evidence_module,
                "solve_terminal_reference_b_concurrently",
                side_effect=failure,
            ),
            patch.object(
                evidence_module,
                "solve_terminal_reference_b_with_trace",
                side_effect=AssertionError("serial fallback was attempted"),
            ),
            patch.object(
                evidence_module,
                "source_validate_terminal_reference_b_record",
                side_effect=AssertionError("serial source validation was attempted"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected concurrent worker failure"):
                evaluate_terminal_evidence_descriptor(
                    self.descriptor,
                    self.mdp,
                    self.belief,
                    concurrent_reference_b=True,
                )


if __name__ == "__main__":
    unittest.main()
