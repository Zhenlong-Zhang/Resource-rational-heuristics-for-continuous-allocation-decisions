"""Strict JSON and generated-stage checks for descriptive finite-task readers."""
import json
import math


def _constant(value):
    raise ValueError("Nonfinite JSON number: "+value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: "+key)
        result[key] = value
    return result


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON quantity")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def parse(raw):
    value = json.loads(raw, parse_constant=_constant, object_pairs_hook=_object)
    _finite_tree(value)
    return value


def read(path):
    return parse(path.read_text())


def read_lines(stream):
    for line in stream:
        yield parse(line)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    require(type(value) in (float, int) and math.isfinite(value), "Finite numeric quantity required")
    return value


def stage(directory, expected, settings):
    completion = read(directory/"completion.json")
    require(completion.get("stage") == expected and completion.get("complete") is True,
            "Complete public "+expected+" stage required")
    require(completion.get("machine_certified_arithmetic") is False,
            "Numerical, not machine-certified, input required")
    require(read(directory/"settings.json") == settings, "Scientific settings differ between inputs")
    return completion


def template_records(directory, expected):
    records = read(directory/"templates.json")
    ids = [r["template_id"] for r in records]
    require(bool(ids) and len(set(ids)) == len(ids), "Empty or duplicate template metadata")
    require(all(expected.get(r["template_id"]) == r for r in records),
            "Input template model differs from requested scientific design")
    return {r["template_id"]: r for r in records}


def panel_identities(panel, expected, full=False):
    require(panel["schema"] == "finite-tutoring-paired-panel-v1"
            and panel["policy"] == "repeated_exact_one_step_myopic", "Wrong predecessor model")
    n = panel["episodes_per_template"]
    require(type(n) is int and n > 0 and type(panel["root_seed"]) is int, "Invalid panel recipe")
    records = panel["templates"]
    ids = [r["template_id"] for r in records]
    require(bool(ids) and len(ids) == len(set(ids)), "Duplicate or empty panel templates")
    require(all(r == expected.get(r["template_id"]) for r in records), "Panel template mismatch")
    identities = {(tid, episode) for tid in ids for episode in range(n)}
    require(panel["records"] == len(identities), "Panel denominator differs from recipe")
    if full:
        require(set(ids) == set(expected) and n == 400 and panel["root_seed"] == 2026100901,
                "Complete six-template x400 source-seed panel required, not smoke")
    return identities


def terminal_rows(directory, filename, completion, expected, full=False):
    summary = read(directory/"summary.json")
    require(completion.get("input_panel") == summary["input_panel"], "Terminal panel metadata mismatch")
    identities = panel_identities(summary["input_panel"], expected, full)
    require(summary["records"] == len(identities) and completion["records"] == len(identities),
            "Terminal input denominator mismatch")
    require(summary["new_random_scenarios"] == 0
            and summary["new_cost_aware_trajectory_claimed"] is False
            and summary["whole_policy_optimality_claimed"] is False, "Conditional terminal scope required")
    seen = set()
    with (directory/filename).open() as stream:
        for row in read_lines(stream):
            require(type(row["episode_id"]) is int, "Integer episode identity required")
            identity = (row["template_id"], row["episode_id"])
            require(identity in identities and identity not in seen, "Duplicate/out-of-scope terminal identity")
            seen.add(identity)
            yield row
    require(seen == identities, "Missing terminal identities")
