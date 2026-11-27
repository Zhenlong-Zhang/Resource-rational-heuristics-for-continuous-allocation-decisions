"""Portable input bindings and independent C5 preparation checks.

The adopted lookup and supplied prefix have fixed public byte pins. The legacy
preparation manifest, if present, is trusted only as part of a reviewed
repository revision. A byte match is not original-writer provenance proof.
"""

import hashlib
import json
import re
from pathlib import Path

from . import c5_final_reference
from . import c5_public_inputs as inputs


class AuthorityUnavailable(ValueError):
    """A required scientific authority or evidence is unavailable."""


MANIFEST_PATH = "configs/c5_scientific_authority.json"
MANIFEST_SCHEMA = "c5-scientific-authority-v1"
REQUIRED_BINDINGS = (
    "initial_lookup", "latent_config", "runtime_profile", "historical_rep1",
    "source_chain", "preparation_evidence",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[2]

# Accepted materialization bytes, not caller-supplied hashes. The lookup is an
# adopted policy definition; the prefix is externally supplied outcome data.
ADOPTED_INITIAL_SHA256 = "68b2fd36ef99aefcadb78e1831a9167134b62731787e4588d149668699c6d1a0"
ACCEPTED_PREFIX_SHA256 = "926d3b69ba90d107a2ad1c2a8150f68b7e8e3c7cb06490342e1d38523ccd0f73"


def load_adopted_inputs(*, initial_lookup, historical_prefix):
    """Bind the adopted lookup and supplied rep1 to accepted materialization bytes.

    The lookup is fixed scientific configuration, not a fresh calculation. The
    prefix remains external scientific data and must be ignored if in a checkout.
    These pins and the loader's schema/mapping checks identify only these exact
    bytes and their internal consistency. They do not prove original-writer
    provenance, reconstruct an unknown historical dispatch seed, establish CPU,
    floating-point or runtime equivalence, or prove high-price policy behavior.
    They do not authorize sampling or historical final reduction.
    """
    fixed = inputs.load_fixed_inputs(initial_lookup, ADOPTED_INITIAL_SHA256)
    prefix = inputs.load_historical_prefix(
        historical_prefix, ACCEPTED_PREFIX_SHA256, repository_root=_ROOT, lookup=fixed)
    return {
        "initialLookup": fixed,
        "historicalPrefix": prefix,
        "initialLookupSHA256": ADOPTED_INITIAL_SHA256,
        "historicalPrefixSHA256": ACCEPTED_PREFIX_SHA256,
        "initialLookupIdentity": "adopted-fixed-scientific-configuration",
        "prefixIdentity": "accepted-materialization-bytes-supplied-data",
        "historicalWriterProvenance": "not-established",
        "historicalDispatchReconstruction": "not-established",
        "highPricePolicyReplay": "not-established",
        "runtimeEquivalence": "not-established",
        "samplingAuthorized": False,
        "finalReductionAllowed": False,
    }


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member: " + key)
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _manifest():
    path = _ROOT / MANIFEST_PATH
    if not path.is_file() or path.is_symlink():
        raise AuthorityUnavailable("Repository scientific authority manifest is absent")
    manifest = _json(path.read_bytes())
    if (type(manifest) is not dict or set(manifest) != {"schema", "target", "bindings"}
            or manifest["schema"] != MANIFEST_SCHEMA or manifest["target"] != inputs.TARGET
            or type(manifest["bindings"]) is not dict
            or set(manifest["bindings"]) != set(REQUIRED_BINDINGS)
            or any(type(v) is not str or not _SHA.fullmatch(v)
                   for v in manifest["bindings"].values())):
        raise AuthorityUnavailable("Invalid repository scientific authority manifest")
    return manifest


def _bound_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Authority input byte SHA256 mismatch")
    return _json(raw)


def _path_free(value):
    if isinstance(value, str):
        return "/" not in value and "\\" not in value
    if type(value) is dict:
        return all(_path_free(k) and _path_free(v) for k, v in value.items())
    if type(value) is list:
        return all(_path_free(v) for v in value)
    return value is None or type(value) in (bool, int)


def check_supplied_geometry(designs, summary, certificate):
    """Check supplied geometry only; never establish its provenance."""
    return c5_final_reference.check_supplied_geometry(designs, summary, certificate)


def inspect_scientific_authority(*, initial_lookup, latent_config, runtime_profile,
                                 historical_rep1, source_chain, preparation_evidence):
    """Verify fixed bytes and independent preparation; not historical admission.

    Manifest bindings are SHA256 of the six named files. The preparation JSON
    has exactly schema, roots, summary and certificate; each of its 90 roots has
    exactly root, high and queryCount. High is the actual decision/terminal input
    accepted by c5_reproduction.law_record, not a reported PASS or symmetry flag.
    queryCount is a pinned observation, not proof of the underlying traversal;
    the source-chain file is byte-bound but its original-writer linkage and
    independent high-price replay are not yet checked. No historical or future numerical equivalence,
    runtime compatibility, sampling permission, or final admission follows.
    """
    manifest = _manifest()
    pins = manifest["bindings"]
    fixed = inputs.load_fixed_inputs(initial_lookup, pins["initial_lookup"])
    latent = _bound_json(latent_config, pins["latent_config"])
    profile = _bound_json(runtime_profile, pins["runtime_profile"])
    _bound_json(source_chain, pins["source_chain"])
    evidence = _bound_json(preparation_evidence, pins["preparation_evidence"])
    prefix = inputs.load_historical_prefix(historical_rep1, pins["historical_rep1"],
                                           repository_root=_ROOT, lookup=fixed)
    if (type(profile) is not dict or profile.get("schema") != "c5-portable-scientific-profile-v1"
            or not _path_free(profile)):
        raise ValueError("Expected path-free scientific runtime profile")
    if (type(latent) is not dict or latent.get("schema") != "fixed-c5-latent-53bit-v1"
            or latent.get("denominator") != 2**53 or latent.get("recipientOrder") != ["A", "B"]
            or type(latent.get("roots")) is not list or len(latent["roots"]) != 90):
        raise ValueError("Invalid fixed latent configuration")
    if (type(evidence) is not dict or set(evidence) != {"schema", "roots", "summary", "certificate"}
            or evidence["schema"] != "c5-authority-preparation-evidence-v1"
            or type(evidence["roots"]) is not list or len(evidence["roots"]) != 90):
        raise ValueError("Invalid complete preparation evidence")

    # Independent finite-law and geometry code is loaded only when evidence is
    # present. This may perform numerical work and is not an import-time check.
    from . import c5_reproduction as reproduction
    from . import c5_reference as reference

    records = []
    for expected, fixed_row, latent_row, item in zip(inputs.ROSTER, fixed["roots"],
                                                      latent["roots"], evidence["roots"]):
        if (type(item) is not dict or set(item) != {"root", "high", "queryCount"}
                or type(item["queryCount"]) is not int or item["queryCount"] != 0):
            raise ValueError("Missing actual no-question preparation evidence")
        record = reproduction.law_record(item["root"], item["high"], independent=True)
        laws = latent_row.get("recipientLaws") if type(latent_row) is dict else None
        if (tuple(record["rootKey"]) != expected or latent_row.get("rootKey") != list(expected)
                or record["rootSHA256"] != fixed_row["rootSHA256"]
                or record["logicalToDisplayed"] != fixed_row["logicalToDisplayed"]
                or type(laws) is not list or len(laws) != 2
                or any(record["laws"][i]["counts"] != laws[i]["counts"] for i in range(2))
                or record["laws"][0]["counts"] != record["laws"][1]["counts"]):
            raise ValueError("Root, mapping or fixed recipient law mismatch")
        records.append(record)
    designs = [reference.design_reference(records[10*v:10*v+10]) for v in range(9)]
    reference.verify_summary(evidence["summary"], designs, True)
    certificate = check_supplied_geometry(designs, evidence["summary"], evidence["certificate"])
    return {"status": "pinned_inputs_and_preparation_checked_not_historical_authority",
            "target": inputs.TARGET, "rootCount": len(records),
            "prefixSHA256": prefix["contentSHA256"], "bindings": dict(pins),
            "geometry": certificate, "highPricePolicyReplay": "not-established",
            "historicalWriterProvenance": "not-established",
            "runtimeEquivalence": "not-established", "futureOutputIdentity": "new-unverified",
            "samplingAuthorized": False, "finalReductionAllowed": False}


def require_scientific_authority(**kwargs):
    """Fail closed even after partial checks until writer provenance is implemented."""
    inspect_scientific_authority(**kwargs)
    raise AuthorityUnavailable("Original writer/source-chain provenance is not independently verified; "
                               "historical final reduction is not authorized")
