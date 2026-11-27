"""Redraw C5 results from conditional, display, or pinned saved-summary inputs.

The CSV and conditional entries do not authenticate inputs. The CSV route uses already-rounded display
values and remains an unverified draft; a checksum cannot establish authority.
The conditional route checks the producer contract, not historical provenance.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
from pathlib import Path
import re


_FIELDS = ("target", "view", "estimate", "band_lower", "band_upper", "band_kind")
_TARGETS = ("gap", "partial_low")
_VIEWS = ("low_price_0.001", "high_price_0.01", "high_minus_low")
ACCEPTED_PLOT_SUMMARY_SHA256 = "06a3c0136ec0753bededce3297b0a29a4d626da110c6959920fea1c8707208ae"

_CONDITIONAL_FIELDS = frozenset((
    "schema", "targets", "halfWidth", "familyAlpha", "inferenceLooks",
    "orderedPopulationRowsSHA256", "prefixContentSHA256", "lookupSHA256",
    "highLawRecordsSHA256", "latentBytesSHA256", "inputByteSHA256",
    "sourceByteSHA256", "runtimeMetadataVerified", "historicalExactResultEstablished",
    "runtimeEquivalenceEstablished", "scope", "assumptions"))
_RESULT_FIELDS = frozenset(("lowPriceEstimate", "lowPriceInterval",
                            "highPricePopulationSlope", "highMinusLowEstimate",
                            "highMinusLowInterval"))
_ASSUMPTIONS = [
    "The supplied prefix is bounded but not authenticated as historical.",
    "Future outcomes follow the fixed model in their actual execution environments.",
    "Root-by-replicate random variables are independent; key checks do not prove independence.",
    "High-price recipient draws are independent and follow the checked integer laws.",
]


def _fraction(value):
    if (not isinstance(value, list) or len(value) != 2 or
            any(not isinstance(x, str) or not re.fullmatch(r"-?(0|[1-9][0-9]*)", x)
                or x == "-0" for x in value)):
        raise ValueError("Expected an exact rational as two integer strings")
    numerator, denominator = map(int, value)
    if denominator <= 0:
        raise ValueError("Rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if [str(result.numerator), str(result.denominator)] != value:
        raise ValueError("Rational must be reduced")
    try:
        finite = math.isfinite(float(result))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("Rational cannot be displayed")
    return result


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def load_verified_saved_summary(path):
    """Read the externally supplied accepted bytes and return exact plot rows.

    This authenticates only the pinned saved summary, not a fresh simulation or
    numerical runtime. Its inference remains conditional on its assumptions.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACCEPTED_PLOT_SUMMARY_SHA256:
        raise ValueError("Saved summary does not match the accepted SHA256")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid saved summary JSON") from exc
    keys = {"schema", "sourceSHA256", "scope", "simultaneousConfidence", "halfWidth",
            "conditionalAssumptions", "originalExactFields", "extraction"}
    if (not isinstance(data, dict) or set(data) != keys or
            data["schema"] != "c5-fixed-repetition-final-plot-summary-v1" or
            not _digest(data["sourceSHA256"]) or
            data["scope"] != "supplemental finite-model population projection; fixed canonical target; retained bounded historical prefix; one final look" or
            data["simultaneousConfidence"] != "95% for the two precommitted low-price targets" or
            _fraction(data["halfWidth"]) != Fraction(1, 10) or
            data["conditionalAssumptions"] != ["recorded canonical numerical implementation",
                "full-support share semantics", "future root-by-repetition independence"] or
            data["extraction"] != "Original JSON strings and fields copied; no local coefficient, interval, or decimal calculation."):
        raise ValueError("Unexpected saved summary schema or scope")
    values = data["originalExactFields"]
    if (not isinstance(values, dict) or set(values) != {"fixedReplicates", "roots", "nonTie", "exactTies", "actualUsedEvents", "estimates", "bands", "highPrice", "highMinusLow", "keyDigest", "seedDigest"}):
        raise ValueError("Unexpected exact fields")
    for key in ("fixedReplicates", "roots", "nonTie", "exactTies", "actualUsedEvents"):
        if type(values[key]) is not int or values[key] < 0:
            raise ValueError("Invalid count")
    if (values["fixedReplicates"] != 1840 or values["roots"] != 165600 or
            values["nonTie"] + values["exactTies"] != values["roots"] or
            not _digest(values["keyDigest"]) or not _digest(values["seedDigest"])):
        raise ValueError("Inconsistent saved summary counts or identities")
    contrast = values["highMinusLow"]
    if not isinstance(contrast, dict) or set(contrast) != {"estimates", "bands"}:
        raise ValueError("Unexpected contrast schema")
    for mapping in (values["estimates"], values["bands"], values["highPrice"],
                    contrast["estimates"], contrast["bands"]):
        if not isinstance(mapping, dict) or set(mapping) != set(_TARGETS):
            raise ValueError("Expected both targets")

    def interval(value):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("Expected two exact interval endpoints")
        return tuple(_fraction(endpoint) for endpoint in value)

    rows = []
    for target in _TARGETS:
        low = _fraction(values["estimates"][target])
        high = _fraction(values["highPrice"][target])
        difference = _fraction(contrast["estimates"][target])
        band = interval(values["bands"][target])
        difference_band = interval(contrast["bands"][target])
        if (high != 0 or band != (low - Fraction(1, 10), low + Fraction(1, 10)) or
                difference != high - low or difference_band != (-band[1], -band[0])):
            raise ValueError("Inconsistent saved estimate, band, or contrast")
        rows.extend((DisplayRow(target, _VIEWS[0], low, *band, "simultaneous_95_percent"),
                     DisplayRow(target, _VIEWS[1], high, None, None, "analytic_exact"),
                     DisplayRow(target, _VIEWS[2], difference, *difference_band, "simultaneous_95_percent")))
    return tuple(rows)


def plot_verified_saved_summary(path, *, ax=None):
    """Return (figure, axes) for a verified historical saved-result redraw."""
    import matplotlib.pyplot as plt

    rows = load_verified_saved_summary(path)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure
    colors = ("#166b67", "#925526", "#343f75")
    labels = {"gap": "Need gap", "partial_low": "Partial lower need"}
    views = {"low_price_0.001": "Low price (0.001)",
             "high_price_0.01": "High price (0.01), analytic zero",
             "high_minus_low": "High minus low"}
    for index, row in enumerate(rows):
        y = len(rows) - index - 1
        color = colors[index % 3]
        if row.lower is not None:
            ax.hlines(y, float(row.lower), float(row.upper), color=color, linewidth=2)
            ax.plot([float(row.lower), float(row.upper)], [y, y], "|", color=color)
        ax.plot(float(row.estimate), y, marker="D" if row.lower is None else "o",
                color=color, linestyle="none")
    ax.set_yticks(range(len(rows) - 1, -1, -1),
                  [f"{labels[row.target]}: {views[row.view]}" for row in rows])
    ax.axvline(0, color="0.65", linewidth=0.8, zorder=0)
    ax.set_xlabel("Projection coefficient")
    ax.set_title("C5 accepted historical saved-result redraw")
    fig.text(0.02, 0.01, "Conditional saved-result redraw, not a fresh simulation; no cross-platform equivalence claimed.\n"
             "Dependent variable: share of remaining tutoring to higher true need; predictors scaled by initial budget.\n"
             "Finite-model population projection, not historical OLS or schedule SD. "
             "Simultaneous 95% inference assumes the recorded canonical implementation, full-support share semantics "
             "and future root-by-repetition independence. High-price zero is analytic, not a confidence interval.",
             fontsize="x-small", wrap=True)
    fig.subplots_adjust(bottom=0.25)
    return fig, ax


def conditional_rows(result):
    """Validate the conditional-final contract and return exact plotting rows.

    Structural validation does not independently authenticate the input or
    establish the assumptions asserted by the producer.
    """
    if not isinstance(result, dict) or set(result) != _CONDITIONAL_FIELDS:
        raise ValueError("Unexpected conditional final schema")
    if (result["schema"] != "c5-conditional-final-v1" or
            result["scope"] != "conditional_on_supplied_prefix_future_runtime_fixed_model_and_independence" or
            result["historicalExactResultEstablished"] is not False or
            result["runtimeEquivalenceEstablished"] is not False or
            result["runtimeMetadataVerified"] is not False or
            type(result["inferenceLooks"]) is not int or result["inferenceLooks"] != 1 or
            _fraction(result["halfWidth"]) != Fraction(1, 10) or
            _fraction(result["familyAlpha"]) != Fraction(1, 20)):
        raise ValueError("Unexpected conditional inference scope")
    for key in ("orderedPopulationRowsSHA256", "prefixContentSHA256", "lookupSHA256",
                "highLawRecordsSHA256", "latentBytesSHA256"):
        if not _digest(result[key]):
            raise ValueError("Invalid identity digest")
    for key in ("inputByteSHA256", "sourceByteSHA256"):
        value = result[key]
        if (not isinstance(value, dict) or not value or
                any(not isinstance(k, str) or not k or not _digest(v) for k, v in value.items())):
            raise ValueError("Invalid byte identity mapping")
    if result["assumptions"] != _ASSUMPTIONS:
        raise ValueError("Unexpected limitations")
    targets = result["targets"]
    if not isinstance(targets, dict) or set(targets) != set(_TARGETS):
        raise ValueError("Expected both conditional targets")
    rows = []
    for target in _TARGETS:
        item = targets[target]
        if not isinstance(item, dict) or set(item) != _RESULT_FIELDS:
            raise ValueError("Unexpected target schema")
        low = _fraction(item["lowPriceEstimate"])
        contrast = _fraction(item["highMinusLowEstimate"])
        intervals = []
        for key in ("lowPriceInterval", "highMinusLowInterval"):
            value = item[key]
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Expected two interval endpoints")
            intervals.append(tuple(_fraction(v) for v in value))
        lower, upper = intervals[0]
        if (_fraction(item["highPricePopulationSlope"]) != 0 or
                (lower, upper) != (low - Fraction(1, 10), low + Fraction(1, 10)) or
                contrast != -low or intervals[1] != (-upper, -lower)):
            raise ValueError("Inconsistent conditional interval or analytic contrast")
        rows.extend((DisplayRow(target, _VIEWS[0], low, lower, upper, "simultaneous_95_percent"),
                     DisplayRow(target, _VIEWS[1], Fraction(0), None, None, "analytic_exact"),
                     DisplayRow(target, _VIEWS[2], contrast, *intervals[1], "simultaneous_95_percent")))
    return tuple(rows)


def plot_conditional_projection(result, *, ax=None):
    """Redraw an actual conditional-final output, without certifying its origin."""
    import matplotlib.pyplot as plt

    rows = conditional_rows(result)
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 8))
    else:
        fig = ax.figure
    labels = {"gap": "Gap", "partial_low": "Partial lower need"}
    views = {"low_price_0.001": "Low price (0.001)",
             "high_price_0.01": "High price (0.01), analytic zero",
             "high_minus_low": "High minus low"}
    colors = {"low_price_0.001": "#166b67", "high_price_0.01": "#925526",
              "high_minus_low": "#343f75"}
    for index, row in enumerate(rows):
        y = len(rows) - index - 1
        color = colors[row.view]
        if row.lower is not None:
            ax.hlines(y, float(row.lower), float(row.upper), color=color, linewidth=2)
            ax.plot([float(row.lower), float(row.upper)], [y, y], "|", color=color)
            exact = f"{row.estimate}  [{row.lower}, {row.upper}]"
        else:
            exact = "0 (analytic; no CI)"
        ax.plot(float(row.estimate), y, marker="D" if row.lower is None else "o",
                color=color, linestyle="none")
        ax.text(1.02, y, exact, transform=ax.get_yaxis_transform(), va="center", fontsize="x-small")
    ax.set_yticks(range(len(rows) - 1, -1, -1),
                  [f"{labels[row.target]}: {views[row.view]}" for row in rows])
    ax.axvline(0, color="0.65", linewidth=0.8, zorder=0)
    ax.set_xlabel("Projection coefficient (exact fractions at right)")
    ax.set_title("C5 projection: CONDITIONAL, NOT HISTORICAL EXACT")
    ax.text(0.5, 0.5, "UNVERIFIED INPUT", transform=ax.transAxes, ha="center", va="center",
            fontsize=32, color="0.5", alpha=0.2, rotation=25)
    limitations = ("Simultaneous 95% family, one look; high-price analytic zero is not a CI.\n"
                   "Projection coefficients, not historical OLS or schedule SD.\n"
                   "Input provenance and runtime metadata are unverified; historical identity and runtime equivalence are not established.\n" +
                   "\n".join(result["assumptions"]))
    fig.text(0.02, 0.01, limitations, ha="left", va="bottom", fontsize="x-small", wrap=True)
    fig.subplots_adjust(left=0.24, right=0.68, bottom=0.29)
    return fig, ax


@dataclass(frozen=True)
class DisplayRow:
    target: str
    view: str
    estimate: Decimal
    lower: Decimal | None
    upper: Decimal | None
    band_kind: str


def _number(value):
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Invalid display number") from exc
    if not result.is_finite():
        raise ValueError("Display numbers must be finite")
    return result


def load_display_csv(path):
    """Read six display rows; validation is structural, not authentication."""
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _FIELDS:
            raise ValueError("Unexpected display CSV schema")
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError("Malformed display row")
            key = (raw["target"], raw["view"])
            if key in rows or key[0] not in _TARGETS or key[1] not in _VIEWS:
                raise ValueError("Duplicate or unknown display row")
            estimate = _number(raw["estimate"])
            if key[1] == "high_price_0.01":
                if (raw["band_kind"] != "analytic_exact" or raw["band_lower"]
                        or raw["band_upper"] or estimate != 0):
                    raise ValueError("High-price row must be analytic zero without a band")
                lower = upper = None
            else:
                if raw["band_kind"] != "simultaneous_95_percent":
                    raise ValueError("Expected simultaneous 95% band")
                lower, upper = _number(raw["band_lower"]), _number(raw["band_upper"])
                if not lower <= estimate <= upper:
                    raise ValueError("Estimate is outside its display band")
            rows[key] = DisplayRow(*key, estimate, lower, upper, raw["band_kind"])
    if set(rows) != {(target, view) for target in _TARGETS for view in _VIEWS}:
        raise ValueError("Expected all six target/view rows")
    return tuple(rows[(target, view)] for target in _TARGETS for view in _VIEWS)


def plot_projection(path, *, ax=None):
    """Plot supplied display rows and return (figure, axes).

    Always watermarked UNVERIFIED: no receipt verification is implemented.
    Four intervals show eight displayed endpoints; high-price analytic zeros
    are markers without interval bars. No sampling, fitting or inference occurs.
    """
    import matplotlib.pyplot as plt

    rows = load_display_csv(path)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure
    labels = {"gap": "Gap", "partial_low": "Partial lower need"}
    views = {"low_price_0.001": "Low price (0.001)",
             "high_price_0.01": "High price (0.01), analytic zero",
             "high_minus_low": "High minus low"}
    colors = {"low_price_0.001": "#166b67", "high_price_0.01": "#925526",
              "high_minus_low": "#343f75"}
    for index, row in enumerate(rows):
        y = len(rows) - index - 1
        color = colors[row.view]
        if row.lower is not None:
            ax.hlines(y, float(row.lower), float(row.upper), color=color, linewidth=2)
            ax.plot([float(row.lower), float(row.upper)], [y, y], "|", color=color)
        ax.plot(float(row.estimate), y, marker="D" if row.lower is None else "o",
                color=color, linestyle="none")
    ax.set_yticks(range(len(rows) - 1, -1, -1),
                  [f"{labels[row.target]}: {views[row.view]}" for row in rows])
    ax.axvline(0, color="0.65", linewidth=0.8, zorder=0)
    ax.set_xlabel("Projection coefficient (display-rounded)")
    ax.set_title("C5 supplemental projection: UNVERIFIED DRAFT")
    ax.text(0, -0.18, "Low-price simultaneous 95% family; contrasts are reflected bands.\n"
            "High-price zero is analytic, not a zero-width CI. Not historical OLS or schedule SD.",
            transform=ax.transAxes, ha="left", va="top", fontsize="small")
    return fig, ax
