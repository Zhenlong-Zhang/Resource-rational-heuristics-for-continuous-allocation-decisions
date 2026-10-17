"""Independent finite schedule, stream and likelihood reconstruction.

The historical seed tokens are retained because they determine scientific draws.
No engine, normalizer, metric or target producer is imported here.
"""
from decimal import Decimal
import json
import math
import struct

PROFILE = "pilot-r9-design-v1"
ENGINE = "tutoring-task-0.1.0"
PILOT = "pilot-4p10m-v1"
VERSIONS = {
    "resolvedRecipeVersion": "r9-c1-gap1-unscaled-v1",
    "instructionVersion": "r9-instructions-v1",
    "displayVersion": "r9-finite95-channel80-decimal-v1",
    "checkVersion": "r9-understanding-sets-v1",
    "feedbackStyle": "r9-outcome-two-decimal-v1",
}
GUIDE = "r9-c3-report50-guide-v1"
RECIPES = {
    "C1": ([50., 50.], [3., 3.], 30., 120., 1.2, [0]),
    "C2": ([45., 75.], [2., 2.], 6., 150., 1.2, [-5, 0, 5]),
    "C3": ([50., 70.], [18., 18.], 6., 170., .5, [0]),
    "C4": ([45., 65.], [2., 2.], 6., 55., .6, [-5, 0, 5]),
    "C5": ([55., 55.], [8., 8.], 3., 55., .3, [0]),
}
SCALES = [.95, 1., 1.05]
NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
         "Avery", "Quinn", "Sam", "Charlie", "Drew", "Blake", "Robin",
         "Cameron", "Reese", "Rowan", "Parker", "Finley", "Skyler", "Dakota",
         "Emerson", "Hayden", "Sage", "Remy", "Jules", "Kendall", "Lane"]
SUBJECTS = ["mathematics", "biology", "chemistry", "physics", "history",
            "literature", "geography"]


def encode(value):
    # Seed arrays consist only of strings, bools and integers. Their JSON spelling
    # is identical to ECMAScript; this is not a general Number.toString clone.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def hash_text(text):
    """Independent unsigned32 translation of the declared non-security hash."""
    units = text.encode("utf-16-le", errors="surrogatepass")
    a, b, c, d = 1779033703, 3144134277, 1013904242, 2773480762
    mask = (1 << 32)-1
    for k in struct.unpack("<"+"H"*(len(units)//2), units):
        a = (b ^ ((a ^ k)*597399067)) & mask
        b = (c ^ ((b ^ k)*2869860233)) & mask
        c = (d ^ ((c ^ k)*951274213)) & mask
        d = (a ^ ((d ^ k)*2716044179)) & mask
    a = ((c ^ (a >> 18))*597399067) & mask
    b = ((d ^ (b >> 22))*2869860233) & mask
    c = ((a ^ (c >> 17))*951274213) & mask
    d = ((b ^ (d >> 19))*2716044179) & mask
    return [a ^ b ^ c ^ d, b ^ a, c ^ a, d ^ a]


def words(value):
    return hash_text(encode(value))


def stable(value):
    """Stable same-input JSON for bounded R9 semantic values, not scientific rounding.

    Python's shortest binary64 representation is converted to ECMAScript's fixed
    versus exponent spelling. Agreement of this checksum is never authentication.
    """
    if isinstance(value, list):
        return "["+",".join(stable(x) for x in value)+"]"
    if isinstance(value, dict):
        return "{"+",".join(encode(k)+":"+stable(value[k]) for k in sorted(value))+"}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError("Nonfinite semantic JSON number")
        if value == 0:
            return "0"
        representation = repr(value)
        if 1e-6 <= abs(value) < 1e21:
            text = format(Decimal(representation), "f")
            return text.rstrip("0").rstrip(".") if "." in text else text
        mantissa, exponent = representation.lower().split("e")
        mantissa = mantissa.removesuffix(".0")
        n = int(exponent)
        return mantissa+"e"+("+" if n >= 0 else "-")+str(abs(n))
    return encode(value)


def fingerprint(value):
    return "".join(f"{word:08x}" for word in hash_text(stable(value)))


class Random:
    def __init__(self, seed):
        self.a, self.b, self.c, self.d = words(seed)
        for _ in range(12):
            self.next()

    def next(self):
        mask = (1 << 32)-1
        a, b, c, d = self.a, self.b, self.c, self.d
        t = (a+b) & mask
        self.a = b ^ (b >> 9)
        self.b = (c+(c << 3)) & mask
        c = ((c << 21) | (c >> 11)) & mask
        self.d = (d+1) & mask
        t = (t+self.d) & mask
        self.c = (c+t) & mask
        return (t+.5)/4294967296


def shuffle(values, random):
    result = list(values)
    for i in range(len(result)-1, 0, -1):
        j = math.floor(random.next()*(i+1))
        result[i], result[j] = result[j], result[i]
    return result


def seed(meta, namespace):
    return [ENGINE, str(meta["seed"]), meta["participantId"], meta["sessionId"],
            meta["conditionId"], namespace, PILOT]


def normalize(logs):
    if len(logs) != 100 or not all(math.isfinite(x) for x in logs):
        raise AssertionError("Expected100finite log weights")
    m = max(logs)
    z = m+math.log(math.fsum(math.exp(x-m) for x in logs))
    return [x-z for x in logs]


def prior(centre, spread):
    return normalize([-.5*((n-centre)/spread)**2 for n in range(1, 101)])


_channels = {}


def channel(width):
    if width not in _channels:
        _channels[width] = [normalize([-.5*((y-n)/width)**2
                                       for y in range(1, 101)])
                            for n in range(1, 101)]
    return _channels[width]


def update(logs, report, width):
    ll = channel(width)
    return normalize([x+ll[n][report-1] for n, x in enumerate(logs)])


def probabilities(logs):
    return [math.exp(x) for x in normalize(logs)]


def mean(ps):
    return math.fsum((i+1)*p for i, p in enumerate(ps))


def draw(logs, random):
    u = random.next()
    cumulative = 0.
    for i, p in enumerate(probabilities(logs), 1):
        cumulative += p
        if u < cumulative:
            return i
    return 100


def schedule(meta):
    condition = meta["conditionId"]
    centres, spreads, width, budget, cost, gaps = RECIPES[condition]
    variant = meta["scheduleVariant"]
    items = []
    for si, scale in enumerate(SCALES):
        for gi, gap in enumerate(gaps):
            count = (3+(si == variant % 3) if len(gaps) == 1
                     else 1+(si == variant//3 and gi == variant % 3))
            template = (f"C1_s{si}_d1000" if condition == "C1"
                        else f"{condition}_s{si}_g{gi}")
            items.extend({"scale": scale, "gap": gap, "templateId": template}
                         for _ in range(count))
    random = Random(seed(meta, "main-order"))
    for _ in range(10000):
        ordered = shuffle(items, random)
        if not any(len({x["scale"] for x in ordered[i-3:i+1]}) == 1
                   for i in range(3, len(ordered))):
            break
    else:
        raise AssertionError("No legal schedule")
    sides = shuffle([False]*5+[True]*5, Random(seed(meta, "main-sides")))
    practice_sides = shuffle([False, False, True, True],
                             Random(seed(meta, "practice-sides")))
    pkey = ["trial-presentation-v1", str(meta["seed"]), meta["participantId"],
            meta["sessionId"]]
    names = shuffle(NAMES, Random(pkey+["names"]))
    sr = Random(pkey+["subjects"])
    for _ in range(10000):
        subjects = shuffle(SUBJECTS+SUBJECTS, sr)
        if all(subjects[i] != subjects[i-1] for i in range(1, 14)):
            break
    else:
        raise AssertionError("No legal subject sequence")
    practice = [{"scale": 1., "gap": 0,
                 "templateId": "C1_practice_r9_d1000" if condition == "C1"
                 else condition+"_practice_central"} for _ in range(4)]
    output = []
    for i, item in enumerate(practice+ordered):
        scale, gap = item["scale"], item["gap"]
        stage, number = ("practice", i+1) if i < 4 else ("main", i-3)
        output.append({**item, "sideReversed": (practice_sides+sides)[i],
                       "stage": stage, "stageIndex": number, "scheduleIndex": i,
                       "trialId": f"{stage}-{number:02d}",
                       "priorCentres": [50*scale-.5, 50*scale+.5] if condition == "C1"
                       else [(centres[0]+gap)*scale, (centres[1]-gap)*scale],
                       "priorSpreads": [x*scale for x in spreads],
                       "reportWidth": width*scale,
                       "budgetMillihours": math.floor(budget*scale*1000+.5),
                       "costMillihours": math.floor(cost*scale*1000+.5),
                       "presentation": {"version": "trial-presentation-v1",
                                        "names": names[2*i:2*i+2],
                                        "subject": subjects[i]},
                       "profileId": PROFILE,
                       "resolvedRecipeVersion": VERSIONS["resolvedRecipeVersion"],
                       "displayVersion": VERSIONS["displayVersion"],
                       "gapConvention": "C1-total-centre-gap-unscaled;other-conditions-scaled-row-offset",
                       **({"c1TotalCentreGapMinutes": 1} if condition == "C1" else {})})
    return output


def scenario(meta, definition):
    prefix = seed(meta, "scenario")+[definition["trialId"]]
    needs = [draw(prior(c, definition["priorSpreads"][i]),
                  Random(prefix+["latent", i]))
             for i, c in enumerate(definition["priorCentres"])]
    capacity = definition["budgetMillihours"]//definition["costMillihours"]
    ll = channel(definition["reportWidth"])
    reports = [[draw(ll[n-1], Random(prefix+["report", i, k]))
                for k in range(capacity)] for i, n in enumerate(needs)]
    mapping = [1, 0] if definition["sideReversed"] else [0, 1]
    return [needs[i] for i in mapping], [reports[i] for i in mapping]
