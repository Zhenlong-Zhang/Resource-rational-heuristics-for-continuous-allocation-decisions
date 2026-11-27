"""Assemble C5 inputs without drawing outcomes or asserting runtime authority.

Roots must come from the public personal-time input generator, and the lookup
is the fixed initial policy, not a newly evaluated decision. Numerical laws
are checked against the adopted integer partitions; a mismatch is not repaired.
"""
import hashlib
import json

from . import c5_moments, c5_public_inputs, c5_public_streams, c5_reference
from . import personal_time_inputs, personal_time_sensitivity

LATENT_BYTES_SHA256 = 'ae28df66e90981b68cba0b38e7fedaa5303330fa2da875e770e6ba41e629ee5c'
LOOKUP_BYTES_SHA256 = '68b2fd36ef99aefcadb78e1831a9167134b62731787e4588d149668699c6d1a0'


def assemble(lookup_path, lookup_sha256, latent_bytes, roots, context, report_products):
    """Build nine trajectory geometry certificates from complete source inputs.

    ``roots`` is either the envelope returned by ``personal_time_inputs.prepare``
    or its previously validated ordered root list (as used by the CLI).
    ``context`` is the sensitivity context. ``report_products`` maps width hex
    to generated product bytes. The fixed lookup's accepted byte identity is
    pinned; its retained candidate label is not historical outcome authority.
    The certificate's ``latentConfigSHA256`` is the canonical JSON object
    digest expected by the trajectory interface, not the input file byte SHA.
    The adopted, byte-pinned integer laws are used directly, not recomputed
    from floating-point priors on the current machine. The validator checks
    their roster, counts and CDF encoding. This does not authenticate historical
    outcomes, prove runtime equivalence, or grant permission to sample.
    """
    if lookup_sha256 != LOOKUP_BYTES_SHA256:
        raise ValueError('Accepted fixed initial lookup byte identity required')
    fixed = c5_public_inputs.load_fixed_inputs(lookup_path, lookup_sha256)
    if type(latent_bytes) is not bytes or hashlib.sha256(latent_bytes).hexdigest() != LATENT_BYTES_SHA256:
        raise ValueError('Adopted latent configuration byte identity required')
    latent = c5_public_streams.ValidatedLatentConfig(json.loads(latent_bytes))
    if type(roots) is dict:
        selected = personal_time_inputs.selection(
            [dict(zip(personal_time_inputs.FIELDS, key)) for key in c5_public_inputs.ROSTER])
        roots = personal_time_inputs.validate_roots(roots, selected, context['reference'])
    if type(roots) is not list or len(roots) != 90:
        raise ValueError('Complete ordered 90-root source input required')
    records, widths = [], set()
    for position, (root, row, expected) in enumerate(zip(roots, fixed['roots'], c5_public_inputs.ROSTER)):
        if type(root) is not dict or personal_time_inputs.coordinates(root) != expected:
            raise ValueError('Root roster or order mismatch')
        cfg = personal_time_sensitivity.rate_config(context, root, 'rho_0.001')
        definition, task = root['definition'], cfg['task']
        if (c5_public_inputs._digest(root) != row['rootSHA256']
                or c5_public_inputs._digest(cfg) != row['configSHA256']
                or cfg != row['config'] or root['logicalToDisplayed'] != row['logicalToDisplayed']
                or definition['reportWidth'] != row['reportWidth']
                or task['report_width'] != row['reportWidth']
                or definition['budgetMillihours'] != task['total_millihours']
                or definition['costMillihours'] != task['cost_millihours']):
            raise ValueError('Root, configuration or display mapping mismatch')
        budget, cost, maximum = task['total_millihours'], task['cost_millihours'], task['max_assessments']
        if (any(type(x) is not int for x in (budget, cost, maximum)) or cost <= 0
                or maximum < 0 or divmod(budget, cost)[0] != maximum
                or not 0 < budget % cost < cost):
            raise ValueError('All-path positive budget certificate failed')
        if type(root['initialLogs']) is not list or len(root['initialLogs']) != 2:
            raise ValueError('Two root prior laws required')
        laws = [{'counts': list(latent.roots[position][index][0])} for index in range(2)]
        records.append({'rootKey': list(expected), 'budget': budget, 'laws': laws})
        widths.add(float(row['reportWidth']).hex())
    if type(report_products) is not dict or set(report_products) != widths:
        raise ValueError('Complete and exact report-width product set required')
    products = {}
    for width in sorted(widths):
        table = c5_public_streams.ValidatedReportTable(report_products[width], width)
        if not table.accepted_product_bytes:
            raise ValueError('Report product is a new unverified numerical identity')
        products[width] = table.product_sha256
    certificates = []
    for variant in range(9):
        group = records[10*variant:10*variant+10]
        design = c5_moments.design_payload(group)
        if design != c5_reference.design_reference(group):
            raise ValueError('Independent schedule geometry mismatch')
        if any(design[name]['status'] != 'available' for name in ('gap', 'partial_low')):
            raise ValueError('Singular schedule geometry')
        certificates.append({'scheduleVariant': variant, 'lookupSHA256': lookup_sha256,
                             'latentConfigSHA256': latent.input_sha256, 'design': design})
    return {'lookup': fixed, 'latent': latent, 'geometryCertificates': certificates,
            'reportProductSHA256': products,
            'lookupIdentity': 'accepted-fixed-policy-bytes-not-historical-outcome-authority',
            'latentIdentity': 'adopted-fixed-integer-input',
            'latentBytesSHA256': LATENT_BYTES_SHA256,
            'latentCanonicalObjectSHA256': latent.input_sha256,
            'lawIdentity': 'adopted-fixed-integer-partitions-not-recomputed',
            'historicalRuntimeEquivalence': False, 'samplingAuthorized': False}
