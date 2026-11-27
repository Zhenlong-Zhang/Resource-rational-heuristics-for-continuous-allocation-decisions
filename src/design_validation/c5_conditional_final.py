"""Conditional C5 final from directly replayed inputs, not historical certification.

The fixed lookup and latent bytes define the adopted target. The supplied
prefix and the future execution environment are not historically authenticated.
The family guarantee additionally assumes the declared root-by-replicate
independence and the checked fixed model; distinct keys do not prove it.

The high-price proof replays both source evaluators for all 90 roots and
requires their existing complete producer and independent preparation stages.
The intended source replay uses Python 3.10 and NumPy 1.23.5; this function
does not observe or certify the executing runtime or cross-runtime equivalence.
"""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from . import c5_final, c5_final_reference, c5_high_price_proof
from . import c5_input_assembly, c5_moments, c5_public_authority
from . import c5_public_inputs, c5_public_reproduction, c5_public_streams, c5_reference
from . import personal_time_inputs


def _bind(audit, proof, lookup, latent, summary, certificate):
    """Bind actual replay and proof outputs; never accept this as an authority API."""
    records = proof['checkedLawRecords']
    roots = audit['rootBindings']
    if (audit['population'] != c5_final.POPULATION or len(roots) != 90 or len(records) != 90 or
            audit['lookupSHA256'] != c5_public_authority.ADOPTED_INITIAL_SHA256 or
            proof['adoptedLookupSHA256'] != c5_public_authority.ADOPTED_INITIAL_SHA256 or
            personal_time_inputs.digest(records) != proof['rootsSHA256']):
        raise ValueError('Replay, proof, or fixed lookup binding mismatch')
    for index, (root, record, fixed) in enumerate(zip(roots, records, lookup['roots'])):
        key = ['C5', index // 10, index % 10 + 1]
        if (root['rootKey'] != key or record['rootKey'] != key or fixed['rootKey'] != key or
                root['rootSHA256'] != fixed['rootSHA256'] or
                root['configSHA256'] != fixed['configSHA256'] or
                root['logicalToDisplayed'] != fixed['logicalToDisplayed'] or
                record['logicalToDisplayed'] != fixed['logicalToDisplayed'] or
                root['reportWidth'] != fixed['reportWidth'] or
                root['budget'] != record['budget'] or
                root['budget'] != fixed['config']['task']['total_millihours'] or
                root['latentConfigSHA256'] != latent.input_sha256 or
                root['latentCounts'] != [list(item[0]) for item in latent.roots[index]] or
                root['latentCounts'] != [item['counts'] for item in record['laws']] or
                root['reportProductSHA256'] != c5_public_streams.ACCEPTED_REPORT_BYTES.get(
                    float(fixed['reportWidth']).hex())):
            raise ValueError('Replayed root, law, report, or fixed mapping mismatch')
    designs = []
    for variant in range(9):
        group = records[10 * variant:10 * (variant + 1)]
        reference = c5_reference.design_reference(group)
        if c5_moments.design_payload(group) != reference:
            raise ValueError('Independent design reconstruction mismatch')
        for root in roots[10 * variant:10 * (variant + 1)]:
            if root['design'] != {name: reference[name]['inverse'][1] for name in c5_final.TARGETS}:
                raise ValueError('Replayed design differs from reconstructed law geometry')
        designs.append(reference)
    c5_final_reference.check_supplied_geometry(designs, summary, certificate)
    return designs


def conditional_final(population, *, high_output, high_config, adopted_lookup,
                      latent_config, geometry_summary, geometry_certificate,
                      policy=None, output=None, shard_dirs=None, temp_dir=None):
    """Replay, bind and calculate conditional single-look simultaneous intervals.

    Population and proof are verified here, not admitted as caller PASS flags.
    Supplied-prefix provenance, historical identity, runtime equivalence, and
    the root-by-replicate independence assumption are not established.
    All 90 checked high-price preparation stages must already be available;
    their two source evaluators are replayed in the intended Python 3.10 /
    NumPy 1.23.5 scientific environment. Caller runtime metadata is not emitted.
    """
    raw = Path(latent_config).read_bytes()
    if hashlib.sha256(raw).hexdigest() != c5_input_assembly.LATENT_BYTES_SHA256:
        raise ValueError('Adopted latent byte pin mismatch')
    latent = c5_public_streams.ValidatedLatentConfig(json.loads(raw))
    lookup = c5_public_inputs.load_fixed_inputs(adopted_lookup, c5_public_authority.ADOPTED_INITIAL_SHA256)
    audit = c5_public_reproduction.audit_population(population, output, shard_dirs=shard_dirs,
                                                    temp_dir=temp_dir)
    proof = c5_high_price_proof.prove(high_output, high_config, adopted_lookup=adopted_lookup,
                                      policy=policy)
    _bind(audit, proof, lookup, latent, geometry_summary, geometry_certificate)
    identity = population['contexts'][0]['identity']
    if (identity['inputs']['lookup'] != c5_public_authority.ADOPTED_INITIAL_SHA256 or
            identity['inputs']['latent'] != c5_input_assembly.LATENT_BYTES_SHA256 or
            identity['inputs']['prefix'] != population['prefix_sha256']):
        raise ValueError('Population input byte identity mismatch')
    result = {}
    for name in c5_final.TARGETS:
        if c5_final.rational(proof['highPricePopulationSlopes'][name]) != 0:
            raise ValueError('Checked high-price slope is not zero')
        estimate = c5_final.rational(audit['arithmetic']['results'][name]['lowPriceEstimate'])
        lower, upper = estimate - Fraction(1, 10), estimate + Fraction(1, 10)
        result[name] = {'lowPriceEstimate': c5_final.encode(estimate),
                        'lowPriceInterval': [c5_final.encode(lower), c5_final.encode(upper)],
                        'highPricePopulationSlope': ['0', '1'],
                        'highMinusLowEstimate': c5_final.encode(-estimate),
                        'highMinusLowInterval': [c5_final.encode(-upper), c5_final.encode(-lower)]}
    return {'schema': 'c5-conditional-final-v1', 'targets': result,
            'halfWidth': ['1', '10'], 'familyAlpha': ['1', '20'], 'inferenceLooks': 1,
            'orderedPopulationRowsSHA256': audit['orderedPopulationRowsSHA256'],
            'prefixContentSHA256': audit['prefixContentSHA256'],
            'lookupSHA256': audit['lookupSHA256'], 'highLawRecordsSHA256': proof['rootsSHA256'],
            'latentBytesSHA256': c5_input_assembly.LATENT_BYTES_SHA256,
            'inputByteSHA256': dict(identity['inputs']), 'sourceByteSHA256': dict(identity['sources']),
            'runtimeMetadataVerified': False,
            'historicalExactResultEstablished': False, 'runtimeEquivalenceEstablished': False,
            'scope': 'conditional_on_supplied_prefix_future_runtime_fixed_model_and_independence',
            'assumptions': ['The supplied prefix is bounded but not authenticated as historical.',
                            'Future outcomes follow the fixed model in their actual execution environments.',
                            'Root-by-replicate random variables are independent; key checks do not prove independence.',
                            'High-price recipient draws are independent and follow the checked integer laws.']}
