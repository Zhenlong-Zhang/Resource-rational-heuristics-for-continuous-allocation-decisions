"""Original supplemental key domain audit only; no PRNG draw or rep1 runner."""
from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from . import personal_time_sensitivity_streams as P
from . import personal_time_sensitivity_reference as R

# These literal namespaces are scientific randomization identities, not paths.
GROUP = 'r14-c5-paired-projection-private-stream-v1'
TERMINAL_DOMAIN = 'r14-c5-paired-projection-terminal-v1'
OLD_GROUP = 'r10-original-root-private-stream-candidate-v1'
POPULATION = 'r10-working-five-v0-rep0-candidate-not-population-estimate'
STREAM = 'r10-private-need-report-v1'


def private_key(variant, main, rep, logical, kind='latent', index=0):
    I.need(type(variant) is int and variant in range(9) and type(main) is int
           and main in range(1, 11) and type(rep) is int and 1 <= rep <= 256,
           'Original supplemental domain required')
    I.need(logical in ('A', 'B') and kind in ('latent', 'report') and type(index) is int
           and index >= 0 and (kind != 'latent' or index == 0), 'Typed stream index')
    return dict(streamVersion=STREAM, populationId=POPULATION, seed=20261019,
                randomizationGroup=GROUP, conditionId='C5', scheduleVariant=variant,
                replicateId=rep, mainIndex=main, logicalRecipient=logical,
                streamKind=kind, reportIndex=index)


def audit(roots, context):
    hashes, paired = set(), []
    for root in roots:
        _, variant, main = I.coordinates(root)
        for rep in (1, 256):
            ids = [P.stable_identity(S.rate_config(context, root, price), 20261019,
                                    POPULATION, 'C5', variant, rep, main)
                   for price in ('rho_0.001', 'rho_0.01')]
            I.need(ids[0]['trialId'] != ids[1]['trialId'], 'Price-bearing trial identities')
            keys = [P.make_key({'identity': x, 'bindings': {'populationId': POPULATION,
                               'randomizationGroup': GROUP}}, 'A', 'latent', 0) for x in ids]
            I.need(keys[0] == keys[1] == private_key(variant, main, rep, 'A'),
                   'Original paired latent keys')
            paired.append(dict(rootKey=list(I.coordinates(root)), replicate=rep,
                               priceIdentities=ids, sharedPrivateKey=keys[0]))
        for rep in range(1, 257):
            minimal = {'identity': {'seed': 20261019, 'conditionId': 'C5', 'variant': variant,
                                   'replicateId': rep, 'mainIndex': main},
                       'bindings': {'populationId': POPULATION, 'randomizationGroup': GROUP}}
            for logical in ('A', 'B'):
                key = private_key(variant, main, rep, logical)
                I.need(P.make_key(minimal, logical, 'latent', 0) == key
                       and R.make_key(minimal, logical, 'latent', 0) == key,
                       'Both public key sources must match')
                digest = I.digest(key)
                I.need(digest not in hashes, 'Latent key collision')
                hashes.add(digest)
                old = dict(key, randomizationGroup=OLD_GROUP, replicateId=0)
                I.need(I.digest(old) != digest, 'Original rep0 domain collision')
                for index in (0, 1):
                    report = private_key(variant, main, rep, logical, 'report', index)
                    I.need(P.make_key(minimal, logical, 'report', index) == report
                           and R.make_key(minimal, logical, 'report', index) == report,
                           'Report key source mismatch')
    I.need(len(hashes) == len(roots)*256*2, 'Complete selected latent domain')
    return dict(registeredGroup=GROUP, latentUniqueHashes=len(hashes),
                pairedIdentityBoundaries=paired, registeredScopeOnly=True,
                unregisteredHistoryNonuseClaim=False, registeredTerminalDomain=TERMINAL_DOMAIN,
                terminalDomainPending=True, terminalImplementationReviewed=False,
                wholeReplicateRNGReady=False, newScientificDraws=0,
                reportIndexDisjointness='exact typed nonnegative integer reportIndex; no draw',
                independence='declared independent-recipient model, not proved by unique keys')
