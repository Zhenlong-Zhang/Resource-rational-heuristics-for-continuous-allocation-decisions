"""Purpose: protect signed integration, H-only actions, cost ownership and resume."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest

from src.design_validation.operation_costs import scalars as S, reference_math as R
from src.design_validation.operation_costs import actors, actor_reference, noask, fees, inputs
from src.design_validation.operation_costs.checkpoint import Workspace
from src.design_validation.operation_costs import noask_reference
from src.design_validation.strategy_cards import external_bounds


def h():
    return dict(stateId='test|initial',budgetInteger=100000,meanStrings=['20.0','60.0'],
                resourceString='100.000',ranges=[dict(lower=1,upper=100)]*2,
                channel=dict(source='task_width_metadata_not_actor_operand',reportWidthHex=(10.).hex()),
                latest=[None,None],history=[])


class SignedReductionTests(unittest.TestCase):
    def test_single_use_generators(self):
        for rows,expected in [([(F(2),F(3))],(F(2),F(3))),
                              ([(F(-4),F(-2))],(F(-4),F(-2))),
                              ([(F(-3),F(-1)),(F(2),F(5))],(F(-1),F(4)))]:
            self.assertEqual(S.total(x for x in rows),expected)
        self.assertEqual(R.generator_regression_reference()['status'],'PASS')

    def test_signed_joint_and_opposite_baseline(self):
        p=[(F(1,2),)*2]*2+[(F(0),)*2]*98
        u=[(F(-4),F(-2)),(F(2),F(6))]+[(F(0),)*2]*98
        self.assertEqual(S.dot(p,u),(F(-1),F(2)))
        self.assertEqual(S.report_value(dict(gpa_baseline=1,gpa_scale=1),p,(F(1),)*2,p,u,u),
                         (F(0),F(6)))

    def test_zero_lower_mass_retained(self):
        p=[(F(1,100),)*2]*100
        channel=[[(F(0),F(1,100))]*100 for _ in range(100)]
        joint,mass=S.report_joint(p,channel,1)
        self.assertEqual(mass,(F(0),F(1,100)))
        self.assertEqual(len(joint),100)


class ActorTests(unittest.TestCase):
    def test_five_rules_and_independent_counts(self):
        H=h()
        for family in actors.FAMILIES:
            a=actors.actor(H,family);ref=actor_reference.actor_reference(H,family)
            self.assertEqual(a['target'],ref['target'])
            self.assertEqual(a['counts'],ref['counts'])
        self.assertEqual(noask.ledger(actors.actor(H,'ES'))['oneSecondMemberSeconds'],'73/20')
        self.assertEqual(actors.actor(H,'EO_raw_surplus')['target']['index'],300)

    def test_private_or_stale_h_refused(self):
        for edit in (lambda z:z.update(latentNeed=20),lambda z:z.update(budgetInteger=99999),
                     lambda z:z.update(latest=[1,None])):
            H=h();edit(H)
            with self.assertRaises(ValueError):actors.actor(H,'EO_raw_surplus')

    def test_postread_ownership(self):
        H=h();H.update(latest=[20,None],history=[dict(slot=0,report=20)])
        a=actors.actor(H,'EO_raw_surplus');rec=dict(H=H,orderedSlot=0,report=20)
        ledger=fees.compose({},rec,a)
        self.assertEqual(ledger,R.ledger({},rec,a,actor_reference))
        self.assertEqual(sum(not x['charged'] for x in ledger['originalTraceWithOwnership']),3)
        self.assertEqual(ledger['counts']['common_execution'],1)
        self.assertEqual(ledger['counts']['request_execution'],1)

    def test_source_recipient_mapping(self):
        root=dict(id=['C3',0,1],orderedPhysical=['B','A'],
                  task=dict(total_millihours=100000,cost_millihours=500,report_width=10.),
                  display=dict(root=['20.0','60.0'],singleReport=[['21.0']*100,['61.0']*100]))
        H=inputs.public_h(root,'A',9)
        self.assertEqual(H['meanStrings'],['20.0','61.0'])
        self.assertEqual(H['latest'],[None,9])
        self.assertEqual(H['budgetInteger'],99500)


class ResumeTests(unittest.TestCase):
    def test_missing_only_and_conflict_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=Workspace(tmp,{'seed':1});context={'root':'original'}
            self.assertEqual(w.obtain('a',context,lambda:{'x':1}),{'x':1})
            resumed=Workspace(tmp,{'seed':1})
            self.assertEqual(resumed.obtain('a',context,lambda:self.fail('regenerated existing phase')),{'x':1})
            self.assertIsNone(resumed.get('missing',context))
            self.assertEqual(resumed.obtain('missing',context,lambda:2),2)
            with self.assertRaises(ValueError):Workspace(tmp,{'seed':2}).get('a',context)
            with self.assertRaises(ValueError):resumed.put('a',context,{'x':2})
            path=Path(tmp)/'checkpoints/a.json';value=json.loads(path.read_text())
            value['result']['x']=4;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):resumed.get('a',context)

    def test_truncation_and_unsafe_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=Workspace(tmp,{});w.put('a',{},1)
            (Path(tmp)/'checkpoints/a.json').write_text('{')
            with self.assertRaises(ValueError):w.get('a',{})
            with self.assertRaises(ValueError):w.get('../private',{})


class IndependentReaderTests(unittest.TestCase):
    def test_rehashed_semantic_mutations_are_refused(self):
        root=dict(id=['C5',0,1],logsHex=[[(0.).hex()]*100]*2,
                  task=dict(alpha=.5,loss_multiplier=2,allocation_denominator=1000,
                            total_millihours=100000,gpa_baseline=2.,gpa_scale=.1))
        H=h()
        with tempfile.TemporaryDirectory() as tmp:
            p=noask.produce(root,H,external_bounds,Workspace(tmp,{'fixture':True}))
            self.assertEqual(noask_reference.verify(root,H,p)['status'],'PASS')
            self.assertEqual(noask.winners(p),noask_reference.winners(p))
            edits=[lambda q:q['pairwise'][0]['region'].update(status='forged'),
                   lambda q:q['actors'][0]['originalActor'].update(actorInputPermission='private_posterior'),
                   lambda q:q['actors'][0]['originalActor']['trace'][0].update(inputOrigin='private'),
                   lambda q:q['actors'][0]['originalActor']['trace'][0].update(lifetime='persistent'),
                   lambda q:q['actors'][0]['originalActor']['trace'][0].update(count=True),
                   lambda q:q['actors'][0]['prices'][0]['criterionInterval'].__setitem__(0,'999')]
            for edit in edits:
                q=copy.deepcopy(p);edit(q)
                with self.assertRaises(ValueError):noask_reference.verify(root,H,q)

    def test_overlap_is_not_an_exact_tie(self):
        prices=[dict(rho=r,criterionInterval=['0','2']) for r in inputs.price_values()]
        row=dict(id=['fixture',0,1],actors=[dict(family='a',index=400,
                 ledger=dict(oneSecondMemberSeconds='1'),prices=prices),
                 dict(family='b',index=600,ledger=dict(oneSecondMemberSeconds='1'),prices=prices)])
        self.assertTrue(all(r['status']=='indeterminate' for r in noask.winners(row)))
        row['actors'][1]['index']=400
        self.assertTrue(all(r['status']=='exact_shared_action_effective_fee_tie' for r in noask.winners(row)))
        self.assertEqual(noask.winners(row),noask_reference.winners(row))


if __name__=='__main__':unittest.main()
