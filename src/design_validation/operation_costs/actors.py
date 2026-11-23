"""H-only literal actors, typed costs, affine regions and strict menu selection."""
import math
import re
from fractions import Fraction

FAMILIES=('ES','EO_raw_surplus','PN','AL','GN')
CHEAP=FAMILIES
H_FIELDS={'stateId','budgetInteger','meanStrings','ranges','resourceString','channel','latest','history'}
KNOWN={'read_scalar':.5,'retrieve_rule':.25,'add_subtract':.5,'multiply_divide':1.,'compare':.25,'clip':.25,'common_execution':2.65,'request_execution':2.65}
UNKNOWN=('local_access','abs','floor','ceil','native_select','store','cache_access','recipe_access','exp','log','power2','sqrt',
         'EGPA_whole','OPT_whole','random_whole')


def need(ok,msg):
 if not ok:raise ValueError(msg)


def coefficients():
 result={}
 for name,value in KNOWN.items():
  result[name]={'kind':'known','value':value,'units':'seconds_per_event',
   'sourceClass':'KLM_referenced_transfer_assumption' if name in ('common_execution','request_execution') else 'assumed_sensitivity_profile',
   'boundary':('read_visible_scalar_including_parse_not_retrieval_or_cache' if name=='read_scalar' else name),
   'transfer':'unmeasured_for_this_task','lifetime':'actual_invocation'}
 for name in UNKNOWN:
  result[name]={'kind':'symbol','symbol':'tau_'+name,'value':None,'units':'seconds_per_event',
   'sourceClass':'unidentified_task_duration','boundary':('bounded_scratch_control_version_arithmetic_excluding_named_NS_adds_and_persistent_store' if name=='local_access' else 'unresolved_algorithm_including_reads_inference_native_local_or_RNG_excluding_common_response' if name in ('EGPA_whole','OPT_whole','random_whole') else name),'nonnegative':True,'lifetime':'actual_use'}
 return result


def validate_coefficients(c):
 need(set(c)==set(KNOWN)|set(UNKNOWN),'Complete coefficient roster')
 for name,r in c.items():
  need(r['units']=='seconds_per_event' and r.get('sourceClass') and r.get('boundary') and r.get('lifetime'),'Typed coefficient ownership')
  if r['kind']=='known':
   need(type(r['value']) in (int,float) and math.isfinite(r['value']) and r['value']>=0,'Finite nonnegative nonboolean time')
   need(name in KNOWN and r==coefficients()[name],'Only original complete labelled assumption records')
  else:need(r['kind']=='symbol' and name in UNKNOWN and r['value'] is None and r['nonnegative'] is True and r==coefficients()[name],'Unknown time cannot become zero/default')
 return c


class Trace:
 def __init__(self,prefix,origin='task_supplied_H'):self.prefix=prefix;self.events=[];self.origin=origin
 def add(self,op,n=1,block='target',ref='local'):
  need(op in KNOWN or op in UNKNOWN,'Declared primitive')
  need(type(n) is int and n>0,'Positive exact event count')
  self.events.append({'id':self.prefix+':'+str(len(self.events)),'operation':op,'coefficientId':op,
   'count':n,'block':block,'inputOrigin':('declared_algorithm' if op=='retrieve_rule' else self.origin if block=='input' else 'history_access' if block=='history' else 'algorithm_local'),
   'inputRef':ref,'outputRef':self.prefix+':value:'+str(len(self.events)),'stateVersion':self.prefix.split(':')[0],'lifetime':'invocation'})
 def read(self,ref,value):self.add('read_scalar',block='input',ref=ref);return value
 def finish(self):self.add('local_access',ref='bounded_invocation_workspace');self.add('native_select',block='native');self.add('common_execution',block='response')


def counts(events):
 out={}
 for e in events:out[e['operation']]=out.get(e['operation'],0)+e['count']
 return out


def validate_trace(events):
 need(len({e['id'] for e in events})==len(events),'Duplicate trace event identity')
 for e in events:
  need(type(e['count']) is int and e['count']>0 and e['operation']==e['coefficientId'],'Trace count/ownership')
  need(e['operation'] in KNOWN or e['operation'] in UNKNOWN,'Undeclared primitive')
  need(e.get('inputRef') and e.get('outputRef') and e.get('stateVersion') and e['lifetime']=='invocation','Bound trace references/lifetime')
  if e['operation']=='common_execution':need(e['block']=='response','Response sole cost owner')
  if e['operation']=='request_execution':need(e['block']=='request','Request separate cost owner')
  if e['operation']=='clip':need(e['block']=='native','Native clip sole owner')
  if e['operation']=='retrieve_rule':need(e['inputOrigin']=='declared_algorithm','Rule is not screen supplied')
  need(e['block'] in ('input','target','native','response','history','reconstruction','cache','request'),'Cost block')
  need(e['inputOrigin'] in ('task_supplied_H','algorithm_local','recipe_supplied','history_access','declared_algorithm'),'No private actor origin')
 need(sum(e['count'] for e in events if e['operation']=='common_execution')<=1,'Duplicate final response')
 need(sum(e['count'] for e in events if e['operation']=='clip')<=1,'Duplicate clipping')


def affine(events,c):
 validate_coefficients(c);validate_trace(events);known=0.;symbols={}
 for e in events:
  r=c[e['coefficientId']]
  if r['kind']=='known':known+=e['count']*r['value']
  else:symbols[r['symbol']]=symbols.get(r['symbol'],0)+e['count']
 return {'knownSeconds':known,'symbols':symbols,'units':'personal_seconds'}


def exact_seconds(expression):
 need(not expression['symbols'],'Exact duration requires every used coefficient')
 return expression['knownSeconds']


def near(a,b,t):
 t.add('add_subtract');delta=abs(a-b);t.add('abs');t.add('compare',2)
 maximum=max(1,a,b);t.add('multiply_divide');epsilon=1e-9*maximum;t.add('compare')
 return delta<=epsilon


def project(x,t):
 t.add('clip',block='native');x=min(1.,max(0.,x));t.add('multiply_divide',block='native');z=x*1000
 t.add('floor',block='native');lo=math.floor(z);t.add('ceil',block='native');hi=math.ceil(z)
 t.add('compare',block='native');js=sorted({lo,hi});dist={}
 for j in js:t.add('add_subtract',block='native');t.add('abs',block='native');dist[j]=abs(j-z)
 if len(js)>1:t.add('compare',len(js)-1,block='native')
 minimum=min(dist.values());t.add('add_subtract',block='native');threshold=minimum+1e-12
 chosen=[]
 for j in js:
  t.add('compare',block='native')
  if dist[j]<=threshold:chosen.append(j)
 for j in chosen:t.add('add_subtract',block='native');t.add('abs',block='native')
 if len(chosen)>1:t.add('compare',len(chosen)-1,block='native')
 return min(chosen,key=lambda j:(abs(j-500),j))


def validate_H(H):
 need(type(H) is dict and set(H)==H_FIELDS,'Actor accepts H only; private/missing/extra operand')
 need(type(H['budgetInteger']) is int and H['budgetInteger']>0,'Fixed-scope positive integer resource')
 need(type(H['meanStrings']) is list and len(H['meanStrings'])==2,'Two display means')
 for value in H['meanStrings']:
  need(type(value) is str and re.fullmatch(r'[0-9]+\.[0-9]',value) is not None and 1<=float(value)<=100,'Finite one-decimal display mean')
 resource=H['resourceString']
 need(type(resource) is str and re.fullmatch(r'[0-9]+(?:\.[0-9]{1,3})?',resource) is not None,'Exact resource display string')
 need(Fraction(resource)*1000==H['budgetInteger'],'Resource display/metadata binding')
 need(type(H['stateId']) is str and len(H['stateId'])>0,'Bound state identity')
 need(type(H['ranges']) is list and len(H['ranges'])==2 and all(type(r) is dict and set(r)=={'lower','upper'} and type(r['lower']) is int and type(r['upper']) is int and 1<=r['lower']<=r['upper']<=100 for r in H['ranges']),'Finite integer display ranges')
 need(type(H['channel']) is dict and set(H['channel'])=={'source','reportWidthHex'} and H['channel']['source']=='task_width_metadata_not_actor_operand' and math.isfinite(float.fromhex(H['channel']['reportWidthHex'])) and float.fromhex(H['channel']['reportWidthHex'])>0,'Bound channel metadata')
 need(type(H['latest']) is list and len(H['latest'])==2 and all(x is None or type(x) is int and 1<=x<=100 for x in H['latest']),'Attributed latest reports')
 need(type(H['history']) is list and all(type(e) is dict and set(e)=={'slot','report'} and type(e['slot']) is int and e['slot'] in (0,1) and type(e['report']) is int and 1<=e['report']<=100 for e in H['history']),'Attributed accessible history')
 latest=[None,None]
 for e in H['history']:latest[e['slot']]=e['report']
 need(latest==H['latest'],'Latest/history identity')
 return H


def valid_rho(rho):need(type(rho) in (int,float) and math.isfinite(rho) and rho>=0,'Finite nonnegative nonboolean rho')


def actor(H,family,c=None):
 if c is None:c=coefficients()
 validate_H(H)
 need(family in CHEAP,'Only declared cheap actor can execute')
 need(type(H['budgetInteger']) is int and H['budgetInteger']>0,'Positive exact resource')
 t=Trace(H['stateId']+':'+family);t.add('retrieve_rule',block='input',ref=family)
 a=b=B=None
 if family not in ('ES','MFN_B'):a=t.read('meanStrings[0]',float(H['meanStrings'][0]))
 if family not in ('ES','MFN_A'):b=t.read('meanStrings[1]',float(H['meanStrings'][1]))
 if family in ('EO_raw_surplus','ML','MFN_A','MFN_B'):B=t.read('resourceString',float(H['resourceString']))
 known=None;x=None
 if family=='ES':known=500
 elif family=='EO_raw_surplus':
  t.add('add_subtract',ref='B+a -> numerator1');n1=B+a
  t.add('add_subtract',ref='numerator1-b -> numerator');n=n1-b
  t.add('add_subtract',ref='B+B -> denominator');d=B+B
  t.add('multiply_divide',ref='numerator/denominator -> rawShare');x=n/d
 elif family=='PN':t.add('add_subtract');d=a+b;t.add('multiply_divide');x=a/d
 elif family in ('AL','GN','ML'):
  if near(a,b,t):known=500
  else:
   t.add('compare');lower=a<b
   if family=='AL':known=1000 if lower else 0
   elif family=='GN':known=0 if lower else 1000
   else:
    t.add('compare');v=min(B,a if lower else b);t.add('multiply_divide');x=v/B
    if not lower:t.add('add_subtract');x=1-x
 elif family=='MFN_A':t.add('compare');v=min(B,a);t.add('multiply_divide');x=v/B
 elif family=='MFN_B':t.add('compare');v=min(B,b);t.add('multiply_divide');x=v/B;t.add('add_subtract');x=1-x
 j=known if known is not None else project(x,t);t.finish()
 return {'family':family,'algorithmId':'H-literal-bplusb-v1','status':'identified_target_symbolic_cost',
  'target':{'kind':'KnownIndex' if known is not None else 'Fraction','index':j,'rawShare':x},
  'arithmeticTree':({'numerator1':n1,'numerator':n,'denominator':d,'rawShare':x,'expression':'((B+a)-b)/(B+B)'} if family=='EO_raw_surplus' else None),
  'trace':t.events,'counts':counts(t.events),'durationExpression':affine(t.events,c),
  'actorInputPermission':'H_only','actor_target_certified':True}
