"""Frozen synthetic method calibration, not a finite-task/human policy generator.

Production estimators remain read-only. Gaussian endpoint summaries are an abstract
statistical idealization; bounded row parity is a distinct input/arithmetic layer.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np

from .inference import (CONDITIONS, CORE, FAMILIES, contrast, fit, multiplicity,
                        participant_slope, scipy_stats, synthetic_example_rows)
from .planning import covariance_matrix, one_sample_power, weighted_contrast_power, joint_bounds

ENDPOINTS={"anyQuestion":"q","EOResemblance":"R","lowerNeedDirectional":"L","slope_initial":"initial"}
WINDOWS=("primary","early","late","available_history")


def resolved_cells(config):
    if config.get("purpose")!="synthetic_statistical_calibration_not_human_evidence":raise ValueError("Invalid calibration purpose")
    cells=[{**config["defaults"],**c} for c in config["cells"]]
    if not isinstance(config["batchSize"],int) or isinstance(config["batchSize"],bool) or config["batchSize"]<1:raise ValueError("Invalid fixed batch size")
    if set(config["policies"])!={"global_holm","within_family_holm","four_family_alpha_budget"} or len(config["policies"])!=3:raise ValueError("All three fixed policies required")
    if not math.isfinite(config["alpha"]) or not 0<config["alpha"]<1:raise ValueError("Invalid alpha")
    if len({c["id"] for c in cells})!=len(cells) or len({c["seed"] for c in cells})!=len(cells):raise ValueError("Duplicate cell identity/seed")
    for c in cells:
        if not isinstance(c["replications"],int) or c["replications"]<1:raise ValueError("Invalid replication count")
        if not isinstance(c["seed"],int) or c["seed"]<0:raise ValueError("Invalid seed")
        if set(c["people"])!=set(CONDITIONS) or any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in c["people"].values()):raise ValueError("Positive integer condition counts required")
        if c["generator"] not in ("gaussian","bounded","t5","gamma","invalid"):raise ValueError("Unknown generator")
        if c["eligibility"]=="complete_ten" and c["dose"]!=10:raise ValueError("Current primary dose must be ten")
        for p in ("participantRetention","trialRetention","endpointMissingProbability","commonEndpointComponent"):
            if not 0<=c[p]<=1:raise ValueError("Invalid retention probability")
        if any(not math.isfinite(c[p]) for p in ("randomSlopeVariance","residualVariance","learning","orderTrend")) or min(c["randomSlopeVariance"],c["residualVariance"])<0:raise ValueError("Invalid variance/trend parameters")
        if any(not math.isfinite(s) or s<0 for s in c["meanTrialSD"].values()):raise ValueError("Invalid endpoint SD")
        covariance_matrix(c["dose"],c["residualVariance"],c["correlation"],c["covarianceKind"])
    return cells


def clip_normal_mean(mu,sd,lo,hi):
    """Exact marginal E[clip(N(mu,sd²),lo,hi)], with deterministic limit."""
    mu,sd=np.broadcast_arrays(np.asarray(mu,float),np.asarray(sd,float))
    if np.any(sd<0) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)) or lo>=hi:raise ValueError("Invalid clipped-normal parameters")
    stats=scipy_stats();safe=np.where(sd>0,sd,1.)
    a=(lo-mu)/safe;b=(hi-mu)/safe
    result=lo*stats.norm.cdf(a)+mu*(stats.norm.cdf(b)-stats.norm.cdf(a))+safe*(stats.norm.pdf(a)-stats.norm.pdf(b))+hi*stats.norm.sf(b)
    return np.where(sd>0,result,np.clip(mu,lo,hi))


def binomial_interval(successes,attempts):
    if not isinstance(successes,(int,np.integer)) or not isinstance(attempts,(int,np.integer)) or not 0<=successes<=attempts:raise ValueError("Invalid binomial counts")
    if attempts==0:return {"rate":None,"interval":None,"mcse":None,"numerator":int(successes),"denominator":0}
    ci=scipy_stats().binomtest(int(successes),int(attempts)).proportion_ci(.95,method="exact")
    rate=successes/attempts
    return {"rate":rate,"interval":[float(ci.low),float(ci.high)],"mcse":math.sqrt(rate*(1-rate)/attempts),
            "numerator":int(successes),"denominator":int(attempts),"intervalMethod":"two_sided_95_Clopper_Pearson_exact"}


def _innovations(rng,shape,kind):
    if kind=="t5":return rng.standard_t(5,size=shape)*math.sqrt(3/5)
    if kind=="gamma":return (rng.gamma(.5,size=shape)-.5)/math.sqrt(.5)
    return rng.standard_normal(shape)


def _effects(cell,cid):
    q=.3;beta=0.;R=0.;L=0.
    mode=cell["effects"]
    if mode in ("signed","opposite"):
        sign=1 if mode=="signed" else -1
        if cid in ("C2","C4"):q=.3-.1*sign
        if cid in ("C3","C5"):q=.3+.1*sign
        if cid in ("C2","C3"):beta=.2*sign;R=2.*sign
        if cid=="C4":beta=-.2*sign
        if cid in ("C4","C5"):L=.04*sign
    elif mode=="mixed":
        if cid=="C2":q=.2;beta=.2;R=2.
        if cid=="C3":q=.4
    elif mode!="null":raise ValueError("Unknown frozen effects")
    return q,beta,R,L


def _schedule(cell,cid,n):
    m=cell["dose"];x=np.linspace(-.2,.2,m)
    if cid=="C3" or cell["predictorKind"]=="two_level":x=np.where(np.arange(m)%2,.15,-.15)
    if cell["predictorKind"]=="recipe_heterogeneous" and cid in ("C2","C4"):x=np.asarray(cell["recipeX"],float)
    if cell["predictorKind"]=="narrow":x=x*.01
    if cid in ("C1","C5") or cell["predictorKind"]=="constant":x=np.zeros(m)
    xs=np.repeat(x[None,:],n,axis=0)
    if cell["predictorKind"]=="recipe_heterogeneous":xs[1::2]*=.5
    return xs


def _mean_summary(data,valid,expected,cov,sd,keep,clip_bounds=None,shared_coefficient=None,mnar=False):
    count=valid.sum(axis=-1);weights=valid/np.maximum(count[...,None],1)
    values=np.sum(np.where(valid,data,0.),axis=-1)/np.maximum(count,1)
    truth=np.sum(weights*expected,axis=-1)
    variance=sd*sd*np.einsum("bni,ij,bnj->bn",weights,cov,weights)
    coefficient=sd*np.einsum("bni,i->bn",weights,shared_coefficient) if shared_coefficient is not None else np.zeros_like(values)
    selected_truth=truth+coefficient*math.sqrt(2/math.pi) if mnar else truth.copy()
    if mnar:variance=variance-(2/math.pi)*coefficient**2
    if clip_bounds is not None:
        values=np.clip(values,*clip_bounds)
        truth=clip_normal_mean(truth,np.sqrt(np.maximum(variance,0)),*clip_bounds)
        selected_truth=truth.copy();variance=np.full_like(variance,np.nan) # postclip planning variance deliberately not supplied
    available=(count>0)&keep
    return {"values":np.where(available,values,np.nan),"truth":truth,"selectedTruth":selected_truth,
            "variance":variance,"observations":count,"available":available}


def _slope_summary(y,x,valid,expected,cov,tau_variance,keep,selected_shift=0.,model="unadjusted",scale=None,gap=None):
    b,n,m=y.shape;count=valid.sum(axis=-1)
    if model=="unadjusted":
        average=np.sum(np.where(valid,x[None,:,:],0.),axis=-1)/np.maximum(count,1)
        centered=np.where(valid,x[None,:,:]-average[...,None],0.)
        sxx=np.sum(centered**2,axis=-1)
        weights=centered/np.maximum(sxx[...,None],1e-300)
        available=(count>=3)&(sxx>1e-12)&keep
        values=np.sum(weights*np.where(valid,y,0.),axis=-1)
        truth=np.sum(weights*(expected-np.mean(expected,axis=-1,keepdims=True)),axis=-1)
        variance=np.einsum("bni,ij,bnj->bn",weights,cov,weights)+tau_variance
    else:
        values=np.full((b,n),np.nan);truth=np.zeros((b,n));variance=np.full((b,n),np.nan);available=np.zeros((b,n),bool)
        sxx=np.sum((x-x.mean(axis=-1,keepdims=True))**2,axis=-1)[None,:].repeat(b,axis=0)
        for p in range(n):
            columns=[np.ones(m),x[p]-x[p].mean(),np.arange(1,m+1)-np.mean(np.arange(1,m+1))]
            if model=="order_recipe":columns.extend([scale-np.mean(scale),gap-np.mean(gap)])
            X=np.column_stack(columns)
            if np.linalg.matrix_rank(X)!=X.shape[1] or m<=X.shape[1]:continue
            weights=np.linalg.pinv(X)[1]
            values[:,p]=y[:,p]@weights
            truth[:,p]=float(weights@(expected[0,p]-np.mean(expected[0,p])))
            variance[:,p]=float(weights@cov@weights)+tau_variance
            available[:,p]=keep[:,p]&np.all(valid[:,p],axis=-1)
    truth=np.where(np.abs(truth)<1e-12,0.,truth)
    return {"values":np.where(available,values,np.nan),"truth":truth,"selectedTruth":truth+selected_shift,
            "variance":variance,"observations":count,"Sxx":sxx,"available":available}


def generate_batch(cell,batch_index,batch_size):
    """SeedSequence(seed,batch_index), fixed batch shape, independent conditions/persons."""
    rng=np.random.default_rng(np.random.SeedSequence([cell["seed"],batch_index]))
    m=cell["dose"];corr=covariance_matrix(m,1.,cell["correlation"],cell["covarianceKind"])
    factor=np.linalg.cholesky(corr);a=cell.get("commonEndpointComponent",0.)
    if not 0<=a<=1:raise ValueError("Invalid shared component")
    shared_coefficient=math.sqrt(a)*factor[:,0]
    populations={w:{} for w in WINDOWS}
    if cell.get("stochasticAdjustedModels"):
        for model in cell["stochasticAdjustedModels"]:populations[model]={}
    raw={}
    for cid in CONDITIONS:
        n=cell["people"][cid];q,beta,R,L=_effects(cell,cid)
        multiplier=cell.get("conditionSDMultiplier",{}).get(cid,1.)
        shared=rng.standard_normal((batch_size,n))
        person_keep=rng.random((batch_size,n))<cell["participantRetention"]
        if cell.get("mnarPositiveSharedZ"):person_keep&=shared>0
        records=rng.random((batch_size,n,m))<cell["trialRetention"]
        fixed=np.zeros(m,bool);fixed[cell.get("retainedIndices",list(range(m)))]=True
        records&=fixed[None,None,:]
        endpoint_masks={key:rng.random((batch_size,n,m))>=cell["endpointMissingProbability"] for key in ("q","R","L","initial")}
        xs=_schedule(cell,cid,n);time=np.linspace(-1,1,m)
        expected_y=.5+beta*(1+cell["learning"]*time)[None,:]*xs+cell["orderTrend"]*time[None,:]
        residual_variance=cell["residualVariance"]*multiplier**2;tau_variance=cell["randomSlopeVariance"]*multiplier**2
        slope_noise=_innovations(rng,(batch_size,n,m),cell["generator"])@factor.T*math.sqrt(residual_variance)
        random_slope=math.sqrt(tau_variance)*(math.sqrt(a)*shared+math.sqrt(1-a)*_innovations(rng,(batch_size,n),cell["generator"]))
        y=expected_y[None,:,:]+random_slope[...,None]*xs[None,:,:]+slope_noise
        if cell["generator"]=="bounded":
            y=np.clip(y,0,1)
            expected_y=clip_normal_mean(expected_y,np.sqrt(residual_variance+tau_variance*xs**2),0.,1.)
        mean_data={};means_expected={"q":q,"R":R,"L":L};sds={}
        for key,mu in means_expected.items():
            sd=cell["meanTrialSD"][key]*multiplier;sds[key]=sd
            z=_innovations(rng,(batch_size,n,m),cell["generator"])
            z[:,:,0]=math.sqrt(a)*shared+math.sqrt(1-a)*z[:,:,0]
            mean_data[key]=mu+sd*(z@factor.T)
        if cell["generator"]=="bounded":
            rho=cell["correlation"]
            p=rng.beta(q*(1/rho-1),(1-q)*(1/rho-1),size=(batch_size,n)) if rho>0 else np.full((batch_size,n),q)
            mean_data["q"]=(rng.random((batch_size,n,m))<p[...,None]).astype(float)
        if cell["generator"]=="invalid":
            mean_data["R"][:]=0.;mean_data["L"][:]=0.
            if cid=="C5":endpoint_masks["q"][:]=False
        expected=np.repeat(expected_y[None,:,:],batch_size,axis=0)
        for name in populations:
            select=np.ones(m,bool)
            if name=="early":select=np.isin(np.arange(m),[0,1,2,3,4])
            if name=="late":select=np.isin(np.arange(m),[5,6,7,8,9])
            keep=person_keep.copy()
            if name!="available_history" and cell["eligibility"]=="complete_ten":keep&=np.all(records,axis=-1)
            valid_records=records&select[None,None,:]
            group={}
            for key,mu in means_expected.items():
                mask=valid_records&endpoint_masks[key]
                bounds=(-20.,20.) if key=="R" else (-.5,.5) if key=="L" else None
                if cell["generator"]!="bounded":bounds=None
                g=_mean_summary(mean_data[key],mask,np.full((batch_size,n,m),mu),corr,sds[key],keep,bounds,shared_coefficient,cell.get("mnarPositiveSharedZ",False))
                if cell["generator"]=="bounded" and key=="q":
                    g["variance"]=q*(1-q)*(cell["correlation"]+(1-cell["correlation"])/np.maximum(g["observations"],1))
                if cell["generator"]=="invalid" and key in ("R","L"):g["variance"][:]=0.
                group[key]=g
            model=name if name in ("order_only","order_recipe") else "unadjusted"
            scale=np.asarray(cell.get("recipeScale",np.resize([.8,1,1.2],m)),float)
            gap=np.asarray(cell.get("recipeGap",np.resize([-12,-12,0,0,12,12],m)),float)
            if cid=="C3":gap=np.zeros(m)
            shift=math.sqrt(tau_variance*a)*math.sqrt(2/math.pi) if cell.get("mnarPositiveSharedZ") else 0.
            slope=_slope_summary(y,xs,valid_records&endpoint_masks["initial"],expected,residual_variance*corr,tau_variance,keep,shift,model,scale,gap)
            if cell.get("mnarPositiveSharedZ"):slope["variance"]-=(2/math.pi)*tau_variance*a
            if cell["generator"]=="bounded":slope["variance"][:]=np.nan
            if cid in ("C1","C5"):slope["available"][:]=False;slope["values"][:]=np.nan
            group["initial"]=slope;populations[name][cid]=group
        raw[cid]={"y":y,"x":xs,"records":records,"endpointMasks":endpoint_masks,"personKeep":person_keep,
                  "expectedY":expected_y,"scale":scale,"gap":gap}
    return {"populations":populations,"raw":raw}


def _group(values,truth,selected_truth):
    valid=np.isfinite(values);n=valid.sum(axis=1)
    mean=np.sum(np.where(valid,values,0.),axis=1)/np.maximum(n,1)
    centered=np.where(valid,values-mean[:,None],0.)
    variance=np.sum(centered**2,axis=1)/np.maximum(n-1,1)
    t=np.sum(np.where(valid,truth,0.),axis=1)/np.maximum(n,1)
    selected=np.sum(np.where(valid,selected_truth,0.),axis=1)/np.maximum(n,1)
    return n,mean,variance,t,selected


def vector_contrast(groups,endpoint,weights,side):
    pieces=[];denominator=[];estimate=0.;truth=0.;selected=0.;enough=True;estimable=True
    for cid,w in weights.items():
        g=groups[cid][endpoint];n,mu,variance,t,ts=_group(g["values"],g["truth"],g["selectedTruth"])
        estimate=estimate+w*mu;truth=truth+w*t;selected=selected+w*ts;enough=enough&(n>=2);estimable=estimable&(n>=1)
        piece=w*w*variance/np.maximum(n,1);pieces.append(piece);denominator.append(piece*piece/np.maximum(n-1,1))
    total=np.sum(pieces,axis=0);se=np.sqrt(np.maximum(total,0.));df=total*total/np.maximum(np.sum(denominator,axis=0),1e-300)
    available=enough&(total>0)&np.isfinite(total)&np.isfinite(df)&(df>0)
    statistic=estimate/np.maximum(se,1e-300);stats=scipy_stats();safe_df=np.where(available,df,1.)
    p=stats.t.sf(statistic,safe_df) if side=="greater" else stats.t.cdf(statistic,safe_df)
    half=stats.t.ppf(.975,safe_df)*se
    reason=np.where(~enough,1,np.where(total<=0,2,np.where(~available,3,0))).astype(np.int8)
    return {"estimate":np.where(estimable,estimate,np.nan),"se":np.where(available,se,np.nan),"df":np.where(available,df,np.nan),
            "statistic":np.where(available,statistic,np.nan),"p":np.where(available,p,np.nan),"low":np.where(available,estimate-half,np.nan),
            "high":np.where(available,estimate+half,np.nan),"available":available,"reason":reason,"truth":np.where(estimable,truth,np.nan),
            "selectedTruth":np.where(estimable,selected,np.nan),"null":(truth<=1e-12 if side=="greater" else truth>=-1e-12),
            "selectedNull":(selected<=1e-12 if side=="greater" else selected>=-1e-12)}


def core_statistics(groups):
    results=[vector_contrast(groups,ENDPOINTS[endpoint],weights,side) for _,_,endpoint,weights,side in CORE]
    return {k:np.column_stack([r[k] for r in results]) for k in results[0]}


def vector_holm(p,available,policy,alpha=.05):
    effective=np.where(available,p,1.);adjusted=np.full_like(p,np.nan);reject=np.zeros_like(available)
    groups=[(list(range(9)),alpha)] if policy=="global_holm" else [([i for i,d in enumerate(CORE) if d[1]==f],alpha/4 if policy=="four_family_alpha_budget" else alpha) for f in FAMILIES]
    for indices,budget in groups:
        values=effective[:,indices];order=np.argsort(values,axis=1,kind="stable");ranked=np.take_along_axis(values,order,axis=1)
        size=len(indices);multipliers=np.arange(size,0,-1)
        adj=np.minimum(1.,np.maximum.accumulate(ranked*multipliers,axis=1))
        rej=np.logical_and.accumulate(ranked<=budget/multipliers,axis=1)
        inverse=np.argsort(order,axis=1);adjusted[:,indices]=np.take_along_axis(adj,inverse,axis=1);reject[:,indices]=np.take_along_axis(rej,inverse,axis=1)
    adjusted=np.where(available,adjusted,np.nan);reject&=available
    return adjusted,reject


def scalar_parity(cell,batch,statistics,policy,index,alpha=.05,population="primary"):
    groups=batch["populations"][population];comparisons=[];core=[]
    for j,(identifier,family,endpoint,weights,side) in enumerate(CORE):
        values=[]
        for cid in weights:
            for person,value in enumerate(groups[cid][ENDPOINTS[endpoint]]["values"][index]):
                values.append({"participantId":f"{cid}-{person}","conditionId":cid,"status":"available" if np.isfinite(value) else "unavailable","value":float(value) if np.isfinite(value) else None})
        r=contrast(values,weights,side);r.update(id=identifier,family=family);core.append(r)
        for scalar,key in (("estimate","estimate"),("standardError","se"),("df","df"),("statistic","statistic"),("pValue","p")):
            expected=r[scalar];actual=statistics[key][index,j]
            if expected is None:
                if np.isfinite(actual):raise AssertionError(f"Unavailable scalar parity {identifier}/{key}")
            elif not np.isclose(expected,actual,atol=1e-10,rtol=1e-9):raise AssertionError(f"Scalar parity {identifier}/{key}: {expected} != {actual}")
        if r["interval"] is not None:
            if not np.allclose(r["interval"],[statistics["low"][index,j],statistics["high"][index,j]],atol=1e-10,rtol=1e-9):raise AssertionError("Interval parity")
        comparisons.append({"id":identifier,"status":r["status"],"reason":r["reason"]})
    correction=multiplicity(core,policy,alpha);adj,rej=vector_holm(statistics["p"],statistics["available"],policy,alpha)
    for j,r in enumerate(correction["tests"]):
        if r["reject"]!=bool(rej[index,j]):raise AssertionError("Holm reject parity")
        if r["adjustedP"] is not None and not np.isclose(r["adjustedP"],adj[index,j],atol=1e-10):raise AssertionError("Holm adjusted parity")
    # Gaussian trial person projection is compared with the unchanged production OLS.
    for cid in ("C2","C3","C4"):
        raw=batch["raw"][cid]
        for person in (0,cell["people"][cid]-1):
            valid=raw["records"][index,person]&raw["endpointMasks"]["initial"][index,person]
            if population=="early":valid&=np.arange(cell["dose"])<5
            if population=="late":valid&=(np.arange(cell["dose"])>=5)&(np.arange(cell["dose"])<10)
            rows=[{"xInitial":float(raw["x"][person,t]),"shareA":float(raw["y"][index,person,t]),"stageIndex":t+1,
                   "scale":float(raw["scale"][t]),"gapOffset":float(raw["gap"][t])} for t in range(cell["dose"]) if valid[t]]
            model=population if population in ("order_only","order_recipe") else "unadjusted"
            r=participant_slope(rows,"xInitial",model)
            actual=groups[cid]["initial"]["values"][index,person]
            admitted=raw["personKeep"][index,person] and (population=="available_history" or cell["eligibility"]!="complete_ten" or np.all(raw["records"][index,person]))
            if admitted and r["status"]=="available" and not np.isclose(actual,r["value"],atol=1e-10,rtol=1e-9):raise AssertionError("Person OLS parity")
            if (not admitted or r["status"]!="available") and np.isfinite(actual):raise AssertionError("Unavailable person OLS parity")
    return {"cell":cell["id"],"population":population,"replica":index,"policy":policy,"tests":comparisons,"status":"PASS"}


def full_row_parity(config):
    """Bounded metric-row layer, separate from abstract Gaussian person endpoints."""
    import copy
    evidence=[];settings=config["fullRowParity"]
    for replica in settings["replicaIndices"]:
        rng=np.random.default_rng(np.random.SeedSequence([settings["seed"],replica]))
        baseline=synthetic_example_rows(6)
        for r in baseline:
            r["shareA"]=float(np.clip(r["shareA"]+rng.normal(0,.001),0.,1.))
            target=.5+.18*r["xTerminal"]
            r["distancePP_equal_split"]=100*abs(r["shareA"]-.5);r["distancePP_equal_outcome"]=100*abs(r["shareA"]-target)
            r["EOResemblance"]=r["distancePP_equal_split"]-r["distancePP_equal_outcome"]
            r["lowerNeedDirectional"]=0. if r["terminalTie"] else .5-r["shareA"] if r["xTerminal"]>0 else r["shareA"]-.5
        for case in settings["cases"]:
            rows=copy.deepcopy(baseline)
            if case=="partial_records":rows=[r for r in rows if not(r["participantId"].endswith("00") and r["trialId"]=="main-10")]
            if case=="endpoint_missing":
                for r in rows:
                    if r["trialId"]=="main-01" and r["participantId"].endswith("00"):r["EOResemblance"]=None;r["distancePP_equal_split"]=None;r["distancePP_equal_outcome"]=None;r["shareA"]=None
            if case=="all_ties":
                for r in rows:
                    r["terminalTie"]=True;r["xTerminal"]=0.;r["lowerNeedDirectional"]=0.
                    r["distancePP_equal_outcome"]=r["distancePP_equal_split"];r["EOResemblance"]=0.
            if case=="rank_deficient":
                for r in rows:r["xInitial"]=0.;r["initialNeedGapRaw"]=0.
            if case=="zero_uncertainty":
                for r in rows:r["anyQuestion"]=0;r["questions"]=0;r["shareA"]=.5;r["distancePP_equal_split"]=0.;r["distancePP_equal_outcome"]=0.;r["EOResemblance"]=0.;r["lowerNeedDirectional"]=0.;r["terminalTie"]=True;r["xTerminal"]=0.
            outputs={policy:fit(rows,policy=policy) for policy in config["policies"]}
            reference=outputs["global_holm"]
            for policy,out in outputs.items():
                if len(out["tests"])!=84:raise AssertionError("Full roster changed")
                by_id={r["id"]:r for r in reference["tests"]}
                for r in out["tests"]:
                    q=by_id[r["id"]]
                    if r["status"]!=q["status"] or r["estimate"]!=q["estimate"] or r["pValue"]!=q["pValue"]:raise AssertionError("Policy changed base estimator")
                # Independent OLS design arithmetic for every available slope/sensitivity.
                for p in out["participantEstimates"]:
                    if p["status"]!="available" or not p["endpoint"].startswith("slope_"):continue
                    selected=[r for r in rows if r["participantId"]==p["participantId"] and r["analysisEligible"] and (p["window"]=="all" or (r["stageIndex"]<=5 if p["window"]=="early" else r["stageIndex"]>5))]
                    predictor={"slope_initial":"xInitial","slope_initial_raw":"initialNeedGapRaw","slope_terminal":"xTerminal"}[p["endpoint"]]
                    selected=[r for r in selected if r["shareA"] is not None and r[predictor] is not None]
                    names=[] if p["model"]=="unadjusted" else ["stageIndex"] if p["model"]=="order_only" else ["stageIndex","scale","gapOffset"]
                    X=np.column_stack([np.ones(len(selected)),*[np.asarray([r[k] for r in selected])-np.mean([r[k] for r in selected]) for k in [predictor,*names]]])
                    value=float(np.linalg.pinv(X)[1]@np.asarray([r["shareA"] for r in selected]))
                    if not np.isclose(value,p["value"],atol=1e-10,rtol=1e-9):raise AssertionError("Full-row slope arithmetic parity")
                evidence.append({"case":case,"replica":replica,"policy":policy,"routes":len(out["tests"]),"participants":out["participants"],
                                 "coreUnavailable":out["multiplicity"]["retainedUnavailableSlots"],"status":"PASS",
                                 "unavailableRoutes":[r["id"] for r in out["tests"] if r["status"]!="available"],
                                 "participantEstimates":out["participantEstimates"],"tests":out["tests"]})
    return {"seed":settings["seed"],"origin":"bounded_metric_row_arithmetic_parity_not_finite_export_or_calibration","cases":evidence}


def _count(mask):
    return np.sum(mask,axis=0).astype(int).tolist()


def evaluate_batch(cell,config,batch_index,size):
    """Reduce one fixed batch; no calibration decisions change generator/support."""
    batch=generate_batch(cell,batch_index,size);populations={};parity=[]
    start=batch_index*config["batchSize"]
    requested={cell["replications"]-1 if i=="last" else i for i in config["replicaParityIndices"]}
    alpha=config["alpha"]
    for name,groups in batch["populations"].items():
        s=core_statistics(groups);a=s["available"];finite=np.isfinite(s["estimate"])
        covered=a&(s["low"]<=s["truth"])&(s["truth"]<=s["high"])
        selected_covered=a&(s["low"]<=s["selectedTruth"])&(s["selectedTruth"]<=s["high"])
        data={"attempts":size,"available":_count(a),"estimable":_count(finite),
              "unavailableInsufficientPeople":_count(s["reason"]==1),"unavailableZeroUncertainty":_count(s["reason"]==2),
              "unexpectedNumericalFailure":_count(s["reason"]==3),"coverage":_count(covered),"selectedCoverage":_count(selected_covered),
              "trueNullAttempts":_count(s["null"]),"selectedNullAttempts":_count(s["selectedNull"]),
              "rawReject":_count(a&(s["p"]<=alpha)),"rawFalseReject":_count(a&(s["p"]<=alpha)&s["null"]),
              "rawSelectedFalseReject":_count(a&(s["p"]<=alpha)&s["selectedNull"]),
              "firstGlobalThresholdReject":_count(a&(s["p"]<=alpha/9)),
              "estimateSum":np.nansum(s["estimate"],axis=0).tolist(),"truthSum":np.nansum(s["truth"],axis=0).tolist(),
              "selectedTruthSum":np.nansum(s["selectedTruth"],axis=0).tolist(),"policies":{},"personSupport":{}}
        for cid,gg in groups.items():
            data["personSupport"][cid]={}
            for endpoint,g in gg.items():
                v=g["available"];observed=g["observations"]
                data["personSupport"][cid][endpoint]={"personAttempts":int(v.size),"availablePeople":int(v.sum()),
                    "observedCountHistogram":np.bincount(observed.ravel(),minlength=cell["dose"]+1).astype(int).tolist(),
                    "availableObservationSum":int(np.sum(np.where(v,observed,0))),
                    "knownVarianceSum":float(np.nansum(np.where(v,g["variance"],np.nan))),
                    "knownVarianceCount":int(np.sum(v&np.isfinite(g["variance"]))),
                    "knownTruthMin":float(np.min(g["truth"][v])) if v.any() else None,
                    "knownTruthMax":float(np.max(g["truth"][v])) if v.any() else None}
        for policy in config["policies"]:
            _,reject=vector_holm(s["p"],a,policy,alpha)
            alternative=~s["null"];selected_alt=~s["selectedNull"]
            has_alt=alternative.any(axis=1);has_selected_alt=selected_alt.any(axis=1)
            record={"reject":_count(reject),"falseReject":_count(reject&s["null"]),"selectedFalseReject":_count(reject&s["selectedNull"]),
                    "anyCoreReject":int(reject.any(axis=1).sum()),"allCoreReject":int(reject.all(axis=1).sum()),
                    "anyTrueNullReject":int((reject&s["null"]).any(axis=1).sum()),
                    "anySelectedNullReject":int((reject&s["selectedNull"]).any(axis=1).sum()),
                    "alternativeSubsetAttempts":int(has_alt.sum()),
                    "anyAlternativeDiscovery":int((reject&alternative).any(axis=1).sum()),
                    "allAlternativeDiscovery":int((has_alt&np.all(reject|~alternative,axis=1)).sum()),
                    "selectedAlternativeSubsetAttempts":int(has_selected_alt.sum()),
                    "anySelectedAlternativeDiscovery":int((reject&selected_alt).any(axis=1).sum()),
                    "allSelectedAlternativeDiscovery":int((has_selected_alt&np.all(reject|~selected_alt,axis=1)).sum()),"families":{}}
            thresholds=np.asarray([alpha/9 if policy=="global_holm" else alpha/FAMILIES[t[1]]/(4 if policy=="four_family_alpha_budget" else 1) for t in CORE])
            record["firstThresholdReject"]=_count(a&(s["p"]<=thresholds))
            for family in FAMILIES:
                js=[j for j,t in enumerate(CORE) if t[1]==family]
                record["families"][family]={"anyTrueNullReject":int((reject[:,js]&s["null"][:,js]).any(axis=1).sum()),
                                           "anySelectedNullReject":int((reject[:,js]&s["selectedNull"][:,js]).any(axis=1).sum())}
            data["policies"][policy]=record
            for i in sorted(requested):
                if start<=i<start+size:
                    item=scalar_parity(cell,batch,s,policy,i-start,alpha,name);item["replica"]=i;parity.append(item)
        populations[name]=data
    return {"cell":cell["id"],"batchIndex":batch_index,"firstReplica":start,"attempts":size,"seed":cell["seed"],
            "populations":populations,"productionParity":parity}


def merge_counts(records):
    """Only sum counters/sufficient sums; retain truth extrema, never average rates."""
    if not records:return None
    def merge(a,b,key=""):
        if isinstance(a,dict):return {k:merge(a[k],b[k],k) for k in a}
        if isinstance(a,list):return [merge(x,y,key) for x,y in zip(a,b)]
        if key.endswith("Min"):return b if a is None else a if b is None else min(a,b)
        if key.endswith("Max"):return b if a is None else a if b is None else max(a,b)
        return a+b
    import copy
    output=copy.deepcopy(records[0])
    for r in records[1:]:output=merge(output,r)
    return output


def power_comparison(cell,config,aggregated):
    """Use analytical generator moments; no simulated variance/effect planning estimate."""
    variable=(cell["generator"]!="gaussian" or cell["participantRetention"]!=1 or cell["trialRetention"]!=1
              or cell["endpointMissingProbability"]!=0 or cell.get("mnarPositiveSharedZ") or cell["predictorKind"]=="recipe_heterogeneous")
    if variable:return {"status":"not_comparable","reason":"Non-Gaussian or heterogeneous selected-person/count/moment scenario; no fictitious exact power.","comparisons":[]}
    reference=generate_batch(cell,0,1)["populations"]["primary"];comparisons=[]
    policy_marginals={p:{} for p in config["policies"]}
    n=aggregated["attempts"];alpha=config["alpha"]
    for j,(identifier,family,endpoint,weights,side) in enumerate(CORE):
        key=ENDPOINTS[endpoint];groups={cid:{"n":cell["people"][cid],"mean":float(reference[cid][key]["truth"][0,0]),
                                                 "variance":float(reference[cid][key]["variance"][0,0])} for cid in weights}
        powers={};thresholds=[(alpha,"raw_alpha",aggregated["rawReject"][j])]
        for policy in config["policies"]:
            threshold=alpha/9 if policy=="global_holm" else alpha/FAMILIES[family]/(4 if policy=="four_family_alpha_budget" else 1)
            thresholds.append((threshold,"first_"+policy,aggregated["policies"][policy]["firstThresholdReject"][j]))
        for threshold,label,count in thresholds:
            if len(weights)==1:
                cid=next(iter(weights));g=groups[cid]
                result=one_sample_power(g["mean"]*weights[cid],g["variance"]*weights[cid]**2,g["n"],threshold,side)
            else:result=weighted_contrast_power(groups,weights,threshold,side)
            empirical=binomial_interval(count,n)
            difference=None if result["power"] is None else abs(empirical["rate"]-result["power"])
            comparisons.append({"id":identifier,"threshold":label,"planning":result,"empirical":empirical,"absoluteDifference":difference,
                                "withinFrozenDifferenceTolerance":None if difference is None else difference<=config["criteria"]["powerAbsoluteDifference"]})
            powers[label]=result
        for policy in config["policies"]:
            ps=powers["first_"+policy];raw=powers["raw_alpha"]
            policy_marginals[policy][identifier]={"status":"available" if ps["status"]==raw["status"]=="available" else "unavailable",
                               "firstThresholdPower":ps["power"],"rawAlphaPower":raw["power"]}
    policy_bounds={}
    for policy in config["policies"]:
        bounds=joint_bounds(policy_marginals[policy],policy,alpha)
        policy_bounds[policy]={"bounds":bounds,"empiricalAny":aggregated["policies"][policy]["anyCoreReject"]/n,
                              "empiricalAll":aggregated["policies"][policy]["allCoreReject"]/n}
    return {"status":"conditional_normal_or_Welch_approximation","origin":"analytical_declared_generator_moments_not_QA_or_simulated_moments",
            "comparisons":comparisons,"policyJointBounds":policy_bounds,
            "jointBoundsComparisonIsMonteCarloAndWelchApproximate":True,"humanSampleSizeSelected":False}


def summarize_cell(cell,config,batches):
    populations={};criteria=config["criteria"]
    for name in batches[0]["populations"]:
        total=merge_counts([b["populations"][name] for b in batches]);n=total["attempts"];tests=[]
        for j,(identifier,family,endpoint,weights,side) in enumerate(CORE):
            available=total["available"][j];estimable=total["estimable"][j]
            cov=binomial_interval(total["coverage"][j],available);selcov=binomial_interval(total["selectedCoverage"][j],available)
            rawerror=binomial_interval(total["rawFalseReject"][j],n);lowtail=binomial_interval(total["firstGlobalThresholdReject"][j],n)
            tests.append({"id":identifier,"family":family,"side":side,"attempts":n,"available":available,"unavailable":n-available,
                "estimable":estimable,"insufficientPeople":total["unavailableInsufficientPeople"][j],"zeroUncertainty":total["unavailableZeroUncertainty"][j],
                "unexpectedNumericalFailure":total["unexpectedNumericalFailure"][j],"coverage":cov,"selectedCoverage":selcov,
                "coverageMissingIntervals":n-available,"rawFalseReject":rawerror,
                "rawSelectedFalseReject":binomial_interval(total["rawSelectedFalseReject"][j],n),"rawReject":binomial_interval(total["rawReject"][j],n),
                "trueNullAttempts":total["trueNullAttempts"][j],"selectedNullAttempts":total["selectedNullAttempts"][j],
                "firstGlobalThresholdReject":lowtail,"meanEstimate":total["estimateSum"][j]/estimable if estimable else None,
                "meanTruth":total["truthSum"][j]/estimable if estimable else None,"meanSelectedTruth":total["selectedTruthSum"][j]/estimable if estimable else None,
                "coverageCriterionPassed":None if not available else cov["interval"][0]>=criteria["coverageLower"],
                "rawErrorCriterionPassed":None if not total["trueNullAttempts"][j] else rawerror["interval"][1]<=criteria["ordinaryErrorUpper"],
                "designatedLowTailCriterionPassed":lowtail["interval"][1]<=config["alpha"]/9+criteria["lowTailErrorAllowance"] if cell.get("designatedLowTail") else None})
        policy_results={}
        for policy,r in total["policies"].items():
            null_event=binomial_interval(r["anyTrueNullReject"],n)
            result={"anyCoreReject":binomial_interval(r["anyCoreReject"],n),"allCoreReject":binomial_interval(r["allCoreReject"],n),
                    "trueNullFWER":null_event,"selectedNullFWER":binomial_interval(r["anySelectedNullReject"],n),
                    "alternativeSubsetAttempts":r["alternativeSubsetAttempts"],"alternativeDiscoveryPowerStatus":"available" if r["alternativeSubsetAttempts"] else "unavailable_empty_alternative_subset",
                    "anyAlternativeDiscovery":binomial_interval(r["anyAlternativeDiscovery"],n) if r["alternativeSubsetAttempts"] else None,
                    "allAlternativeDiscovery":binomial_interval(r["allAlternativeDiscovery"],n) if r["alternativeSubsetAttempts"] else None,
                    "anySelectedAlternativeDiscovery":binomial_interval(r["anySelectedAlternativeDiscovery"],n) if r["selectedAlternativeSubsetAttempts"] else None,
                    "allSelectedAlternativeDiscovery":binomial_interval(r["allSelectedAlternativeDiscovery"],n) if r["selectedAlternativeSubsetAttempts"] else None,
                    "reject":[binomial_interval(k,n) for k in r["reject"]],"falseReject":[binomial_interval(k,n) for k in r["falseReject"]],
                    "firstThresholdReject":[binomial_interval(k,n) for k in r["firstThresholdReject"]],
                    "globalErrorCriterionPassed":null_event["interval"][1]<=criteria["ordinaryErrorUpper"] if policy!="within_family_holm" else None,
                    "globalGuarantee":policy!="within_family_holm","families":{}}
            for family,f in r["families"].items():
                event=binomial_interval(f["anyTrueNullReject"],n)
                result["families"][family]={"trueNullFWER":event,"criterionPassed":event["interval"][1]<=criteria["ordinaryErrorUpper"]}
            policy_results[policy]=result
        unexpected=sum(total["unexpectedNumericalFailure"])
        # Non-random anticipated-invalid support is not a claimed-supported calculation.
        if cell["nominalStatus"]!="anticipated_invalid":unexpected+=sum(total["unavailableZeroUncertainty"])
        populations[name]={"attempts":n,"tests":tests,"policies":policy_results,"personSupport":total["personSupport"],
                           "unexpectedSupportedFailures":unexpected,"supportCriterionPassed":unexpected==0,
                           "pointwiseCoverageNotSimultaneous":True,"rejectionDenominator":"all attempted replicas"}
    raw_primary=merge_counts([b["populations"]["primary"] for b in batches])
    return {"cell":cell,"populations":populations,"powerComparison":power_comparison(cell,config,raw_primary),
            "productionParity":[p for b in batches for p in b["productionParity"]],"origin":"synthetic_method_calibration_not_human_evidence"}
