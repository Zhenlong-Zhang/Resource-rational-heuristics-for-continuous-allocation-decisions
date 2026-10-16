"""Conditional participant-level power and design geometry; no final human N.

These are hypothetical sensitivities, not power estimated from software QA. PSD and
marginal moment guards do not establish bounded joint-generative realizability.
"""
from __future__ import annotations

import math
import numpy as np
from .inference import CORE, FAMILIES, POLICIES, scipy_stats

PSD_TOL = 1e-10


def unavailable(reason, **metadata):
    return {"status":"unavailable","reason":reason,"power":None,**metadata}


def moment_check(mean, variance, bounds=None):
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (mean,variance)) or variance<0:
        raise ValueError("Mean/variance must be finite and variance nonnegative")
    if bounds is not None:
        if len(bounds)!=2 or not all(math.isfinite(v) for v in bounds) or bounds[0]>=bounds[1]: raise ValueError("Invalid endpoint bounds")
        lo,hi=bounds
        if not lo<=mean<=hi or variance>(hi-mean)*(mean-lo)+1e-12: raise ValueError("Impossible bounded endpoint marginal moments")
    return {"necessaryMarginalMomentGuardPassed":True,"jointBoundedGenerativeFeasibilityProven":False}


def covariance_matrix(dose, residual_variance, correlation=0., kind="exchangeable"):
    if isinstance(dose,bool) or not isinstance(dose,int) or dose<1: raise ValueError("Positive integer dose required")
    if not math.isfinite(residual_variance) or residual_variance<0 or not math.isfinite(correlation): raise ValueError("Invalid covariance parameters")
    if kind=="exchangeable":
        if not (-1/(dose-1) if dose>1 else -1)<=correlation<=1: raise ValueError("Exchangeable correlation outside PSD domain")
        return residual_variance*((1-correlation)*np.eye(dose)+correlation*np.ones((dose,dose)))
    if kind=="ar1":
        if not -1<correlation<1: raise ValueError("AR1 correlation must lie strictly between -1 and1")
        return residual_variance*correlation**np.abs(np.arange(dose)[:,None]-np.arange(dose)[None,:])
    raise ValueError("Unknown covariance model")


def validate_covariance(matrix, dose):
    a=np.asarray(matrix,dtype=float)
    if a.shape!=(dose,dose) or not np.all(np.isfinite(a)): raise ValueError("Covariance finite shape mismatch")
    scale=max(1.,float(np.max(np.abs(a))))
    if not np.allclose(a,a.T,atol=PSD_TOL*scale,rtol=0): raise ValueError("Covariance not symmetric")
    if float(np.linalg.eigvalsh(a).min()) < -PSD_TOL*scale: raise ValueError("Covariance not PSD within stated tolerance")
    return a


def summary_geometry(expected_y, covariance, predictor=None, retained=None, random_slope_variance=0.,
                     eligibility="available_history", endpoint_bounds=None):
    """Explicit projection and covariance, with no ICC-only slope shortcut."""
    try:
        y=np.asarray(expected_y,dtype=float); dose=len(y)
        if y.ndim!=1 or dose<1 or not np.all(np.isfinite(y)): raise ValueError("Expected response must be a finite vector")
        cov=validate_covariance(covariance,dose)
        if not math.isfinite(random_slope_variance) or random_slope_variance<0: raise ValueError("Invalid random-slope variance")
        selected=list(range(dose)) if retained is None else list(retained)
        if not selected or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<dose for i in selected) or len(set(selected))!=len(selected): raise ValueError("Invalid retained trial indices")
        if eligibility not in ("complete_ten","available_history","hypothetical_future_profile"): raise ValueError("Unknown planning eligibility")
        if eligibility=="complete_ten" and (dose!=10 or set(selected)!=set(range(10))): raise ValueError("Incomplete/30-dose geometry is not current complete-ten primary")
        if endpoint_bounds is not None:
            for mu,var in zip(y,np.diag(cov)): moment_check(float(mu),float(var),endpoint_bounds)
        ys=y[selected]; cs=cov[np.ix_(selected,selected)]
        if predictor is None:
            weights=np.ones(len(selected))/len(selected); span=None; sxx=None; kind="participant_mean"
            if random_slope_variance: raise ValueError("Random-slope term requires a predictor")
        else:
            x=np.asarray(predictor,dtype=float)
            if x.shape!=y.shape or not np.all(np.isfinite(x)): raise ValueError("Predictor shape/finite mismatch")
            xs=x[selected]; sxx=float(np.sum((xs-xs.mean())**2)); span=float(np.ptp(xs));kind="participant_unadjusted_slope"
            if len(selected)<3 or len(set(xs))<2 or sxx<=1e-12 or not math.isfinite(sxx) or not math.isfinite(span): raise ValueError("Insufficient finite varying predictor support")
            weights=(xs-xs.mean())/sxx
        effect=float(weights@ys);raw_residual_projection=float(weights@cs@weights)
        # Dot products can put exact intercept cancellation microscopically below zero.
        # This bound concerns float-operation roundoff, NOT the much larger raw PSD tolerance.
        rounding_bound=float(np.finfo(float).eps*8*len(selected)*np.max(np.abs(cs))*np.sum(np.abs(weights))**2)
        if not all(math.isfinite(v) for v in (effect,raw_residual_projection,rounding_bound)): raise ValueError("Nonfinite projected effect/variance/roundoff bound")
        if raw_residual_projection < -rounding_bound: raise ValueError("Projected residual variance is negative; random-slope variance cannot mask amplified tolerated eigenvalue errors")
        residual_projection=max(0.,raw_residual_projection)
        variance=residual_projection+random_slope_variance
        if not math.isfinite(variance): raise ValueError("Nonfinite total projected variance")
        return {"status":"available","reason":None,"summaryKind":kind,"designDose":dose,"retainedDose":len(selected),
                "retainedIndices":selected,"eligibility":eligibility,"effect":effect,"variance":variance,"weights":weights.tolist(),
                "Sxx":sxx,"predictorSpan":span,"projectedResidualVariance":residual_projection,
                "rawProjectedResidualVariance":raw_residual_projection,"projectedResidualRoundingBound":rounding_bound,
                "projectedResidualNumericalZero":raw_residual_projection<0,"allocationChangeOverSpan":None if span is None else effect*span,
                "allocationChangePPOverSpan":None if span is None else 100*effect*span,"randomSlopeVariance":random_slope_variance,
                "covariancePsdTolerance":PSD_TOL,"jointBoundedGenerativeFeasibilityProven":False,
                "iidCommonPersonVarianceAssumedForExactNct":True,
                "limitation":"Representative common geometry is an idealization; varied schedules/retention/variance require approximate planning and generative calibration."}
    except (ValueError,TypeError,IndexError,OverflowError) as exc: return unavailable(str(exc),summaryKind="geometry")


def one_sample_power(effect, variance, n, alpha=.05, side="greater", common_iid_normal=True):
    try:
        if side not in ("greater","less") or not math.isfinite(alpha) or not 0<alpha<1: raise ValueError("Invalid alpha/side")
        if isinstance(n,bool) or not isinstance(n,int) or n<2: raise ValueError("At least2 independent participants required")
        moment_check(effect,variance)
        if variance<=0: raise ValueError("Zero uncertainty has no supported noncentral-t planning result")
        se=math.sqrt(variance/n);df=n-1;nc=effect/se;stats=scipy_stats()
        critical=float(stats.t.ppf(1-alpha,df) if side=="greater" else stats.t.ppf(alpha,df))
        power=float(stats.nct.sf(critical,df,nc) if side=="greater" else stats.nct.cdf(critical,df,nc))
        return {"status":"available","reason":None,"power":power,"effect":effect,"variance":variance,"participants":n,
                "alpha":alpha,"side":side,"standardError":se,"df":df,"noncentrality":nc,"criticalValue":critical,
                "method":"noncentral_t_one_sample","exactUnderDeclaredIidNormalCommonVariance":bool(common_iid_normal),
                "approximate":not common_iid_normal,"assumptions":"Independent normal person summaries with common geometry/variance; heterogeneity makes this a planning approximation.",
                "finalNSelected":False}
    except (ValueError,TypeError) as exc: return unavailable(str(exc),effect=effect,participants=n,alpha=alpha,side=side)


def weighted_contrast_power(groups, weights, alpha=.05, side="greater"):
    try:
        if side not in ("greater","less") or not math.isfinite(alpha) or not 0<alpha<1: raise ValueError("Invalid alpha/side")
        if not weights or any(not math.isfinite(w) or w==0 for w in weights.values()): raise ValueError("Invalid fixed condition weights")
        pieces=[];terms=[];effect=0.
        for condition,w in weights.items():
            g=groups[condition];n=g["n"];mu=g["mean"];variance=g["variance"]
            if isinstance(n,bool) or not isinstance(n,int) or n<2: raise ValueError("At least2 independent participants in each required condition")
            moment_check(mu,variance,g.get("bounds"))
            effect+=w*mu;p=w*w*variance/n;pieces.append(p);terms.append(p*p/(n-1))
        total=sum(pieces)
        if total<=0: raise ValueError("Zero contrast uncertainty is unavailable")
        se=math.sqrt(total);df=total**2/sum(terms);nc=effect/se;stats=scipy_stats()
        critical=float(stats.t.ppf(1-alpha,df) if side=="greater" else stats.t.ppf(alpha,df))
        p=float(stats.nct.sf(critical,df,nc) if side=="greater" else stats.nct.cdf(critical,df,nc))
        return {"status":"available","reason":None,"power":p,"effect":effect,"weights":weights,"groups":groups,
                "standardError":se,"df":df,"noncentrality":nc,"criticalValue":critical,"alpha":alpha,"side":side,
                "method":"approximate_Welch_Satterthwaite_noncentral_t","approximate":True,
                "exactJointPower":False,"finalNSelected":False}
    except (ValueError,TypeError,KeyError) as exc:return unavailable(str(exc),weights=weights,alpha=alpha,side=side)


def usable_people(enrolled, participant_retention=1., trial_retention=1., dose=10, eligibility="complete_ten"):
    if isinstance(enrolled,bool) or not isinstance(enrolled,int) or enrolled<0: raise ValueError("Enrolled count must be nonnegative integer")
    if not all(math.isfinite(v) and 0<=v<=1 for v in (participant_retention,trial_retention)): raise ValueError("Invalid retention probability")
    if not isinstance(dose,int) or dose<1: raise ValueError("Invalid dose")
    if eligibility not in ("complete_ten","available_history","hypothetical_future_profile"):raise ValueError("Unknown retention eligibility")
    if eligibility=="complete_ten" and dose!=10:raise ValueError("Current primary requires dose10")
    probability=participant_retention*(trial_retention**dose if eligibility=="complete_ten" else 1-(1-trial_retention)**dose)
    expected=enrolled*probability
    return {"enrolled":enrolled,"eligibility":eligibility,"dose":dose,"participantRetention":participant_retention,
            "independentTrialRetention":trial_retention,"usableProbability":probability,"expectedUsablePeople":expected,
            "planningCountFloor":math.floor(expected+1e-12),"countMethod":"floor of expected usable count; plug-in approximation, not guaranteed realized N",
            "availableHistoryQualification":"At least one retained trial for means; this is NOT sufficient for a slope. Specific retained x/support must be checked separately.",
            "assumptions":"MCAR independent participant/trial loss. Available-history slope support must be checked separately; informative loss unresolved."}


def joint_bounds(marginals, policy="global_holm", alpha=.05):
    """Bounds from first-step sufficient probabilities and raw-alpha probabilities."""
    if policy not in POLICIES or not 0<alpha<1:raise ValueError("Invalid multiplicity policy/alpha")
    if set(marginals)!={x[0] for x in CORE}:raise ValueError("Power needs the fixed nine core slots")
    q=[];raw=[];unavailable_slots=[]
    for identifier,family,*_ in CORE:
        m=marginals[identifier]
        if m["status"]!="available":q.append(0.);raw.append(0.);unavailable_slots.append(identifier);continue
        values=(m["firstThresholdPower"],m["rawAlphaPower"])
        if any(not math.isfinite(v) or not 0<=v<=1 for v in values) or values[0]>values[1]+1e-10:raise ValueError("Invalid nested marginal powers")
        q.append(values[0]);raw.append(values[1])
    return {"policy":policy,"alpha":alpha,"anyCoreLower":max(q),"anyCoreUpper":min(1.,sum(q)),
            "allCoreLower":0. if unavailable_slots else max(0.,sum(q)-8),"allCoreUpper":min(raw),
            "unavailableSlots":unavailable_slots,"exactJointPower":False,"globalAnyGuarantee":policy!="within_family_holm",
            "anyEvent":"union of each correction family's first-threshold events",
            "allLowerEvent":"intersection at first thresholds is sufficient, not necessary",
            "allUpperEvent":"all rejections require every raw p<=alpha",
            "conditionalOnValidMarginalProbabilities":True,"welchApproximationMayAffectBounds":True}


def planning_report(config):
    """Execute frozen hypothetical sensitivities; never derive assumptions from QA."""
    if config.get("purpose")!="hypothetical_sensitivity" or config.get("assumptionsOrigin")!="declared_not_QA_estimated":raise ValueError("Planning requires explicit hypothetical non-QA assumptions")
    scenarios=[]
    for supplied in config["scenarios"]:
        scenario={**config.get("defaults",{}),**supplied}
        alpha=scenario.get("alpha",.05);policy=scenario.get("policy","global_holm")
        if policy not in POLICIES or not 0<alpha<1:raise ValueError("Invalid scenario policy/alpha")
        dose=scenario.get("dose",10);eligibility=scenario.get("eligibility","complete_ten")
        x=np.linspace(-.2,.2,dose);order=np.linspace(-1,1,dose)
        if scenario.get("predictorKind")=="two_level":x=np.asarray([-.15 if i%2 else .15 for i in range(dose)])
        elif scenario.get("predictorKind")=="constant":x=np.zeros(dose)
        elif scenario.get("predictorKind")=="narrow":x=x*.01
        retained=scenario.get("retainedIndices",list(range(dose)))
        cov=covariance_matrix(dose,scenario.get("residualVariance",.01),scenario.get("correlation",.3),scenario.get("covarianceKind","exchangeable"))
        magnitude=scenario.get("slopeMagnitude",.2)
        learning=scenario.get("learning",0.);trend=scenario.get("orderTrend",0.)
        geometry={}
        for cid,sign in (("C2",1),("C3",1),("C4",-1)):
            y=.5+sign*magnitude*(1+learning*order)*x+trend*order
            geometry[cid]=summary_geometry(y,cov,x,retained,scenario.get("randomSlopeVariance",.01),eligibility,(0.,1.))
            if eligibility!="complete_ten" and scenario.get("trialRetention",1.)<1.:
                geometry[cid]=unavailable("Random incomplete slope support not specified by a fixed retained-index pattern")
        counts={};retention={}
        for cid in ("C1","C2","C3","C4","C5"):
            enrolled=scenario["enrolledPerCondition"].get(cid,0)
            info=usable_people(enrolled,scenario.get("participantRetention",1.),scenario.get("trialRetention",1.),dose,eligibility)
            retention[cid]=info;counts[cid]=info["planningCountFloor"]
        acquisition=scenario.get("acquisitionMeans",{"C2":.2,"C3":.4,"C4":.2,"C5":.4})
        acquisition_variance=scenario.get("acquisitionParticipantVariance",.04)
        resemblance=scenario.get("resemblanceMeanPP",2.);rv=scenario.get("resemblanceParticipantVariancePP2",9.)
        bound=scenario.get("fixedTargetSpacingPP",20.)
        lower=scenario.get("lowerMeanShare",.04);lv=scenario.get("lowerParticipantVariance",.01)
        mean_geometry={}
        def mean_model(name,mu,direct_variance,bounds,learning,trial_variance):
            expected=mu+learning*order
            if scenario.get("meanVarianceModel","declared_participant_variance")=="trial_covariance":
                # Binary indicator diagonals are exactly p_t(1-p_t); joint binary realizability remains unproven.
                std=np.sqrt(expected*(1-expected)) if name.startswith("acquisition_") and np.all((expected>=0)&(expected<=1)) else np.full(dose,math.sqrt(trial_variance))
                correlation=covariance_matrix(dose,1.,scenario.get("correlation",.3),scenario.get("covarianceKind","exchangeable"))
                result=summary_geometry(expected,correlation*std[:,None]*std[None,:],retained=retained,eligibility=eligibility,endpoint_bounds=bounds)
            else:
                result=summary_geometry(expected,np.zeros((dose,dose)),retained=retained,eligibility=eligibility,endpoint_bounds=bounds)
                if result["status"]=="available":
                    try:moment_check(result["effect"],direct_variance,bounds)
                    except ValueError as exc:result=unavailable(str(exc))
                    else:result.update(variance=direct_variance,varianceModel="declared_participant_variance_not_dose_derived")
            if result["status"]=="available":
                try:moment_check(result["effect"],result["variance"],bounds)
                except ValueError as exc:result=unavailable(str(exc))
            mean_geometry[name]=result
            return result
        acquisition_geometry={c:mean_model("acquisition_"+c,mu,acquisition_variance,[0.,1.],scenario.get("acquisitionLearning",0.),mu*(1-mu)) for c,mu in acquisition.items()}
        r_geometry=mean_model("resemblance",resemblance,rv,[-bound,bound],scenario.get("resemblanceLearningPP",0.),scenario.get("resemblanceTrialVariancePP2",100.))
        l_geometry=mean_model("lower",lower,lv,[-.5,.5],scenario.get("lowerLearningShare",0.),scenario.get("lowerTrialVariance",.04))
        marginals={};results=[]
        for identifier,family,endpoint,weights,side in CORE:
            threshold=alpha/9 if policy=="global_holm" else alpha/FAMILIES[family] if policy=="within_family_holm" else alpha/(4*FAMILIES[family])
            if endpoint=="anyQuestion":
                if any(acquisition_geometry[c]["status"]!="available" for c in weights):first=raw=unavailable("Required acquisition mean geometry unavailable")
                else:
                    groups={c:{"n":counts[c],"mean":acquisition_geometry[c]["effect"],"variance":acquisition_geometry[c]["variance"],"bounds":[0.,1.]} for c in weights}
                    first=weighted_contrast_power(groups,weights,threshold,side);raw=weighted_contrast_power(groups,weights,alpha,side)
            else:
                cid=next(iter(weights))
                if endpoint=="slope_initial":
                    g=geometry[cid]
                    if g["status"]!="available": first=raw=unavailable(g["reason"])
                    else:
                        first=one_sample_power(g["effect"],g["variance"],counts[cid],threshold,side,common_iid_normal=scenario.get("participantRetention",1.)==1. and scenario.get("trialRetention",1.)==1.)
                        raw=one_sample_power(g["effect"],g["variance"],counts[cid],alpha,side,common_iid_normal=scenario.get("participantRetention",1.)==1. and scenario.get("trialRetention",1.)==1.)
                else:
                    g=r_geometry if endpoint=="EOResemblance" else l_geometry
                    if g["status"]!="available":first=raw=unavailable(g["reason"])
                    else:
                        # Bounded R/L summaries are not literally normal: this is a normal-summary planning approximation.
                        first=one_sample_power(g["effect"],g["variance"],counts[cid],threshold,side,common_iid_normal=False)
                        raw=one_sample_power(g["effect"],g["variance"],counts[cid],alpha,side,common_iid_normal=False)
            available=first["status"]==raw["status"]=="available"
            marginals[identifier]={"status":"available" if available else "unavailable","firstThresholdPower":first["power"],"rawAlphaPower":raw["power"]}
            results.append({"id":identifier,"family":family,"side":side,"firstThreshold":threshold,"first":first,"rawAlpha":raw})
        scenarios.append({"id":scenario["id"],"declaredScenario":scenario,"retention":retention,"slopeGeometry":geometry,"meanGeometry":mean_geometry,
                          "coreMarginals":results,"jointBounds":joint_bounds(marginals,policy,alpha),
                          "designGeometry":"Illustrative declared x vectors; not a proposed or evaluated finite recipe",
                          "resemblanceBound":"Declared hypothetical fixed spacing, not the old initial/no-search1.7 bound",
                          "finalNSelected":False})
    return {"schema":"tutoring-conditional-power-v1","purpose":"hypothetical_sensitivity","humanVarianceEstimated":False,
            "statisticalCalibrationPerformed":False,"scenarios":scenarios,
            "limitations":["No final N, human effects, achieved power, or exact joint Holm power.",
                           "Welch nct and floor-expected usable counts are approximations.",
                           "PSD and marginal bounded moments do not prove joint bounded-generative feasibility.",
                           "Different recipe/Sxx/person variances violate the common-variance exact iid-normal idealization.",
                           "Dose30 is hypothetical future profile; retained<10 is sensitivity, not current complete-ten primary."]}
