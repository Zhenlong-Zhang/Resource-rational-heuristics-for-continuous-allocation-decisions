# Literal Falk mixed model and explicitly provisional initial/percent H10.

args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(jsonlite))
request <- read_json(args[1], simplifyVector=TRUE)
emit <- function(x) write_json(x,args[2],auto_unbox=TRUE,pretty=TRUE,null="null",digits=NA)
if(identical(request$model,"runtime_probe")) {
  expected <- request$runtime
  stopifnot(isTRUE(expected$passed), R.version.string==expected$R,
            normalizePath(R.home())==normalizePath(expected$Rhome))
  observed <- list()
  for(name in names(expected$pins)) {
    stopifnot(requireNamespace(name,quietly=FALSE),read.dcf(file.path(find.package(name),"DESCRIPTION"))[1,"Version"]==expected$pins[[name]],
              normalizePath(find.package(name))==normalizePath(expected$steps[[name]]$library))
    observed[[name]] <- list(version=as.character(packageVersion(name)),path=find.package(name))
  }
  for(name in names(expected$sharedPins)) {
    stopifnot(requireNamespace(name,quietly=FALSE),read.dcf(file.path(find.package(name),"DESCRIPTION"))[1,"Version"]==expected$sharedPins[[name]],
              normalizePath(find.package(name))==normalizePath(file.path(expected$sharedLibrary,name)))
    observed[[name]] <- list(version=as.character(packageVersion(name)),path=find.package(name))
  }
  emit(list(passed=TRUE,R=R.version.string,Rhome=R.home(),packages=observed))
  quit(status=0)
}
tryCatch({
  suppressPackageStartupMessages(library(lmerTest))
  d <- as.data.frame(request$rows)
  d$participant <- factor(d$participant)
  d$condition <- factor(d$condition,levels=c(request$reference,setdiff(c("C1","C2","C3","C4"),request$reference)))
  h10 <- request$model == "H10"
  formula <- if(h10) y ~ condition*need_A + condition*need_B + (1|participant) else y ~ delta + (1+trial_c|participant)
  # Only observed conditions enter the H10 model (C4 is not an extra empty term).
  d$condition <- droplevels(d$condition)
  warnings <- character()
  fit <- withCallingHandlers(lmerTest::lmer(formula,data=d,REML=TRUE,
     control=lme4::lmerControl(optimizer="nloptwrap",calc.derivs=TRUE,
       check.rankX="stop.deficient",optCtrl=list(xtol_abs=1e-8,ftol_abs=1e-8,maxeval=10000))),
     warning=function(w){warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")})
  beta <- lme4::fixef(fit)
  V <- as.matrix(vcov(fit))
  derivs <- fit@optinfo$derivs
  eigenvalues <- if(is.null(derivs$Hessian)) numeric() else eigen(derivs$Hessian,symmetric=TRUE,only.values=TRUE)$values
  messages <- unlist(fit@optinfo$conv$lme4$messages)
  singular <- lme4::isSingular(fit,tol=1e-4)
  converged <- all(fit@optinfo$conv$opt == 0) && length(messages)==0
  hessianOK <- length(eigenvalues)>0 && all(is.finite(eigenvalues)) && min(eigenvalues)>0
  gradientOK <- !is.null(derivs$gradient) && all(is.finite(derivs$gradient))
  usable <- converged && !singular && hessianOK && gradientOK && all(is.finite(beta)) && all(is.finite(V))
  reference_test <- function(L) {
    z <- lmerTest::contest1D(fit,L,ddf="Satterthwaite",confint=TRUE,level=.95)
    # Save direct beta/vcov reconstruction independently of contest1D output.
    list(estimate=unname(z[1,"Estimate"]),SE=unname(z[1,"Std. Error"]),
         df=unname(z[1,"df"]),statistic=unname(z[1,"t value"]),
         twoSidedP=unname(z[1,"Pr(>|t|)"]),CI=c(z[1,"lower"],z[1,"upper"]),
         L=unname(L),coefficientNames=names(beta),
         reconstructedEstimate=unname(sum(L*beta)),reconstructedSE=unname(sqrt(drop(t(L)%*%V%*%L))))
  }
  references <- list()
  slopes <- list()
  if(!h10) {
    L <- setNames(rep(0,length(beta)),names(beta)); L["delta"] <- 1
    references$delta <- reference_test(L)
  } else {
    fixed <- delete.response(terms(lme4::nobars(formula)))
    slope_vector <- function(arm,recipient) {
      pair <- data.frame(condition=factor(c(arm,arm),levels=levels(d$condition)),need_A=c(0,0),need_B=c(0,0))
      pair[2,paste0("need_",recipient)] <- 1
      mm <- model.matrix(fixed,pair,contrasts.arg=attr(lme4::getME(fit,"X"),"contrasts"))
      unname(mm[2,names(beta)]-mm[1,names(beta)])
    }
    for(arm in c("C1","C2","C3")) for(recipient in c("A","B")) {
      key <- paste0(arm,":",recipient)
      slopes[[key]] <- reference_test(slope_vector(arm,recipient))
    }
    for(other in c("C2","C3")) for(recipient in c("A","B")) {
      key <- paste0("C1-",other,":",recipient)
      references[[key]] <- reference_test(slope_vector("C1",recipient)-slope_vector(other,recipient))
    }
  }
  emit(list(status="fitted",formula=paste(deparse(formula),collapse=" "),REML=TRUE,
    method="lmerTest_Satterthwaite_candidate",versions=list(R=R.version.string,
      lme4=as.character(packageVersion("lme4")),lmerTest=as.character(packageVersion("lmerTest")),Matrix=as.character(packageVersion("Matrix"))),
    optimizer=fit@optinfo$optimizer,coefficients=as.list(beta),coefficientNames=names(beta),vcov=unname(V),
    randomCovariance=unname(as.matrix(lme4::VarCorr(fit)$participant)),residualVariance=sigma(fit)^2,
    diagnostics=list(converged=converged,singular=singular,gradient=unname(derivs$gradient),
      hessianEigenvalues=eigenvalues,convergenceMessages=as.list(messages),warnings=as.list(warnings),
      usableCandidateInference=usable),references=references,conditionSlopes=slopes,
    note="Software candidate only; no human-effect, calibration, or power claim"))
},error=function(e) emit(list(status="unavailable",reason="backend_fit_error",detail=conditionMessage(e),p=NULL,candidateNominalReject=FALSE)))
