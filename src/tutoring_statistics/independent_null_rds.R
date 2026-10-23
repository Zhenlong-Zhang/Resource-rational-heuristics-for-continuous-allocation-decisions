#!/usr/bin/env Rscript
# Independent read-only extraction from already committed same-input RDS fits.

arguments <- commandArgs(trailingOnly=TRUE)
if (length(arguments)!=2L) stop("Expected REQUEST.json OUTPUT.json")
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(jsonlite))
request <- read_json(arguments[[1L]],simplifyVector=FALSE)
if (!identical(request$schema,"mixed-null-repair-independent-rds-request-v1") ||
    !is.list(request$rows) || file.exists(arguments[[2L]])) stop("Invalid independent RDS request/output")
keys <- vapply(request$rows,function(row) row$key,character(1))
if (anyDuplicated(keys)) stop("Duplicate independent RDS key")
rows <- lapply(request$rows,function(row) {
  if (!row$methodID %in% c("baseline_replay","strict_nlopt_restart","alternate_PORT") ||
      !is.character(row$path) || length(row$path)!=1L || !startsWith(row$path,"/"))
    stop("Invalid independent RDS identity")
  fit <- readRDS(row$path)
  if (!inherits(fit,"lmerMod")) stop("RDS does not contain a literal mixed-model fit")
  theta <- getME(fit,"theta"); lower <- unname(getME(fit,"lower"))
  beta <- fixef(fit); V <- as.matrix(vcov(fit)); derivative <- fit@optinfo$derivs
  eig <- if (is.null(derivative$Hessian)) numeric() else
    eigen(derivative$Hessian,symmetric=TRUE,only.values=TRUE)$values
  messages <- unlist(fit@optinfo$conv$lme4$messages)
  codes <- fit@optinfo$conv$opt
  converged <- all(codes==0) && length(messages)==0L
  singular <- isSingular(fit,tol=1e-4)
  usable <- converged && !singular && length(eig)>0L && all(is.finite(eig)) &&
    min(eig)>0 && !is.null(derivative$gradient) && all(is.finite(derivative$gradient)) &&
    all(is.finite(beta)) && all(is.finite(V))
  L <- setNames(rep(0,length(beta)),names(beta))
  if (!setequal(names(beta),c("(Intercept)","delta"))) stop("Changed fitted coefficient names")
  L["delta"] <- 1
  z <- contest1D(fit,L,ddf="Satterthwaite",confint=TRUE,level=.95)
  reference <- list(estimate=unname(z[1,"Estimate"]),SE=unname(z[1,"Std. Error"]),
    df=unname(z[1,"df"]),statistic=unname(z[1,"t value"]),twoSidedP=unname(z[1,"Pr(>|t|)"]),
    CI=c(z[1,"lower"],z[1,"upper"]),L=unname(L),coefficientNames=names(beta),
    reconstructedEstimate=unname(sum(L*beta)),reconstructedSE=unname(sqrt(drop(t(L)%*%V%*%L))))
  frame <- model.frame(fit)
  if (!all(c("y","delta","trial_c","participant") %in% names(frame)))
    stop("Missing fitted input columns")
  if (!is.character(row$actualRequestPath) || length(row$actualRequestPath)!=1L ||
      !startsWith(row$actualRequestPath,"/")) stop("Unbound actual request path")
  actualRequest <- read_json(row$actualRequestPath,simplifyVector=FALSE)
  expected <- actualRequest$rows
  if (length(expected)!=nrow(frame)) stop("RDS input row count differs")
  numericInput <- lapply(c("y","delta","trial_c"),function(field) {
    wanted <- vapply(expected,function(value) value[[field]],numeric(1))
    observed <- frame[[field]]
    if (any(!is.finite(wanted)) || any(!is.finite(observed)) ||
        !isTRUE(all.equal(unname(observed),unname(wanted),tolerance=1e-12)))
      stop(paste("RDS input differs",field))
    list(field=field,maxAbsoluteError=max(abs(observed-wanted)))
  })
  if (!identical(as.character(frame$participant),
                 vapply(expected,function(value) value$participant,character(1))))
    stop("RDS participant order differs")
  list(key=row$key,path=normalizePath(row$path,mustWork=TRUE),methodID=row$methodID,
    theta=list(names=names(theta),values=unname(theta),lower=lower,
      lowerKinds=ifelse(is.finite(lower),"finite",ifelse(lower<0,"-Inf","Inf")),
      lowerFinite=unname(is.finite(lower)),lowerNegativeInfinite=unname(is.infinite(lower)&lower<0)),
    raw=list(formula=paste(deparse(formula(fit)),collapse=" "),REML=isREML(fit),
      optimizer=fit@optinfo$optimizer,coefficientNames=names(beta),coefficients=as.list(beta),
      vcov=unname(V),randomCovariance=unname(as.matrix(VarCorr(fit)$participant)),
      residualVariance=sigma(fit)^2,
      diagnostics=list(converged=converged,singular=singular,gradient=unname(derivative$gradient),
        hessianEigenvalues=unname(eig),convergenceMessages=as.list(messages),usableCandidateInference=usable),
      references=list(delta=reference)),
    optimizerCode=codes,controls=fit@optinfo$control,
    Hessian=unname(derivative$Hessian),remlCriterion=-2*as.numeric(logLik(fit,REML=TRUE)),
    logLik=as.numeric(logLik(fit,REML=TRUE)),
    inputCheck=list(actualRequestPath=normalizePath(row$actualRequestPath,mustWork=TRUE),
      rowCount=nrow(frame),participantOrderEqual=TRUE,numericColumns=numericInput,passed=TRUE))
})
versionRaw <- function(package) read.dcf(file.path(find.package(package),"DESCRIPTION"))[1L,"Version"]
result <- list(schema="mixed-null-repair-independent-rds-v1",rows=rows,
  runtime=list(R=R.version.string,Rhome=normalizePath(R.home(),mustWork=TRUE),
    packages=lapply(c("lme4","lmerTest","jsonlite"),function(name)
      list(name=name,version=versionRaw(name),path=normalizePath(find.package(name),mustWork=TRUE)))),
  newFits=0L,newScientificInputs=0L)
temporary <- tempfile(".independent-rds-",tmpdir=dirname(arguments[[2L]]))
writeLines(toJSON(result,auto_unbox=TRUE,null="null",na="null",digits=NA),temporary,useBytes=TRUE)
if (!file.link(temporary,arguments[[2L]])) { unlink(temporary); stop("Cannot atomically save independent RDS extraction") }
unlink(temporary)
