#!/usr/bin/env Rscript
# Portable admission around the shared, unchanged fixed-population kernels.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L || file.exists(args[2L])) stop("Expected new REQUEST RECEIPT")
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(clubSandwich))
request <- read_json(args[1L],simplifyVector=FALSE)
if (!identical(request$schema,"statistical-comparison-r-request-v1") ||
    !request$stage %in% c("baseline","inference")) stop("Wrong comparison request")
root <- dirname(normalizePath(sub("^--file=", "", grep("^--file=",commandArgs(),value=TRUE)[1L])))
ha <- new.env(parent=globalenv()); hi <- new.env(parent=globalenv())
sys.source(file.path(root,"joint_fit.R"),envir=ha)
sys.source(file.path(root,"cluster_fit.R"),envir=hi)
for (pin in request$sources) ha$ha_check(pin)
for (name in c("manifest","spec","backend")) ha$ha_check(request[[name]])
if (!identical(as.character(getRversion()),"4.2.2")) stop("Expected R 4.2.2")
runtime <- list(R=as.character(getRversion()),Rhome=normalizePath(R.home()))
for (name in names(request$packages)) {
  version <- unname(read.dcf(file.path(find.package(name),"DESCRIPTION"),fields="Version")[1L,1L])
  if (!identical(version,request$packages[[name]])) stop(paste("Changed package",name))
  runtime[[name]] <- version
}
hi$hi_runtime <- runtime
spec <- ha$ha_read(request$spec$path)
if (!ha$ha_same(spec$manifest,request$manifest) || spec$case$N!=154L ||
    !identical(spec$fitKey,request$fitKey)) stop("Changed fixed-population spec")
ha$ha_check(spec$request); ha$ha_check(spec$generation)
baselinePath <- file.path(request$baselineDir,"fits",spec$fitKey,"record.json")
resumed <- file.exists(baselinePath)
if (!resumed) {
  if (!identical(request$stage,"baseline")) stop("Inference requires the saved baseline fit")
  ha$ha_one(spec,list(outputDir=request$baselineDir,manifest=request$manifest,
                     batch=request$spec,backend=request$backend))
}
baselinePin <- ha$ha_pin(baselinePath); ha$ha_check(baselinePin)
old <- ha$ha_read(baselinePath)
if (!identical(old$manifestSha256,request$manifest$sha256) ||
    !identical(old$batchSha256,request$spec$sha256) ||
    !identical(old$fitKey,spec$fitKey) || !identical(old$caseID,spec$caseID) || old$repID!=spec$repID ||
    !ha$ha_same(old$request,spec$request) || !ha$ha_same(old$generation,spec$generation)) stop("Changed saved baseline identity")
for (name in c("rawArtifact","fitArtifact")) if (!is.null(old[[name]])) ha$ha_check(old[[name]])
candidatePin <- NULL; candidateResumed <- FALSE
if (identical(request$stage,"inference")) {
  candidatePath <- file.path(request$candidateDir,"records",sprintf("rep_%04d",spec$repID),paste0(spec$caseID,".json"))
  candidateResumed <- file.exists(candidatePath)
  if (!candidateResumed) {
    selected <- c(spec,list(baseline=baselinePin,sourceManifest=request$manifest,
                           rawArtifact=old$rawArtifact,fitArtifact=old$fitArtifact))
    hi$hi_one(selected,list(outputDir=request$candidateDir,manifest=request$manifest,repID=spec$repID))
  }
  candidatePin <- hi$hi_pin(candidatePath); hi$hi_check(candidatePin)
  candidate <- hi$hi_read(candidatePath)
  if (!ha$ha_same(candidate$baseline,baselinePin) || !ha$ha_same(candidate$manifest,request$manifest) ||
      !ha$ha_same(candidate$request,spec$request) || !ha$ha_same(candidate$generation,spec$generation) ||
      !ha$ha_same(candidate$fitArtifact,old$fitArtifact) || !identical(candidate$fitKey,spec$fitKey) ||
      !identical(candidate$caseID,spec$caseID) || candidate$repID!=spec$repID) stop("Changed saved inference identity")
}
for (pin in request$sources) ha$ha_check(pin)
ha$ha_closure(); hi$hi_closure()
ha$ha_write(list(schema="statistical-comparison-r-receipt-v1",stage=request$stage,
  manifest=request$manifest,spec=request$spec,fitKey=spec$fitKey,baseline=baselinePin,candidate=candidatePin,
  baselineResumed=resumed,candidateResumed=candidateResumed,
  newFitInvocations=if(resumed)0L else old$fitInvocationCount,
  newCandidateCalculations=if(identical(request$stage,"inference") && !candidateResumed)1L else 0L,
  runtime=runtime,sourceClosure=TRUE),args[2L])
