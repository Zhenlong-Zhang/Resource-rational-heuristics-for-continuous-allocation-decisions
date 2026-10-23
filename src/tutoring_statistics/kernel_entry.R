#!/usr/bin/env Rscript
# Portable entrypoint. Scientific functions below are loaded unchanged.

args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop("Expected REQUEST.json RECEIPT.json")
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(clubSandwich))
request <- read_json(args[1L],simplifyVector=FALSE)
if (!identical(request$schema,"round10-statistical-relocated-r-request-v1") ||
    !request$family %in% c("null","h10") || file.exists(args[2L])) stop("Wrong relocation request/output")
digest <- function(path) {
  x <- system2(Sys.which("sha256sum"),shQuote(path),stdout=TRUE,stderr=TRUE)
  if (!is.null(attr(x,"status")) || length(x)!=1L || !grepl("^[0-9a-f]{64} [ *]",x)) stop("Cannot bind source")
  substr(x,1L,64L)
}
check <- function(pin) {
  if (!identical(digest(pin$path),pin$sha256)) stop(paste("Changed source/input",pin$path))
}
for (pin in request$sources) check(pin)
if (!identical(as.character(getRversion()),"4.2.2")) stop("Expected R 4.2.2")
runtime <- list(R=as.character(getRversion()),Rhome=normalizePath(R.home()))
for (name in names(request$packages)) {
  version <- unname(read.dcf(file.path(find.package(name),"DESCRIPTION"),fields="Version")[1L,1L])
  if (!identical(version,request$packages[[name]])) stop(paste("Changed package",name,version))
  runtime[[name]] <- version
}
# Load exact original assignment expressions, preserving their function bodies,
# defaults and closure. Top-level historical driver/root admission is replaced
# by this explicit new reproduction identity, never a manufactured old receipt.
load_original <- function(path,functions,constants=character(),env=new.env(parent=globalenv())) {
  found <- character()
  for (expr in parse(file=path,keep.source=TRUE)) {
    if (!is.call(expr) || length(expr)!=3L || !identical(expr[[1L]],as.name("<-")) || !is.symbol(expr[[2L]])) next
    name <- as.character(expr[[2L]])
    if (!(name %in% c(functions,constants))) next
    if (name %in% found) stop("Duplicate original assignment")
    if (name %in% functions && (!is.call(expr[[3L]]) || !identical(expr[[3L]][[1L]],as.name("function")))) stop("Non-function kernel")
    eval(expr,envir=env); found <- c(found,name)
  }
  if (!setequal(found,c(functions,constants))) stop("Missing original assignment")
  env
}
if (request$family=="null") {
  # These three files only define functions; ni_run retains original scientific
  # case/seed/input/resume checks and original three-route fitting semantics.
  for (path in request$definitionFiles) sys.source(path,envir=globalenv())
  ni_run(request$producerRequest,args[2L])
} else {
  ha <- load_original(request$baselineKernel,
    c("ha_read","ha_sha","ha_pin","ha_same","ha_check","ha_write","ha_save","ha_execution","ha_closure","ha_one"),"ha_observed")
  hi <- load_original(request$candidateKernel,
    c("hi_read","hi_sha","hi_pin","hi_check","hi_closure","hi_write","hi_execution","hi_same_numeric",
      "hi_formula","hi_extract","hi_design","hi_one"),c("hi_observed","hi_ids","hi_method"))
  hi$hi_runtime <- runtime
  if (identical(request$scientificFamily,"power")) {
    # Accepted variable-population/nonfinite-guard adapter, unchanged.
    parts <- load_original(request$powerKernel,"pd_extract")
    environment(parts$pd_extract) <- hi
    hi$hi_extract <- parts$pd_extract
  } else if (!identical(request$scientificFamily,"paired")) stop("Unknown H10 scientific family")
  records <- lapply(request$fits,function(spec) {
    if (identical(request$scientificFamily,"paired") &&
        (!setequal(names(spec$populationCounts),c("C1","C2","C3")) ||
         any(unlist(spec$populationCounts)!=154))) stop("Paired population must be exactly154 per arm")
    run <- list(outputDir=request$outputDir,manifest=request$manifest,batch=request$batch,backend=request$backend)
    oldPath <- file.path(request$outputDir,"fits",spec$fitKey,"record.json")
    if (!file.exists(oldPath)) {
      oldPin <- ha$ha_one(spec,run)
      baseline <- list(path=oldPin$path,sha256=oldPin$sha256)
    } else {
      baseline <- ha$ha_pin(oldPath)
    }
    old <- ha$ha_read(oldPath)
    # A committed fit may resume only under this exact input and new manifest.
    if (!identical(old$request,spec$request) || !identical(old$generation,spec$generation) ||
        !identical(old$manifestSha256,request$manifest$sha256) || !identical(old$fitKey,spec$fitKey)) stop("Changed resumed baseline identity")
    for (name in c("request","generation","rawArtifact","fitArtifact")) if (!is.null(old[[name]])) ha$ha_check(old[[name]])
    selected <- c(spec,list(baseline=baseline,sourceManifest=request$manifest,rawArtifact=old$rawArtifact,fitArtifact=old$fitArtifact))
    candidatePath <- file.path(request$outputDir,"records",sprintf("rep_%04d",spec$repID),paste0(spec$caseID,".json"))
    if (!file.exists(candidatePath)) {
      if (identical(request$scientificFamily,"power") && identical(old$raw$status,"fitted") &&
          identical(old$raw$diagnostics$usableCandidateInference,FALSE)) {
        # Same original power.R guard-unavailable path: verify the actual frame
        # and matching nonfinite positions, retain statistical unavailability,
        # and never send unusable fitted values into CR2.
        started <- proc.time()[["elapsed"]]
        checked <- hi$hi_extract(selected,old)
        skipped <- list(schema="h10-inference-record-v1",manifest=request$manifest,method=hi$hi_method,
          caseID=spec$caseID,repID=spec$repID,fitKey=spec$fitKey,baseline=baseline,request=spec$request,
          generation=spec$generation,fitArtifact=old$fitArtifact,technicalCompleted=TRUE,technicalError=NULL,
          rawCandidate=list(status="unavailable",reason="original_guard_unavailable",diagnostics=old$raw$diagnostics),
          candidateMetadata=list(calculationSkipped=TRUE,skipReason="original_guard_unavailable",runtime=hi$hi_runtime,
            originalGuardVerified=TRUE,frameVerified=TRUE,frameRowCount=nrow(checked$frame),requestDescriptor=spec$request,
            CR2CovarianceComputed=FALSE),elapsedSeconds=unname(proc.time()[["elapsed"]]-started),
          execution=hi$hi_execution(),newFitInvocations=0L,newScientificInputs=0L)
        candidate <- hi$hi_write(skipped,candidatePath)
      } else {
        made <- hi$hi_one(selected,list(outputDir=request$outputDir,manifest=request$manifest,repID=spec$repID))
        candidate <- list(path=made$path,sha256=made$sha256)
      }
    } else {
      candidate <- hi$hi_pin(candidatePath)
      saved <- hi$hi_read(candidatePath)
      if (!identical(saved$baseline,baseline) || !identical(saved$request,spec$request) ||
          !identical(saved$generation,spec$generation) || !identical(saved$manifest,request$manifest)) stop("Changed resumed candidate")
    }
    list(fitKey=spec$fitKey,baseline=baseline,candidate=candidate)
  })
  ha$ha_closure(); hi$hi_closure()
  ha$ha_write(list(schema="round10-statistical-relocated-r-receipt-v1",records=records,
    manifest=request$manifest,runtime=runtime,execution=ha$ha_execution()),args[2L])
}
for (pin in request$sources) check(pin)
