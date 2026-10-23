# Accepted numerical definitions; portable execution is in kernel_entry.R.
ha_read <- function(path) jsonlite::read_json(path,simplifyVector=FALSE)

ha_sha <- function(path) {
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) stop("sha256sum unavailable")
  result <- system2(executable,shQuote(path),stdout=TRUE,stderr=TRUE)
  if (!is.null(attr(result,"status")) || length(result)!=1L || !grepl("^[0-9a-f]{64} ",result))
    stop("Cannot bind bytes")
  substr(result,1L,64L)
}

ha_pin <- function(path) list(path=normalizePath(path,mustWork=TRUE),sha256=ha_sha(path))

ha_same <- function(a,b) identical(as.character(toJSON(a,auto_unbox=TRUE,null="null",na="null",digits=NA)),
                                   as.character(toJSON(b,auto_unbox=TRUE,null="null",na="null",digits=NA)))

ha_observed <- new.env(parent=emptyenv())

ha_check <- function(descriptor) {
  if (!is.character(descriptor$path) || !grepl("^/",descriptor$path) ||
      !identical(ha_sha(descriptor$path),descriptor$sha256)) stop("Changed bound file")
  if (exists(descriptor$path,envir=ha_observed,inherits=FALSE) &&
      !identical(get(descriptor$path,envir=ha_observed),descriptor$sha256)) stop("Contradictory pins")
  assign(descriptor$path,descriptor$sha256,envir=ha_observed)
  invisible(TRUE)
}

ha_write <- function(value,path) {
  if (file.exists(path)) stop("Refusing overwrite")
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  temporary <- tempfile(".uncommitted-",tmpdir=dirname(path));on.exit(unlink(temporary))
  write_json(value,temporary,auto_unbox=TRUE,pretty=TRUE,null="null",na="null",digits=NA)
  if (!file.link(temporary,path)) stop("Atomic JSON commit failed")
  ha_pin(path)
}

ha_save <- function(fit,path) {
  if (file.exists(path)) stop("Refusing RDS overwrite")
  temporary <- tempfile(".uncommitted-",tmpdir=dirname(path));on.exit(unlink(temporary))
  saveRDS(fit,temporary)
  if (!file.link(temporary,path)) stop("Atomic RDS commit failed")
  ha_pin(path)
}

ha_execution <- function() list(pid=Sys.getpid())

ha_closure <- function() {
  for(path in ls(ha_observed,all.names=TRUE)) {
    if (!identical(ha_sha(path),get(path,envir=ha_observed))) stop("Source changed at closure")
  }
  TRUE
}

ha_one <- function(fitSpec,runRequest) {
  key <- fitSpec$fitKey
  directory <- file.path(runRequest$outputDir,"fits",key)
  recordPath <- file.path(directory,"record.json")
  if (file.exists(recordPath)) stop("R receives missing records only")
  ha_check(fitSpec$request)
  ha_check(fitSpec$generation)
  actual <- ha_read(fitSpec$request$path)
  if (!identical(actual$model,"H10") || !identical(actual$reference,"C2")) stop("Wrong scientific request")
  artifact <- NULL; rawArtifact <- NULL; count <- 0L; elapsed <- 0
  if (!isTRUE(fitSpec$support$fitEligible)) {
    raw <- list(status="unavailable",reason="design_support_unavailable",p=NULL,candidateNominalReject=FALSE)
  } else {
    attempt <- file.path(directory,"attempt")
    if (dir.exists(attempt)) stop("Partial attempt exists; do not refit")
    dir.create(attempt,recursive=TRUE)
    intent <- list(schema="h10-calibration-production-fit-intent-v1",caseID=fitSpec$caseID,
      repID=fitSpec$repID,fitKey=key,manifest=runRequest$manifest,batch=runRequest$batch,
      request=fitSpec$request,generation=fitSpec$generation,backend=runRequest$backend,execution=ha_execution())
    ha_write(intent,file.path(attempt,"intent.json"))
    rawPath <- file.path(attempt,"raw.json")
    actualPath <- fitSpec$request$path
    environment <- new.env(parent=globalenv())
    environment$commandArgs <- function(trailingOnly=FALSE) c(actualPath,rawPath)
    started <- proc.time()[["elapsed"]]
    # The original script still reads actual JSON, fits the original formula,
    # applies the original guard and emits its original raw result.
    sys.source(runRequest$backend$path,envir=environment)
    elapsed <- unname(proc.time()[["elapsed"]]-started)
    count <- 1L
    if (!file.exists(rawPath)) stop("Frozen backend did not emit a result")
    raw <- ha_read(rawPath)
    if (!(identical(raw$status,"fitted") ||
          (identical(raw$status,"unavailable") && identical(raw$reason,"backend_fit_error"))))
      stop("Unexpected backend protocol status")
    rawArtifact <- ha_pin(rawPath)
    if (exists("fit",envir=environment,inherits=FALSE)) {
      savedFit <- get("fit",envir=environment,inherits=FALSE)
      if (!inherits(savedFit,"merMod")) stop("Unexpected fit object")
      artifact <- ha_save(savedFit,file.path(attempt,"fit.rds"))
    }
    if (identical(raw$status,"fitted") && is.null(artifact)) stop("Fitted output lacks actual RDS")
    ha_check(rawArtifact)
    if (!is.null(artifact)) ha_check(artifact)
  }
  ha_closure()
  value <- list(schema="h10-calibration-production-fit-v1",caseID=fitSpec$caseID,repID=fitSpec$repID,
    fitKey=key,manifestSha256=runRequest$manifest$sha256,batchSha256=runRequest$batch$sha256,
    request=fitSpec$request,generation=fitSpec$generation,status=raw$status,raw=raw,
    rawArtifact=rawArtifact,fitArtifact=artifact,execution=ha_execution(),elapsedSeconds=elapsed,
    fitInvocationCount=count)
  descriptor <- ha_write(value,recordPath)
  list(caseID=fitSpec$caseID,repID=fitSpec$repID,fitKey=key,origin="production",
    path=descriptor$path,sha256=descriptor$sha256,resumed=FALSE,fitInvocationCount=count)
}
