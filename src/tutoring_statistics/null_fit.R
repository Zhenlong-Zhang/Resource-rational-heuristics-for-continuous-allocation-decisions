# Fresh-input adapter around frozen DGP and fitting functions; portable execution.
ni_run <- function(requestPath,receiptPath) {

  suppressPackageStartupMessages(library(jsonlite))
  request <- jsonlite::read_json(requestPath,simplifyVector=FALSE)
  sys.source(request$paths$generator,envir=globalenv())
  sys.source(request$paths$diagnostics,envir=globalenv())
  for (path in names(request$newBinding$frozenDependencies))
    if (!identical(md_sha256(path),request$newBinding$frozenDependencies[[path]])) stop("Frozen dependency changed")
  if (!identical(request$schema,"mixed-independent-null-producer-request-v1") ||
      !identical(request$namespace,"r10_refined_independent_null_v1_20261019") ||
      !identical(request$evidenceRole,"independent_null_calibration") ||
      !request$mode %in% c("produce","resume_probe")) stop("Invalid fresh producer authority")
  mc_validate_plan(request$plan)
  if (request$plan$seedBase!=120261019 || request$plan$seedStride!=10000 ||
      !identical(vapply(request$plan$cases,function(x) as.integer(x$caseIndex),integer(1)),
                 c(1L,4L,5L,7L,9L,12L,13L,15L,17L,20L,21L,23L)) ||
      any(vapply(request$plan$cases,function(x) x$beta!=0 || x$nReplicates!=2000,logical(1))))
    stop("Fresh population/seed domain changed")
  case <- request$case
  choices <- Filter(function(x) identical(x$caseID,case$caseID),request$plan$cases)
  if (length(choices)!=1L || !md_equal_json(choices[[1L]],case)) stop("Fresh case differs from plan")
  ids <- vapply(request$repIDs,function(x) mc_integer(x,"repID",upper=2000),integer(1))
  if (!length(ids) || length(ids)>100L || anyDuplicated(ids)) stop("Invalid fresh ID roster")
  if (request$mode=="resume_probe" && length(ids)!=1L) stop("Resume probe needs one existing input")
  suppressPackageStartupMessages(library(lmerTest))
  defaults <- md_numderiv_defaults()
  if (!identical(md_sha256(request$paths$design),request$designSha256)) stop("Design changed")
  design <- mc_design(request$paths$design)
  items <- lapply(ids,function(repID) {
    cat("FRESH_NULL",case$caseID,repID,"\n"); flush.console()
    value <- ni_one(request,case,repID,design,defaults)
    invisible(gc()); value
  })
  for (path in names(request$newBinding$frozenDependencies))
    if (!identical(md_sha256(path),request$newBinding$frozenDependencies[[path]])) stop("Frozen dependency changed during run")
  if (!identical(md_sha256(request$paths$design),request$designSha256)) stop("Design changed during run")
  receipt <- list(schema="mixed-independent-null-producer-receipt-v1",namespace=request$namespace,
    evidenceRole=request$evidenceRole,newBinding=request$newBinding,requestSha256=md_sha256(requestPath),
    shardIndex=request$shardIndex,shardID=request$shardID,mode=request$mode,recordCount=length(items),
    records=items,scientificAdoption=FALSE)
  md_atomic_json(receipt,receiptPath)
  invisible(receipt)
}

ni_identity <- function(value,request,case,repID,schema) {
  if (!identical(value$schema,schema) || !identical(value$namespace,request$namespace) ||
      !identical(value$evidenceRole,request$evidenceRole) ||
      !md_equal_json(value$newBinding,request$newBinding) || !md_equal_json(value$case,case) ||
      !identical(value$caseID,case$caseID) || !isTRUE(value$repID==repID) ||
      !isTRUE(value$seed==mc_seed(request$plan,case,repID)) ||
      any(c("oldRawSha256","oldBackend","baselineParity") %in% names(value)))
    stop("Fresh record identity/source mismatch")
}

ni_input <- function(request,case,repID,design) {
  relative <- file.path("inputs",case$caseID,sprintf("rep_%06d.json.gz",repID))
  path <- file.path(request$outputDir,relative); existed <- file.exists(path)
  if (!existed) {
    if (request$mode=="resume_probe") stop("Resume probe cannot generate a missing input")
    generated <- mc_generate(request$plan,case,repID,design)
    value <- c(list(schema="mixed-independent-null-input-v1",namespace=request$namespace,
                   evidenceRole=request$evidenceRole,newBinding=request$newBinding,
                   caseID=case$caseID,repID=repID,case=case),generated)
    mc_atomic_json(value,path,compress=TRUE)
  }
  value <- md_json(path)
  ni_identity(value,request,case,repID,"mixed-independent-null-input-v1")
  if (!identical(value$RNGkind,as.list(c("Mersenne-Twister","Inversion","Rejection"))) ||
      length(value$request$rows)!=10*case$nParticipants ||
      length(value$scheduleIDs)!=case$nParticipants || length(value$scheduleIndices)!=case$nParticipants ||
      !identical(value$request$model,case$H) || !identical(value$request$reference,case$condition) ||
      !isTRUE(value$generation$nullKnown) || !isTRUE(value$generation$truthBeta==0))
    stop("Fresh generated input schema incomplete")
  list(value=value,path=relative,sha256=md_sha256(path),resumed=existed)
}

ni_one <- function(request,case,repID,design,defaults) {
  generated <- ni_input(request,case,repID,design); input <- generated$value
  relative <- file.path("diagnostics",case$caseID,sprintf("rep_%06d.json.gz",repID))
  path <- file.path(request$outputDir,relative)
  if (file.exists(path)) {
    saved <- md_json(path); ni_identity(saved,request,case,repID,"mixed-independent-null-diagnostic-v1")
    if (!identical(saved$inputPath,generated$path) || !identical(saved$inputSha256,generated$sha256) ||
        !md_equal_json(saved$request,input$request) ||
        !setequal(names(saved$methods),c("baseline_replay","strict_nlopt_restart","alternate_PORT")))
      stop("Committed fresh diagnostic input differs")
    if (!identical(md_sha256(file.path(request$outputDir,saved$actualRequestPath)),saved$actualRequestSha256))
      stop("Actual request artifact changed")
    for (method in saved$methods) {
      if (!method$status %in% c("fitted","error","dependency_unavailable")) stop("Incomplete method")
      if (!is.null(method$fitArtifact) &&
          !identical(md_sha256(file.path(request$outputDir,method$fitArtifact$path)),method$fitArtifact$sha256))
        stop("Fit artifact changed")
    }
    return(list(caseID=case$caseID,repID=repID,path=path,sha256=md_sha256(path),resumed=TRUE,
                inputResumed=generated$resumed))
  }
  if (request$mode=="resume_probe") stop("Resume probe cannot fit a missing diagnostic")
  artifactRelative <- file.path("artifacts",paste0(case$caseID,"_",repID,"_",Sys.getpid()))
  artifactDir <- file.path(request$outputDir,artifactRelative)
  dir.create(dirname(artifactDir),recursive=TRUE,showWarnings=FALSE)
  if (!dir.create(artifactDir,showWarnings=FALSE)) stop("Artifact directory already exists")
  requestPath <- file.path(artifactDir,"actual_request.json")
  jsonlite::write_json(input$request,requestPath,auto_unbox=TRUE,null="null",digits=NA)
  preparationPhase <- "input_preparation"; preparationError <- NULL
  d <- NULL; devfun <- NULL
  tryCatch({
    d <- md_data(requestPath)
    preparationPhase <- "common_objective_setup"
    devfun <- lme4::lmer(y ~ delta + (1+trial_c|participant),data=d,REML=TRUE,devFunOnly=TRUE,
                         control=lme4::lmerControl(check.rankX="stop.deficient"))
  },error=function(e) { preparationError <<- conditionMessage(e) })
  methods <- list(); baselineFit <- NULL; baselineMethodFailure <- FALSE
  for (methodID in c("baseline_replay","strict_nlopt_restart","alternate_PORT")) {
    began <- proc.time()[["elapsed"]]; phase <- "fit"
    result <- list(methodID=methodID,status="error",raw=NULL,extra=NULL,richardson=NULL,fitWarnings=list(),
      fitArtifact=NULL,error=NULL,errorPhase=NULL,methodFailure=FALSE,elapsedSeconds=NULL)
    if (!is.null(preparationError) && is.null(d)) {
      result$error <- preparationError; result$errorPhase <- preparationPhase
    } else if (methodID=="strict_nlopt_restart" && is.null(baselineFit)) {
      result$status <- "dependency_unavailable"; result$error <- "Baseline theta unavailable"
      result$errorPhase <- "dependency"; result$methodFailure <- baselineMethodFailure
    } else {
      result <- tryCatch({
        if (methodID=="baseline_replay") {
          phase <- "baseline_protocol"
          rawOutput <- file.path(artifactDir,"baseline_raw_emit.json")
          isolated <- new.env(parent=globalenv())
          isolated$commandArgs <- function(trailingOnly=FALSE) {
            if (!isTRUE(trailingOnly)) stop("Unexpected full commandArgs")
            c(requestPath,rawOutput)
          }
          sys.source(request$paths$backend,envir=isolated)
          if (!file.exists(rawOutput)) stop("Baseline did not emit output")
          result$raw <- md_json(rawOutput)
          if (identical(result$raw$status,"unavailable") && identical(result$raw$reason,"backend_fit_error")) {
            phase <- "fit"; baselineMethodFailure <- TRUE
            stop("Original backend emitted backend_fit_error")
          }
          if (!exists("fit",envir=isolated,inherits=FALSE) || is.null(isolated$fit)) stop("Baseline has no fit")
          fit <- isolated$fit; baselineFit <- fit
        } else {
          fitted <- md_fit_route(methodID,d,baselineFit)
          result$fitWarnings <- as.list(fitted$warnings)
          if (is.null(fitted$fit)) stop(fitted$error)
          fit <- fitted$fit
        }
        phase <- "artifact"
        fitPath <- file.path(artifactDir,paste0(methodID,".rds")); saveRDS(fit,fitPath,compress=TRUE)
        result$fitArtifact <- list(path=file.path(artifactRelative,paste0(methodID,".rds")),sha256=md_sha256(fitPath))
        phase <- "core"
        if (methodID!="baseline_replay") result$raw <- md_raw_fit(fit,fitted$warnings)
        if (is.null(devfun)) { phase <- preparationPhase; stop(preparationError) }
        result$extra <- md_extra(fit,devfun)
        result$richardson <- md_richardson(fit,devfun,defaults)
        result$status <- "fitted"; result
      },error=function(e) {
        result$status <- "error"; result$error <- conditionMessage(e); result$errorPhase <- phase
        result$methodFailure <- identical(phase,"fit"); result
      })
    }
    result$elapsedSeconds <- proc.time()[["elapsed"]]-began; methods[[methodID]] <- result
  }
  output <- list(schema="mixed-independent-null-diagnostic-v1",namespace=request$namespace,
    evidenceRole=request$evidenceRole,newBinding=request$newBinding,caseID=case$caseID,repID=repID,
    seed=input$seed,case=case,inputPath=generated$path,inputSha256=generated$sha256,request=input$request,
    actualRequestPath=file.path(artifactRelative,"actual_request.json"),actualRequestSha256=md_sha256(requestPath),
    methods=methods,numDerivDefaults=defaults,scientificAdoption=FALSE,
    execution=list(pid=Sys.getpid()))
  mc_atomic_json(output,path,compress=TRUE)
  list(caseID=case$caseID,repID=repID,path=path,sha256=md_sha256(path),resumed=FALSE,
       inputResumed=generated$resumed)
}
