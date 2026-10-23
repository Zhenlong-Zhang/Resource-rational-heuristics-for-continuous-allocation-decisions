# Accepted numerical definitions; portable execution is in kernel_entry.R.
hi_read <- function(path) read_json(path,simplifyVector=FALSE)

hi_sha <- function(path) {
  value <- system2(Sys.which("sha256sum"),shQuote(path),stdout=TRUE,stderr=TRUE)
  if (!is.null(attr(value,"status")) || length(value)!=1L || !grepl("^[0-9a-f]{64} [ *]",value)) stop("Cannot bind bytes")
  substr(value,1L,64L)
}

hi_pin <- function(path) list(path=normalizePath(path,mustWork=TRUE),sha256=hi_sha(path))

hi_observed <- new.env(parent=emptyenv())

hi_check <- function(value) {
  if (!is.character(value$path) || length(value$path)!=1L || !grepl("^/",value$path) ||
      !identical(hi_sha(value$path),value$sha256)) stop("Changed bound file")
  if (exists(value$path,hi_observed,inherits=FALSE) &&
      !identical(get(value$path,hi_observed),value$sha256)) stop("Contradictory file pins")
  assign(value$path,value$sha256,hi_observed); invisible(TRUE)
}

hi_closure <- function() {
  for (path in ls(hi_observed,all.names=TRUE))
    if (!identical(hi_sha(path),get(path,hi_observed))) stop("Source changed at closure")
  TRUE
}

hi_write <- function(value,path) {
  if (file.exists(path)) stop("Refusing overwrite")
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  temporary <- tempfile(".uncommitted-",tmpdir=dirname(path)); on.exit(unlink(temporary))
  write_json(value,temporary,auto_unbox=TRUE,pretty=TRUE,null="null",na="null",digits=NA)
  if (!file.link(temporary,path)) stop("Atomic commit failed")
  hi_pin(path)
}

hi_execution <- function() list(pid=Sys.getpid())

hi_ids <- c("C1-C2:A","C1-C2:B","C1-C3:A","C1-C3:B")

hi_method <- "participant_CR2_Satterthwaite_v1"

hi_same_numeric <- function(a,b) {
  a <- as.numeric(a); b <- as.numeric(b)
  length(a)==length(b) && all(is.finite(a)) && all(is.finite(b)) && all(abs(a-b)<=1e-10*(1+abs(b)))
}

hi_formula <- function(x) gsub("[[:space:]]","",paste(deparse(x),collapse=""))

hi_extract <- function(spec,old) {
  hi_check(spec$request); hi_check(spec$fitArtifact)
  original <- read_json(spec$request$path,simplifyVector=TRUE)
  if (!identical(original$model,"H10") || !identical(original$reference,"C2")) stop("Wrong original request")
  data <- as.data.frame(original$rows,stringsAsFactors=FALSE)
  fit <- readRDS(spec$fitArtifact$path)
  if (!inherits(fit,"lmerModLmerTest") || !isTRUE(lme4::isREML(fit)) ||
      !identical(hi_formula(formula(fit)),"y~condition*need_A+condition*need_B+(1|participant)")) stop("Changed fitted model")
  if (!is.null(fit@call$weights)) stop("Prior weights are outside this candidate")
  frame <- model.frame(fit)
  required <- c("participant","condition","need_A","need_B","y")
  if (!setequal(names(data),required) || !all(required %in% names(frame)) || nrow(frame)!=4620L || nrow(data)!=4620L)
    stop("Changed actual frame shape")
  for (key in c("participant","condition")) if (!identical(as.character(frame[[key]]),as.character(data[[key]]))) stop("Changed actual frame identity")
  for (key in c("need_A","need_B","y")) if (!identical(as.numeric(frame[[key]]),as.numeric(data[[key]]))) stop("Changed actual frame values")
  if (!identical(levels(frame$condition),c("C2","C1","C3"))) stop("Changed reference levels")
  cnms <- lme4::getME(fit,"cnms")
  if (!identical(names(cnms),"participant") || !identical(cnms$participant,"(Intercept)")) stop("Changed random structure")
  cluster <- frame$participant
  if (length(unique(cluster))!=462L || any(table(cluster)!=10L)) stop("Changed participant clusters")
  for (arm in c("C1","C2","C3")) if (length(unique(cluster[frame$condition==arm]))!=154L) stop("Changed arm population")
  beta <- lme4::fixef(fit); V <- as.matrix(vcov(fit)); X <- lme4::getME(fit,"X")
  oldV <- do.call(rbind,lapply(old$raw$vcov,unlist))
  if (!is.null(attr(X,"col.dropped")) || !identical(names(beta),colnames(X)) ||
      !identical(names(beta),unlist(old$raw$coefficientNames)) ||
      !hi_same_numeric(beta,unlist(old$raw$coefficients[names(beta)])) ||
      !identical(dim(V),dim(oldV)) || !hi_same_numeric(V,oldV)) {
    stop("Changed original fitted coefficients or covariance")
  }
  opt <- fit@optinfo
  if (!identical(opt$optimizer,"nloptwrap")) stop("Changed original optimizer")
  expected <- list(xtol_abs=1e-8,ftol_abs=1e-8,maxeval=10000)
  for (key in names(expected)) if (!isTRUE(opt$control[[key]]==expected[[key]])) stop("Changed original controls")
  extras <- setdiff(names(opt$control),names(expected))
  if (length(extras) && (!identical(extras,"print_level") || !isTRUE(opt$control$print_level==0))) stop("Unexpected optimizer control")
  gradient <- opt$derivs$gradient; H <- opt$derivs$Hessian
  eigenvalues <- if (is.null(H)) numeric() else eigen(H,symmetric=TRUE,only.values=TRUE)$values
  converged <- all(opt$conv$opt==0) && length(unlist(opt$conv$lme4$messages))==0L
  singular <- lme4::isSingular(fit,tol=1e-4)
  usable <- converged && !singular && length(eigenvalues)>0L && all(is.finite(eigenvalues)) && min(eigenvalues)>0 &&
    !is.null(gradient) && all(is.finite(gradient)) && all(is.finite(beta)) && all(is.finite(V))
  if (!identical(converged,old$raw$diagnostics$converged) || !identical(singular,old$raw$diagnostics$singular) ||
      !identical(usable,old$raw$diagnostics$usableCandidateInference)) stop("Changed original guard")
  terms <- delete.response(terms(lme4::nobars(formula(fit))))
  slope <- function(arm,need) {
    d <- data.frame(condition=factor(rep(arm,2L),levels=levels(frame$condition)),need_A=c(0,0),need_B=c(0,0))
    d[[paste0("need_",need)]][2L] <- 1
    mm <- model.matrix(terms,d,contrasts.arg=attr(X,"contrasts"))
    if (!identical(colnames(mm),names(beta))) stop("Changed contrast model matrix")
    unname(mm[2L,]-mm[1L,])
  }
  L <- do.call(rbind,lapply(hi_ids,function(id) {
    pieces <- strsplit(id,":",fixed=TRUE)[[1L]]
    slope("C1",pieces[2L])-slope(substr(pieces[1L],4L,5L),pieces[2L])
  }))
  dimnames(L) <- list(hi_ids,names(beta))
  for (id in hi_ids) if (!identical(as.numeric(L[id,]),as.numeric(unlist(old$raw$references[[id]]$L)))) stop("Changed original contrast")
  list(fit=fit,frame=frame,beta=beta,V=V,X=X,L=L,cluster=cluster)
}

hi_design <- function(parts,case) {
  X <- parts$X; frame <- parts$frame; beta <- parts$beta
  e <- as.numeric(frame$y-X%*%beta)
  theta <- as.numeric(lme4::getME(parts$fit,"theta"))
  if (length(theta)!=1L || !is.finite(theta)) stop("Unexpected random intercept theta")
  sigma2 <- sigma(parts$fit)^2
  blocks <- split(seq_len(nrow(X)),parts$cluster)
  breadInv <- matrix(0,ncol(X),ncol(X)); meatTrue <- breadInv
  byArm <- list()
  for (arm in c("C1","C2","C3")) {
    ix <- which(frame$condition==arm); a <- frame$need_A[ix]; b <- frame$need_B[ix]
    byArm[[arm]] <- list(rows=length(ix),participants=length(unique(parts$cluster[ix])),
      needARange=range(a),needBRange=range(b),needCorrelation=cor(a,b),
      marginalResidualMean=mean(e[ix]),marginalResidualMeanSquare=mean(e[ix]^2),
      marginalResidualRMS=sqrt(mean(e[ix]^2)),designGram=matrix(0,3L,3L))
  }
  for (ix in blocks) {
    n <- length(ix); arm <- as.character(frame$condition[ix[1L]])
    if (length(unique(frame$condition[ix]))!=1L) stop("Participant crosses condition")
    W <- diag(n)-matrix(theta^2/(1+n*theta^2),n,n)
    Xi <- X[ix,,drop=FALSE]; A <- cbind(1,frame$need_A[ix],frame$need_B[ix])
    breadInv <- breadInv+crossprod(Xi,W%*%Xi)
    trueV <- diag(as.numeric(case$parameters$sigmaByArm[[arm]])^2,n)+matrix(as.numeric(case$parameters$sdU)^2,n,n)
    meatTrue <- meatTrue+crossprod(Xi,W%*%trueV%*%W%*%Xi)
    byArm[[arm]]$designGram <- byArm[[arm]]$designGram+crossprod(A,W%*%A)
  }
  for (arm in names(byArm)) {
    G <- byArm[[arm]]$designGram
    byArm[[arm]]$needAAdjustedInformation <- as.numeric(G[2,2]-G[2,c(1,3),drop=FALSE]%*%solve(G[c(1,3),c(1,3)],G[c(1,3),2,drop=FALSE]))
    byArm[[arm]]$needBAdjustedInformation <- as.numeric(G[3,3]-G[3,c(1,2),drop=FALSE]%*%solve(G[c(1,2),c(1,2)],G[c(1,2),3,drop=FALSE]))
    byArm[[arm]]$designGram <- unname(G)
  }
  B <- solve(breadInv); oracleV <- B%*%meatTrue%*%B
  list(clusterCount=length(blocks),rowCount=nrow(X),sigmaSquared=sigma2,theta=unname(theta),
    randomInterceptVariance=sigma2*theta^2,ICC=theta^2/(1+theta^2),byCondition=byArm,
    designInformationInterpretation="weighted_need_information_adjusted_for_intercept_and_other_need",
    residualInterpretation="marginal_y_minus_Xbeta_no_degrees_of_freedom_bias_correction",
    oracle=list(interpretation="known_DGP_covariance_with_estimated_working_W_held_fixed_not_candidate_inference",
      coefficientVcov=unname(oracleV),contrastSE=as.list(setNames(sqrt(diag(parts$L%*%oracleV%*%t(parts$L))),hi_ids))))
}

hi_one <- function(spec,request) {
  path <- file.path(request$outputDir,"records",sprintf("rep_%04d",request$repID),paste0(spec$caseID,".json"))
  if (file.exists(path)) stop("R receives missing records only")
  for (name in c("baseline","request","generation","rawArtifact","fitArtifact")) if (!is.null(spec[[name]])) hi_check(spec[[name]])
  old <- hi_read(spec$baseline$path)
  if (!identical(old$fitKey,spec$fitKey) || !identical(old$request$sha256,spec$request$sha256) ||
      !identical(old$manifestSha256,spec$sourceManifest$sha256)) stop("Wrong baseline identity")
  started <- proc.time()[["elapsed"]]; metadata <- NULL; error <- NULL; technical <- TRUE
  if (!identical(old$raw$status,"fitted")) {
    raw <- old$raw
  } else {
    parts <- hi_extract(spec,old) # Source/model/frame contradictions are hard failures.
    warnings <- character(); phase <- "CR2_covariance"
    raw <- tryCatch(withCallingHandlers({
      covariance <- clubSandwich::vcovCR(parts$fit,cluster=parts$cluster,type="CR2",inverse_var=TRUE)
      phase <- "Satterthwaite_contrasts"
      ci <- clubSandwich::linear_contrast(parts$fit,vcov=covariance,contrasts=parts$L,
        level=.95,test="Satterthwaite",p_values=TRUE,adjustment_method="none")
      if (!all(c("Est","SE","df","CI_L","CI_U","p_val") %in% names(ci)) || nrow(ci)!=4L) stop("Unexpected released linear_contrast protocol")
      refs <- list()
      for (i in seq_along(hi_ids)) {
        L <- parts$L[i,]
        refs[[hi_ids[i]]] <- list(L=unname(L),coefficientNames=names(parts$beta),estimate=unname(ci$Est[i]),
          SE=unname(ci$SE[i]),df=unname(ci$df[i]),statistic=unname(ci$Est[i]/ci$SE[i]),
          CI=unname(c(ci$CI_L[i],ci$CI_U[i])),packageTwoSidedP=unname(ci$p_val[i]),
          reconstructedEstimate=sum(L*parts$beta),reconstructedSE=sqrt(as.numeric(t(L)%*%covariance%*%L)))
      }
      phase <- "descriptive_diagnostics"
      metadata <- list(runtime=hi_runtime,target="fitted_scaled_marginal_Theta",inverseVariance=TRUE,
        cluster="participant",residual="y-Xbeta",priorWeights=FALSE,warnings=as.list(warnings),
        modelBasedVcov=unname(parts$V),contrastCovariance=unname(parts$L%*%covariance%*%t(parts$L)),
        frameVerified=TRUE,frameRowCount=nrow(parts$frame),requestDescriptor=spec$request,
        designDiagnostics=hi_design(parts,spec$case))
      list(status="fitted",formula=old$raw$formula,coefficients=as.list(parts$beta),
        coefficientNames=names(parts$beta),vcov=unname(as.matrix(covariance)),diagnostics=old$raw$diagnostics,
        references=refs,method=hi_method)
    },warning=function(w) {warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
    error=function(e) {technical <<- FALSE;error <<- list(phase=phase,message=conditionMessage(e),class=as.list(class(e)));NULL})
    if (!is.null(metadata)) metadata$warnings <- as.list(warnings)
  }
  value <- list(schema="h10-inference-record-v1",manifest=request$manifest,method=hi_method,
    caseID=spec$caseID,repID=spec$repID,fitKey=spec$fitKey,baseline=spec$baseline,request=spec$request,
    generation=spec$generation,fitArtifact=spec$fitArtifact,technicalCompleted=technical,technicalError=error,
    rawCandidate=raw,candidateMetadata=metadata,elapsedSeconds=unname(proc.time()[["elapsed"]]-started),
    execution=hi_execution(),newFitInvocations=0L,newScientificInputs=0L)
  # Each input closes its own immutable files. The final receipt separately
  # closes all accumulated pins, including earlier records and runtime sources.
  for (name in c("baseline","request","generation","rawArtifact","fitArtifact")) if (!is.null(spec[[name]])) hi_check(spec[[name]])
  descriptor <- hi_write(value,path);hi_check(descriptor)
  c(list(caseID=spec$caseID,repID=spec$repID,fitKey=spec$fitKey),descriptor,list(resumed=FALSE))
}
