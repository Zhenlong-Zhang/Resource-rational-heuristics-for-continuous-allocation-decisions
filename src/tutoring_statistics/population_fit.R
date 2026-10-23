# Accepted numerical definitions; portable execution is in kernel_entry.R.
pd_extract <- function(spec,old) {
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
  counts <- unlist(spec$populationCounts)
  if (!setequal(names(counts),c("C1","C2","C3")) || any(counts<1 | counts>193 | counts!=floor(counts))) stop("Changed expected population")
  if (!setequal(names(data),required) || !all(required %in% names(frame)) || nrow(frame)!=10L*sum(counts) || nrow(data)!=10L*sum(counts))
    stop("Changed actual frame shape")
  for (key in c("participant","condition")) if (!identical(as.character(frame[[key]]),as.character(data[[key]]))) stop("Changed actual frame identity")
  for (key in c("need_A","need_B","y")) if (!identical(as.numeric(frame[[key]]),as.numeric(data[[key]]))) stop("Changed actual frame values")
  if (!identical(levels(frame$condition),c("C2","C1","C3"))) stop("Changed reference levels")
  cnms <- lme4::getME(fit,"cnms")
  if (!identical(names(cnms),"participant") || !identical(cnms$participant,"(Intercept)")) stop("Changed random structure")
  cluster <- frame$participant
  if (length(unique(cluster))!=sum(counts) || any(table(cluster)!=10L)) stop("Changed participant clusters")
  for (arm in c("C1","C2","C3")) if (length(unique(cluster[frame$condition==arm]))!=counts[[arm]]) stop("Changed arm population")
  beta <- lme4::fixef(fit); V <- as.matrix(vcov(fit)); X <- lme4::getME(fit,"X")
  # JSON null is the frozen backend's nonfinite scalar representation. Only
  # the already-unusable path admits corresponding nonfinite positions; it
  # never supplies those values to CR2 or converts them into inference.
  scalar <- function(value) if(is.null(value))NA_real_ else as.numeric(value)
  oldV <- do.call(rbind,lapply(old$raw$vcov,function(row)vapply(row,scalar,numeric(1))))
  oldBeta <- vapply(old$raw$coefficients[names(beta)],scalar,numeric(1))
  same <- function(a,b) {
    if (isTRUE(old$raw$diagnostics$usableCandidateInference)) return(hi_same_numeric(a,b))
    a<-as.numeric(a);b<-as.numeric(b)
    if(length(a)!=length(b) || !identical(is.finite(a),is.finite(b)))return(FALSE)
    good<-is.finite(a)
    !any(good) || hi_same_numeric(a[good],b[good])
  }
  if (!is.null(attr(X,"col.dropped")) || !identical(names(beta),colnames(X)) ||
      !identical(names(beta),unlist(old$raw$coefficientNames)) ||
      !same(beta,oldBeta) || !identical(dim(V),dim(oldV)) || !same(V,oldV)) {
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
