# Independent package-based references for one authenticated saved H10 fit.
# No fitting, input generation or Python contrast/df implementation is used.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L || file.exists(args[[2L]])) stop("Expected new REQUEST OUTPUT")
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(clubSandwich))
request <- read_json(args[[1L]], simplifyVector=FALSE)
if (!identical(request$schema,"h10-magnitude-independent-r-request-v1")) stop("Wrong request")
observed <- new.env(parent=emptyenv())
check <- function(pin) {
  if (!setequal(names(pin),c("path","sha256")) || !grepl("^/",pin$path)) stop("Wrong pin")
  hash_program <- Sys.which("sha256sum")
  if (!nzchar(hash_program)) stop("GNU sha256sum is required")
  line <- system2(hash_program,shQuote(pin$path),stdout=TRUE,stderr=TRUE)
  if (!is.null(attr(line,"status")) || length(line)!=1L ||
      !grepl("^[0-9a-f]{64} [ *]",line) || !identical(substr(line,1L,64L),pin$sha256)) stop("Changed input")
  if (exists(pin$path,observed,inherits=FALSE) &&
      !identical(get(pin$path,observed),pin$sha256)) stop("Conflicting pin")
  assign(pin$path,pin$sha256,observed)
}
for (pin in request$sources) check(pin)
check(request$request); check(request$fitArtifact)
runtime <- list(R=as.character(getRversion()),Rhome=normalizePath(R.home()))
for (package in names(request$packageVersions)) {
  description <- normalizePath(file.path(find.package(package),"DESCRIPTION"),mustWork=TRUE)
  expected <- request$runtimeFiles[[description]]
  if (is.null(expected)) stop("Loaded package DESCRIPTION is not bound")
  check(list(path=description,sha256=expected))
  version <- unname(read.dcf(description,fields="Version")[1L,1L])
  if (!identical(version,request$packageVersions[[package]])) stop("Changed package version")
  runtime[[package]] <- version
}
if (!identical(runtime$R,"4.2.2") || !identical(runtime$clubSandwich,"0.7.0")) stop("Changed numerical runtime")
original <- read_json(request$request$path,simplifyVector=TRUE)
if (!identical(original$model,"H10") || !identical(original$reference,"C2")) stop("Wrong original model")
d <- as.data.frame(original$rows,stringsAsFactors=FALSE)
required <- c("participant","condition","need_A","need_B","y")
if (!setequal(names(d),required)) stop("Wrong persisted row fields")
fit <- readRDS(request$fitArtifact$path)
compact <- function(x) gsub("[[:space:]]","",paste(deparse(x),collapse=""))
if (!inherits(fit,"lmerModLmerTest") || !isTRUE(lme4::isREML(fit)) ||
    !identical(compact(formula(fit)),"y~condition*need_A+condition*need_B+(1|participant)") ||
    !is.null(fit@call$weights)) stop("Unsupported actual model")
frame <- model.frame(fit)
if (!all(required %in% names(frame)) || nrow(frame)!=nrow(d)) stop("Wrong actual frame")
for (name in c("participant","condition"))
  if (!identical(as.character(frame[[name]]),as.character(d[[name]]))) stop("Changed frame identity")
for (name in c("need_A","need_B","y"))
  if (!identical(as.numeric(frame[[name]]),as.numeric(d[[name]]))) stop("Changed frame value")
if (!identical(levels(frame$condition),c("C2","C1","C3"))) stop("Changed condition coding")
cnms <- lme4::getME(fit,"cnms"); flist <- lme4::getME(fit,"flist")
if (!identical(names(cnms),"participant") || !identical(cnms$participant,"(Intercept)") ||
    !identical(names(flist),"participant")) stop("Changed participant random intercept")
cluster <- as.character(frame$participant)
if (any(table(cluster)!=10L)) stop("Incomplete participant")
arms <- c("C1","C2","C3"); actualK <- list()
for (arm in arms) actualK[[arm]] <- length(unique(cluster[as.character(frame$condition)==arm]))
for (indices in split(seq_len(nrow(frame)),cluster))
  if (length(unique(as.character(frame$condition[indices])))!=1L) stop("Participant crosses arm")
if (!identical(unname(unlist(actualK)),as.integer(unlist(request$actualK[arms])))) stop("Changed actual K")
beta <- lme4::fixef(fit); modelV <- as.matrix(vcov(fit)); X <- lme4::getME(fit,"X")
if (!is.null(attr(X,"col.dropped")) || !identical(names(beta),colnames(X)) ||
    !identical(names(beta),rownames(modelV)) || !identical(names(beta),colnames(modelV))) stop("Changed coefficient design")
# Rebuild the source design in R before any JSON numeric transport.
d$condition <- factor(d$condition,levels=levels(frame$condition))
sourceX <- model.matrix(delete.response(terms(lme4::nobars(formula(fit)))),d,
                       contrasts.arg=attr(X,"contrasts"))
if (!identical(colnames(sourceX),colnames(X)) || !identical(dim(sourceX),dim(X)) ||
    !identical(as.numeric(sourceX),as.numeric(X)) ||
    !identical(as.numeric(lme4::getME(fit,"y")),as.numeric(d$y))) stop("Changed exact source design")
opt <- fit@optinfo
if (!identical(opt$optimizer,"nloptwrap")) stop("Changed optimizer")
controls <- list(xtol_abs=1e-8,ftol_abs=1e-8,maxeval=10000)
for (name in names(controls)) if (!isTRUE(opt$control[[name]]==controls[[name]])) stop("Changed optimizer control")
extra <- setdiff(names(opt$control),names(controls))
if (length(extra) && (!identical(extra,"print_level") || !isTRUE(opt$control$print_level==0))) stop("Unexpected optimizer control")
gradient <- opt$derivs$gradient; H <- opt$derivs$Hessian
eigenvalues <- if (is.null(H)) numeric() else eigen(H,symmetric=TRUE,only.values=TRUE)$values
guard <- list(converged=all(opt$conv$opt==0) && length(unlist(opt$conv$lme4$messages))==0L,
  singular=lme4::isSingular(fit,tol=1e-4),
  hessianOK=length(eigenvalues)>0L && all(is.finite(eigenvalues)) && min(eigenvalues)>0,
  gradientOK=!is.null(gradient) && all(is.finite(gradient)),
  finiteBeta=all(is.finite(beta)),finiteV=all(is.finite(modelV)))
guard$usableCandidateInference <- guard$converged && !guard$singular && guard$hessianOK &&
  guard$gradientOK && guard$finiteBeta && guard$finiteV
fixed <- delete.response(terms(lme4::nobars(formula(fit))))
slope <- function(arm,need) {
  points <- data.frame(condition=factor(rep(arm,2L),levels=levels(frame$condition)),need_A=c(0,0),need_B=c(0,0))
  points[[paste0("need_",need)]][[2L]] <- 1
  design <- model.matrix(fixed,points,contrasts.arg=attr(X,"contrasts"))
  if (!identical(colnames(design),names(beta))) stop("Changed slope model matrix")
  unname(design[2L,]-design[1L,])
}
comparisons <- c("C1-C2:A","C1-C2:B","C1-C3:A","C1-C3:B")
vectors <- list()
for (id in comparisons) {
  pieces <- strsplit(id,":",fixed=TRUE)[[1L]]
  a <- slope("C1",pieces[[2L]]); b <- slope(substr(pieces[[1L]],4L,5L),pieces[[2L]])
  vectors[[paste0(id,":D")]] <- a-b
  vectors[[paste0(id,":S")]] <- a+b
}
L <- do.call(rbind,vectors); dimnames(L) <- list(names(vectors),names(beta))
make_reference <- function(id,estimate,se,df,ci,packageP) {
  if (!all(is.finite(c(estimate,se,df,ci,packageP))) || se<=0 || df<=0)
    stop("Undefined independent package reference; technical failure, not statistical unavailability")
  statistic <- estimate/se
  list(linearID=id,status="available",reason=NULL,L=unname(L[id,]),coefficientNames=names(beta),
    estimate=unname(estimate),SE=unname(se),df=unname(df),statistic=unname(statistic),CI=unname(ci),
    lessP=unname(pt(statistic,df)),greaterP=unname(pt(statistic,df,lower.tail=FALSE)),
    twoSidedP=unname(packageP))
}
methods <- list(); warnings <- character()
if (isTRUE(guard$usableCandidateInference)) {
  candidateV <- clubSandwich::vcovCR(fit,cluster=frame$participant,type="CR2",inverse_var=TRUE)
  # Separate package invocation for each L avoids borrowing any difference df for a sum.
  baseline <- list(); candidate <- list()
  for (id in names(vectors)) {
    z <- lmerTest::contest1D(fit,L[id,],ddf="Satterthwaite",confint=TRUE,level=.95)
    if (!all(c("Estimate","Std. Error","df","lower","upper","Pr(>|t|)") %in% names(z)) || nrow(z)!=1L)
      stop("Incomplete lmerTest reference")
    baseline[[id]] <- make_reference(id,z[["Estimate"]],z[["Std. Error"]],z[["df"]],
      as.numeric(unlist(z[1L,c("lower","upper")],use.names=FALSE)),z[["Pr(>|t|)"]])
    z <- withCallingHandlers(clubSandwich::linear_contrast(fit,vcov=candidateV,
      contrasts=L[id,,drop=FALSE],level=.95,test="Satterthwaite",p_values=TRUE,adjustment_method="none"),
      warning=function(w) {warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
    if (!all(c("Est","SE","df","CI_L","CI_U","p_val") %in% names(z)) || nrow(z)!=1L)
      stop("Incomplete clubSandwich reference")
    candidate[[id]] <- make_reference(id,z$Est,z$SE,z$df,c(z$CI_L,z$CI_U),z$p_val)
  }
  methods <- list(baseline=list(covariance=unname(modelV),contrastCovariance=unname(L%*%modelV%*%t(L)),references=baseline),
    candidate=list(covariance=unname(as.matrix(candidateV)),contrastCovariance=unname(L%*%candidateV%*%t(L)),references=candidate))
} else {
  unavailable <- lapply(names(vectors),function(id) list(linearID=id,status="statistical_unavailable",
    reason="original_fit_guard_false",L=unname(L[id,]),coefficientNames=names(beta),estimate=NULL,SE=NULL,df=NULL,
    statistic=NULL,CI=NULL,lessP=NULL,greaterP=NULL,twoSidedP=NULL))
  names(unavailable) <- names(vectors)
  methods <- list(baseline=list(covariance=NULL,contrastCovariance=NULL,references=unavailable),
                  candidate=list(covariance=NULL,contrastCovariance=NULL,references=unavailable))
}
for (name in ls(observed,all.names=TRUE)) check(list(path=name,sha256=get(name,observed)))
result <- list(schema="h10-magnitude-independent-r-v1",target=request$target,
  request=request$request,fitArtifact=request$fitArtifact,
  coefficientNames=names(beta),coefficients=unname(beta),actualK=actualK,guard=guard,
  residualVariance=sigma(fit)^2,theta=unname(lme4::getME(fit,"theta")),
  randomCovariance=unname(as.matrix(lme4::VarCorr(fit)$participant)),
  linearIDs=names(vectors),methods=methods,frameVerified=TRUE,exactSourceDesignInR=TRUE,
  rowCount=nrow(frame),clusterCount=length(unique(cluster)),warnings=as.list(warnings),
  runtime=runtime,observedInputs=as.list(observed),sourceClosure=TRUE,
  newFits=0L,newDraws=0L,RDSReads=1L)
temporary <- tempfile(".magnitude-readback-",tmpdir=dirname(args[[2L]]))
write_json(result,temporary,auto_unbox=TRUE,pretty=TRUE,null="null",na="null",digits=NA)
if (!file.link(temporary,args[[2L]])) {unlink(temporary);stop("Output exists")}
unlink(temporary)
