# KR interval diagnostic on an authenticated saved REML fit.
# Two interfaces share pbkrtest; this is not an independent KR algorithm.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L || file.exists(args[[2L]])) stop("Expected new REQUEST OUTPUT")
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(clubSandwich))
suppressPackageStartupMessages(library(pbkrtest))
request <- read_json(args[[1L]], simplifyVector=FALSE)
if (!identical(request$schema,"closed-interval-r-request-v1")) stop("Wrong request")
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

if (!identical(runtime$pbkrtest,"0.5.2")) stop("Changed KR implementation")
vectors <- if(request$caseIndex==1L)
  list("C1-C2:B:D"=slope("C1","B")-slope("C2","B")) else if(request$caseIndex==2L)
  list("C1-C2:A:S"=slope("C1","A")+slope("C2","A"),
       "C1-C3:A:S"=slope("C1","A")+slope("C3","A")) else stop("Wrong frozen law")
if (!setequal(names(vectors),names(request$references))) stop("Changed target roster")
strict <- function(code) withCallingHandlers(force(code),warning=function(w)
  stop(paste("KR numerical warning:",conditionMessage(w))))
near <- function(a,b) length(a)==length(b) && all(is.finite(c(a,b))) &&
  all(abs(a-b)<=pmax(2e-9,2e-8*pmax(abs(a),abs(b))))
reference <- function(id,L,truth,estimate,variance,df,scale,critical,ci,p) {
  if(any(!is.finite(c(L,truth,estimate,variance,df,scale,critical,ci,p))) ||
     min(variance,df,scale,critical)<=0 || length(ci)!=2L || ci[1L]>=ci[2L] ||
     p<0 || p>1) stop("Undefined KR reference")
  ambiguous <- any(vapply(ci,function(end) near(truth,end),logical(1)))
  list(linearID=id,L=unname(L),truth=truth,status="available",reason=NULL,
    estimate=unname(estimate),variance=unname(variance),SE=unname(sqrt(variance)),
    df=unname(df),scale=unname(scale),critical=unname(critical),CI=unname(ci),
    twoSidedPAtTruth=unname(p),covered=if(ambiguous) NULL else ci[1L]<=truth && truth<=ci[2L],
    coverageNumericallyUnresolved=ambiguous)
}
producer <- independent <- list()
if (isTRUE(guard$usableCandidateInference)) {
  adjustedV <- strict(pbkrtest::vcovAdj(fit))
  for(id in names(vectors)) {
    L <- vectors[[id]];truth <- request$references[[id]]$truth
    if (!identical(L,as.numeric(unlist(request$references[[id]]$L)))) stop("Changed named L")
    estimate <- sum(L*beta);variance <- as.numeric(t(L)%*%adjustedV%*%L)
    betaH <- L*truth/sum(L^2)
    if(!near(sum(L*betaH),truth)) stop("Changed KR null")
    z <- strict(getFromNamespace(".KR_adjust","pbkrtest")(
      adjustedV,modelV,matrix(L,nrow=1L),beta,betaH))
    df <- unname(z$ddf);scale <- unname(z$F.scaling);critical <- qf(.95,1,df)
    if(z$ndf!=1L || !is.finite(z$Fstat) || z$Fstat<0) stop("Undefined KR F")
    half <- sqrt(variance*critical/scale)
    producer[[id]] <- reference(id,L,truth,estimate,variance,df,scale,critical,
                                estimate+c(-half,half),z$p.value)
  }
  # Independently construct named slopes from a full grid and invert public
  # KRmodcomp at an off-estimate null. No producer variance/df/scale is reused.
  grid <- expand.grid(condition=c("C2","C1","C3"),need_A=c(0,1),need_B=c(0,1),
                      stringsAsFactors=FALSE)
  grid$condition <- factor(grid$condition,levels=c("C2","C1","C3"))
  gridX <- model.matrix(fixed,grid,contrasts.arg=attr(X,"contrasts"))
  if(!identical(colnames(gridX),names(beta))) stop("Wrong independent columns")
  grid_slope <- function(arm,variable) {
    i <- which(grid$condition==arm & grid$need_A==0 & grid$need_B==0)
    j <- which(grid$condition==arm & grid[[variable]]==1 &
      grid[[if(variable=="need_A") "need_B" else "need_A"]]==0)
    if(length(i)!=1L || length(j)!=1L) stop("Wrong slope grid")
    unname(gridX[j,]-gridX[i,])
  }
  for(id in names(request$references)) {
    arm <- substr(id,4L,5L);need <- substr(id,7L,7L);kind <- substr(id,9L,9L)
    left <- grid_slope("C1",paste0("need_",need))
    right <- grid_slope(arm,paste0("need_",need))
    L <- left + (if(kind=="S") right else -right)
    if(!identical(L,as.numeric(unlist(request$references[[id]]$L)))) stop("Changed independent L")
    truth <- request$references[[id]]$truth
    estimate <- drop(crossprod(L,beta));probe <- estimate+1
    betaH <- L*probe/drop(crossprod(L));actualNull <- drop(crossprod(L,betaH))
    if(!near(actualNull,probe)) stop("Changed public null constraint")
    z <- strict(pbkrtest::KRmodcomp(fit,matrix(L,nrow=1L),betaH=betaH))$stats
    nu <- unname(z$ddf);s <- unname(z$F.scaling)
    if(z$ndf!=1L || any(!is.finite(c(z$FstatU,z$Fstat,z$p.value,nu,s))) ||
       z$FstatU<=0 || nu<=0 || s<=0 || !near(z$Fstat,s*z$FstatU) ||
       !near(z$p.value,pf(z$Fstat,1,nu,lower.tail=FALSE))) stop("Undefined public KR")
    v <- (estimate-actualNull)^2/unname(z$FstatU);se <- sqrt(v)
    tcrit <- qt(.975,nu);half <- se*tcrit/sqrt(s)
    statistic <- sqrt(s)*(estimate-truth)/se
    independent[[id]] <- reference(id,L,truth,estimate,v,nu,s,tcrit^2,
      c(estimate-half,estimate+half),2*pt(abs(statistic),nu,lower.tail=FALSE))
  }
} else {
  for(id in names(vectors)) {
    value <- list(linearID=id,L=unname(vectors[[id]]),truth=request$references[[id]]$truth,
      status="statistical_unavailable",reason="accepted_original_guard_false",
      covered=NULL,coverageNumericallyUnresolved=FALSE)
    producer[[id]] <- independent[[id]] <- value
  }
}
# Pin every loaded package file for resume admission, not just its version.
packageFiles <- list()
for(package in loadedNamespaces()) {
  location <- find.package(package)
  for(path in list.files(location,recursive=TRUE,full.names=TRUE,all.files=TRUE)) {
    if(dir.exists(path)) next
    path <- normalizePath(path,mustWork=TRUE)
    line <- system2(Sys.which("sha256sum"),shQuote(path),stdout=TRUE,stderr=TRUE)
    if(!is.null(attr(line,"status")) || length(line)!=1L ||
       !grepl("^[0-9a-f]{64} [ *]",line)) stop("Cannot bind loaded package")
    p <- list(path=path,sha256=substr(line,1L,64L));check(p)
    packageFiles[[path]] <- p$sha256
  }
}
for(path in ls(observed,all.names=TRUE)) check(list(path=path,sha256=get(path,observed)))
value <- list(schema="closed-interval-r-v1",target=request$target,
  request=request$request,fitArtifact=request$fitArtifact,actualK=actualK,guard=guard,
  producer=producer,independent=independent,frameVerified=TRUE,exactSourceDesignInR=TRUE,
  sharedPackageCore=TRUE,independentKRAlgorithm=FALSE,runtime=runtime,
  observedInputs=as.list(observed),sourceClosure=TRUE,newFits=0L,newDraws=0L)
tmp <- tempfile(".interval-",tmpdir=dirname(args[[2L]]))
write_json(value,tmp,auto_unbox=TRUE,pretty=TRUE,null="null",na="null",digits=NA)
if(!file.link(tmp,args[[2L]])){unlink(tmp);stop("Refuse existing output")}
unlink(tmp)
