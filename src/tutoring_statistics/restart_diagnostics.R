# Accepted numerical definitions; portable execution is in kernel_entry.R.
md_sha256 <- function(path) {
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) stop("sha256sum unavailable")
  result <- system2(executable,shQuote(path),stdout=TRUE,stderr=TRUE)
  if (!is.null(attr(result,"status")) || length(result)!=1L ||
      !grepl("^[0-9a-f]{64} ",result)) stop("Cannot bind file bytes")
  substr(result,1L,64L)
}

md_json <- function(path) {
  connection <- if (grepl("[.]gz$",path)) gzfile(path,"rt") else file(path,"rt")
  on.exit(close(connection))
  jsonlite::fromJSON(paste(readLines(connection,warn=FALSE),collapse="\n"),simplifyVector=FALSE)
}

md_atomic_json <- function(value,path) {
  if (file.exists(path)) stop(paste("Refusing overwrite",path))
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  temporary <- tempfile(".uncommitted-",tmpdir=dirname(path)); on.exit(unlink(temporary))
  jsonlite::write_json(value,temporary,auto_unbox=TRUE,pretty=TRUE,null="null",na="null",digits=NA)
  if (!file.link(temporary,path)) stop(paste("Atomic commit failed",path))
  invisible(path)
}

md_equal_json <- function(x,y) {
  identical(as.character(jsonlite::toJSON(x,auto_unbox=TRUE,null="null",digits=NA)),
            as.character(jsonlite::toJSON(y,auto_unbox=TRUE,null="null",digits=NA)))
}

md_derivative_check <- function(g,H,thetaNames) {
  answer <- list(gradient=unname(g),Hessian=unname(H),hessianEigenvalues=NULL,
    scaledGradient=NULL,combinedComponents=NULL,combinedMax=NULL,combinedComponentIndex=NULL,
    combinedComponentName=NULL,gradientThreshold=.002,passesGradientCheck=NULL,
    calculationStatus="unavailable",error=NULL)
  tryCatch({
    if (!is.numeric(g) || !length(g) || any(!is.finite(g)) || !is.matrix(H) ||
        !identical(dim(H),c(length(g),length(g))) || any(!is.finite(H)))
      stop("Missing/nonfinite or incompatible derivative dimensions")
    answer$hessianEigenvalues <- unname(eigen(H,symmetric=TRUE,only.values=TRUE)$values)
    scaled <- solve(chol(H),g)
    combined <- pmin(abs(scaled),abs(g))
    if (any(!is.finite(combined))) stop("Nonfinite combined gradient")
    answer$scaledGradient <- unname(scaled)
    answer$combinedComponents <- unname(combined)
    answer$combinedMax <- max(combined)
    answer$combinedComponentIndex <- which.max(combined)
    answer$combinedComponentName <- thetaNames[[which.max(combined)]]
    answer$passesGradientCheck <- max(combined)<=.002
    answer$calculationStatus <- "computed"
    answer
  },error=function(e) {answer$error <- conditionMessage(e); answer})
}

md_numderiv_defaults <- function() {
  # Resolve actual defaults from the locked functions, rather than assume a
  # rolling manual's defaults. Only evaluate their literal named default lists.
  findDefaults <- function(fun,requireShowDetails=TRUE) {
    candidates <- list()
    walk <- function(expression) {
      if (!is.call(expression)) return(invisible(NULL))
      if (identical(expression[[1]],as.name("<-")) && length(expression)==3L &&
          identical(expression[[2]],as.name("args")) && is.call(expression[[3]]) &&
          identical(expression[[3]][[1]],as.name("list"))) {
        fields <- names(as.list(expression[[3]]))[-1L]
        if (all(c("eps","d","zero.tol","r","v") %in% fields) &&
            (!requireShowDetails || "show.details" %in% fields))
          candidates[[length(candidates)+1L]] <<- eval(expression[[3]],envir=environment(fun))
      }
      parts <- as.list(expression)
      for (i in seq_along(parts)[-1L]) {
        # Subsetting a one-element list does not evaluate an R missing argument
        # (e.g. the empty row subscript in a[,i]); skip it before [[ extraction.
        boxed <- unname(parts[i])
        if (!identical(boxed,unname(alist(x = )))) walk(boxed[[1L]])
      }
    }
    walk(body(fun))
    if (length(candidates)!=1L) stop("Cannot unambiguously resolve locked numDeriv defaults")
    candidates[[1L]]
  }
  namespace <- asNamespace("numDeriv")
  grad <- getS3method("grad","default",envir=namespace)
  hessian <- getS3method("hessian","default",envir=namespace)
  genD <- getS3method("genD","default",envir=namespace)
  gradientDefaults <- findDefaults(grad); hessianDefaults <- findDefaults(hessian)
  version <- read.dcf(file.path(find.package("numDeriv"),"DESCRIPTION"))[1,"Version"]
  if (version!="2016.8-1.1" || gradientDefaults$d!=.0001 || hessianDefaults$d!=.1 ||
      gradientDefaults$eps!=1e-4 || hessianDefaults$eps!=1e-4 ||
      gradientDefaults$r!=4 || hessianDefaults$r!=4 || gradientDefaults$v!=2 || hessianDefaults$v!=2)
    stop("Locked numDeriv Richardson defaults differ from the reviewed source")
  list(version=version,gradient=gradientDefaults,hessian=hessianDefaults,
       genDBaseDefaults=findDefaults(genD,requireShowDetails=FALSE),
       implementations=list(grad=deparse(body(grad)),hessian=deparse(body(hessian)),genD=deparse(body(genD))))
}

md_raw_fit <- function(fit,warnings) {
  beta <- lme4::fixef(fit); V <- as.matrix(vcov(fit)); derivs <- fit@optinfo$derivs
  eigenvalues <- if (is.null(derivs$Hessian)) numeric() else
    eigen(derivs$Hessian,symmetric=TRUE,only.values=TRUE)$values
  messages <- unlist(fit@optinfo$conv$lme4$messages)
  singular <- lme4::isSingular(fit,tol=1e-4)
  converged <- all(fit@optinfo$conv$opt==0) && length(messages)==0
  usable <- converged && !singular && length(eigenvalues)>0 &&
    all(is.finite(eigenvalues)) && min(eigenvalues)>0 && !is.null(derivs$gradient) &&
    all(is.finite(derivs$gradient)) && all(is.finite(beta)) && all(is.finite(V))
  L <- setNames(rep(0,length(beta)),names(beta)); L["delta"] <- 1
  z <- lmerTest::contest1D(fit,L,ddf="Satterthwaite",confint=TRUE,level=.95)
  reference <- list(estimate=unname(z[1,"Estimate"]),SE=unname(z[1,"Std. Error"]),
    df=unname(z[1,"df"]),statistic=unname(z[1,"t value"]),twoSidedP=unname(z[1,"Pr(>|t|)"]),
    CI=c(z[1,"lower"],z[1,"upper"]),L=unname(L),coefficientNames=names(beta),
    reconstructedEstimate=unname(sum(L*beta)),reconstructedSE=unname(sqrt(drop(t(L)%*%V%*%L))))
  list(status="fitted",formula=paste(deparse(formula(fit)),collapse=" "),REML=TRUE,
    method="lmerTest_Satterthwaite_candidate",versions=list(R=R.version.string,
      lme4=as.character(packageVersion("lme4")),lmerTest=as.character(packageVersion("lmerTest")),
      Matrix=as.character(packageVersion("Matrix"))),optimizer=fit@optinfo$optimizer,
    coefficients=as.list(beta),coefficientNames=names(beta),vcov=unname(V),
    randomCovariance=unname(as.matrix(lme4::VarCorr(fit)$participant)),residualVariance=sigma(fit)^2,
    diagnostics=list(converged=converged,singular=singular,gradient=unname(derivs$gradient),
      hessianEigenvalues=eigenvalues,convergenceMessages=as.list(messages),warnings=as.list(warnings),
      usableCandidateInference=usable),references=list(delta=reference),conditionSlopes=list(),
    note="Isolated same-input numerical diagnostic; historical inference unchanged")
}

md_matrix_diagnostics <- function(fit) {
  X <- lme4::getME(fit,"X"); Z <- lme4::getME(fit,"Z")
  xn <- sqrt(colSums(X^2)); zn <- sqrt(Matrix::colSums(Z^2))
  scaled <- sweep(X,2L,ifelse(xn>0,xn,1),"/")
  singular <- svd(scaled,nu=0,nv=0)$d
  tolerance <- max(dim(X))*.Machine$double.eps*max(singular)
  list(matrixX=list(dimensions=dim(X),columnNames=colnames(X),columnNorms=unname(xn),
                    scaledSingularValues=singular,rankTolerance=tolerance,
                    scaledRank=sum(singular>tolerance),zeroColumnCount=sum(xn==0)),
       matrixZ=list(dimensions=dim(Z),columnNames=colnames(Z),columnNorms=unname(zn),
                    zeroColumnCount=sum(zn==0)))
}

md_extra <- function(fit,devfun) {
  theta <- lme4::getME(fit,"theta")
  matrices <- md_matrix_diagnostics(fit)
  lower <- unname(lme4::getME(fit,"lower"))
  list(theta=list(names=names(theta),values=unname(theta),lower=lower,
                 lowerKinds=ifelse(is.finite(lower),"finite",ifelse(lower<0,"-Inf","Inf"))),
    optinfo=fit@optinfo,optimizerCode=fit@optinfo$conv$opt,
    optimizerMessage=fit@optinfo$message,feval=fit@optinfo$feval,controls=fit@optinfo$control,
    objective=list(commonAtTheta=devfun(theta),remlCriterion=-2*as.numeric(logLik(fit,REML=TRUE)),
                   logLik=as.numeric(logLik(fit,REML=TRUE))),
    finiteDifference=md_derivative_check(fit@optinfo$derivs$gradient,fit@optinfo$derivs$Hessian,names(theta)),
    matrixX=matrices$matrixX,matrixZ=matrices$matrixZ)
}

md_richardson <- function(fit,devfun,defaults) {
  theta <- lme4::getME(fit,"theta"); warnings <- character()
  result <- list(status="error",method="Richardson",methodArgs=list(gradient=defaults$gradient,hessian=defaults$hessian),
    gradient=NULL,Hessian=NULL,check=NULL,objectiveBefore=NULL,objectiveAfter=NULL,warnings=list(),error=NULL)
  tryCatch({
    result$objectiveBefore <- devfun(theta)
    g <- withCallingHandlers(numDeriv::grad(devfun,theta,method="Richardson",method.args=defaults$gradient),
      warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")})
    H <- withCallingHandlers(numDeriv::hessian(devfun,theta,method="Richardson",method.args=defaults$hessian),
      warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")})
    result$gradient <- unname(g); result$Hessian <- unname(H)
    result$check <- md_derivative_check(g,H,names(theta))
    result$objectiveAfter <- devfun(theta); result$status <- "computed"
    result$warnings <- as.list(warnings); result
  },error=function(e) {
    result$error <- conditionMessage(e); result$warnings <- as.list(warnings)
    result$objectiveAfter <- tryCatch(devfun(theta),error=function(e) NULL); result
  })
}

md_data <- function(requestPath) {
  request <- jsonlite::read_json(requestPath,simplifyVector=TRUE)
  if (!request$model %in% c("H3","H4","H5")) stop("Only literal H3-H5 requests are permitted")
  d <- as.data.frame(request$rows)
  d$participant <- factor(d$participant)
  d$condition <- droplevels(factor(d$condition,levels=c(request$reference,
                               setdiff(c("C1","C2","C3","C4"),request$reference))))
  d
}

md_fit_route <- function(methodID,d,baselineFit=NULL) {
  warnings <- character()
  optimizer <- if (methodID=="strict_nlopt_restart") "nloptwrap" else "nlminbwrap"
  controls <- if (methodID=="strict_nlopt_restart") list(xtol_abs=1e-10,ftol_abs=1e-10,maxeval=100000) else
    list(eval.max=20000,iter.max=10000,rel.tol=1e-10,x.tol=1e-10)
  control <- lme4::lmerControl(optimizer=optimizer,calc.derivs=TRUE,check.rankX="stop.deficient",optCtrl=controls)
  literal <- y ~ delta + (1+trial_c|participant)
  error <- NULL
  fit <- tryCatch(withCallingHandlers({
    if (methodID=="strict_nlopt_restart") {
      lmerTest::lmer(literal,data=d,REML=TRUE,control=control,start=list(theta=lme4::getME(baselineFit,"theta")))
    } else lmerTest::lmer(literal,data=d,REML=TRUE,control=control)
  },warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")}),
  error=function(e) {error <<- conditionMessage(e); NULL})
  list(fit=fit,warnings=warnings,error=error)
}
