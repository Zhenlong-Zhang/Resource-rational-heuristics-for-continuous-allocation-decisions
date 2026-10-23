# Independent CR2 input extraction from actual H10 fitted objects.
# No source(), lmer(), optimizer, generator or production contrast is called.

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Usage: READER REQUEST OUTPUT")
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(lmerTest))
request <- read_json(args[[1L]], simplifyVector=FALSE)
if (!identical(request$schema, "h10-inference-independent-rds-request-v1")) stop("Wrong independent request schema")
if (file.exists(args[[2L]])) stop("Independent R output already exists")
ids <- c("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
compact_formula <- function(x) gsub("[[:space:]]", "", paste(deparse(x), collapse=""))
numeric_equal <- function(a, b) {
  is.numeric(a) && is.numeric(b) && identical(dim(a), dim(b)) &&
    identical(as.numeric(a), as.numeric(b))
}
extract <- function(spec) {
  original <- read_json(spec$requestPath, simplifyVector=TRUE)
  if (!identical(original$model, "H10") || !identical(original$reference, "C2")) stop("Wrong persisted H10 request")
  d <- as.data.frame(original$rows, stringsAsFactors=FALSE)
  if (!identical(names(d), c("participant", "condition", "need_A", "need_B", "y")) &&
      !setequal(names(d), c("participant", "condition", "need_A", "need_B", "y"))) stop("Wrong request row fields")
  fit <- readRDS(spec$rdsPath)
  if (!inherits(fit, "lmerModLmerTest")) stop("Actual lmerTest fitted object required")
  if (!identical(compact_formula(formula(fit)), "y~condition*need_A+condition*need_B+(1|participant)")) stop("Unexpected actual model formula")
  if (!isTRUE(lme4::isREML(fit))) stop("Actual model must use REML")
  if (!is.null(fit@call$weights)) stop("Prior weights are outside this CR2 candidate")
  frame <- model.frame(fit)
  required <- c("participant", "condition", "need_A", "need_B", "y")
  if (!all(required %in% names(frame)) || nrow(frame) != nrow(d)) stop("Actual model frame shape mismatch")
  for (key in c("participant", "condition")) {
    if (!identical(as.character(frame[[key]]), as.character(d[[key]]))) stop(paste("Actual model frame identity mismatch", key))
  }
  for (key in c("need_A", "need_B", "y")) {
    if (!numeric_equal(frame[[key]], d[[key]])) stop(paste("Actual model frame numeric mismatch", key))
  }
  if (!identical(levels(frame$condition), c("C2", "C1", "C3"))) stop("Actual condition/reference levels mismatch")
  flist <- lme4::getME(fit, "flist")
  cnms <- lme4::getME(fit, "cnms")
  if (!identical(names(flist), "participant") || !identical(names(cnms), "participant") ||
      !identical(cnms$participant, "(Intercept)")) stop("Actual random effects differ from participant intercept")
  opt <- fit@optinfo
  if (!identical(opt$optimizer, "nloptwrap")) stop("Actual optimizer changed")
  controls <- opt$control
  expected <- list(xtol_abs=1e-8, ftol_abs=1e-8, maxeval=10000)
  if (!all(names(expected) %in% names(controls))) stop("Missing actual optimizer controls")
  for (key in names(expected)) if (!isTRUE(controls[[key]] == expected[[key]])) stop("Actual optimizer control changed")
  extra <- setdiff(names(controls), names(expected))
  if (length(extra) && (!identical(extra, "print_level") || !isTRUE(controls$print_level == 0))) stop("Unrecognized actual optimizer control")
  beta <- lme4::fixef(fit)
  V <- as.matrix(vcov(fit))
  X <- lme4::getME(fit, "X")
  if (!is.null(attr(X, "col.dropped")) || !identical(names(beta), colnames(X)) ||
      !identical(names(beta), rownames(V)) || !identical(names(beta), colnames(V))) stop("Actual fixed coefficient design mismatch")
  gradient <- opt$derivs$gradient
  H <- opt$derivs$Hessian
  eigenvalues <- if (!is.null(H)) eigen(H, symmetric=TRUE, only.values=TRUE)$values else numeric()
  messages <- unlist(opt$conv$lme4$messages)
  code <- opt$conv$opt
  converged <- all(code == 0) && length(messages) == 0L
  singular <- lme4::isSingular(fit, tol=1e-4)
  hessian_ok <- length(eigenvalues) > 0L && all(is.finite(eigenvalues)) && min(eigenvalues) > 0
  gradient_ok <- !is.null(gradient) && all(is.finite(gradient))
  usable <- converged && !singular && hessian_ok && gradient_ok && all(is.finite(beta)) && all(is.finite(V))
  fixed_terms <- delete.response(terms(lme4::nobars(formula(fit))))
  slope_vector <- function(condition, need) {
    pair <- data.frame(condition=factor(rep(condition, 2L), levels=levels(frame$condition)),
                       need_A=c(0, 0), need_B=c(0, 0))
    pair[[need]][[2L]] <- 1
    mm <- model.matrix(fixed_terms, pair, contrasts.arg=attr(X, "contrasts"))
    if (!identical(colnames(mm), names(beta))) stop("Independent contrast design changed")
    as.numeric(mm[2L, ] - mm[1L, ])
  }
  slopes <- list()
  vectors <- list()
  for (condition in c("C1", "C2", "C3")) for (need in c("A", "B")) {
    L <- slope_vector(condition, paste0("need_", need))
    key <- paste(condition, need, sep=":")
    vectors[[key]] <- L
    slopes[[key]] <- list(L=L, estimate=sum(L * beta), SE=sqrt(as.numeric(t(L) %*% V %*% L)))
  }
  comparisons <- list()
  Ls <- matrix(0, nrow=4L, ncol=length(beta))
  for (i in seq_along(ids)) {
    pieces <- strsplit(ids[[i]], ":", fixed=TRUE)[[1L]]
    other <- substr(pieces[[1L]], 4L, 5L)
    need <- pieces[[2L]]
    L <- vectors[[paste0("C1:", need)]] - vectors[[paste0(other, ":", need)]]
    Ls[i, ] <- L
    estimate <- sum(L * beta)
    SE <- sqrt(as.numeric(t(L) %*% V %*% L))
    z <- lmerTest::contest1D(fit, L, ddf="Satterthwaite", confint=TRUE, level=.95)
    comparisons[[ids[[i]]]] <- list(L=L, coefficientNames=names(beta), estimate=estimate, SE=SE,
      df=as.numeric(z[["df"]]), statistic=estimate/SE,
      packageEstimate=as.numeric(z[["Estimate"]]), packageSE=as.numeric(z[["Std. Error"]]),
      packageStatistic=as.numeric(z[["t value"]]), packageTwoSidedP=as.numeric(z[["Pr(>|t|)"]]),
      packageConfidenceInterval=as.numeric(unlist(z[1L, c("lower", "upper")], use.names=FALSE)))
  }
  list(fitKey=spec$fitKey, formula=compact_formula(formula(fit)), REML=lme4::isREML(fit),
    optimizer=opt$optimizer, optimizerControls=controls, coefficientNames=names(beta),
    coefficients=unname(beta), fixedVcov=unname(V), contrastCovariance=unname(Ls %*% V %*% t(Ls)),
    conditionSlopes=slopes, references=comparisons,
    diagnostics=list(converged=converged, singular=singular, gradient=unname(gradient),
      hessianEigenvalues=unname(eigenvalues), optimizerMessages=as.list(messages),
      usableCandidateInference=usable),
    frameRowCount=nrow(frame), requestRowCount=nrow(d), verifiedFields=as.list(required), verified=TRUE,
    requestDescriptor={
      if (!is.character(spec$requestPath) || length(spec$requestPath)!=1L ||
          !grepl("^/",spec$requestPath)) stop("Absolute verified request path required")
      requestHash <- system2(Sys.which("sha256sum"),shQuote(spec$requestPath),stdout=TRUE,stderr=TRUE)
      if (!is.null(attr(requestHash,"status")) || length(requestHash)!=1L ||
          !grepl("^[0-9a-f]{64} [ *]",requestHash)) stop("Cannot bind verified request bytes")
      list(path=spec$requestPath,sha256=substr(requestHash,1L,64L))
    },
    residualVariance=sigma(fit)^2, randomCovariance=unname(as.matrix(lme4::VarCorr(fit)$participant)),
    theta=unname(lme4::getME(fit, "theta")),
    originalSatterthwaite=list(varianceParameterCovariance=unname(fit@vcov_varpar),
      covarianceJacobians=lapply(fit@Jac_list, unname)),
    design=list(X=unname(X), y=unname(lme4::getME(fit, "y")),
      participant=as.list(as.character(frame$participant)),
      condition=as.list(as.character(frame$condition))))
}
fits <- structure(list(), names=character())
for (spec in request$fits) {
  if (!is.null(fits[[spec$fitKey]])) stop("Duplicate independent fit key")
  fits[[spec$fitKey]] <- extract(spec)
}
result <- list(schema="h10-inference-independent-rds-v1", fits=fits,
  runtime=list(R=as.character(getRversion()), Rhome=normalizePath(R.home()),
    lme4=as.character(packageVersion("lme4")), lmerTest=as.character(packageVersion("lmerTest"))))
temporary <- tempfile(".independent-rds-", tmpdir=dirname(args[[2L]]))
write_json(result, temporary, auto_unbox=TRUE, pretty=TRUE, null="null", na="null", digits=NA)
if (!file.link(temporary, args[[2L]])) {
  unlink(temporary)
  stop("Independent R output commit failed")
}
unlink(temporary)
