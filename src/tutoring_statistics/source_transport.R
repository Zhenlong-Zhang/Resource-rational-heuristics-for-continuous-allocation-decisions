# Exact source/actual-X check before the frozen jsonlite numeric transport.
# No source generation, fitting, optimization, CR2 or testing is performed here.

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(lmerTest))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L, !file.exists(args[[2L]]))
request <- read_json(args[[1L]], simplifyVector=FALSE)
stopifnot(identical(request$schema, "h10-power-source-transport-request-v1"))
check_pin <- function(pin) {
  stopifnot(setequal(names(pin), c("path", "sha256")), grepl("^/", pin$path))
  value <- system2(Sys.which("sha256sum"), shQuote(pin$path), stdout=TRUE, stderr=TRUE)
  stopifnot(is.null(attr(value,"status")), length(value)==1L,
            identical(substr(value,1L,64L), pin$sha256))
}
exact_matrix <- function(actual, expected) {
  if (!identical(dim(actual), dim(expected)) ||
      !identical(as.numeric(actual), as.numeric(expected))) stop("Exact source/actual matrix mismatch")
}
fits <- structure(list(), names=character())
for (spec in request$fits) {
  stopifnot(is.null(fits[[spec$fitKey]]))
  check_pin(spec$request); check_pin(spec$fitArtifact)
  original <- read_json(spec$request$path, simplifyVector=TRUE)
  stopifnot(identical(original$model,"H10"), identical(original$reference,"C2"))
  d <- as.data.frame(original$rows, stringsAsFactors=FALSE)
  stopifnot(setequal(names(d),c("participant","condition","need_A","need_B","y")))
  c1 <- as.numeric(d$condition=="C1"); c3 <- as.numeric(d$condition=="C3")
  expected <- cbind(1,c1,c3,d$need_A,d$need_B,c1*d$need_A,c3*d$need_A,c1*d$need_B,c3*d$need_B)
  fit <- readRDS(spec$fitArtifact$path)
  stopifnot(inherits(fit,"lmerModLmerTest"))
  actual <- lme4::getME(fit,"X")
  exact_matrix(actual, expected)
  stopifnot(identical(as.numeric(lme4::getME(fit,"y")),as.numeric(d$y)))
  # One ULP at the exact intercept1 must fail the actual same-R check.
  changed <- actual; changed[1L,1L] <- 1+.Machine$double.eps
  rejected <- tryCatch({exact_matrix(changed,expected);FALSE},error=function(e)TRUE)
  stopifnot(isTRUE(rejected))
  fits[[spec$fitKey]] <- list(request=spec$request,fitArtifact=spec$fitArtifact,
    exactMatrixInR=TRUE,exactResponseInR=TRUE,oneULPMatrixMutationRejected=TRUE,
    design=list(X=unname(expected),y=unname(d$y),
      participant=as.list(as.character(d$participant)),condition=as.list(as.character(d$condition))))
  check_pin(spec$request); check_pin(spec$fitArtifact)
}
value <- list(schema="h10-power-source-transport-v1",fits=fits,
  runtime=list(R=as.character(getRversion()),jsonlite=as.character(packageVersion("jsonlite"))),
  newScientificInputs=0L,newFitInvocations=0L,newCandidateCalculations=0L)
temporary <- tempfile(".source-transport-",tmpdir=dirname(args[[2L]]))
write_json(value,temporary,pretty=TRUE,auto_unbox=TRUE,digits=NA,null="null",na="null")
if (!file.link(temporary,args[[2L]])) {unlink(temporary);stop("Source transport commit failed")}
unlink(temporary)
