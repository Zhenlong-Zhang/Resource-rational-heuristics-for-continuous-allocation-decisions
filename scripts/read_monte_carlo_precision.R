#!/usr/bin/env Rscript
# Independent exhaustive exact-binomial numerical planning with R's math library.
# No producer import, random draw, model fit or old-result modification.
# Usage: Rscript read_monte_carlo_precision.R INPUT OUTPUT [PYTHON_EXECUTABLE]
# Python is used only for standard-library SHA-256, not numerical calculations.
args <- commandArgs(trailingOnly=TRUE)
if (!(length(args) %in% c(2L, 3L))) stop("input/output and optional Python executable required")
stopifnot(as.character(getRversion()) == "4.2.2",
          as.character(utils::packageVersion("jsonlite")) == "1.8.8")
python <- if (length(args) == 3L) args[[3L]] else "python3"
sha256 <- function(path) {
    code <- "import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())"
    value <- system2(python, c("-c", shQuote(code), shQuote(path)), stdout=TRUE, stderr=TRUE)
    if (!is.null(attr(value, "status")) || length(value) != 1L ||
        !grepl("^[a-f0-9]{64}$", value)) stop("Cannot verify SHA-256")
    value[[1L]]
}
input <- normalizePath(args[[1]], mustWork=TRUE)
output <- args[[2]]
if (file.exists(output)) stop("output already exists")
if (!dir.create(output, recursive=FALSE)) stop("cannot create output directory")
file_arg <- grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)
stopifnot(length(file_arg) == 1L)
reader_source <- normalizePath(sub("^--file=", "", file_arg), mustWork=TRUE)
producer_source <- normalizePath(file.path(dirname(reader_source),
    "plan_monte_carlo_precision.py"), mustWork=TRUE)
observed_paths <- c(binding=file.path(input, "binding.json"),
                    result=file.path(input, "result.json"), trace=file.path(input, "trace.csv"),
                    producer=producer_source, reader=reader_source)
before_sha256 <- vapply(observed_paths, sha256, character(1))
binding <- jsonlite::fromJSON(observed_paths[["binding"]], simplifyVector=FALSE)
p <- jsonlite::fromJSON(file.path(input, "result.json"), simplifyVector=FALSE)
binding_keys <- c("schema", "rule", "numericalGuards", "searchBound", "source",
                  "readerSource", "runtime", "jobID", "newDraws", "newFits",
                  "powerReductions", "blockRounding")
result_keys <- c(binding_keys, "selected", "scannedIntegers", "traceSha256",
                 "status", "scientificAdoption", "calibrationEstablished")
stopifnot(identical(sort(names(binding)), sort(binding_keys)),
          identical(sort(names(p)), sort(result_keys)))
for (key in binding_keys) {
    if (!identical(binding[[key]], p[[key]])) stop("producer metadata mismatch")
}
check_source <- function(pin, path, digest) {
    stopifnot(is.list(pin), identical(sort(names(pin)), c("path", "sha256")),
              is.character(pin$path), length(pin$path) == 1L,
              identical(normalizePath(pin$path, mustWork=TRUE), path),
              identical(pin$sha256, digest))
}
check_source(p$source, producer_source, before_sha256[["producer"]])
check_source(p$readerSource, reader_source, before_sha256[["reader"]])
stopifnot(identical(p$traceSha256, before_sha256[["trace"]]),
          identical(p$runtime, list(python="3.10.9", numpy="1.23.5", scipy="1.10.0")),
          identical(p$status, "NUMERICAL_PLAN_COMPLETE_PENDING_INDEPENDENT_READBACK"),
          identical(p$rule, list(confidence=.95, trueProbability=.05,
              maximumHalfWidth=.01, assurance=.90,
              selection="first_integer_in_exhaustive_ascending_scan")),
          identical(p$numericalGuards, list(width=1e-12, mass=1e-10)))
trace <- read.csv(file.path(input, "trace.csv"), check.names=FALSE)
stopifnot(identical(p$schema, "h10-magnitude-precision-plan-v1"),
          p$rule$confidence == .95, p$rule$trueProbability == .05,
          p$rule$maximumHalfWidth == .01, p$rule$assurance == .90,
          p$rule$selection == "first_integer_in_exhaustive_ascending_scan",
          p$numericalGuards$width == 1e-12, p$numericalGuards$mass == 1e-10,
          p$newDraws == 0, p$newFits == 0, p$powerReductions == 0,
          !p$blockRounding, !p$scientificAdoption, !p$calibrationEstablished,
          p$selected$B == p$scannedIntegers,
          p$selected$B == nrow(trace),
          all(trace$B == seq_len(nrow(trace))),
          p$searchBound == ceiling(log(40)/(2*.01^2)),
          p$selected$B <= p$searchBound)
needed <- c("B", "assurance", "minimumWidthMargin", "totalMass", "passingCounts")
stopifnot(identical(names(trace), needed), all(is.finite(as.matrix(trace))))
stopifnot(nrow(trace) > 0L, identical(sort(names(p$selected)), sort(needed)))
rows <- vector("list", nrow(trace))
for (n in seq_len(nrow(trace))) {
    k <- 0:n
    lower <- c(0, qbeta(.025, k[-1], n-k[-1]+1))
    upper <- c(qbeta(.975, k[-length(k)]+1, n-k[-length(k)]), 1)
    mass <- dbinom(k, n, .05)
    stopifnot(all(is.finite(c(lower, upper, mass))), all(mass >= 0),
              all(lower >= 0), all(upper <= 1), all(lower <= upper),
              abs(sum(mass)-1) <= 1e-10)
    half <- (upper-lower)/2
    margin <- min(abs(half-.01))
    if (margin <= 1e-12) stop("NUMERIC_UNRESOLVED: width threshold")
    passing <- half <= .01
    assurance <- sum(mass[passing])
    if (assurance < 0 || assurance > 1+1e-10 || abs(assurance-.90) <= 1e-10)
        stop("NUMERIC_UNRESOLVED: assurance threshold")
    stopifnot(abs(assurance-trace$assurance[n]) <= 1e-10,
              abs(margin-trace$minimumWidthMargin[n]) <= 1e-12,
              abs(sum(mass)-trace$totalMass[n]) <= 1e-10,
              sum(passing) == trace$passingCounts[n])
    if (n < nrow(trace)) stopifnot(assurance < .90)
    else stopifnot(assurance >= .90)
    rows[[n]] <- data.frame(B=n, assurance=assurance, minimumWidthMargin=margin,
                            totalMass=sum(mass), passingCounts=sum(passing))
}
verified <- do.call(rbind, rows)
stopifnot(p$selected$B == tail(verified$B, 1),
          abs(p$selected$assurance-tail(verified$assurance, 1)) <= 1e-10,
          abs(p$selected$minimumWidthMargin-tail(verified$minimumWidthMargin, 1)) <= 1e-12,
          abs(p$selected$totalMass-tail(verified$totalMass, 1)) <= 1e-10,
          p$selected$passingCounts == tail(verified$passingCounts, 1))
if (!identical(before_sha256, vapply(observed_paths, sha256, character(1))))
    stop("Precision source or input changed during readback")
write.csv(verified, file.path(output, "independent_trace.csv"), row.names=FALSE)
answer <- list(schema="h10-magnitude-precision-independent-v1", passed=TRUE,
               producerInputDirectory=input, producerTraceSha256=p$traceSha256,
               producerSource=p$source, readerSource=p$readerSource,
               inputSha256=as.list(before_sha256[c("binding", "result", "trace")]),
               sourceClosure=TRUE,
               selectedB=nrow(verified), assurance=tail(verified$assurance, 1),
               scannedIntegers=nrow(verified), allSmallerIntegersFail=TRUE,
               minimumWidthMargin=min(verified$minimumWidthMargin),
               minimumAssuranceMargin=min(abs(verified$assurance-.90)),
               method="R qbeta/dbinom complete enumeration, no producer import",
               runtime=list(R=R.version.string, jsonlite=list(
                   version=as.character(utils::packageVersion("jsonlite")),
                   path=find.package("jsonlite"))), jobID=Sys.getenv("JOB_ID"),
               newDraws=0L, newFits=0L, powerReductions=0L,
               scientificAdoption=FALSE, calibrationEstablished=FALSE)
jsonlite::write_json(answer, file.path(output, "independent.json"),
                     auto_unbox=TRUE, pretty=TRUE, digits=NA)
cat("INDEPENDENT_PRECISION_PLAN_COMPLETE", nrow(verified), "\n")
