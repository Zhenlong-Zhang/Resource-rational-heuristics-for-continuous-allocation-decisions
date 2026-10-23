# Accepted numerical definitions; portable execution is in kernel_entry.R.
mc_integer <- function(x, name, lower=1, upper=.Machine$integer.max - 1) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x != floor(x) ||
      x < lower || x > upper) stop(paste("Invalid integer", name))
  as.integer(x)
}

mc_same_json <- function(a, b) {
  identical(as.character(jsonlite::toJSON(a, auto_unbox=TRUE, null="null", digits=NA)),
            as.character(jsonlite::toJSON(b, auto_unbox=TRUE, null="null", digits=NA)))
}

mc_read_json <- function(path) {
  connection <- if (grepl("[.]gz$", path)) gzfile(path, "rt") else file(path, "rt")
  on.exit(close(connection))
  jsonlite::fromJSON(paste(readLines(connection, warn=FALSE), collapse="\n"),
                     simplifyVector=FALSE)
}

mc_atomic_json <- function(value, path, compress=TRUE) {
  if (file.exists(path)) stop(paste("Refusing overwrite", path))
  dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE)
  temporary <- tempfile(pattern=".uncommitted-", tmpdir=dirname(path))
  on.exit(unlink(temporary))
  connection <- if (compress) gzfile(temporary, "wt") else file(temporary, "wt")
  tryCatch(writeLines(jsonlite::toJSON(value, auto_unbox=TRUE, null="null", digits=NA),
                      connection), finally=close(connection))
  # A same-filesystem link is an atomic create-if-absent: it cannot replace a
  # concurrently committed record. No partial JSON is ever the stable target.
  if (!file.link(temporary, path)) stop(paste("Atomic no-overwrite commit failed", path))
  invisible(path)
}

mc_validate_plan <- function(plan) {
  if (!identical(plan$schema, "mixed-calibration-plan-v1") ||
      !is.list(plan$cases) || !length(plan$cases)) stop("Invalid plan schema/cases")
  if (!isTRUE(plan$replicateIndexBase==1) ||
      !identical(plan$rngKind,as.list(c("Mersenne-Twister","Inversion","Rejection"))))
    stop("Unsupported declared replicate/RNG convention")
  base <- mc_integer(plan$seedBase, "seedBase")
  stride <- mc_integer(plan$seedStride, "seedStride")
  ids <- character(); indices <- integer()
  for (case in plan$cases) {
    if (!is.character(case$caseID) || length(case$caseID) != 1L ||
        !grepl("^[A-Za-z0-9_-]+$", case$caseID)) stop("Invalid caseID")
    index <- mc_integer(case$caseIndex, "caseIndex")
    n <- mc_integer(case$nParticipants, "nParticipants", lower=3)
    B <- mc_integer(case$nReplicates, "nReplicates")
    if (B >= stride) stop("seedStride must exceed every nReplicates")
    mc_integer(as.double(base) + as.double(index)*stride + B, "largest seed")
    expected <- c(C2="H3", C3="H4", C4="H5")
    if (!case$condition %in% names(expected) ||
        !identical(case$H, unname(expected[case$condition]))) stop("Hypothesis/condition mismatch")
    if (!case$responseTransform %in% c("gaussian_exact", "bounded_grid")) stop("Unknown response transform")
    for (field in c("intercept", "beta", "sdIntercept", "sdTrial", "rho", "sigma",
                    "fixedTrialTrend", "gridStep")) {
      x <- case[[field]]
      if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) stop(paste("Invalid", field))
    }
    if (case$sdIntercept < 0 || case$sdTrial < 0 || abs(case$rho) > 1 ||
        case$sigma <= 0 || case$gridStep <= 0) stop("Invalid DGP variance/grid")
    if (case$fixedTrialTrend != 0) stop("This plan requires zero fixed trial trend")
    if (case$responseTransform == "bounded_grid" &&
        abs(100/case$gridStep - round(100/case$gridStep)) > 1e-10)
      stop("Grid must divide the 0-to-100 response range")
    ids <- c(ids, case$caseID); indices <- c(indices, index)
  }
  if (anyDuplicated(ids) || anyDuplicated(indices)) stop("Duplicate caseID or caseIndex")
  invisible(plan)
}

mc_seed <- function(plan, case, repID) {
  repID <- mc_integer(repID, "repID", upper=case$nReplicates)
  mc_integer(as.double(plan$seedBase) + as.double(case$caseIndex)*plan$seedStride + repID,
             "replicate seed")
}

mc_design <- function(path) {
  panel <- jsonlite::read_json(path, simplifyVector=TRUE)
  if (!identical(panel$schema, "tutoring-statistics-fixture-rows-v1") ||
      !identical(panel$origin, "synthetic_software_fixture")) stop("Expected accepted synthetic design panel")
  d <- as.data.frame(panel$rows)
  required <- c("participant", "session", "condition", "mainIndex", "meanAInitial", "meanBInitial")
  if (!all(required %in% names(d))) stop("Missing design fields")
  result <- list()
  for (condition in c("C2", "C3", "C4")) {
    arm <- d[d$condition == condition, , drop=FALSE]
    ids <- sort(unique(arm$participant), method="radix")
    if (length(ids) != 9L || nrow(arm) != 90L) stop("Expected nine complete design schedules per condition")
    schedules <- lapply(ids, function(id) {
      x <- arm[arm$participant == id, , drop=FALSE]
      x <- x[order(x$mainIndex), , drop=FALSE]
      if (nrow(x) != 10L || !identical(as.integer(x$mainIndex), 1:10) ||
          length(unique(x$session)) != 1L ||
          any(!is.finite(x$meanAInitial)) || any(!is.finite(x$meanBInitial)))
        stop("Incomplete or nonfinite design schedule")
      x
    })
    names(schedules) <- ids; result[[condition]] <- schedules
  }
  result
}

mc_transform <- function(value, case) {
  if (case$responseTransform == "gaussian_exact") return(value)
  clipped <- pmin(100,pmax(0,value))
  # The plan specifies half-up, not R's round-to-even convention.
  case$gridStep*floor((1/case$gridStep)*clipped+0.5)
}

mc_generate <- function(plan, case, repID, design) {
  seed <- mc_seed(plan, case, repID)
  RNGkind(kind="Mersenne-Twister", normal.kind="Inversion", sample.kind="Rejection")
  set.seed(seed)
  n <- as.integer(case$nParticipants)
  schedules <- design[[case$condition]]
  selected <- sample.int(length(schedules), n, replace=TRUE)
  # Frozen draw order: N schedule choices, each person's z0,z1, then person/trial eps.
  z <- matrix(rnorm(2L*n), ncol=2L, byrow=TRUE)
  b0 <- case$sdIntercept*z[,1]
  b1 <- case$sdTrial*(case$rho*z[,1] + sqrt(1-case$rho^2)*z[,2])
  epsilon <- matrix(rnorm(10L*n, sd=case$sigma), nrow=n, ncol=10L, byrow=TRUE)
  rows <- vector("list", n)
  latent <- observed <- numeric(10L*n)
  clippedCount <- roundedCount <- 0L
  for (i in seq_len(n)) {
    s <- schedules[[selected[i]]]
    delta <- s$meanAInitial-s$meanBInitial
    trial <- as.numeric(s$mainIndex)-5.5
    value <- case$intercept+case$beta*delta+case$fixedTrialTrend*trial+b0[i]+b1[i]*trial+epsilon[i,]
    clipped <- if (case$responseTransform == "bounded_grid") pmin(100,pmax(0,value)) else value
    y <- mc_transform(value,case)
    clippedCount <- clippedCount+sum(value < 0 | value > 100)
    roundedCount <- roundedCount+sum(y != clipped)
    positions <- (10L*(i-1L)+1L):(10L*i)
    latent[positions] <- value; observed[positions] <- y
    rows[[i]] <- data.frame(participant=sprintf("sim_%06d", i), condition=case$condition,
                            y=y, delta=delta, trial_c=trial,
                            need_A=s$meanAInitial, need_B=s$meanBInitial)
  }
  exact <- case$responseTransform == "gaussian_exact"
  nullKnown <- case$beta == 0 && case$fixedTrialTrend == 0 && (exact || case$intercept == 50)
  list(seed=seed, RNGkind=as.list(RNGkind()), scheduleIDs=as.list(names(schedules)[selected]),
       scheduleIndices=as.list(selected),
       request=list(model=case$H, reference=case$condition, rows=do.call(rbind, rows)),
       generation=list(truthBeta=if (exact) case$beta else if (nullKnown) 0 else NULL, latentBeta=case$beta,
         nullKnown=nullKnown, betaNull=case$beta==0,
         truthInterpretation=if (exact) "Gaussian conditional fixed need coefficient" else if (nullKnown)
           "bounded_null_symmetry" else
           "Latent coefficient; transformed observed-scale nonzero coefficient is not declared known",
         responseTransform=case$responseTransform, totalRows=length(latent),
         clippedCount=if (exact) 0L else clippedCount, latentOutsideRangeCount=clippedCount,
         endpointZeroCount=sum(observed==0),endpointHundredCount=sum(observed==100),
         roundedChangedCount=roundedCount, gridStep=if (exact) NULL else case$gridStep,
         minLatent=min(latent), maxLatent=max(latent), minObserved=min(observed),
         maxObserved=max(observed), observedMean=mean(observed),
         intercept=case$intercept, sdIntercept=case$sdIntercept, sdTrial=case$sdTrial,
         rho=case$rho, sigma=case$sigma, fixedTrialTrend=case$fixedTrialTrend,
         randomEffects=list(intercept=as.list(b0), trial=as.list(b1))))
}
