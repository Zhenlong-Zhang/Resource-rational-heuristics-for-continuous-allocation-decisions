# Independently profile restricted likelihood using a dense covariance,
# Cholesky whitening and QR. Compare all six fixed slopes and four contrasts.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
rows <- read.csv(args[1])
reported <- read.csv(args[2])
review <- lapply(unique(rows$rate), function(rate_id) {
  # Explicit indexing avoids masking the rate by the CSV's numeric price column.
  d <- rows[rows$rate == rate_id & rows$condition %in% c("C1", "C2", "C3"), , drop = FALSE]
  r <- reported[reported$rate == rate_id, , drop = FALSE]
  stopifnot(nrow(d) == 270L, nrow(r) == 1L, all(table(d$person) == 10L))
  group <- outer(d$condition, c("C1", "C2", "C3"), "==") * 1
  x <- cbind(group, group * d$need_a, group * d$need_b)
  z <- outer(d$person, unique(d$person), "==") * 1
  n <- nrow(x)
  k <- ncol(x)
  stopifnot(qr(x)$rank == k)
  block <- tcrossprod(z)
  restricted <- function(ratio, coefficients = FALSE) {
    lower <- t(chol(diag(n) + ratio * block))
    design_qr <- qr(forwardsolve(lower, x))
    y <- forwardsolve(lower, d$amount_a)
    if (coefficients) return(qr.coef(design_qr, y))
    residual_sum <- sum(qr.resid(design_qr, y)^2)
    (n - k) * log(residual_sum / (n - k)) +
      2 * sum(log(diag(lower))) +
      2 * sum(log(abs(diag(qr.R(design_qr)))))
  }
  profile <- optimize(restricted, c(0, 1), tol = 1e-10)
  outer_profile <- optimize(restricted, c(1, 100), tol = 1e-7)
  ratio <- if (restricted(0) <= profile$objective) 0 else profile$minimum
  stopifnot(restricted(ratio) < outer_profile$objective)
  coefficients <- restricted(ratio, coefficients = TRUE)
  a <- coefficients[4:6]
  b <- coefficients[7:9]
  reported_slopes <- unlist(r[c("A_C1", "A_C2", "A_C3", "B_C1", "B_C2", "B_C3")],
                            use.names = FALSE)
  contrasts <- c(abs(a[2]) - abs(a[1]), abs(b[2]) - abs(b[1]),
                 abs(a[3]) - abs(a[1]), abs(b[3]) - abs(b[1]))
  reported_contrasts <- unlist(r[c("M_A2", "M_B2", "M_A3", "M_B3")], use.names = FALSE)
  max_error <- max(abs(c(a, b) - reported_slopes), abs(contrasts - reported_contrasts))
  stopifnot(max_error < 1e-7, (ratio == 0) == r$boundary)
  data.frame(rate = rate_id, ratio = ratio, max_error = max_error, boundary = ratio == 0)
})
stopifnot(length(review) == 6L)
print(do.call(rbind, review), row.names = FALSE)
