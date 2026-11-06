# Finite descriptive projection, using the specified random-intercept model.
args <- commandArgs(trailingOnly = TRUE)
rows <- do.call(rbind, lapply(args, read.csv))
rows <- subset(rows, condition %in% c("C1", "C2", "C3"))
coefficients <- lapply(split(rows, rows$rate), function(panel) {
  stopifnot(nrow(panel) == 270L, length(unique(panel$person)) == 27L,
            all(table(panel$person) == 10L), all(panel$budget_milli > 0),
            !anyDuplicated(panel[c("person", "main")]))
  panel$condition <- factor(panel$condition, levels = c("C1", "C2", "C3"))
  fixed <- amount_a ~ 0 + condition + condition:need_a + condition:need_b
  design <- model.matrix(fixed, panel)
  stopifnot(qr(design)$rank == 9L)
  fit <- lme4::lmer(update(fixed, . ~ . + (1 | person)), panel, REML = TRUE)
  beta <- lme4::fixef(fit)
  # Check the fitted fixed effects by independently solving the GLS equations.
  ratio <- as.numeric(lme4::VarCorr(fit)$person) / stats::sigma(fit)^2
  whiten <- function(value) {
    averages <- rowsum(value, panel$person, reorder = TRUE) / 10
    value - (1 - 1/sqrt(1 + 10 * ratio)) * averages[panel$person, , drop = FALSE]
  }
  gls <- qr.solve(whiten(design), whiten(as.matrix(panel$amount_a)))
  stopifnot(max(abs(gls - beta)) < 1e-8)
  convergence <- fit@optinfo$conv$lme4$messages
  stopifnot(is.null(convergence) || (length(convergence) == 1L &&
            grepl("^boundary \\(singular\\) fit", convergence)))
  a <- unname(beta[paste0("conditionC", 1:3, ":need_a")])
  b <- unname(beta[paste0("conditionC", 1:3, ":need_b")])
  data.frame(rate = panel$rate[1], A_C1 = a[1], A_C2 = a[2], A_C3 = a[3],
             B_C1 = b[1], B_C2 = b[2], B_C3 = b[3],
             M_A2 = abs(a[2]) - abs(a[1]), M_B2 = abs(b[2]) - abs(b[1]),
             M_A3 = abs(a[3]) - abs(a[1]), M_B3 = abs(b[3]) - abs(b[1]),
             intercept_sd = sqrt(as.numeric(lme4::VarCorr(fit)$person)),
             residual_sd = stats::sigma(fit), boundary = lme4::isSingular(fit),
             gls_max_error = max(abs(gls - beta)))
})
write.csv(do.call(rbind, coefficients), row.names = FALSE)
