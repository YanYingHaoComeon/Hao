# SPDX-License-Identifier: MIT
select_boruta <- function(x, y, options, seed) {
  if (!requireNamespace("Boruta", quietly = TRUE)) stop("Install Boruta first")
  set.seed(seed)
  fit <- Boruta::Boruta(x = x, y = factor(y, levels = c(0, 1)),
    pValue = options$p_value, mcAdj = options$adjust_multiplicity,
    maxRuns = as.integer(options$max_runs), doTrace = 0, holdHistory = TRUE,
    getImp = Boruta::getImpRfZ, num.threads = 1)
  if (isTRUE(options$rough_fix)) fit <- Boruta::TentativeRoughFix(fit)
  Boruta::getSelectedAttributes(fit, withTentative = FALSE)
}
