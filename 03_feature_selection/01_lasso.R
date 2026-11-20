# SPDX-License-Identifier: MIT
select_lasso <- function(x, y, options, seed) {
  if (!requireNamespace("glmnet", quietly = TRUE)) stop("Install glmnet first")
  k <- as.integer(options$folds)
  if (k < 3L || min(table(y)) < k) stop("LASSO requires at least three folds and sufficient class counts")
  set.seed(seed)
  fold_id <- integer(length(y))
  for (level in unique(y)) {
    ids <- which(y == level)
    fold_id[ids] <- sample(rep(seq_len(k), length.out = length(ids)))
  }
  cv <- glmnet::cv.glmnet(as.matrix(x), y, family = "binomial", alpha = 1,
    foldid = fold_id, type.measure = "deviance", standardize = TRUE)
  rule <- options$rule
  if (!rule %in% c("lambda.1se", "lambda.min")) stop("Invalid lambda selection rule")
  coefficients <- as.matrix(stats::coef(cv, s = rule))
  setdiff(rownames(coefficients)[coefficients[, 1] != 0], "(Intercept)")
}
