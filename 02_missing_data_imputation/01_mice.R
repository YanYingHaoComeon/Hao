# SPDX-License-Identifier: MIT
"Run only on authorized or synthetic data; never publish generated artifacts."
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) stop("Expected train, assessment, labels, config, output, repository")
if (!requireNamespace("mice", quietly = TRUE)) stop("Install mice first")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install jsonlite first")
cfg <- jsonlite::fromJSON(args[4], simplifyVector = FALSE)
output <- args[5]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
train <- read.csv(args[1], check.names = FALSE)
assessment <- read.csv(args[2], check.names = FALSE)
y <- read.csv(args[3])$target
stopifnot(identical(names(train), names(assessment)), length(y) == nrow(train))
z <- rbind(train, assessment)
for (name in unlist(cfg$schema$categorical)) {
  levels <- sort(unique(train[[name]][!is.na(train[[name]])]))
  z[[name]] <- factor(z[[name]], levels = levels)
}
initial <- z
for (name in names(z)) {
  observed <- train[[name]][!is.na(train[[name]])]
  if (!length(observed)) stop(paste("Entirely missing training column", name))
  replacement <- if (is.factor(z[[name]])) names(which.max(table(z[[name]][seq_len(nrow(train))]))) else median(observed)
  initial[[name]][is.na(initial[[name]])] <- replacement
}
predictors <- mice::make.predictorMatrix(z)
methods <- setNames(rep("", ncol(z)), names(z))
methods[colSums(is.na(z)) > 0] <- "pmm"
passive <- cfg$imputation$passive
if (length(passive)) {
  for (name in names(passive)) {
    if (!name %in% names(z)) stop("Unknown passive variable")
    methods[name] <- passive[[name]]
    predictors[, name] <- 0
  }
}
ignored <- c(rep(FALSE, nrow(train)), rep(TRUE, nrow(assessment)))
fit <- mice::mice(z, m = cfg$imputation$m, maxit = cfg$imputation$iterations,
  method = methods, predictorMatrix = predictors, ignore = ignored, data.init = initial,
  seed = cfg$imputation$seed, donors = cfg$imputation$donors, matchtype = 1,
  ridge = cfg$imputation$ridge, remove.constant = FALSE, remove.collinear = FALSE,
  printFlag = FALSE)
source(file.path(args[6], "03_feature_selection", "01_lasso.R"))
source(file.path(args[6], "03_feature_selection", "02_boruta.R"))
selection <- list()
for (i in seq_len(cfg$imputation$m)) {
  completed <- mice::complete(fit, i)
  # Matrix conversion restores the configured training category codes.
  for (name in unlist(cfg$schema$categorical)) completed[[name]] <- as.numeric(as.character(completed[[name]]))
  a <- completed[seq_len(nrow(train)), , drop = FALSE]
  b <- completed[nrow(train) + seq_len(nrow(assessment)), , drop = FALSE]
  mode <- cfg$selection$mode
  if (mode == "none") {
    selected <- names(a); lasso <- boruta <- character()
  } else if (mode == "fixed") {
    selected <- unlist(cfg$selection$predictors)
    if (!all(selected %in% names(a))) stop("Unknown fixed predictor")
    lasso <- boruta <- character()
  } else if (mode == "intersection") {
    lasso <- select_lasso(a, y, cfg$selection$lasso, cfg$selection$seed + i)
    boruta <- select_boruta(a, y, cfg$selection$boruta, cfg$selection$seed + 1000L + i)
    selected <- names(a)[names(a) %in% intersect(lasso, boruta)]
  } else stop("Unknown selection mode")
  write.csv(a, file.path(output, paste0("train_", i, ".csv")), row.names = FALSE)
  write.csv(b, file.path(output, paste0("assessment_", i, ".csv")), row.names = FALSE)
  selection[[i]] <- list(lasso = lasso, boruta = boruta, retained = selected)
}
saveRDS(fit, file.path(output, "mids.rds"))
jsonlite::write_json(selection, file.path(output, "selection.json"), auto_unbox = FALSE, pretty = TRUE)
write.csv(as.data.frame.table(fit$chainMean), file.path(output, "chain_mean.csv"), row.names = FALSE)
write.csv(as.data.frame.table(fit$chainVar), file.path(output, "chain_variance.csv"), row.names = FALSE)
if (!is.null(fit$loggedEvents)) write.csv(fit$loggedEvents, file.path(output, "logged_events.csv"), row.names = FALSE)
