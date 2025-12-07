library(testthat)

test_that("package exposes CCM_fit and diagnostics", {
  expect_true(exists("CCM_fit", mode = "function"))
  expect_true(exists("CCM_traceplot", mode = "function"))
  expect_true(exists("CCM_density_compare", mode = "function"))
})

test_that("CCM_fit errors gracefully if reticulate missing", {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    expect_error(CCM_fit(list("Edge"), list("NP"), list()), regexp = "reticulate")
  } else {
    skip("reticulate available; smoke-run not executed in CI.")
  }
})
