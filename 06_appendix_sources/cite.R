# cite.R: numbered citations from one shared list (sources.csv in this folder).
#
# In a section's setup chunk:   source("../06_appendix_sources/cite.R")
# In the text:                  ... later in fiscal 2027. `r cite("tbac")`
# Several at once:              `r cite("census_nrs", "nahb_hmi")`   ->  [3][16]
# Three or more in a row:       `r cite("freddie_pmms", "fed_h15", "census_nrs")`   ->  [1]-[3]
# On the Sources page (chunk with results='asis'):   sources_list()
#
# Numbers come from the "n" column of sources.csv, so every section shows the
# same number for the same source. A misspelled key stops the knit with a message.

.cite_file <- function() {
  here <- c("sources.csv", "06_appendix_sources/sources.csv", "../06_appendix_sources/sources.csv")
  hit <- here[file.exists(here)]
  if (length(hit) == 0) stop("cite.R: cannot find sources.csv")
  hit[1]
}

.cite_tbl <- function() {
  read.csv(.cite_file(), stringsAsFactors = FALSE, encoding = "UTF-8", na.strings = character(0))
}

cite <- function(...) {
  keys <- c(...)
  s <- .cite_tbl()
  bad <- setdiff(keys, s$key)
  if (length(bad) > 0) stop("cite(): no such source key: ", paste(bad, collapse = ", "),
                            ". Keys are in 06_appendix_sources/sources.csv")
  n <- sort(unique(s$n[match(keys, s$key)]))
  # runs of three or more consecutive numbers print as a range: [1]-[3][5]
  runs <- split(n, cumsum(c(1, diff(n) != 1)))
  paste0(vapply(runs, function(r) if (length(r) >= 3) sprintf("[%d]-[%d]", r[1], r[length(r)])
                                  else paste0("[", r, "]", collapse = ""), character(1)), collapse = "")
}

sources_list <- function() {
  s <- .cite_tbl()
  s <- s[order(s$n), ]
  for (i in seq_len(nrow(s))) {
    url <- if (nzchar(s$url[i])) paste0(" <", s$url[i], ">") else ""
    cat(sprintf("\\[%d\\] %s%s\n\n", s$n[i], s$citation[i], url))
  }
  invisible(NULL)
}
