.get_stat_col <- function(stats, prefer = c("Edge","edges")) {
  cols <- colnames(stats)
  for (p in prefer) if (p %in% cols) return(p)
  return(cols[1])
}
