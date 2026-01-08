
generate_initial_graph_CCMnet <- function(G, max_degree, ER_prob) {
  Gen_Net_counter = 1
  G_max_degree_bool = FALSE
  if (is.null(G)) {
    print("Generating Random Initial Network...")
    while (!G_max_degree_bool) {
      g <- sample_gnp(n = population, p = ER_prob, directed = FALSE)
      V(g)$CovAttribute <- covPattern
      ER_prob = ER_prob/2
      G_max_degree_bool = max(degree(g)) <= max_degree
      Gen_Net_counter =   Gen_Net_counter + 1
    }
    print("COMPLETED: Generated Random Initial Network")
    P = g
  } else {
    g = G
    P = G
  }
  return(list(P, g))
}