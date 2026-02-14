# Load package
library(igraph)
library(causaloptim)
set.seed(42)

# results <- specify_graph()

graph <- igraph::graph_from_literal(M -+Z, Z -+ X, X -+ Y,
                                    Ur -+ X, Ur -+ Y)
V(graph)$leftside <- c(1, 1, 0, 0, 0)
V(graph)$latent <- c(0, 0, 0, 0, 1)
V(graph)$nvals <- c(2, 2, 2, 2, 2)
E(graph)$rlconnect <- c(0, 0, 0, 0, 0)
E(graph)$edge.monotone <- c(0, 0,0, 0, 0)


V(graph)

# 3. Specify the causal effect and compute bounds
riskdiff <- "p{Y(X = 1) = 1}"
obj <- analyze_graph(graph, constraints = NULL, effectt = riskdiff)
print(obj[4])

bounds <- optimize_effect_2(obj)
boundsfunc <- interpret_bounds(bounds = bounds$bounds, obj$parameters)

boundsfunc
bounds

help("causaloptim")

