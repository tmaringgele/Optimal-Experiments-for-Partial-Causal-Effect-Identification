# Load package
library(igraph)
library(causaloptim)
set.seed(42)


# 2. Define the causal graph with Ur as latent
graph <- graph_from_literal(Z -+ X, X -+ Y, U -+ X, U -+ Y, W -+Z, W-+Y, M -+ Y,X-+M)
#Z, X, Y, U, W, M
V(graph)$leftside <- c(0, 0, 0, 0, 0, 0)
V(graph)$latent   <- c(0, 0, 0, 1, 1, 0)
V(graph)$nvals    <- c(2, 2, 2, 2, 2, 2)
E(graph)$rlconnect     <- c(0, 0, 0, 0, 0, 0, 0, 0)
E(graph)$edge.monotone <- c(0, 0, 0, 0, 0, 0, 0, 0)

V(graph)

# 3. Specify the causal effect and compute bounds
riskdiff <- "p{Y(X = 1) = 1} - p{Y(X = 0) = 1}"
obj <- analyze_graph(graph, constraints = NULL, effectt = riskdiff)
bounds <- optimize_effect_2(obj)
boundsfunc <- interpret_bounds(bounds = bounds$bounds, obj$parameters)

boundsfunc
bounds

help("causaloptim")

