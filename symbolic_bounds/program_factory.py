"""
ProgramFactory class for generating constraint systems from causal DAGs.
Implements Algorithm 1 from the paper.
"""

from typing import List, Tuple, Dict, Set
import numpy as np
import itertools
from .dag import DAG
from .node import Node
from .response_type import ResponseType
from .constraints import Constraints
from .scm import SCM
from .linear_program import LinearProgram


class ProgramFactory:
    """
    Factory class for generating linear constraint systems from causal DAGs.
    
    Implements Algorithm 1 which creates a system of linear equations relating:
    - Joint probabilities p* = P(W_L, W_R) (observational data)
    - Conditional probabilities p = P(W_R | W_L) (conditional on observed W_L)
    - Decision variable q (response type probabilities)
    """
    
    @staticmethod
    def write_constraints(dag: DAG) -> Constraints:
        """
        Generate the constraint system for a given DAG using Algorithm 1.
        
        ALGORITHM 1 IMPLEMENTATION (from paper):
        Input: Causal graph G with vertex partition (W_L, W_R)
        Output: Systems of linear equations relating p* and p to q
        
        Variable naming convention matching Algorithm 1:
        - B: Number of configurations of (W_L, W_R)
        - ℵᴿ (aleph_R): Number of response type combinations γ
        - b: Index for configuration (w_{b,L}, w_{b,R})
        - γ (gamma): Index for response type combination r_γ
        - ω (omega): Simulated values of W_R given w_{b,L} and r_γ
        - gᵂⁱ: Response function for variable Wⁱ
        
        Args:
            dag: The causal DAG with nodes partitioned into W_L and W_R.
        
        Returns:
            Constraints object containing matrices P, P*, Λ, p.
        """
        constraints = Constraints()
        
        # =========================================================================
        # ALGORITHM 1 - INITIALIZATION: Enumerate response types for all vertices
        # =========================================================================
        # For each vertex, enumerate all response types
        all_response_types = dag.generate_all_response_types()
        
        # Get all nodes (needed for compatibility checking)
        all_nodes = sorted(dag.get_all_nodes(), key=lambda n: n.name)
        
        # ℵᴿ = number of response type combinations for W_R ONLY (not all nodes!)
        # q represents response type combinations for nodes in W_R
        w_r_nodes = sorted(dag.W_R, key=lambda n: n.name)
        w_r_response_type_lists = [all_response_types[node] for node in w_r_nodes]
        w_r_response_type_combinations = list(itertools.product(*w_r_response_type_lists))
        
        # ℵᴿ = |supp(R_R)| = number of response type combinations for W_R
        aleph_R = len(w_r_response_type_combinations)
        
        # Store index mapping: response type combination r_γ -> γ (0-indexed)
        # Note: r_γ only includes response types for nodes in W_R
        for gamma, rt_combo in enumerate(w_r_response_type_combinations):
            constraints.response_type_index[rt_combo] = gamma
            # Create human-readable label for r_γ (only W_R nodes)
            label_parts = []
            for node, rt in zip(w_r_nodes, rt_combo):
                rt_num = node.response_types.index(rt) + 1
                label_parts.append(f"r_{node.name}^{rt_num}")
            constraints.response_type_labels.append(", ".join(label_parts))
        
        # =========================================================================
        # ALGORITHM 1 - MAIN LOOP: for b ∈ {1, …, B} do
        # =========================================================================
        # Initialize P as a B × ℵᴿ matrix of 0s
        # Initialize P* as a B × ℵᴿ matrix of 0s  
        # Initialize Λ as a B × B matrix of 0s
        # Where B = number of configurations (w_{b,L}, w_{b,R})
        #
        # The outer loop iterates over b (configurations)
        # The inner loop iterates over γ (response type combinations for W_R)
        constraints.P, constraints.P_star, constraints.Lambda_matrix, constraints.joint_prob_index, constraints.joint_prob_labels = \
            ProgramFactory._generate_joint_constraints(dag, all_nodes, all_response_types, 
                                                      w_r_nodes, w_r_response_type_combinations)
        
        # =========================================================================
        # Generate Λ matrices for conditional probabilities
        # =========================================================================
        # For each distinct w_{b,L}, create corresponding Λ matrix entries
        w_l_nodes = sorted(dag.W_L, key=lambda n: n.name)
        
        # Generate all configurations of W_L (the conditioning variables)
        w_l_supports = [node.support for node in w_l_nodes]
        w_l_configs = list(itertools.product(*w_l_supports))
        
        for w_l_values in w_l_configs:
            # Create label for this W_L configuration
            w_l_config = tuple((node, value) for node, value in zip(w_l_nodes, w_l_values))
            w_l_label = ", ".join(f"{node.name}={value}" for node, value in w_l_config)
            condition_name = f"W_L=({w_l_label})"
            
            P_lambda, p_lambda, cond_index, cond_labels = \
                ProgramFactory._generate_conditional_constraints(
                    dag, all_nodes, all_response_types,
                    w_l_nodes, w_l_values, w_r_nodes, w_r_response_type_combinations
                )
            
            constraints.Lambda[condition_name] = P_lambda
            constraints.p_Lambda[condition_name] = p_lambda
            constraints.conditional_prob_index[condition_name] = cond_index
            constraints.conditional_prob_labels[condition_name] = cond_labels
        
        return constraints
    
    @staticmethod
    def _generate_joint_constraints(dag: DAG, all_nodes: List[Node],
                                    all_response_types: Dict[Node, List[ResponseType]],
                                    w_r_nodes: List[Node],
                                    w_r_response_type_combinations: List[Tuple[ResponseType, ...]]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, List[str]]:
        """
        Generate constraints for joint probabilities following Algorithm 1.
        
        ALGORITHM 1 - Main double loop structure:
        
        Initialize P as a B × ℵᴿ matrix of 0s
        Initialize P* as a B × ℵᴿ matrix of 0s
        Initialize Λ as a B × B matrix of 0s
        
        for b ∈ {1, …, B} do                    # For each configuration
            for γ ∈ {1, …, ℵᴿ} do               # For each response type combination (W_R only!)
                Initialize ω as empty vector
                for i ∈ R do
                    Set ωᵢ := gᵂⁱ(w_{b,L}, r_γ)  # Simulate W_R values
                end
                if ω = w_{b,R} then              # Check compatibility
                    P_{b,γ} := 1
                    Λ_{b,b} := p{W_L = w_{b,L}}
                    P*_{b,γ} := p{W_L = w_{b,L}}
                end
            end
        end
        
        Args:
            dag: The causal DAG.
            all_nodes: Ordered list of all nodes (both W_L and W_R).
            all_response_types: Response types for all nodes.
            w_r_nodes: Ordered list of nodes in W_R.
            w_r_response_type_combinations: All r_γ for γ ∈ {1, ..., ℵᴿ} (W_R only).
        
        Returns:
            Tuple of (P matrix [B × ℵᴿ], P* matrix [B × ℵᴿ], Λ matrix [B × B], index mapping, labels).
        """
        # Generate all possible configurations (w_{b,L}, w_{b,R}) for b ∈ {1, ..., B}
        all_supports = [node.support for node in all_nodes]
        all_configs = list(itertools.product(*all_supports))
        
        B = len(all_configs)  # Number of configurations
        aleph_R = len(w_r_response_type_combinations)  # ℵᴿ = |supp(R_R)|
        
        # Initialize matrices according to Algorithm 1
        P = np.zeros((B, aleph_R))  # P as a B × ℵᴿ matrix of 0s
        P_star = np.zeros((B, aleph_R))  # P* as a B × ℵᴿ matrix of 0s
        Lambda = np.zeros((B, B))  # Λ as a B × B matrix of 0s
        joint_prob_index = {}
        joint_prob_labels = []
        
        # Get W_L nodes to compute marginal probabilities
        w_l_nodes = sorted(dag.W_L, key=lambda n: n.name)
        
        # for b ∈ {1, …, B} do
        for b, value_config in enumerate(all_configs):
            # Configuration (w_{b,L}, w_{b,R})
            config = tuple((node, value) for node, value in zip(all_nodes, value_config))
            joint_prob_index[config] = b
            
            # Create label for configuration b
            label = ", ".join(f"{node.name}={value}" for node, value in config)
            joint_prob_labels.append(label)
            
            # Extract w_{b,L} from the configuration
            config_dict = dict(config)
            w_b_L = tuple((node, config_dict[node]) for node in w_l_nodes)
            
            # for γ ∈ {1, …, ℵᴿ} do (γ indexes response type combinations for W_R only)
            for gamma, r_gamma_wr in enumerate(w_r_response_type_combinations):
                # Check if ω = w_{b,R} where ωᵢ := gᵂⁱ(w_{b,L}, r_γ) for i ∈ R
                # r_gamma_wr contains only response types for W_R nodes
                if ProgramFactory._is_compatible_wr(dag, all_nodes, all_response_types,
                                                     w_r_nodes, r_gamma_wr, config):
                    # if ω = w_{b,R} then:
                    # P_{b,γ} := 1
                    P[b, gamma] = 1.0
                    
                    # Λ_{b,b} := p{W_L = w_{b,L}}
                    # P*_{b,γ} := p{W_L = w_{b,L}}
                    # Note: These are placeholders - actual probability values
                    # would be filled in when data is provided
                    Lambda[b, b] = 1.0  # Placeholder
                    P_star[b, gamma] = 1.0  # Placeholder
        
        return P, P_star, Lambda, joint_prob_index, joint_prob_labels
    
    @staticmethod
    def _generate_conditional_constraints(dag: DAG, all_nodes: List[Node],
                                         all_response_types: Dict[Node, List[ResponseType]],
                                         w_l_nodes: List[Node], w_l_values: Tuple[int, ...],
                                         w_r_nodes: List[Node],
                                         w_r_response_type_combinations: List[Tuple[ResponseType, ...]]) \
            -> Tuple[np.ndarray, np.ndarray, Dict, List[str]]:
        """
        Generate constraints for conditional probabilities P(W_R | W_L).
        
        This generates the Λ matrix entries for a specific w_{b,L} configuration.
        The Λ matrix relates to the conditional probabilities p = P(W_R | W_L).
        
        For the given w_{b,L} (specified by w_l_values):
        - We iterate over all possible w_R configurations
        - For each (w_{b,L}, w_R) pair, we check all response types r_γ (W_R only!)
        - If r_γ produces this configuration, we set the corresponding entry to 1
        
        This uses the same compatibility check as in the main algorithm:
        checking if ω = w_{b,R} where ωᵢ := gᵂⁱ(w_{b,L}, r_γ)
        
        Args:
            dag: The causal DAG.
            all_nodes: Ordered list of all nodes.
            all_response_types: Response types for all nodes.
            w_l_nodes: List of nodes in W_L.
            w_l_values: The specific w_{b,L} values (conditioning values).
            w_r_nodes: List of nodes in W_R.
            w_r_response_type_combinations: All r_γ for γ ∈ {1, ..., ℵᴿ} (W_R only).
        
        Returns:
            Tuple of (P_Lambda matrix, p_Lambda vector, index mapping, labels).
        """
        # Generate all configurations of W_R variables
        w_r_supports = [node.support for node in w_r_nodes]
        w_r_configs = list(itertools.product(*w_r_supports))
        
        n_configs = len(w_r_configs)
        aleph_R = len(w_r_response_type_combinations)  # ℵᴿ = |supp(R_R)|
        
        # Initialize P_Λ matrix
        P_lambda = np.zeros((n_configs, aleph_R))
        p_lambda = np.zeros(n_configs)  # Placeholder for p values
        cond_prob_index = {}
        cond_prob_labels = []
        
        # Fixed w_{b,L} configuration
        w_l_config = tuple((node, value) for node, value in zip(w_l_nodes, w_l_values))
        w_l_dict = dict(w_l_config)
        
        # For each possible w_R (this is like iterating over subset of b indices)
        for config_idx, w_r_value_config in enumerate(w_r_configs):
            # Create configuration w_{b,R}
            w_r_config = tuple((node, value) for node, value in zip(w_r_nodes, w_r_value_config))
            cond_prob_index[w_r_config] = config_idx
            
            # Create label: P(W_R = w_R | W_L = w_{b,L})
            w_r_label = ", ".join(f"{node.name}={value}" for node, value in w_r_config)
            w_l_label = ", ".join(f"{node.name}={value}" for node, value in w_l_config)
            label = f"{w_r_label} | {w_l_label}"
            cond_prob_labels.append(label)
            
            # Build full configuration (w_{b,L}, w_{b,R})
            full_config_dict = {**w_l_dict, **dict(w_r_config)}
            full_config = tuple((node, full_config_dict[node]) for node in all_nodes)
            
            # for γ ∈ {1, …, ℵᴿ} do (γ indexes response type combinations for W_R only)
            for gamma, r_gamma_wr in enumerate(w_r_response_type_combinations):
                # Check if ω = w_{b,R} where ωᵢ := gᵂⁱ(w_{b,L}, r_γ) for i ∈ R
                if ProgramFactory._is_compatible_wr(dag, all_nodes, all_response_types,
                                                     w_r_nodes, r_gamma_wr, full_config):
                    P_lambda[config_idx, gamma] = 1.0
        
        return P_lambda, p_lambda, cond_prob_index, cond_prob_labels
    
    @staticmethod
    def _is_compatible_wr(dag: DAG, all_nodes: List[Node],
                         all_response_types: Dict[Node, List[ResponseType]],
                         w_r_nodes: List[Node],
                         w_r_response_type_combo: Tuple[ResponseType, ...],
                         configuration: Tuple[Tuple[Node, int], ...]) -> bool:
        """
        Check if response type combination r_γ (for W_R only) produces configuration (w_{b,L}, w_{b,R}).
        
        ALGORITHM 1 - Compatibility check (lines within the inner loop):
        
        Initialize ω as an empty vector of length |R| (= n − |L|)
        for i ∈ R do
            Set ωᵢ := gᵂⁱ(w_{b,L}, r_γ)    # Apply response function
        end
        if ω = w_{b,R} then                 # Check if simulated values match
            [set matrix entries]
        end
        
        This function implements the check "if ω = w_{b,R}":
        - r_γ contains response types for W_R nodes only
        - For W_L nodes, we use any response type (they're determined by w_{b,L})
        - For each W_R node Wⁱ, we compute ωᵢ := gᵂⁱ(Pa(Wⁱ), r_γ)
        - We check if the simulated ω matches w_{b,R}
        
        Args:
            dag: The causal DAG.
            all_nodes: Ordered list of all nodes.
            all_response_types: Response types for all nodes.
            w_r_nodes: List of nodes in W_R.
            w_r_response_type_combo: Response type combination r_γ (W_R nodes only).
            configuration: Target configuration (w_{b,L}, w_{b,R}).
        
        Returns:
            True if ω = w_{b,R} (i.e., r_γ produces the W_R part of configuration), False otherwise.
        """
        config_map = dict(configuration)
        w_r_rt_map = dict(zip(w_r_nodes, w_r_response_type_combo))
        
        # For each W_R node, check if ωᵢ := gᵂⁱ(Pa(Wⁱ), r_γ) equals w_{b,R}[i]
        for node in w_r_nodes:
            rt = w_r_rt_map[node]  # Response function gᵂⁱ from r_γ
            target_value = config_map[node]  # Target value from w_{b,R}
            
            # Get parent values from configuration (parents can be in W_L or W_R)
            parents = dag.get_parents(node)
            
            if not parents:
                # Node has no parents: check if gᵂⁱ() = target_value
                if rt.get(()) != target_value:
                    return False
            else:
                # Build parent configuration: values of Pa(Wⁱ) from (w_{b,L}, w_{b,R})
                parent_config = tuple((parent, config_map[parent]) 
                                     for parent in sorted(parents, key=lambda n: n.name))
                
                # Check if ωᵢ := gᵂⁱ(Pa(Wⁱ), r_γ) equals target_value
                try:
                    if rt.get(parent_config) != target_value:
                        return False
                except KeyError:
                    # Response type doesn't have this parent configuration
                    return False
        
        # All W_R nodes match: ω = w_{b,R}
        return True
    
    @staticmethod
    def _is_compatible(dag: DAG, nodes: List[Node],
                      response_types: Tuple[ResponseType, ...],
                      configuration: Tuple[Tuple[Node, int], ...]) -> bool:
        """
        Check if response type combination r_γ produces configuration (w_{b,L}, w_{b,R}).
        
        ALGORITHM 1 - Compatibility check (lines within the inner loop):
        
        Initialize ω as an empty vector of length |R| (= n − |L|)
        for i ∈ R do
            Set ωᵢ := gᵂⁱ(w_{b,L}, r_γ)    # Apply response function
        end
        if ω = w_{b,R} then                 # Check if simulated values match
            [set matrix entries]
        end
        
        This function implements the check "if ω = w_{b,R}":
        - For each node Wⁱ, we compute ωᵢ := gᵂⁱ(parents of Wⁱ, r_γ)
        - We check if the simulated ω matches the target configuration
        
        Args:
            dag: The causal DAG.
            nodes: Ordered list of nodes.
            response_types: Response type combination r_γ.
            configuration: Target configuration (w_{b,L}, w_{b,R}).
        
        Returns:
            True if ω = w_{b,R} (i.e., r_γ produces the configuration), False otherwise.
        """
        # Create mapping from nodes to their response types and target values
        rt_map = dict(zip(nodes, response_types))
        config_map = dict(configuration)
        
        # For each node Wⁱ, check if ωᵢ := gᵂⁱ(Pa(Wⁱ), r_γ) equals the target value
        for node in nodes:
            rt = rt_map[node]  # Response function gᵂⁱ from r_γ
            target_value = config_map[node]  # Target value from (w_{b,L}, w_{b,R})
            
            # Get parent values from configuration
            parents = dag.get_parents(node)
            
            if not parents:
                # Node has no parents: check if gᵂⁱ() = target_value
                if rt.get(()) != target_value:
                    return False
            else:
                # Build parent configuration: values of Pa(Wⁱ) from (w_{b,L}, w_{b,R})
                parent_config = tuple((parent, config_map[parent]) 
                                     for parent in sorted(parents, key=lambda n: n.name))
                
                # Check if ωᵢ := gᵂⁱ(Pa(Wⁱ), r_γ) equals target_value
                try:
                    if rt.get(parent_config) != target_value:
                        return False
                except KeyError:
                    # Response type doesn't have this parent configuration
                    return False
        
        # All nodes match: ω = w_{b,R}
        return True
    
    @staticmethod
    def writeRung2(dag: DAG, Y: Set[Node], X: Set[Node], 
                   Y_values: Tuple[int, ...], X_values: Tuple[int, ...]) -> np.ndarray:
        """
        Construct objective function vector α for causal query P(Y=y | do(X=x)).
        
        This implements the objective function construction for Rung 2 (interventional) queries:
        - Y, X ⊆ W_R (both are subsets of the right partition)
        - Y ∩ X = ∅ (Y and X are disjoint)
        - Query: P(Y=y | do(X=x)) can be expressed as α^T q
        
        ALGORITHM OVERVIEW (based on Algorithm 2):
        The query P(Y=y | do(X=x)) under the structural causal model can be written as:
            P(Y=y | do(X=x)) = ∑_{γ: r_γ compatible} q_γ = α^T q
        
        where r_γ is "compatible" if:
        1. Under intervention do(X=x), we override X nodes to value x
        2. The response functions r_γ produce Y=y (for all possible W_L configurations)
        
        DIMENSIONS:
        - q has dimension ℵᴿ (number of response type combinations for ALL of W_R)
        - α has dimension ℵᴿ (same as q)
        - Each r_γ includes response types for ALL nodes in W_R (including X)
        - Under do(X=x), we simply override X values in simulation, but r_γ still 
          includes response types for X (they just don't affect the outcome)
        
        The coefficient vector α has:
        - α_γ = 1 if response type r_γ produces Y=y under do(X=x) for ALL W_L values
        - α_γ = 0 otherwise
        
        Args:
            dag: The causal DAG with partition (W_L, W_R).
            Y: Set of target/outcome nodes in W_R.
            X: Set of intervention nodes in W_R.
            Y_values: Tuple of target values for Y nodes (in sorted order by node name).
            X_values: Tuple of intervention values for X nodes (in sorted order by node name).
        
        Returns:
            Vector α of length ℵᴿ where α^T q = P(Y=y | do(X=x)).
            Note: ℵᴿ = |supp(R_R)| where R_R are response types for ALL nodes in W_R.
            
        Raises:
            ValueError: If Y or X are not subsets of W_R, or if Y ∩ X ≠ ∅.
        
        Example:
            >>> # For query P(Y=1 | do(X=1)) on DAG with X -> Y in W_R
            >>> dag = DAG()
            >>> X = dag.add_node('X', support={0,1}, partition='R')
            >>> Y = dag.add_node('Y', support={0,1}, partition='R')
            >>> dag.add_edge(X, Y)
            >>> dag.generate_all_response_types()
            >>> alpha = ProgramFactory.writeRung2(dag, {Y}, {X}, (1,), (1,))
            >>> # alpha[γ] = 1 iff response type r_γ produces Y=1 when we set X=1
            >>> # alpha has same dimension as q (both are length ℵᴿ = 8 in this case)
        """
        # Validate inputs
        if not Y.issubset(dag.W_R):
            raise ValueError(f"Y must be a subset of W_R. Got nodes: {[n.name for n in Y]}")
        if not X.issubset(dag.W_R):
            raise ValueError(f"X must be a subset of W_R. Got nodes: {[n.name for n in X]}")
        if not Y.isdisjoint(X):
            raise ValueError(f"Y and X must be disjoint. Overlap: {[n.name for n in Y & X]}")
        
        # Get response types for W_R nodes
        all_response_types = dag.generate_all_response_types()
        w_r_nodes = sorted(dag.W_R, key=lambda n: n.name)
        w_l_nodes = sorted(dag.W_L, key=lambda n: n.name)
        
        w_r_response_type_lists = [all_response_types[node] for node in w_r_nodes]
        w_r_response_type_combinations = list(itertools.product(*w_r_response_type_lists))
        
        aleph_R = len(w_r_response_type_combinations)
        
        # Sort V and Z nodes for consistent ordering
        Y_nodes = sorted(Y, key=lambda n: n.name)
        X_nodes = sorted(X, key=lambda n: n.name)
        
        # Validate value counts
        if len(Y_values) != len(Y_nodes):
            raise ValueError(f"Expected {len(Y_nodes)} values for Y, got {len(Y_values)}")
        if len(X_values) != len(X_nodes):
            raise ValueError(f"Expected {len(X_nodes)} values for X, got {len(X_values)}")
        
        # Initialize coefficient vector
        alpha = np.zeros(aleph_R)
        
        # Create intervention and target configurations
        X_config = dict(zip(X_nodes, X_values))
        Y_target = dict(zip(Y_nodes, Y_values))
        
        # For each response type combination γ
        for gamma, r_gamma in enumerate(w_r_response_type_combinations):
            # Create mapping from W_R nodes to their response types
            rt_map = dict(zip(w_r_nodes, r_gamma))
            
            # Check if this response type produces Y=y under do(X=x)
            # KEY INSIGHT: Under intervention do(X=x), we:
            # 1. Ignore the response types for X nodes (they're overridden by intervention)
            # 2. Use response types for non-X nodes to simulate their values
            # 3. Check if Y nodes take value y
            
            # We need to check this holds for ALL possible W_L configurations
            # (marginalized over W_L)
            
            # Generate all possible W_L configurations
            if w_l_nodes:
                w_l_supports = [node.support for node in w_l_nodes]
                w_l_configs = list(itertools.product(*w_l_supports))
            else:
                w_l_configs = [()]  # Empty configuration if W_L is empty
            
            # Check if r_γ produces Y=y under do(X=x) for ALL W_L configurations
            compatible_for_all = True
            
            for w_l_config_values in w_l_configs:
                # Build W_L configuration
                if w_l_nodes:
                    w_l_config = dict(zip(w_l_nodes, w_l_config_values))
                else:
                    w_l_config = {}
                
                # Simulate W_R values under intervention do(X=x) and given W_L configuration
                simulated_values = {}
                
                # Set intervention values: X = x (this overrides the response types for X)
                for node in X_nodes:
                    simulated_values[node] = X_config[node]
                
                # Compute topological order for W_R nodes to evaluate response functions
                topo_order = ProgramFactory._topological_sort_wr(dag, w_r_nodes)
                
                # Simulate values in topological order
                simulation_failed = False
                for node in topo_order:
                    if node in X_nodes:
                        # Already set by intervention
                        continue
                    
                    # Get response function for this node
                    rt = rt_map[node]
                    
                    # Get parents of this node (could be from W_L or W_R)
                    parents = sorted(dag.get_parents(node), key=lambda n: n.name)
                    
                    if not parents:
                        # No parents: use response type's fixed value
                        simulated_values[node] = rt.get(())
                    else:
                        # Build parent configuration from W_L values and simulated W_R values
                        parent_config = []
                        for parent in parents:
                            if parent in w_l_nodes:
                                # Parent is in W_L - use given W_L configuration
                                parent_config.append((parent, w_l_config[parent]))
                            elif parent in w_r_nodes:
                                # Parent is in W_R - use simulated/intervened value
                                if parent in simulated_values:
                                    parent_config.append((parent, simulated_values[parent]))
                                else:
                                    # Parent not yet computed (shouldn't happen with correct topo order)
                                    simulation_failed = True
                                    break
                        
                        if simulation_failed:
                            break
                        
                        # Evaluate response function
                        parent_config_tuple = tuple(parent_config)
                        try:
                            simulated_values[node] = rt.get(parent_config_tuple)
                        except KeyError:
                            simulation_failed = True
                            break
                
                if simulation_failed:
                    compatible_for_all = False
                    break
                
                # Check if Y nodes have the target values
                for node in Y_nodes:
                    if node not in simulated_values or simulated_values[node] != Y_target[node]:
                        compatible_for_all = False
                        break
                
                if not compatible_for_all:
                    break
            
            # If compatible for all W_L configurations, set α_γ = 1
            if compatible_for_all:
                alpha[gamma] = 1.0
        
        return alpha
    
    @staticmethod
    def _topological_sort_wr(dag: DAG, w_r_nodes: List[Node]) -> List[Node]:
        """
        Compute topological ordering of W_R nodes.
        
        Args:
            dag: The causal DAG.
            w_r_nodes: List of nodes in W_R.
        
        Returns:
            List of nodes in topological order.
        """
        from collections import deque
        
        # Compute in-degrees (counting only edges within W_R)
        in_degree = {node: 0 for node in w_r_nodes}
        for parent, child in dag.edges:
            if parent in w_r_nodes and child in w_r_nodes:
                in_degree[child] += 1
        
        # Start with nodes that have no W_R parents
        queue = deque([node for node in w_r_nodes if in_degree[node] == 0])
        topo_order = []
        
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            
            # Reduce in-degree of children
            for parent, child in dag.edges:
                if parent == node and child in w_r_nodes:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        return topo_order
    
    @staticmethod
    def write_LP(
        scm: SCM,
        Y: Set[Node],
        X: Set[Node],
        Y_values: Tuple[int, ...],
        X_values: Tuple[int, ...]
    ) -> LinearProgram:
        """
        Construct a linear program for computing bounds on P(Y=y | do(X=x)).
        
        This function creates the full LP structure:
            minimize/maximize    α^T q
            subject to           P q = p
                                 q ≥ 0
                                 1^T q = 1
        
        Where:
        - α is the objective vector from writeRung2 (for the query P(Y=y | do(X=x)))
        - P is the constraint matrix from write_constraints (Algorithm 1)
        - p is the right-hand-side vector derived from the observed joint distribution
        - q is the decision variable (response type probabilities)
        
        The constraints q ≥ 0 and 1^T q = 1 are implicit (probability constraints).
        
        Args:
            scm: Structural Causal Model (DAG + observed joint distribution)
            Y: Set of outcome nodes (must be in W_R)
            X: Set of intervention nodes (must be in W_R)
            Y_values: Target values for Y nodes (tuple of ints)
            X_values: Intervention values for X nodes (tuple of ints)
        
        Returns:
            LinearProgram object containing objective, constraints, and RHS
            
        Raises:
            ValueError: If Y or X nodes are not in the DAG or not in W_R
        """
        dag = scm.dag
        
        # Validate that Y and X are in the DAG
        all_nodes = dag.get_all_nodes()
        if not Y.issubset(all_nodes):
            missing = Y - all_nodes
            raise ValueError(f"Y nodes not in DAG: {[n.name for n in missing]}")
        
        if not X.issubset(all_nodes):
            missing = X - all_nodes
            raise ValueError(f"X nodes not in DAG: {[n.name for n in missing]}")
        
        # Validate that Y and X are in W_R (required by writeRung2)
        if not Y.issubset(dag.W_R):
            not_in_wr = Y - dag.W_R
            raise ValueError(f"Y nodes must be in W_R: {[n.name for n in not_in_wr]}")
        
        if not X.issubset(dag.W_R):
            not_in_wr = X - dag.W_R
            raise ValueError(f"X nodes must be in W_R: {[n.name for n in not_in_wr]}")
        
        # Step 1: Generate objective vector α using writeRung2
        # This represents the query P(Y=y | do(X=x))
        alpha = ProgramFactory.writeRung2(dag, Y, X, Y_values, X_values)
        
        # Step 2: Generate constraint matrix P using write_constraints
        # This gives us the mapping from response types to observations
        constraints = ProgramFactory.write_constraints(dag)
        P = constraints.P  # Use only P matrix, not P* or Λ
        
        # Step 3: Construct right-hand-side vector p from observed joint
        # The rows of P correspond to configurations (w_L, w_R)
        # We need to extract the probability for each configuration from observedJoint
        
        # Get the ordered list of configurations from constraints
        p = np.zeros(len(constraints.joint_prob_labels))
        
        for i, config_label in enumerate(constraints.joint_prob_labels):
            # Parse the configuration label to extract node assignments
            # Format: "W_L=(X=0,Z=1), W_R=(Y=0)"
            config_dict = ProgramFactory._parse_config_label(config_label, dag)
            
            # Convert to frozenset of (Node, value) tuples
            config_set = set((node, value) for node, value in config_dict.items())
            
            # Get probability from observed joint
            p[i] = scm.observedJoint.get_probability(config_set)
        
        # Verify that probabilities sum to 1
        if not abs(np.sum(p) - 1.0) < 1e-10:
            raise ValueError(f"RHS probabilities must sum to 1, got {np.sum(p)}")
        
        # Step 4: Create LinearProgram object
        lp = LinearProgram(
            objective=alpha,
            constraint_matrix=P,
            rhs=p,
            variable_labels=constraints.response_type_labels,
            constraint_labels=constraints.joint_prob_labels,
            is_minimization=True  # Default to minimization for lower bound
        )
        
        return lp
    
    @staticmethod
    def _parse_config_label(config_label: str, dag: DAG) -> Dict[Node, int]:
        """
        Parse a configuration label string to extract node assignments.
        
        Example: "W_L=(X=0,Z=1), W_R=(Y=0)" -> {X: 0, Z: 1, Y: 0}
        
        Args:
            config_label: Configuration label string
            dag: DAG to look up node objects
        
        Returns:
            Dictionary mapping Node objects to their values
        """
        import re
        
        config_dict = {}
        
        # Find all "NodeName=value" patterns
        pattern = r'(\w+)=(\d+)'
        matches = re.findall(pattern, config_label)
        
        for node_name, value_str in matches:
            if node_name in dag.nodes:
                node = dag.nodes[node_name]
                value = int(value_str)
                config_dict[node] = value
        
        return config_dict
