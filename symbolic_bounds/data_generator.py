"""
DataGenerator class for generating robust observed distributions from DAGs.

This module implements automatic data generation by:
1. Randomly sampling a probability distribution over response types
2. Computing the observed joint distribution by summing over compatible response types

The key insight is that each response-type configuration produces a deterministic
observable outcome, and the observed distribution is a mixture over these deterministic worlds.
"""

from typing import Dict, Tuple, FrozenSet, Set
import numpy as np
from .dag import DAG
from .node import Node
from .response_type import ResponseType
from itertools import product


class DataGenerator:
    """
    Generates robust observed distributions from a causal DAG.
    
    The generator works by:
    1. Sampling a distribution over response types P(R)
    2. Computing the observed joint P(V) = Σ_{r: g(r)=v} P(R=r)
    
    This ensures that the generated distributions are causally consistent with
    the DAG structure and can be used for both observational and interventional queries.
    
    Attributes:
        dag: The causal DAG
        trueResponseTypes: Dict mapping (Node, ResponseType) -> probability
    """
    
    def __init__(self, dag: DAG, seed: int = None):
        """
        Initialize the data generator with a DAG.
        
        Args:
            dag: The causal DAG
            seed: Random seed for reproducibility (optional)
        
        Raises:
            ValueError: If DAG has no response types generated
        """
        self.dag = dag
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate response types if not already done
        all_nodes = dag.get_all_nodes()
        if not all_nodes:
            raise ValueError("DAG has no nodes")
        
        # Check if response types are generated
        for node in all_nodes:
            if not node.response_types:
                raise ValueError(
                    f"Node {node.name} has no response types. "
                    "Call dag.generate_all_response_types() first."
                )
        
        # Sample distribution over response types
        self.trueResponseTypes = self._sample_response_type_distribution()
    
    def _sample_response_type_distribution(self) -> Dict[Tuple[Node, ResponseType], float]:
        """
        Randomly sample a probability distribution over response types for each node.
        
        For each node independently:
        1. Get all its response types
        2. Sample a probability distribution using Dirichlet distribution
        3. Store as (Node, ResponseType) -> probability
        
        Returns:
            Dictionary mapping (Node, ResponseType) to probability
        """
        distribution = {}
        
        all_nodes = self.dag.get_all_nodes()
        
        for node in all_nodes:
            response_types = node.response_types
            n_types = len(response_types)
            
            if n_types == 0:
                raise ValueError(f"Node {node.name} has no response types")
            
            # Sample from Dirichlet(1, 1, ..., 1) for uniform prior
            # This gives a random probability distribution over the simplex
            probabilities = np.random.dirichlet(np.ones(n_types))
            
            # Assign probability to each response type
            for rt, prob in zip(response_types, probabilities):
                distribution[(node, rt)] = prob
        
        return distribution
    
    def computeObservedJoint(self) -> Dict[FrozenSet[Tuple[Node, int]], float]:
        """
        Compute the observed joint distribution P(V) from the response type distribution.
        
        The computation follows:
            P(V = v) = Σ_{r: g(r)=v} P(R = r)
        
        Where:
        - r is a response-type configuration (one response type per node)
        - g(r) is the deterministic mapping from response types to observable values
        - v is an observable configuration
        
        Algorithm:
        1. Enumerate all response-type configurations
        2. For each configuration, compute its probability (product of individual probabilities, since independent)
        3. Simulate the deterministic values produced by each configuration
        4. Accumulate probabilities for each observable outcome
        
        Returns:
            Dictionary mapping frozenset of (Node, int) tuples to probability
        """
        all_nodes = sorted(self.dag.get_all_nodes(), key=lambda n: n.name)
        
        # Create the observed joint distribution as a dict
        observed_joint: Dict[FrozenSet[Tuple[Node, int]], float] = {}
        
        # Get topological order for simulation
        topo_order = self._topological_sort(all_nodes)
        
        # Enumerate all response-type configurations
        # For each node, get its response types
        node_response_types = []
        for node in topo_order:
            node_response_types.append((node, node.response_types))
        
        # Iterate over all combinations of response types (one per node)
        response_type_configs = product(*[rts for _, rts in node_response_types])
        
        for rt_config in response_type_configs:
            # rt_config is a tuple of ResponseType objects, one per node in topo_order
            
            # Compute the probability of this response-type configuration
            # P(R = r) = Π_i P(R_i = r_i) (assuming independence)
            rt_prob = 1.0
            for node, rt in zip(topo_order, rt_config):
                rt_prob *= self.trueResponseTypes[(node, rt)]
            
            # Skip if probability is negligible
            if rt_prob < 1e-15:
                continue
            
            # Simulate the deterministic values produced by this response-type configuration
            # V_i = g_i(Pa(V_i), R_i)
            simulated_values = self._simulate_values(topo_order, rt_config)
            
            # Convert simulated values to a configuration set
            config = frozenset((node, value) for node, value in simulated_values.items())
            
            # Accumulate probability for this observable configuration
            if config in observed_joint:
                observed_joint[config] += rt_prob
            else:
                observed_joint[config] = rt_prob
        
        return observed_joint
    
    def _simulate_values(
        self, 
        topo_order: list[Node], 
        rt_config: Tuple[ResponseType, ...]
    ) -> Dict[Node, int]:
        """
        Simulate the deterministic values produced by a response-type configuration.
        
        Given a fixed response type for each node, compute the observable values
        by evaluating structural equations in topological order:
            V_i = f_i(Pa(V_i), R_i) = g_i(r_i)[values of parents]
        
        Args:
            topo_order: Nodes in topological order
            rt_config: Tuple of ResponseType objects (one per node in topo_order)
        
        Returns:
            Dictionary mapping Node -> observed value
        """
        simulated_values = {}
        
        for node, rt in zip(topo_order, rt_config):
            # Get parent values
            parents = self.dag.get_parents(node)
            
            if not parents:
                # No parents - response type directly determines value
                # For a node with no parents, response type has a single mapping with key ()
                parent_config = ()
                value = rt.mapping[parent_config]
            else:
                # Get parent values from already-simulated values
                parent_values = []
                for parent in sorted(parents, key=lambda p: p.name):
                    if parent not in simulated_values:
                        raise ValueError(
                            f"Parent {parent.name} not yet simulated when processing {node.name}. "
                            "Topological order may be incorrect."
                        )
                    parent_values.append((parent, simulated_values[parent]))
                
                # Look up the value in the response type mapping
                # Mapping keys are tuples, not frozensets
                parent_config = tuple(sorted(parent_values, key=lambda x: x[0].name))
                
                if parent_config not in rt.mapping:
                    raise ValueError(
                        f"Parent configuration {parent_config} not found in response type "
                        f"mapping for node {node.name}"
                    )
                
                value = rt.mapping[parent_config]
            
            simulated_values[node] = value
        
        return simulated_values
    
    def _topological_sort(self, nodes: list[Node]) -> list[Node]:
        """
        Compute topological ordering of nodes.
        
        Args:
            nodes: List of nodes to sort
        
        Returns:
            List of nodes in topological order
        """
        from collections import deque
        
        node_set = set(nodes)
        
        # Compute in-degrees (counting only edges within the node set)
        in_degree = {node: 0 for node in nodes}
        for parent, child in self.dag.edges:
            if parent in node_set and child in node_set:
                in_degree[child] += 1
        
        # Start with nodes that have no parents
        queue = deque([node for node in nodes if in_degree[node] == 0])
        topo_order = []
        
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            
            # Reduce in-degree of children
            for parent, child in self.dag.edges:
                if parent == node and child in node_set:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        if len(topo_order) != len(nodes):
            raise ValueError("DAG contains a cycle")
        
        return topo_order
    
    def print_true_distribution(self) -> None:
        """
        Print the true distribution over response types.
        """
        print("\n" + "=" * 80)
        print("TRUE RESPONSE TYPE DISTRIBUTION")
        print("=" * 80)
        
        # Group by node
        nodes = sorted(set(node for node, _ in self.trueResponseTypes.keys()), 
                      key=lambda n: n.name)
        
        for node in nodes:
            print(f"\nNode: {node.name}")
            print("-" * 40)
            
            # Get all response types for this node
            rts_and_probs = [(rt, prob) for (n, rt), prob in self.trueResponseTypes.items() 
                            if n == node]
            
            # Sort by response type index
            rts_and_probs.sort(key=lambda x: x[0].index if hasattr(x[0], 'index') else 0)
            
            total_prob = 0.0
            for rt, prob in rts_and_probs:
                print(f"  P(R_{node.name} = {self._format_response_type(rt)}) = {prob:.6f}")
                total_prob += prob
            
            print(f"  Total: {total_prob:.6f}")
        
        print("\n" + "=" * 80)
    
    def _format_response_type(self, rt: ResponseType) -> str:
        """
        Format a response type for display.
        
        Args:
            rt: ResponseType to format
        
        Returns:
            String representation
        """
        mappings = sorted(rt.mapping.items(), key=lambda x: str(x[0]))
        
        if len(mappings) == 1 and not mappings[0][0]:
            # No parents - just show the value
            return f"{mappings[0][1]}"
        
        # Show mappings
        mapping_strs = []
        for parent_config, output in mappings:
            if not parent_config:
                mapping_strs.append(f"→{output}")
            else:
                parent_str = ",".join(f"{p.name}={v}" for p, v in sorted(parent_config, 
                                                                          key=lambda x: x[0].name))
                mapping_strs.append(f"({parent_str})→{output}")
        
        return "[" + ", ".join(mapping_strs) + "]"
