"""
JointDistribution class for representing observed probability distributions.

A JointDistribution maps configurations of nodes to probabilities.
Each configuration is a tuple of (Node, value) pairs.
"""

from typing import Dict, Tuple, Set, FrozenSet
from .node import Node
from itertools import product


class JointDistribution:
    """
    Represents a joint probability distribution over nodes.
    
    The distribution is stored as a mapping from frozenset of (Node, int) tuples
    to probability values. For example:
        frozenset({(Node('X'), 0), (Node('Y'), 1)}) -> 0.3
    represents P(X=0, Y=1) = 0.3
    
    Attributes:
        probabilities: Dict mapping frozenset of (Node, value) tuples to probability
    """
    
    def __init__(self):
        """Initialize an empty joint distribution."""
        self.probabilities: Dict[FrozenSet[Tuple[Node, int]], float] = {}
    
    def set_probability(self, configuration: Set[Tuple[Node, int]], prob: float) -> None:
        """
        Set the probability for a specific configuration.
        
        Args:
            configuration: Set of (Node, value) tuples representing the configuration
            prob: Probability value (must be between 0 and 1)
        
        Raises:
            ValueError: If probability is not in [0, 1]
        """
        if not 0 <= prob <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {prob}")
        
        # Convert to frozenset for hashability
        key = frozenset(configuration)
        self.probabilities[key] = prob
    
    def get_probability(self, configuration: Set[Tuple[Node, int]]) -> float:
        """
        Get the probability for a specific configuration.
        
        Args:
            configuration: Set of (Node, value) tuples
            
        Returns:
            Probability value, or 0.0 if configuration not found
        """
        key = frozenset(configuration)
        return self.probabilities.get(key, 0.0)
    
    def validate(self, nodes: Set[Node]) -> bool:
        """
        Validate that the distribution covers all possible configurations.
        
        Args:
            nodes: Set of nodes that should be covered
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValueError: If distribution is invalid with explanation
        """
        # Check that probabilities sum to 1
        total_prob = sum(self.probabilities.values())
        if not abs(total_prob - 1.0) < 1e-10:
            raise ValueError(f"Probabilities must sum to 1, got {total_prob}")
        
        # Generate all possible configurations
        node_list = sorted(nodes, key=lambda n: n.name)
        supports = [sorted(node.support) for node in node_list]
        expected_configs = []
        
        for values in product(*supports):
            config = frozenset((node, val) for node, val in zip(node_list, values))
            expected_configs.append(config)
        
        # Check that all configurations are present
        expected_set = set(expected_configs)
        actual_set = set(self.probabilities.keys())
        
        if expected_set != actual_set:
            missing = expected_set - actual_set
            extra = actual_set - expected_set
            
            error_msg = []
            if missing:
                error_msg.append(f"Missing configurations: {len(missing)}")
                # Show first few missing
                for i, config in enumerate(list(missing)[:3]):
                    config_str = ", ".join(f"{n.name}={v}" for n, v in sorted(config, key=lambda x: x[0].name))
                    error_msg.append(f"  - {config_str}")
                if len(missing) > 3:
                    error_msg.append(f"  ... and {len(missing) - 3} more")
            
            if extra:
                error_msg.append(f"Extra configurations: {len(extra)}")
            
            raise ValueError("\n".join(error_msg))
        
        return True
    
    def get_marginal(self, nodes_subset: Set[Node]) -> 'JointDistribution':
        """
        Compute the marginal distribution over a subset of nodes.
        
        Args:
            nodes_subset: Nodes to marginalize over
            
        Returns:
            New JointDistribution representing the marginal
        """
        marginal = JointDistribution()
        marginal_probs: Dict[FrozenSet[Tuple[Node, int]], float] = {}
        
        for config, prob in self.probabilities.items():
            # Extract only the nodes in the subset
            marginal_config = frozenset((n, v) for n, v in config if n in nodes_subset)
            
            if marginal_config in marginal_probs:
                marginal_probs[marginal_config] += prob
            else:
                marginal_probs[marginal_config] = prob
        
        marginal.probabilities = marginal_probs
        return marginal
    
    def print_distribution(self, title: str = "Joint Distribution") -> None:
        """
        Print the distribution in a readable format.
        
        Args:
            title: Title to display
        """
        print(f"\n{title}")
        print("=" * 80)
        
        if not self.probabilities:
            print("(empty distribution)")
            return
        
        # Sort configurations for consistent display
        sorted_configs = sorted(self.probabilities.items(), 
                               key=lambda x: sorted((n.name, v) for n, v in x[0]))
        
        for config, prob in sorted_configs:
            config_str = ", ".join(f"{n.name}={v}" for n, v in sorted(config, key=lambda x: x[0].name))
            print(f"  P({config_str}) = {prob:.6f}")
        
        total = sum(self.probabilities.values())
        print(f"\nTotal probability: {total:.6f}")
