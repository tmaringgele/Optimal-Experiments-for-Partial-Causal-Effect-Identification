"""
ResponseType class for representing response types associated with nodes.
"""

from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node


class ResponseType:
    """
    Represents a type of response associated with a node.
    
    In the context of causal graphs, response types enumerate the different
    ways a node can respond to its parent nodes' values. This is used in
    Algorithm 1 to create the system of linear equations relating joint
    probabilities p* and conditional probabilities p to decision variable q.
    
    A response type is essentially a function that maps each configuration of
    parent node values to an output value in the node's support.
    
    Attributes:
        node: The node that this response type belongs to.
        mapping: Dictionary mapping parent configurations to output values.
                 Keys are tuples of (Node, value) pairs representing parent assignments.
                 Values are integers from the node's support.
                 Example: {((Z, 0), (M, 1)): 0, ((Z, 0), (M, 0)): 1, ...}
    """
    
    def __init__(self, node: 'Node', 
                 mapping: Dict[Tuple[Tuple['Node', int], ...], int] = None):
        """
        Initialize a ResponseType.
        
        Args:
            node: The node that this response type belongs to.
            mapping: Dictionary mapping parent configurations to output values.
                     Keys are tuples of (Node, value) pairs.
                     Values are integers from the node's support.
                     If None, an empty mapping is created.
        """
        self.node: 'Node' = node
        self.mapping: Dict[Tuple[Tuple['Node', int], ...], int] = (
            mapping if mapping is not None else {}
        )
    
    def get(self, parent_config: Tuple[Tuple['Node', int], ...]) -> int:
        """
        Get the output value for a given parent configuration.
        
        Args:
            parent_config: Tuple of (Node, value) pairs representing parent assignments.
                          Example: ((Z, 0), (M, 1))
        
        Returns:
            The output value from the node's support.
        
        Raises:
            KeyError: If the parent configuration is not in the mapping.
        """
        return self.mapping[parent_config]
    
    def set(self, parent_config: Tuple[Tuple['Node', int], ...], output_value: int) -> None:
        """
        Set the output value for a given parent configuration.
        
        Args:
            parent_config: Tuple of (Node, value) pairs representing parent assignments.
            output_value: The output value (must be in node's support).
        
        Raises:
            ValueError: If output_value is not in the node's support.
        """
        if output_value not in self.node.support:
            raise ValueError(
                f"Output value {output_value} not in node {self.node.name}'s "
                f"support {self.node.support}"
            )
        self.mapping[parent_config] = output_value
    
    def get_parent_nodes(self) -> Tuple['Node', ...]:
        """
        Get the ordered tuple of parent nodes from the mapping.
        
        Returns:
            Tuple of parent nodes in the order they appear in mapping keys.
            Empty tuple if node has no parents.
        """
        if not self.mapping:
            return ()
        # Get first key and extract parent nodes
        first_key = next(iter(self.mapping.keys()))
        return tuple(node for node, value in first_key)
    
    def is_complete(self, num_parent_configs: int) -> bool:
        """
        Check if this response type has a mapping for all parent configurations.
        
        Args:
            num_parent_configs: Expected number of parent configurations.
        
        Returns:
            True if mapping is complete, False otherwise.
        """
        return len(self.mapping) == num_parent_configs
    
    def __repr__(self) -> str:
        return f"ResponseType(node={self.node.name}, mapping={self.mapping})"
    
    def __str__(self) -> str:
        if not self.mapping:
            return f"RT[{self.node.name}]: empty"
        
        # Check if node has no parents (empty tuple keys)
        first_key = next(iter(self.mapping.keys()))
        if len(first_key) == 0:
            # No parents - just show the constant value
            value = next(iter(self.mapping.values()))
            return f"RT[{self.node.name}]: {self.node.name} = {value}"
        
        # Has parents - show full mapping
        parent_nodes = tuple(node for node, val in first_key)
        parent_names = ", ".join(p.name for p in parent_nodes)
        
        mapping_strs = []
        for config, output in sorted(self.mapping.items(), 
                                     key=lambda x: tuple(v for _, v in x[0])):
            config_str = ", ".join(f"{node.name}={val}" for node, val in config)
            mapping_strs.append(f"({config_str}) -> {output}")
        
        return f"RT[{self.node.name}|{parent_names}]: " + "; ".join(mapping_strs)
