"""
Node class for representing nodes in a causal graph.
"""

from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .response_type import ResponseType


class Node:
    """
    Represents a node in the causal graph.
    
    Attributes:
        name: String identifier of the node.
        support: Set of natural numbers representing the domain/support of this node.
        response_types: List of associated response types for this node.
    """
    
    def __init__(self, name: str, support: Set[int] = None):
        """
        Initialize a Node.
        
        Args:
            name: The identifier of the node.
            support: Set of natural numbers in the node's domain. 
                     Defaults to {0, 1} for binary variables.
        """
        self.name: str = name
        self.support: Set[int] = support if support is not None else {0, 1}
        self.response_types: List['ResponseType'] = []
    
    def add_response_type(self, response_type: 'ResponseType') -> None:
        """
        Add a response type to this node.
        
        Args:
            response_type: The ResponseType object to add.
        """
        self.response_types.append(response_type)
    
    def __repr__(self) -> str:
        return f"Node(name={self.name})"
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.name == other.name
        return False
