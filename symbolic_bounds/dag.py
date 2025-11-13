"""
DAG class for representing directed acyclic graphs in causal models.
"""

from typing import Set, Tuple, Dict, List, Optional
import itertools
from .node import Node
from .response_type import ResponseType


class DAG:
    """
    Represents a directed acyclic graph composed of nodes.
    
    This class implements the graph structure required for Algorithm 1,
    which creates a system of linear equations for causal effect bounds.
    The graph can be partitioned into W_L and W_R as described in
    Proposition 2 of the paper.
    
    Attributes:
        W_L: Set of nodes on the left side of the partition.
        W_R: Set of nodes on the right side of the partition.
        edges: Set of directed edges, each represented as a tuple (parent, child).
    """
    
    def __init__(self):
        """Initialize an empty DAG."""
        self.W_L: Set[Node] = set()
        self.W_R: Set[Node] = set()
        self.edges: Set[Tuple[Node, Node]] = set()
        self._nodes: Dict[str, Node] = {}
    
    def add_node(self, name: str, support: Optional[Set[int]] = None, 
                 partition: Optional[str] = None) -> Node:
        """
        Add a node to the DAG.
        
        Args:
            name: The name of the node to add.
            support: Set of natural numbers in the node's domain. 
                    Defaults to {0, 1} for binary variables.
            partition: Optional partition assignment ('L' or 'R').
        
        Returns:
            The created or existing Node object.
        """
        if name not in self._nodes:
            node = Node(name, support=support)
            self._nodes[name] = node
            
            if partition == 'L':
                self.W_L.add(node)
            elif partition == 'R':
                self.W_R.add(node)
        
        return self._nodes[name]
    
    def get_node(self, name: str) -> Optional[Node]:
        """
        Get a node by name.
        
        Args:
            name: The name of the node.
        
        Returns:
            The Node object if it exists, None otherwise.
        """
        return self._nodes.get(name)
    
    def add_edge(self, parent: Node, child: Node) -> None:
        """
        Add a directed edge from parent to child.
        
        Args:
            parent: The parent node.
            child: The child node.
        """
        self.edges.add((parent, child))
    
    def add_edge_by_name(self, parent_name: str, child_name: str) -> None:
        """
        Add a directed edge using node names.
        
        Args:
            parent_name: The name of the parent node.
            child_name: The name of the child node.
        """
        parent = self.add_node(parent_name)
        child = self.add_node(child_name)
        self.add_edge(parent, child)
    
    def get_parents(self, node: Node) -> Set[Node]:
        """
        Get all parent nodes of a given node.
        
        Args:
            node: The node whose parents to find.
        
        Returns:
            Set of parent nodes.
        """
        return {parent for parent, child in self.edges if child == node}
    
    def get_children(self, node: Node) -> Set[Node]:
        """
        Get all child nodes of a given node.
        
        Args:
            node: The node whose children to find.
        
        Returns:
            Set of child nodes.
        """
        return {child for parent, child in self.edges if parent == node}
    
    def get_all_nodes(self) -> Set[Node]:
        """
        Get all nodes in the DAG.
        
        Returns:
            Set of all nodes.
        """
        return set(self._nodes.values())
    
    def set_partition(self, node: Node, partition: str) -> None:
        """
        Assign a node to a partition (W_L or W_R).
        
        Args:
            node: The node to assign.
            partition: The partition ('L' or 'R').
        """
        # Remove from both partitions first
        self.W_L.discard(node)
        self.W_R.discard(node)
        
        # Add to the specified partition
        if partition == 'L':
            self.W_L.add(node)
        elif partition == 'R':
            self.W_R.add(node)
        else:
            raise ValueError(f"Invalid partition: {partition}. Must be 'L' or 'R'.")
    
    def is_valid_partition(self) -> bool:
        """
        Check if the current partition satisfies the criteria from Proposition 2.
        
        Returns:
            True if the partition is valid, False otherwise.
            
        Note:
            This is a placeholder. The actual validation logic should check
            the specific criteria mentioned in Proposition 2 of the paper.
        """
        # Check that W_L and W_R are disjoint
        if self.W_L.intersection(self.W_R):
            return False
        
        # Additional checks based on Proposition 2 would go here
        # For now, we just check basic consistency
        return True
    
    def enumerate_response_types(self, node: Node, store: bool = True) -> List[ResponseType]:
        """
        Enumerate all possible response types for a given node.
        
        For a node with k parents (each with their own support) and support of size m,
        this generates all possible response functions (mappings from parent 
        configurations to output values).
        
        Args:
            node: The node for which to enumerate response types.
            store: If True, stores the response types in node.response_types.
        
        Returns:
            List of all possible ResponseType objects for this node.
        """
        parents = sorted(self.get_parents(node), key=lambda n: n.name)
        
        # Generate all possible parent configurations
        if not parents:
            # Node has no parents - only one response type mapping () -> each value in support
            response_types = []
            for value in node.support:
                rt = ResponseType(node, mapping={(): value})
                response_types.append(rt)
            if store:
                node.response_types = response_types
            return response_types
        
        # Get all combinations of parent values
        parent_supports = [parent.support for parent in parents]
        parent_value_configs = list(itertools.product(*parent_supports))
        
        # Convert to (Node, value) tuple format
        parent_configs = []
        for value_config in parent_value_configs:
            config = tuple((parent, value) for parent, value in zip(parents, value_config))
            parent_configs.append(config)
        
        # Generate all possible mappings from parent configs to node's support
        # This is support^(# of parent configs)
        num_configs = len(parent_configs)
        all_outputs = itertools.product(node.support, repeat=num_configs)
        
        response_types = []
        for output_assignment in all_outputs:
            mapping = dict(zip(parent_configs, output_assignment))
            rt = ResponseType(node, mapping=mapping)
            response_types.append(rt)
        
        if store:
            node.response_types = response_types
        
        return response_types
    
    def generate_all_response_types(self) -> Dict[Node, List[ResponseType]]:
        """
        Generate all response types for all nodes in the DAG.
        
        Returns:
            Dictionary mapping each node to its list of response types.
        """
        all_response_types = {}
        for node in self.get_all_nodes():
            response_types = self.enumerate_response_types(node)
            node.response_types = response_types
            all_response_types[node] = response_types
        
        return all_response_types
    
    def print_response_type_table(self, node: Node) -> None:
        """
        Print a formatted table of response types for a given node.
        Similar to the style in Section 6.1 of the paper.
        
        Args:
            node: The node for which to print the response type table.
        """
        if not node.response_types:
            print(f"\nNo response types enumerated for {node.name}")
            print("Call enumerate_response_types() first.")
            return
        
        # Get parent nodes from the first response type
        first_rt = node.response_types[0]
        parent_nodes = first_rt.get_parent_nodes()
        
        if not parent_nodes:
            print(f"\nResponse types for {node.name} (no parents):")
            print("=" * 50)
            for i, rt in enumerate(node.response_types, 1):
                value = rt.get(())
                print(f"r_{node.name}^{i}: {node.name} = {value}")
            print()
            return
        
        # Get all parent configurations from the first response type
        parent_configs = sorted(first_rt.mapping.keys(), 
                               key=lambda x: tuple(v for _, v in x))
        
        # Build header with actual parent names
        parent_names = [p.name for p in parent_nodes]
        header = " | ".join(parent_names) + f" | {node.name}"
        separator = "-" * len(header)
        
        print(f"\nResponse types for {node.name} (parents: {', '.join(parent_names)}):")
        print("=" * len(header))
        
        # Print each response type
        for i, rt in enumerate(node.response_types, 1):
            print(f"\nr_{node.name}^{i}:")
            print(header)
            print(separator)
            
            for config in parent_configs:
                output = rt.get(config)
                # Extract just the values from the (Node, value) tuples
                config_str = " | ".join(str(val) for _, val in config)
                print(f"{config_str} | {output}")
        
        print()
    
    def get_response_type_table_string(self, node: Node) -> str:
        """
        Generate a formatted string table of response types for a given node.
        
        Args:
            node: The node for which to generate the response type table.
        
        Returns:
            Formatted string representation of the response type table.
        """
        if not node.response_types:
            return f"\nNo response types enumerated for {node.name}\n"
        
        lines = []
        
        # Get parent nodes from the first response type
        first_rt = node.response_types[0]
        parent_nodes = first_rt.get_parent_nodes()
        
        if not parent_nodes:
            lines.append(f"\nResponse types for {node.name} (no parents):")
            lines.append("=" * 50)
            for i, rt in enumerate(node.response_types, 1):
                value = rt.get(())
                lines.append(f"r_{node.name}^{i}: {node.name} = {value}")
            lines.append("")
            return "\n".join(lines)
        
        # Get all parent configurations from the first response type
        parent_configs = sorted(first_rt.mapping.keys(),
                               key=lambda x: tuple(v for _, v in x))
        
        # Build header
        parent_names = [p.name for p in parent_nodes]
        header = " | ".join(parent_names) + f" | {node.name}"
        separator = "-" * len(header)
        
        lines.append(f"\nResponse types for {node.name} (parents: {', '.join(parent_names)}):")
        lines.append("=" * len(header))
        
        # Generate each response type
        for i, rt in enumerate(node.response_types, 1):
            lines.append(f"\nr_{node.name}^{i}:")
            lines.append(header)
            lines.append(separator)
            
            for config in parent_configs:
                output = rt.get(config)
                # Extract just the values from the (Node, value) tuples
                config_str = " | ".join(str(val) for _, val in config)
                lines.append(f"{config_str} | {output}")
        
        lines.append("")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        nodes_str = ", ".join(node.name for node in self.get_all_nodes())
        edges_str = ", ".join(f"{p.name}->{c.name}" for p, c in self.edges)
        return f"DAG(nodes=[{nodes_str}], edges=[{edges_str}])"
