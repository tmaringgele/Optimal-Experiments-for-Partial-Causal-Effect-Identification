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
                 partition: str = 'R') -> Node:
        """
        Add a node to the DAG.
        
        Args:
            name: The name of the node to add.
            support: Set of natural numbers in the node's domain. 
                    Defaults to {0, 1} for binary variables.
            partition: Partition assignment ('L' or 'R'). Required.
                      Defaults to 'R' if not specified.
        
        Returns:
            The created or existing Node object.
            
        Raises:
            ValueError: If partition is not 'L' or 'R'.
        """
        if partition not in ('L', 'R'):
            raise ValueError(f"Partition must be 'L' or 'R', got '{partition}'")
        
        if name not in self._nodes:
            node = Node(name, support=support)
            self._nodes[name] = node
            
            if partition == 'L':
                self.W_L.add(node)
            else:  # partition == 'R'
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
    
    def add_edge_by_name(self, parent_name: str, child_name: str, 
                         parent_partition: str = 'R', child_partition: str = 'R') -> None:
        """
        Add a directed edge using node names.
        
        Args:
            parent_name: The name of the parent node.
            child_name: The name of the child node.
            parent_partition: Partition for parent node if it doesn't exist ('L' or 'R').
            child_partition: Partition for child node if it doesn't exist ('L' or 'R').
        """
        parent = self.add_node(parent_name, partition=parent_partition)
        child = self.add_node(child_name, partition=child_partition)
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
    
    @property
    def nodes(self) -> Dict[str, Node]:
        """
        Get the dictionary of nodes (name -> Node).
        
        Returns:
            Dictionary mapping node names to Node objects.
        """
        return self._nodes
    
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
    
    def generate_all_response_types(self, R_only: bool = False) -> Dict[Node, List[ResponseType]]:
        """
        Generate all response types for all nodes in the DAG.
        
        Returns:
            Dictionary mapping each node to its list of response types.
        """
        all_response_types = {}
        nodes_to_process = self.W_R if R_only else self.get_all_nodes()
        for node in nodes_to_process:
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
    
    def draw(self, figsize=(10, 6), node_size=2000, font_size=12, 
             with_labels=True, with_legend=False, title=None):
        """
        Draw the DAG using NetworkX and matplotlib.
        
        Args:
            figsize: Figure size as (width, height) tuple.
            node_size: Size of the nodes in the visualization.
            font_size: Font size for node labels.
            with_labels: Whether to show node labels.
            title: Optional title for the plot.
        
        Returns:
            matplotlib Figure object.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "NetworkX and matplotlib are required for drawing. "
                "Install with: pip install networkx matplotlib"
            )
        
        # Create NetworkX directed graph
        G = nx.DiGraph()
        
        # Add nodes with partition information
        for node in self.get_all_nodes():
            partition = 'L' if node in self.W_L else 'R'
            G.add_node(node.name, partition=partition)
        
        # Add edges
        for parent, child in self.edges:
            G.add_edge(parent.name, child.name)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Separate nodes by partition for coloring
        w_l_nodes = [node.name for node in self.W_L]
        w_r_nodes = [node.name for node in self.W_R]
        
        # Use hierarchical layout (left to right)
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Try to use a more structured layout if possible
        try:
            # Separate into levels based on topological sort
            pos = {}
            levels = {}
            for node in nx.topological_sort(G):
                # Calculate level as max parent level + 1
                parent_levels = [levels.get(p, -1) for p in G.predecessors(node)]
                levels[node] = max(parent_levels) + 1 if parent_levels else 0
            
            # Group nodes by level
            level_groups = {}
            for node, level in levels.items():
                level_groups.setdefault(level, []).append(node)
            
            # Assign positions: x is level, y is spread within level
            for level, nodes in level_groups.items():
                for i, node in enumerate(sorted(nodes)):
                    y = (i - (len(nodes) - 1) / 2) * 0.5
                    pos[node] = (level, y)
        except:
            # Fall back to spring layout if topological layout fails
            pass
        
        # Draw W_L nodes (left partition) in blue
        nx.draw_networkx_nodes(G, pos, nodelist=w_l_nodes,
                              node_color='lightblue', node_size=node_size,
                              ax=ax, label='W_L')
        
        # Draw W_R nodes (right partition) in orange
        nx.draw_networkx_nodes(G, pos, nodelist=w_r_nodes,
                              node_color='lightcoral', node_size=node_size,
                              ax=ax, label='W_R')
        
        # Draw edges with clear arrowheads
        # Use FancyArrowPatch for better arrow rendering
        nx.draw_networkx_edges(G, pos, ax=ax, 
                              edge_color='gray',
                              arrows=True, 
                              arrowsize=25,
                              arrowstyle='-|>',
                              node_size=node_size,
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=15,
                              min_target_margin=15,
                              width=2)
        
        # Draw labels
        if with_labels:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                                   font_weight='bold')
        
        # Center the graph vertically by adjusting axis limits
        if pos:
            y_values = [y for x, y in pos.values()]
            y_min, y_max = min(y_values), max(y_values)
            y_range = y_max - y_min
            y_padding = max(0.5, y_range * 0.3)  # Add 30% padding or at least 0.5
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            x_values = [x for x, y in pos.values()]
            x_min, x_max = min(x_values), max(x_values)
            x_range = x_max - x_min
            x_padding = max(0.5, x_range * 0.2)  # Add 20% padding or at least 0.5
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
        
        # Add legend and title
        if with_legend:
            ax.legend(loc='upper left', fontsize=font_size-2)
        if title:
            ax.set_title(title, fontsize=font_size+2, fontweight='bold')
        else:
            ax.set_title(f"DAG with {len(self.W_L)} nodes in W_L, {len(self.W_R)} nodes in W_R",
                        fontsize=font_size+2, fontweight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def __repr__(self) -> str:
        nodes_str = ", ".join(node.name for node in self.get_all_nodes())
        edges_str = ", ".join(f"{p.name}->{c.name}" for p, c in self.edges)
        return f"DAG(nodes=[{nodes_str}], edges=[{edges_str}])"
    
    def get_autobound_info(self) -> dict:
        """
        Generate parameters for autobound package integration.
        
        This method extracts DAG information and formats it for use with the
        autobound package. It automatically adds unobserved confounders U_L and U_R
        based on the W_L/W_R partition.
        
        Returns:
            dict: Dictionary with keys:
                - 'dag_structure': Comma-separated edge list (e.g., "X -> Y, U_R -> X, U_R -> Y")
                - 'node_domains': Dict mapping node names to their support sizes
                - 'unobserved_nodes': Comma-separated list of unobserved nodes (e.g., "U_L,U_R")
        
        Examples:
            >>> dag = DAG()
            >>> # ... add nodes and edges ...
            >>> info = dag.get_autobound_info()
            >>> print(info['dag_structure'])
            'X -> Y, U_R -> X, U_R -> Y'
        """
        # Build node domains dictionary (only observed nodes)
        node_domains = {}
        for node in self.get_all_nodes():
            node_domains[node.name] = len(node.support)
        
        # Build edge list
        edge_list = []
        
        # Add U_L as parent of all W_L nodes if W_L is non-empty
        # Note: U_L is NOT added to node_domains (unobserved nodes excluded)
        if self.W_L:
            for node in self.W_L:
                edge_list.append(f"U_L -> {node.name}")
        
        # Add U_R as parent of all W_R nodes if W_R is non-empty
        # Note: U_R is NOT added to node_domains (unobserved nodes excluded)
        if self.W_R:
            for node in self.W_R:
                edge_list.append(f"U_R -> {node.name}")
        
        # Add all edges from the DAG
        for parent, child in sorted(self.edges, key=lambda e: (e[0].name, e[1].name)):
            edge_list.append(f"{parent.name} -> {child.name}")
        
        # Format as comma-separated string
        dag_structure = ", ".join(edge_list)
        
        # Build unobserved nodes list
        unobserved = []
        if self.W_L:
            unobserved.append("U_L")
        if self.W_R:
            unobserved.append("U_R")
        unobserved_nodes = ",".join(unobserved)
        
        return {
            'dag_structure': dag_structure,
            'node_domains': node_domains,
            'unobserved_nodes': unobserved_nodes
        }
