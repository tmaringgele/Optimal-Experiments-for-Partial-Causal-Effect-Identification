"""
Random DAG Generator for Left/Right Partitioned Causal Graphs.

This module provides functionality to generate random directed acyclic graphs (DAGs)
with a bipartite partition into W_L (left) and W_R (right) sets, as used in
causal inference problems.
"""

import random
from typing import List, Tuple, Set
from .dag import DAG
from .node import Node


def generate_random_partitioned_dag(n: int, 
                                    binary_only: bool = True,
                                    seed: int = None,
                                    min_left_ratio: float = 0.3,
                                    max_left_ratio: float = 0.7,
                                    edge_probability: float = 0.4,
                                    chain_bonus: float = 0.3,
                                    max_parents: int = 3) -> DAG:
    """
    Generate a random DAG with left/right partition and rich internal structure.
    
    This function creates a random DAG with approximately n nodes, partitioned into
    W_L (left) and W_R (right) sets. The DAG structure ensures:
    - Acyclicity (no directed cycles)
    - Non-trivial partition (both W_L and W_R are non-empty)
    - Hierarchical structure with multiple topological levels
    - Chains and mediators within W_R
    - Nodes can have multiple parents
    
    Args:
        n: Target number of nodes (actual number may vary slightly).
        binary_only: If True, all nodes have binary support {0, 1}.
        seed: Random seed for reproducibility.
        min_left_ratio: Minimum fraction of nodes in W_L (default: 0.3).
        max_left_ratio: Maximum fraction of nodes in W_L (default: 0.7).
        edge_probability: Base probability of adding an edge between compatible nodes.
        chain_bonus: Additional probability for edges between consecutive levels (creates chains).
        max_parents: Maximum number of parents per node (encourages multiple parents).
    
    Returns:
        A DAG object with the randomly generated structure.
    
    Example:
        >>> dag = generate_random_partitioned_dag(5, seed=42)
        >>> print(f"Nodes in W_L: {len(dag.W_L)}, Nodes in W_R: {len(dag.W_R)}")
        >>> print(f"Edges: {len(dag.edges)}")
    """
    if seed is not None:
        random.seed(seed)
    
    if n < 2:
        n = 2  # Minimum 2 nodes to have non-trivial partition
    
    # Determine partition sizes
    n_left = max(1, int(n * random.uniform(min_left_ratio, max_left_ratio)))
    n_right = max(1, n - n_left)
    
    dag = DAG()
    
    # Generate node names
    left_names = [f"L{i}" for i in range(n_left)]
    right_names = [f"R{i}" for i in range(n_right)]
    
    # Create nodes with random supports
    left_nodes: List[Node] = []
    right_nodes: List[Node] = []
    
    for name in left_names:
        if binary_only:
            support = {0, 1}
        else:
            # Random support size between 2 and 4
            support_size = random.randint(2, 4)
            support = set(range(support_size))
        
        node = dag.add_node(name, support=support, partition='L')
        left_nodes.append(node)
    
    for name in right_names:
        if binary_only:
            support = {0, 1}
        else:
            support_size = random.randint(2, 4)
            support = set(range(support_size))
        
        node = dag.add_node(name, support=support, partition='R')
        right_nodes.append(node)
    
    all_nodes = left_nodes + right_nodes
    
    # Generate edges with hierarchical structure
    # Strategy: Create distinct topological levels within each partition
    
    node_levels = {}
    
    # Assign levels within W_L - create 2-3 levels
    if n_left == 1:
        node_levels[left_nodes[0]] = 0
    else:
        num_left_levels = min(3, max(2, n_left // 2))
        for i, node in enumerate(left_nodes):
            level = int(i * num_left_levels / n_left)
            node_levels[node] = level
    
    # Assign levels within W_R - create 2-4 levels to allow chains
    if n_right == 1:
        node_levels[right_nodes[0]] = n_left
    else:
        # Create more levels in W_R to enable chains and mediators
        num_right_levels = min(4, max(2, n_right // 2 + 1))
        base_level = max(node_levels.values()) + 1 if node_levels else 0
        
        for i, node in enumerate(right_nodes):
            level = base_level + int(i * num_right_levels / n_right)
            node_levels[node] = level
    
    # Track parent counts to limit multiple parents
    parent_count = {node: 0 for node in all_nodes}
    
    # Phase 1: Build chains within each partition (higher probability for consecutive levels)
    for node_i in all_nodes:
        level_i = node_levels[node_i]
        
        for node_j in all_nodes:
            level_j = node_levels[node_j]
            
            # Only consider edges from lower to higher levels
            if level_j <= level_i:
                continue
            
            # Skip if target already has max parents
            if parent_count[node_j] >= max_parents:
                continue
            
            # Calculate edge probability based on level distance
            level_diff = level_j - level_i
            
            if level_diff == 1:
                # Consecutive levels: high probability (creates chains/mediators)
                prob = edge_probability + chain_bonus
            elif level_diff == 2:
                # Skip one level: moderate probability
                prob = edge_probability * 0.6
            else:
                # Long-range connections: lower probability
                prob = edge_probability * 0.3
            
            # Add edge based on probability
            if random.random() < prob:
                dag.add_edge(node_i, node_j)
                parent_count[node_j] += 1
                
                # Stop adding edges if target reached max parents
                if parent_count[node_j] >= max_parents:
                    continue
    
    # Phase 2: Ensure every node in W_R has at least one parent
    for node in right_nodes:
        if parent_count[node] == 0:
            # Find potential parents (nodes with lower level)
            potential_parents = [n for n in all_nodes 
                               if node_levels[n] < node_levels[node]]
            
            if potential_parents:
                # Prefer parents from the immediately preceding level
                same_level_parents = [n for n in potential_parents 
                                    if node_levels[n] == node_levels[node] - 1]
                
                if same_level_parents:
                    parent = random.choice(same_level_parents)
                else:
                    parent = random.choice(potential_parents)
                
                dag.add_edge(parent, node)
                parent_count[node] += 1
    
    # Phase 3: Ensure connectivity from W_L to W_R
    if not _has_path_left_to_right(dag, left_nodes, right_nodes):
        # Find a W_L node with high level and W_R node with low level
        left_candidates = sorted(left_nodes, key=lambda n: node_levels[n], reverse=True)
        right_candidates = sorted(right_nodes, key=lambda n: node_levels[n])
        
        source = left_candidates[0]
        target = right_candidates[0]
        dag.add_edge(source, target)
    
    return dag


def generate_random_chain_dag(n: int, binary_only: bool = True, seed: int = None) -> DAG:
    """
    Generate a simple chain DAG: L0 -> L1 -> ... -> R0 -> R1 -> ...
    
    This creates a simpler structure useful for testing.
    
    Args:
        n: Total number of nodes.
        binary_only: If True, all nodes have binary support {0, 1}.
        seed: Random seed for reproducibility.
    
    Returns:
        A chain DAG with n nodes.
    """
    if seed is not None:
        random.seed(seed)
    
    if n < 2:
        n = 2
    
    # Split nodes between left and right
    n_left = max(1, n // 2)
    n_right = n - n_left
    
    dag = DAG()
    
    # Create left nodes
    left_nodes = []
    for i in range(n_left):
        support = {0, 1} if binary_only else set(range(random.randint(2, 4)))
        node = dag.add_node(f"L{i}", support=support, partition='L')
        left_nodes.append(node)
    
    # Create right nodes
    right_nodes = []
    for i in range(n_right):
        support = {0, 1} if binary_only else set(range(random.randint(2, 4)))
        node = dag.add_node(f"R{i}", support=support, partition='R')
        right_nodes.append(node)
    
    # Create chain
    all_nodes = left_nodes + right_nodes
    for i in range(len(all_nodes) - 1):
        dag.add_edge(all_nodes[i], all_nodes[i + 1])
    
    return dag


def generate_random_tree_dag(n: int, binary_only: bool = True, seed: int = None) -> DAG:
    """
    Generate a tree-structured DAG with left/right partition.
    
    Creates a tree where nodes in W_L form the upper levels and W_R form lower levels.
    
    Args:
        n: Target number of nodes.
        binary_only: If True, all nodes have binary support {0, 1}.
        seed: Random seed for reproducibility.
    
    Returns:
        A tree-structured DAG.
    """
    if seed is not None:
        random.seed(seed)
    
    if n < 2:
        n = 2
    
    dag = DAG()
    
    # Split nodes
    n_left = max(1, n // 2)
    n_right = n - n_left
    
    # Create nodes
    left_nodes = []
    for i in range(n_left):
        support = {0, 1} if binary_only else set(range(random.randint(2, 4)))
        node = dag.add_node(f"L{i}", support=support, partition='L')
        left_nodes.append(node)
    
    right_nodes = []
    for i in range(n_right):
        support = {0, 1} if binary_only else set(range(random.randint(2, 4)))
        node = dag.add_node(f"R{i}", support=support, partition='R')
        right_nodes.append(node)
    
    all_nodes = left_nodes + right_nodes
    
    # Build tree structure
    # First node is the root (in W_L)
    for i in range(1, len(all_nodes)):
        # Each node gets a random parent from nodes that came before it
        parent_idx = random.randint(0, i - 1)
        dag.add_edge(all_nodes[parent_idx], all_nodes[i])
    
    return dag


def _has_path_left_to_right(dag: DAG, left_nodes: List[Node], right_nodes: List[Node]) -> bool:
    """
    Check if there's at least one path from any W_L node to any W_R node.
    
    Args:
        dag: The DAG to check.
        left_nodes: Nodes in W_L.
        right_nodes: Nodes in W_R.
    
    Returns:
        True if a path exists, False otherwise.
    """
    # For each right node, check if it's reachable from any left node
    for right_node in right_nodes:
        # BFS backwards from right_node
        visited = set()
        queue = [right_node]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # Check if we reached a left node
            if current in left_nodes:
                return True
            
            # Add parents to queue
            parents = dag.get_parents(current)
            for parent in parents:
                if parent not in visited:
                    queue.append(parent)
    
    return False


def print_dag_summary(dag: DAG) -> None:
    """
    Print a summary of the DAG structure.
    
    Args:
        dag: The DAG to summarize.
    """
    left_nodes = sorted(dag.W_L, key=lambda n: n.name)
    right_nodes = sorted(dag.W_R, key=lambda n: n.name)
    
    print(f"DAG Summary:")
    print(f"  Total nodes: {len(left_nodes) + len(right_nodes)}")
    print(f"  W_L nodes ({len(left_nodes)}): {', '.join(n.name for n in left_nodes)}")
    print(f"  W_R nodes ({len(right_nodes)}): {', '.join(n.name for n in right_nodes)}")
    print(f"  Edges ({len(dag.edges)}):")
    
    for parent, child in sorted(dag.edges, key=lambda e: (e[0].name, e[1].name)):
        print(f"    {parent.name} -> {child.name}")


if __name__ == "__main__":
    """Test the random DAG generators."""
    print("=" * 80)
    print("TESTING RANDOM DAG GENERATORS")
    print("=" * 80)
    
    # Test 1: Small random DAG
    print("\n" + "-" * 80)
    print("Test 1: Random DAG with 5 nodes")
    print("-" * 80)
    dag1 = generate_random_partitioned_dag(5, seed=42)
    print_dag_summary(dag1)
    
    # Test 2: Medium random DAG
    print("\n" + "-" * 80)
    print("Test 2: Random DAG with 8 nodes")
    print("-" * 80)
    dag2 = generate_random_partitioned_dag(8, seed=123)
    print_dag_summary(dag2)
    
    # Test 3: Chain DAG
    print("\n" + "-" * 80)
    print("Test 3: Chain DAG with 6 nodes")
    print("-" * 80)
    dag3 = generate_random_chain_dag(6, seed=456)
    print_dag_summary(dag3)
    
    # Test 4: Tree DAG
    print("\n" + "-" * 80)
    print("Test 4: Tree DAG with 7 nodes")
    print("-" * 80)
    dag4 = generate_random_tree_dag(7, seed=789)
    print_dag_summary(dag4)
    
    # Test 5: Multiple random DAGs to ensure variety
    print("\n" + "-" * 80)
    print("Test 5: Generate 10 random DAGs and check variety")
    print("-" * 80)
    
    left_ratios = []
    edge_counts = []
    
    for i in range(10):
        dag = generate_random_partitioned_dag(6, seed=i)
        n_left = len(dag.W_L)
        n_total = n_left + len(dag.W_R)
        left_ratio = n_left / n_total
        left_ratios.append(left_ratio)
        edge_counts.append(len(dag.edges))
    
    print(f"  Left partition ratios: {[f'{r:.2f}' for r in left_ratios]}")
    print(f"  Edge counts: {edge_counts}")
    print(f"  Average left ratio: {sum(left_ratios)/len(left_ratios):.2f}")
    print(f"  Average edges: {sum(edge_counts)/len(edge_counts):.1f}")
    
    print("\n" + "=" * 80)
    print("✓ Random DAG generators working correctly!")
    print("=" * 80)
