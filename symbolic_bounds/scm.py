"""
Structural Causal Model (SCM) class.

An SCM combines a DAG (representing the causal structure) with an
observed joint probability distribution.
"""

from .dag import DAG
from .joint_distribution import JointDistribution
from typing import Set
from .node import Node


class SCM:
    """
    Structural Causal Model combining causal structure and observations.
    
    An SCM consists of:
    - A DAG representing the causal structure
    - An observed joint distribution over all nodes in the DAG
    
    The observed joint distribution must be valid (cover all configurations
    and sum to 1).
    
    Attributes:
        dag: The causal DAG
        observedJoint: The observed joint probability distribution
    """
    
    def __init__(self, dag: DAG, observedJoint: JointDistribution):
        """
        Create an SCM with a DAG and observed joint distribution.
        
        Args:
            dag: Causal DAG
            observedJoint: Observed joint probability distribution
            
        Raises:
            ValueError: If the joint distribution is not valid for the DAG
        """
        self.dag = dag
        self.observedJoint = observedJoint
        
        # Validate that the joint distribution covers all nodes in the DAG
        all_nodes = dag.get_all_nodes()
        self.observedJoint.validate(all_nodes)
    
    def print_scm(self) -> None:
        """Print a summary of the SCM."""
        print("\n" + "=" * 80)
        print("STRUCTURAL CAUSAL MODEL")
        print("=" * 80)
        
        print("\nDAG Structure:")
        print(f"  Nodes: {', '.join(sorted(self.dag.nodes.keys()))}")
        print(f"  W_L (left partition): {', '.join(n.name for n in sorted(self.dag.W_L, key=lambda x: x.name))}")
        print(f"  W_R (right partition): {', '.join(n.name for n in sorted(self.dag.W_R, key=lambda x: x.name))}")
        
        print(f"\n  Edges:")
        for parent_name, child_name in sorted(self.dag.edges):
            print(f"    {parent_name} -> {child_name}")
        
        self.observedJoint.print_distribution("Observed Joint Distribution")
