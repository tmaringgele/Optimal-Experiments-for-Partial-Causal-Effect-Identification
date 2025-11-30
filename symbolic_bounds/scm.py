"""
Structural Causal Model (SCM) class.

An SCM combines a DAG (representing the causal structure) with a
DataGenerator for producing observed distributions.
"""

from .dag import DAG
from typing import Dict, FrozenSet, Tuple, TYPE_CHECKING
from .node import Node

if TYPE_CHECKING:
    from .data_generator import DataGenerator


class SCM:
    """
    Structural Causal Model combining causal structure and data generation.
    
    An SCM consists of:
    - A DAG representing the causal structure
    - A DataGenerator that produces causally consistent distributions
    
    Attributes:
        dag: The causal DAG
        data_generator: The DataGenerator for producing observed distributions
    """
    
    def __init__(self, dag: DAG, data_generator: 'DataGenerator'):
        """
        Create an SCM with a DAG and DataGenerator.
        
        Args:
            dag: Causal DAG
            data_generator: DataGenerator for this DAG
            
        Raises:
            ValueError: If the DataGenerator's DAG doesn't match
        """
        if data_generator.dag is not dag:
            raise ValueError("DataGenerator must be initialized with the same DAG")
        
        self.dag = dag
        self.data_generator = data_generator
    
    def getObservedJoint(self) -> Dict[FrozenSet[Tuple[Node, int]], float]:
        """
        Get the observed joint distribution by invoking the DataGenerator.
        
        Returns:
            Dictionary mapping frozenset of (Node, int) tuples to probability
        """
        return self.data_generator.computeObservedJoint()
    
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
        for parent, child in sorted(self.dag.edges, key=lambda e: (e[0].name, e[1].name)):
            print(f"    {parent.name} -> {child.name}")
        
        print("\n  Data Generation:")
        print(f"    Using DataGenerator with sampled response type distribution")
        
        # Print observed joint
        observed = self.getObservedJoint()
        print("\n  Observed Joint Distribution:")
        for config, prob in sorted(observed.items(), key=lambda x: sorted((n.name, v) for n, v in x[0])):
            config_str = ", ".join(f"{n.name}={v}" for n, v in sorted(config, key=lambda x: x[0].name))
            print(f"    P({config_str}) = {prob:.6f}")
