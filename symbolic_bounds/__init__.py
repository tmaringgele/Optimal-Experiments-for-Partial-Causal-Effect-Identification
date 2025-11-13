"""
Symbolic Bounds Package

This package implements algorithms for deriving tight symbolic bounds on causal effects,
based on the methods described in "A General Method for Deriving Tight Symbolic Bounds 
on Causal Effects".

The implementation provides:
- Algorithm 1: System of linear equations relating joint probabilities (p*) and 
  conditional probabilities (p) to decision variable q (response type enumeration)
- Algorithm 2: (To be implemented)

Classes:
    Node: Represents a node in the causal graph
    ResponseType: Represents response types associated with nodes
    DAG: Represents the directed acyclic graph structure
"""

from .node import Node
from .response_type import ResponseType
from .dag import DAG

__all__ = ['Node', 'ResponseType', 'DAG']
__version__ = '0.1.0'
