"""
Symbolic Bounds Package

This package implements algorithms for deriving tight symbolic bounds on causal effects,
based on the methods described in "A General Method for Deriving Tight Symbolic Bounds 
on Causal Effects".

The implementation provides:
- Algorithm 1: System of linear equations relating joint probabilities (p*) and 
  conditional probabilities (p) to decision variable q (response type enumeration)
- Algorithm 2: Objective function construction for interventional queries
- Linear Programming: Symbolic LP formulation for vertex enumeration

Classes:
    Node: Represents a node in the causal graph
    ResponseType: Represents response types associated with nodes
    DAG: Represents the directed acyclic graph structure
    Constraints: Holds the constraint system (matrices P, Lambda, etc.)
    ProgramFactory: Factory for generating constraints from DAGs (implements Algorithms 1 & 2)
    LinearProgram: Represents LP with symbolic parameters for vertex enumeration
    SymbolicParameter: Represents a symbolic parameter (e.g., P(W_R | W_L))
"""

from .node import Node
from .response_type import ResponseType
from .dag import DAG
from .constraints import Constraints
from .program_factory import ProgramFactory
from .linear_program import LinearProgram, SymbolicParameter
from .vertex_enumeration import (
    VertexEnumerator, 
    BoundResult, 
    compute_causal_bounds
)
from .symbolic_vertex_enum import (
    SymbolicVertexEnumerator,
    SymbolicVertex,
    SymbolicBound,
    compute_symbolic_causal_bounds
)

__all__ = ['Node', 'ResponseType', 'DAG', 'Constraints', 'ProgramFactory', 
           'LinearProgram', 'SymbolicParameter',
           'VertexEnumerator', 'BoundResult', 'compute_causal_bounds',
           'SymbolicVertexEnumerator', 'SymbolicVertex', 'SymbolicBound',
           'compute_symbolic_causal_bounds']
__version__ = '0.1.0'
