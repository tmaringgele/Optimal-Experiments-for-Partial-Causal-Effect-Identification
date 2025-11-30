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
    Constraints: Holds the constraint system (matrices P, Lambda, etc.)
    ProgramFactory: Factory for generating constraints from DAGs (implements Algorithm 1)
"""

from .node import Node
from .response_type import ResponseType
from .dag import DAG
from .constraints import Constraints
from .program_factory import ProgramFactory
from .joint_distribution import JointDistribution
from .scm import SCM
from .linear_program import LinearProgram

__all__ = [
    'Node', 
    'ResponseType', 
    'DAG', 
    'Constraints', 
    'ProgramFactory',
    'JointDistribution',
    'SCM',
    'LinearProgram'
]
__version__ = '0.1.0'
