"""
Test script for writeRung2 function.
"""

from symbolic_bounds import ProgramFactory
from symbolic_bounds.dag import DAG

print("="*80)
print("TESTING writeRung2 FUNCTION")
print("="*80)

# Test 1: Simple chain X -> Y (both in W_R)
print("\nTest 1: Simple chain X -> Y (both in W_R)")
print("-"*80)

dag1 = DAG()
X = dag1.add_node('X', support={0, 1}, partition='R')
Y = dag1.add_node('Y', support={0, 1}, partition='R')
dag1.add_edge(X, Y)

# Generate response types
dag1.generate_all_response_types()

# Print response types for understanding
print("\nResponse types for X (no parents):")
dag1.print_response_type_table(X)

print("\nResponse types for Y (parent: X):")
dag1.print_response_type_table(Y)

# Query: P(Y=1 | do(X=1))
print("\nQuery: P(Y=1 | do(X=1))")
alpha = ProgramFactory.writeRung2(dag1, {Y}, {X}, (1,), (1,))

print(f"\nCoefficient vector α (length {len(alpha)}):")
print(alpha)
print(f"\nNon-zero entries: {sum(alpha > 0)} out of {len(alpha)}")
print(f"Indices with α=1: {[i for i, val in enumerate(alpha) if val > 0]}")

# Test 2: With confounding from W_L
print("\n" + "="*80)
print("Test 2: Confounding with W_L -> X -> Y, W_L -> Y")
print("-"*80)

dag2 = DAG()
Z = dag2.add_node('Z', support={0, 1}, partition='L')
X2 = dag2.add_node('X', support={0, 1}, partition='R')
Y2 = dag2.add_node('Y', support={0, 1}, partition='R')
dag2.add_edge(Z, X2)
dag2.add_edge(Z, Y2)
dag2.add_edge(X2, Y2)

# Generate response types
dag2.generate_all_response_types()

# Query: P(Y=1 | do(X=0))
print("\nQuery: P(Y=1 | do(X=0))")
alpha2 = ProgramFactory.writeRung2(dag2, {Y2}, {X2}, (1,), (0,))

print(f"\nCoefficient vector α (length {len(alpha2)}):")
print(alpha2)
print(f"\nNon-zero entries: {sum(alpha2 > 0)} out of {len(alpha2)}")
print(f"Sum of α (probability): {sum(alpha2)}")

# Test 3: Multiple variables in V
print("\n" + "="*80)
print("Test 3: Multiple outcome variables")
print("-"*80)

dag3 = DAG()
X3 = dag3.add_node('X', support={0, 1}, partition='R')
Y3a = dag3.add_node('Y1', support={0, 1}, partition='R')
Y3b = dag3.add_node('Y2', support={0, 1}, partition='R')
dag3.add_edge(X3, Y3a)
dag3.add_edge(X3, Y3b)

dag3.generate_all_response_types()

# Query: P(Y1=1, Y2=1 | do(X=1))
print("\nQuery: P(Y1=1, Y2=1 | do(X=1))")
alpha3 = ProgramFactory.writeRung2(dag3, {Y3a, Y3b}, {X3}, (1, 1), (1,))

print(f"\nCoefficient vector α (length {len(alpha3)}):")
print(alpha3)
print(f"\nNon-zero entries: {sum(alpha3 > 0)} out of {len(alpha3)}")

print("\n" + "="*80)
print("✓ All tests completed successfully!")
print("="*80)
