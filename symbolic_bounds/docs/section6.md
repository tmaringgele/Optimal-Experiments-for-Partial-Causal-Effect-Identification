
# 6. Examples



The graphs in the following examples are divided into a left
side, which corresponds to the WL set, and a right side, which
corresponds to the WR set, as in Figure 4(a). The left side is
displayed as a violet (dark gray) box, and the right side a yellow
(light gray) box.

## 6.1. Confounded Exposure and Outcome

The basic DAG with two variables that are confounded as shown
in Figure 4(a) conforms to our class of models. In this case, the
variable X is the exposure of interest, and Y the outcome of
interest. X and Y have a common, unmeasured cause U which
we make no assumptions about. We specify X and Y to be
ternary and binary, respectively, so X takes values in {0, 1, 2} and
Y in {0, 1}. Our causal effects of interest are the risk differences
p{Y(X = 2) = 1} − P{Y(X = 0) = 1}, p{Y(X = 2) = 1} −
P{Y(X = 1) = 1} and p{Y(X = 1) = 1} − P{Y(X = 0) = 1},
and we have no additional constraints to specify.

Here we have two variables and therefore two response func-
tion variables. The response function variable formulation of the
graph in Figure 4(b) is an equivalent representation of the causal
model. The following tables define the values of the response
functions and variables:

```
x = fX(rX)
rX = 0   x = 0
rX = 1   x = 1
rX = 2   x = 2

y = fY(x,rY)   x = 0   x = 1   x = 2
rY = 0        y = 0   y = 0   y = 0
rY = 1        y = 1   y = 0   y = 0
rY = 2        y = 0   y = 1   y = 0
rY = 3        y = 1   y = 1   y = 0
rY = 4        y = 0   y = 0   y = 1
rY = 5        y = 1   y = 0   y = 1
rY = 6        y = 0   y = 1   y = 1
rY = 7        y = 1   y = 1   y = 1
```

RX is a random variable that can take on 3 possible values,
and RY is a random variable that can take on 2³ = 8 possible
values. Thus, the joint distribution of (RX, RY) is characterized
by 3 · 8 = 24 parameters, say qi,j, where i ∈ {0, 1, 2} and j ∈
{0, 1, 2, 3, 4, 5, 6, 7}. Applying Algorithm 1, we can relate the 3 ·
2 = 6 observed probabilities to the parameters of the response
function variable distribution as follows:

```
p0,0 := p{X = 0, Y = 0} = q0,0 + q0,2 + q0,4 + q0,6
p1,0 := p{X = 1, Y = 0} = q1,0 + q1,1 + q1,4 + q1,5
p2,0 := p{X = 2, Y = 0} = q2,0 + q2,1 + q2,2 + q2,3
p0,1 := p{X = 0, Y = 1} = q0,1 + q0,3 + q0,5 + q0,7
p1,1 := p{X = 1, Y = 1} = q1,2 + q1,3 + q1,6 + q1,7
p2,1 := p{X = 2, Y = 1} = q2,4 + q2,5 + q2,6 + q2,7
```

We get

```
A = [ X→Y   Y
      X  a   ∅
      Y  ∅   ∅ ],
```

for p{Y(X = a) = 1}, a ∈ {0, 1, 2}

Applying Algorithm 2, we get

```
p{Y(X = 0) = 1} = q0,1 + q0,3 + q0,5 + q0,7 + q1,1
                   + q1,3 + q1,5 + q1,7 + q2,1 + q2,3
                   + q2,5 + q2,7,

p{Y(X = 1) = 1} = q0,2 + q0,3 + q0,6 + q0,7 + q1,2
                   + q1,3 + q1,6 + q1,7 + q2,2 + q2,3
                   + q2,6 + q2,7,

p{Y(X = 2) = 1} = q0,4 + q0,5 + q0,6 + q0,7 + q1,4
                   + q1,5 + q1,6 + q1,7 + q2,4 + q2,5
                   + q2,6 + q2,7,
```

from which the contrasts of interest are easily derived.

Together with the probabilistic constraints, we then have the
fully specified linear programming problem. The bounds are,
after some algebra on the output from the program,

```
p{X = x1, Y = 1} + p{X = x2, Y = 0} − 1
   ≤ p{Y(X = x1) = 1} − p{Y(X = x2) = 1} ≤
1 − p{X = x1, Y = 0} − p{X = x2, Y = 1},
```

for (x1, x2) ∈ {(1, 0),(2, 0),(2, 1)}.

Note that these expressions are not unique and may be repa-
rameterized using that conditional probabilities sum to 1.

## 6.2. Two Instruments

Our next example is shown in the DAG in Figure 5. This extends
the instrumental variable example to the case where there are
two binary variables on the left side that may be associated
with each other and that both have a direct effect on X, but
no direct effect on Y. This situation may arise in Mendelian
randomization studies, wherein multiple genes may be known
to cause changes in an exposure but not directly on the outcome.

The bounds on risk difference p{Y(X = 1)} − p{Y(X =
0)} under this DAG can be computed using our method. In
this problem, there are 16 constraints involving the conditional
probabilities, the distribution of the response function variables
of the R-side has 64 parameters, and the causal query is a
function of 32 of these parameters. The bounds are the extrema
over 112 vertices, and are therefore too long to be presented
simply, but code to reproduce them, as well as details about the
simulation, is included in the supplementary materials.

To illustrate these bounds, we computed them for specific
values of observed probabilities generated from a model
(described fully in Section S2 of the supplementary materials)
which satisfies the DAG in Figure 5. Using these simulations,
we compare our bounds to the classic IV bounds from Balke
and Pearl (1997) for a single binary instrument and to bounds
derived using our method for a single but 4-level categorical
instrument.

The widths of the classic IV bounds and the dual binary
instruments are compared for a subsample of the simulations
in Figure 6. The bounds with two instruments are never wider
than the classic IV bounds with a single binary instrument.
The simulations also verify that a single four level instrument
yields exactly the same bounds as two binary ones. Details and
R code for these simulations are provided in the supplementary
materials.

## 6.3. Measurement Error in the Outcome

Our final example illustrates some additional features of our
method. In Figure 7, we have a binary variable X affecting a
binary variable Y, but Y is not observed. Instead, the binary
variable Y2 which is a child of Y is observed, and the effect of
the true Y on the measured Y2 is confounded. Additionally, we
would like to include a constraint that Y2(Y = 1) ≥ Y2(Y =
0), which is often called the monotonicity constraint. This con-
straint encodes the assumption that the outcome measured
with error would not be equal to 0 unless the true unobserved
outcome is also equal to 0. In terms of the response functions,
this constraint removes the case where fY2(y,rY2) = 1 − y,
thereby reducing the number of possible values that rY2 can take
by 1.

The fact that Y is unobserved implies that we have four
possible conditional probabilities to work with; p{Y2 = y2|X =
x}, for y2, x ∈ {0, 1}. There are 12 parameters that characterize
the distribution of the response function variables of the R-
side, and 4 constraints involving conditional probabilities. The
bounds for the risk difference p{Y(X = 1) = 1} − p{Y(X =
0) = 1} derived using our method are given by

```
max{−1, 2 p{Y2 = 0|X = 0} − 2 p{Y2 = 0|X = 1} − 1}
  ≤ p{Y(X = 1) = 1} − p{Y(X = 0) = 1} ≤
min{1, 2 p{Y2 = 0|X = 0} − 2 p{Y2 = 0|X = 1} + 1}.
```

Except in cases where p{Y2 = 0|X = 0} = p{Y2 = 0|X = 1},
these bounds are informative; meaning they give an interval that
is shorter than the a priori interval [−1, 1].
