import numpy as np
from gurobipy import *
import math

N = 12
M = 72
phi = -2 * math.pi / N
I1 = list(range(1, N + 1))
A = list(range(0, 2))
I2 = list(range(0, M + 1))
c_transport =6


alpha = 36
beta = 2


# failure rate according to weibull distribution
def failure_rate(age):
    return 1 - math.exp(- (age / alpha) ** beta + ((age - 1) / alpha) ** beta)

# transition probabilities for the MDP as described in the paper
def pi(i1, i2, j1, j2, a, m=1):
    if (a == 0):
        if j1 % (N*m) == (i1 + 1) % (N*m) and j2 == i2 + 1 and i2 != 0 and i2 != M:
            return 1 - failure_rate(j2)
        elif j1 % (N*m) == (i1 + 1) % (N*m) and j2 == 0 and i2 != 0 and i2 != M:
            return failure_rate(i2 + 1)
        else:
            return 0
    else:
        if j1 % (N*m) == (i1 + 1) % (N*m) and j2 == 1:
            return 1 - failure_rate(1)
        elif j1 % (N*m) == (i1 + 1) % (N*m) and j2 == 0:
            return failure_rate(1)
        else:
            return 0

## costs for corrective maintenance in period i1
def c_f(i1):
    c_cm = 50 + c_transport + 10 * math.cos(2 * math.pi * i1 / N + phi)
    return c_cm

## costs for preventive maintenance in period i1
def c_p(i1):
    c_pm = 10 + c_transport + 2 * math.cos(2 * math.pi * i1 / N + phi)
    return c_pm

## action space for age i2
def action(i2):
    if i2 == 0 or i2 >= M:
        return [1]
    else:
        return [0, 1]

## function that solves pARP for an I1 and I2 input set, returns gurobi model instances
def solve_parp(I1, I2):
    parp = Model(name='pARP')
    x = parp.addVars(I1, I2, A, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
    parp.addConstrs((quicksum(x[i1, i2, a] for i2 in I2 for a in action(i2)) == (1 / N) for i1 in I1), "equal_period")
    parp.addConstrs((quicksum(x[i1, i2, a] for a in action(i2)) - quicksum(
        x[j1, j2, a] * pi(j1, j2, i1, i2, a) for j1 in I1 for j2 in I2 for a in action(j2)) == 0
                     for i1 in I1 for i2 in I2), "transition_probs")
    parp.setObjective(
        quicksum(x[i1, i2, 1] * c_p(i1) for i1 in I1 for i2 in I2[1:]) + quicksum(x[i1, 0, 1] * c_f(i1) for i1 in I1),
        GRB.MINIMIZE)
    parp.optimize()
    return parp

## function that solves pBRP for an I1 and I2 input set, returns gurobi model instances
def solve_brp(I1, I2):
    m = len(I1) / N
    brp = Model(name='BRP')
    x = brp.addVars(I1, I2, A, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
    y = brp.addVars(I1, vtype=GRB.BINARY, name="y")
    equal_period = brp.addConstrs((quicksum(x[i1, i2, a] for i2 in I2 for a in action(i2)) == 1 / (N * m) for i1 in I1),
                                  "equal_period")
    probs = brp.addConstrs((quicksum(x[i1, i2, a] for a in action(i2)) - quicksum(
        x[j1, j2, a] * pi(j1, j2, i1, i2, a, m) for j1 in I1 for j2 in I2 for a in action(j2)) == 0
                            for i1 in I1 for i2 in I2), "transition_probs")
    no_maintenance = brp.addConstrs((x[i1, i2, 0] + y[i1] <= 1 for i1 in I1 for i2 in I2[1:]), "no_maintenance")
    maintenance = brp.addConstrs((x[i1, i2, 1] - y[i1] <= 0 for i1 in I1 for i2 in I2[1:]), "maintenance")
    brp.setObjective(
        quicksum(x[i1, i2, 1] * c_p(i1) for i1 in I1 for i2 in I2[1:]) + quicksum(x[i1, 0, 1] * c_f(i1) for i1 in I1),
        GRB.MINIMIZE)
    brp.optimize()
    return brp

## function that solves pMBRP for an I1 and I2 input set, returns gurobi model instances
def solve_mbrp(I1, I2):
    m = len(I1) / N
    mbrp = Model(name='MBRP')
    x = mbrp.addVars(I1, I2, A, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
    y = mbrp.addVars(I1, vtype=GRB.BINARY, name="y")
    z = mbrp.addVars(I1, I2, vtype=GRB.BINARY, name="z")
    t = mbrp.addVars(I1, lb=1, vtype=GRB.INTEGER, name="t")
    mbrp.addConstrs((quicksum(x[i1, i2, a] for i2 in I2 for a in action(i2)) == (1 / (N * m)) for i1 in I1),
                    "equal_period")
    mbrp.addConstrs((quicksum(x[i1, i2, a] for a in action(i2)) - quicksum(
        x[j1, j2, a] * pi(j1, j2, i1, i2, a, m) for j1 in I1 for j2 in I2 for a in action(j2)) == 0
                     for i1 in I1 for i2 in I2), "transition_probs")
    mbrp.addConstrs(z[i1, i2] - y[i1] <= 0 for i1 in I1 for i2 in I2)
    mbrp.addConstrs(z[i1, i2] - z[i1, j2] <= 0 for i1 in I1 for i2 in I2 for j2 in I2 if i2 < j2)
    mbrp.addConstrs(t[i1] + j1 * y[j1] + m * N * y[j1] <= (m * N + i1) for i1 in I1 for j1 in I1 if j1 < i1)
    mbrp.addConstrs(t[i1] + j1 * y[j1] <= (m * N + i1) for i1 in I1 for j1 in I1 if j1 > i1)
    mbrp.addConstrs(M * y[i1] - M * z[i1, i2] - t[i1] <= (M - 1 - i2) for i1 in I1 for i2 in I2)
    mbrp.addConstrs(M * z[i1, i2] + t[i1] <= (M + i2) for i1 in I1 for i2 in I2)
    mbrp.setObjective(
        quicksum(x[i1, i2, 1] * c_p(i1) for i1 in I1 for i2 in I2[1:]) + quicksum(x[i1, 0, 1] * c_f(i1) for i1 in I1),
        GRB.MINIMIZE)
    mbrp.optimize()
    return mbrp, z


arp = solve_parp(I1, I2)

## Get the objective funciton
print('Long Term Average Costs: ' + str(arp.ObjVal*12))

######### p-ARP cma extraction ############
vars = np.zeros((len(I1), len(I2), 2))
even = 0
counter = 0

## Storing decision variables
for v in arp.getVars():
    a = v.index % 2
    even = v.index - 1
    i1 = v.index//(2*(1 + M)) + 1
    i2 = counter
    if even % 2 == 0: counter = (counter + 1) % (M + 1)
    vars[(i1-1, i2, a)] = v.X

# create array to store all state and action combinations under the policy
R = [(i1, i2, a) for i1 in I1 for i2 in I2 for a in [0, 1] if vars[(i1-1, i2, a)] > 0]
all = [(i1, i2, a) for i1 in I1 for i2 in I2 for a in [0, 1]]

# add the states that occur to R
while len(R) < N*(M + 1):
    for i1, i2, action in all:
        for j1, j2, a in R:
            if (i1, i2, 0) not in R:
                if (i1, i2, 1) not in R:
                    if (pi(i1, i2, j1, j2, action) > 0): R.append((i1, i2, action))
            if len(R) > N*(M + 1): break
        if len(R) > N*(M + 1): break
    if len(R) > N*(M + 1): break

## retreiving the critical maintenance stages
## for an explanation of the procedure, check the paper section 4.2.1
R.sort()
p = [r for r in R if r[2] != 0 if r[1] != 0]
cma = np.zeros(len(I1))
for (i1, i2, a) in p:
    if cma[i1-1] == 0:
        cma[i1-1] = i2
print(cma)

# variables = arp.getVars()
# lp_solution = []
#
# for x in variables:
#     if x.X > 0:
#         variables.append(x)

### Run MBRP
m = 3
I1 = list(range(1, N * m + 1))
# brp = solve_brp(I1, I2)
mbrp, Z = solve_mbrp(I1, I2)

print(Z.values())

## print variables
for var in mbrp.getVars():
    if var.x > 0:
        print("{}:{}".format(var.x, var))