import time

import sys
start = time.time()

for i in model.component_objects(Set):
    i.pprint()
sys.exit()


# discretize model
discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(model, nfe=1, wrt=model.dis, scheme='FORWARD')
discretizer.apply_to(model, nfe=47, wrt=model.time, scheme='BACKWARD')

# What it should be to match description in paper
# discretizer.apply_to(model,nfe=48,wrt=model.time,scheme='BACKWARD')

TimeStep = model.time[2] - model.time[1]


def supcost_rule(m, k):
    return sum(m.cs * m.s[k, j, t] * (TimeStep) for j in m.SUP for t in m.time.get_finite_elements())


model.supcost = Expression(model.scen, rule=supcost_rule)


def boostcost_rule(m, k):
    return sum(m.ce * m.pow[k, j, t] * (TimeStep) for j in m.LINK_A for t in m.time.get_finite_elements())


model.boostcost = Expression(model.scen, rule=boostcost_rule)


def trackcost_rule(m, k):
    return sum(m.cd * (m.dem[k, j, t] - m.stochd[k, j, t]) ** 2.0 for j in m.DEM for t in m.time.get_finite_elements())


model.trackcost = Expression(model.scen, rule=trackcost_rule)


def sspcost_rule(m, k):
    return sum(
        m.cT * (m.px[k, i, m.time.last(), j] - m.px[k, i, m.time.first(), j]) ** 2.0 for i in m.link for j in m.dis)


model.sspcost = Expression(model.scen, rule=sspcost_rule)


def ssfcost_rule(m, k):
    return sum(
        m.cT * (m.fx[k, i, m.time.last(), j] - m.fx[k, i, m.time.first(), j]) ** 2.0 for i in m.link for j in m.dis)


model.ssfcost = Expression(model.scen, rule=ssfcost_rule)


def cost_rule(m, k):
    return 1e-6 * (m.supcost[k] + m.boostcost[k] + m.trackcost[k] + m.sspcost[k] + m.ssfcost[k])


model.cost = Expression(model.scen, rule=cost_rule)


def mcost_rule(m):
    return (1.0 / m.S) * sum(m.cost[k] for k in m.scen)


model.mcost = Expression(rule=mcost_rule)


def eqcvar_rule(m, k):
    return m.cost[k] - m.nu <= m.phi[k]


model.eqcvar = Constraint(model.scen, rule=eqcvar_rule)


def obj_rule(m):
    return (1.0 - m.cvar_lambda) * m.mcost + m.cvar_lambda * m.cvarcost


model.obj = Objective(rule=obj_rule)

endTime = time.time() - start
print('model creation time = %s' % (endTime,))

for i in model.scen:
    print("Scenario %s = %s" % (
        i, sum(sum(0.5 * value(model.pow[i, j, k])
                   for j in model.LINK_A)
               for k in model.time.get_finite_elements())))

solver = SolverFactory('ipopt')
results = solver.solve(model, tee=True)

for i in model.scen:
    print("Scenario %s = %s" % (
        i, sum(sum(0.5 * value(model.pow[i, j, k])
                   for j in model.LINK_A)
               for k in model.time.get_finite_elements())))
