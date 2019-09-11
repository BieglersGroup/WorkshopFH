import time

from pyomo.environ import *
from pyomo.dae import *
from stochpdegas_automatic import model
import sys
start = time.time()
instance = model.create_instance('stochpdegas_automatic.dat')

for i in instance.component_objects(Set):
    i.pprint()
sys.exit()


# discretize model
discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(instance, nfe=1, wrt=instance.dis, scheme='FORWARD')
discretizer.apply_to(instance, nfe=47, wrt=instance.time, scheme='BACKWARD')

# What it should be to match description in paper
# discretizer.apply_to(instance,nfe=48,wrt=instance.time,scheme='BACKWARD')

TimeStep = instance.time[2] - instance.time[1]


def supcost_rule(m, k):
    return sum(m.cs * m.s[k, j, t] * (TimeStep) for j in m.SUP for t in m.time.get_finite_elements())


instance.supcost = Expression(instance.scen, rule=supcost_rule)


def boostcost_rule(m, k):
    return sum(m.ce * m.pow[k, j, t] * (TimeStep) for j in m.LINK_A for t in m.time.get_finite_elements())


instance.boostcost = Expression(instance.scen, rule=boostcost_rule)


def trackcost_rule(m, k):
    return sum(m.cd * (m.dem[k, j, t] - m.stochd[k, j, t]) ** 2.0 for j in m.DEM for t in m.time.get_finite_elements())


instance.trackcost = Expression(instance.scen, rule=trackcost_rule)


def sspcost_rule(m, k):
    return sum(
        m.cT * (m.px[k, i, m.time.last(), j] - m.px[k, i, m.time.first(), j]) ** 2.0 for i in m.link for j in m.dis)


instance.sspcost = Expression(instance.scen, rule=sspcost_rule)


def ssfcost_rule(m, k):
    return sum(
        m.cT * (m.fx[k, i, m.time.last(), j] - m.fx[k, i, m.time.first(), j]) ** 2.0 for i in m.link for j in m.dis)


instance.ssfcost = Expression(instance.scen, rule=ssfcost_rule)


def cost_rule(m, k):
    return 1e-6 * (m.supcost[k] + m.boostcost[k] + m.trackcost[k] + m.sspcost[k] + m.ssfcost[k])


instance.cost = Expression(instance.scen, rule=cost_rule)


def mcost_rule(m):
    return (1.0 / m.S) * sum(m.cost[k] for k in m.scen)


instance.mcost = Expression(rule=mcost_rule)


def eqcvar_rule(m, k):
    return m.cost[k] - m.nu <= m.phi[k]


instance.eqcvar = Constraint(instance.scen, rule=eqcvar_rule)


def obj_rule(m):
    return (1.0 - m.cvar_lambda) * m.mcost + m.cvar_lambda * m.cvarcost


instance.obj = Objective(rule=obj_rule)

endTime = time.time() - start
print('model creation time = %s' % (endTime,))

for i in instance.scen:
    print("Scenario %s = %s" % (
        i, sum(sum(0.5 * value(instance.pow[i, j, k])
                   for j in instance.LINK_A)
               for k in instance.time.get_finite_elements())))

solver = SolverFactory('ipopt')
results = solver.solve(instance, tee=True)

for i in instance.scen:
    print("Scenario %s = %s" % (
        i, sum(sum(0.5 * value(instance.pow[i, j, k])
                   for j in instance.LINK_A)
               for k in instance.time.get_finite_elements())))
