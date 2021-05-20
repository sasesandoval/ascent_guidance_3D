import numpy as np
import matplotlib.pyplot as plt
import pickle, scipy.io

from numpy import linalg
from openmdao.api import Problem, IndepVarComp, ExecComp, ScipyOptimizer, pyOptSparseDriver
from ozone.api import ODEIntegrator
from ascent_function import AscentFunction
from constraint_group_analytic import OrbitalConditionOne, OrbitalConditionTwo, OrbitalConditionThree, UnitVectorConstraint
from lsdo_viz.api import Problem

num = 51 # number of nodes

# scaling factors
r_scale = 1738100.
v_scale = 1680.598851600226
t_scale = 1034.214677907832
m_scale = 15103.

# ---------------------------------------------------------------------------------------------------------
# integrator group

ode_function = AscentFunction(system_init_kwargs={'g0': 1.625,'R0':1738100.,'vex':3049.87,'w2':1.,'T':1.8335634})
formulation = 'solver-based'
method_name = 'ExplicitMidpoint'
initial_time = 0.
normalized_times = np.linspace(0., 1., num)

# Case 2321 - Abort Time: 150
if 0:
    initial_conditions = {'rx': 0.123789472141163,
                          'ry': 0.110836820117998,
                          'rz':-0.994476455504387,
                          'Vx':-0.611200671810031,
                          'Vy':-0.547145258362872,
                          'Vz':-0.182429517863452,
                          'm': 13371.752600734264/m_scale}

# Case 414 - Abort Time: 160
if 0:
    initial_conditions = {'rx': 0.118541274445888,
                          'ry': 0.106140084796489,
                          'rz':-0.995592878086612,
                          'Vx':-0.601740476640379,
                          'Vy':-0.538628391106065,
                          'Vz':-0.173069838633859,
                          'm': 13268.428447345410/m_scale}

# Case 1885 - Abort Time: 170
if 0:
    initial_conditions = {'rx': 0.112934417419268,
                          'ry': 0.101122622667143,
                          'rz':-0.996748085697020,
                          'Vx':-0.591535866840624,
                          'Vy':-0.529449544391755,
                          'Vz':-0.164163814415230,
                          'm': 13152.164008317570/m_scale}

# Case 728 - Abort Time: 180
if 0:
    initial_conditions = {'rx': 0.107492848759599,
                          'ry': 0.096253355876165,
                          'rz':-0.997807691486522,
                          'Vx':-0.581076368588230,
                          'Vy':-0.520048718866702,
                          'Vz':-0.155560119082495,
                          'm': 13035.028461224259/m_scale}

# Case 876 - Abort Time: 190
if 0:
    initial_conditions = {'rx': 0.102492115319790,
                          'ry': 0.091778817670127,
                          'rz':-0.998705150893710,
                          'Vx':-0.570462734155777,
                          'Vy':-0.510515799456322,
                          'Vz':-0.510515799456322,
                          'm': 12924.608845467152/m_scale}

# Case 2670 - Abort Time: 200
if 0:
    initial_conditions = {'rx': 0.097168425891157,
                          'ry': 0.087015449954683,
                          'rz':-0.999625410233659,
                          'Vx':-0.559106385451942,
                          'Vy':-0.500322602007477,
                          'Vz':-0.138364377978712,
                          'm': 12801.490338704645/m_scale}

# Case 1996 - Abort Time: 210
if 0:
    initial_conditions = {'rx': 0.092068327543631,
                          'ry': 0.082452294681549,
                          'rz':-1.000449212010749,
                          'Vx':-0.547651316096105,
                          'Vy':-0.490046875329196,
                          'Vz':-0.130414173078158,
                          'm': 12679.211941709411/m_scale}


# Case 2387 - Abort Time: 220
if 0:
    initial_conditions = {'rx': 0.087039843244263,
                          'ry': 0.077953327113988,
                          'rz':-1.001204636764928,
                          'Vx':-0.535771322888445,
                          'Vy':-0.479395707953645,
                          'Vz':-0.122615628242980,
                          'm': 12554.281576273783/m_scale}

# Case 1027 - Abort Time: 230
if 0:
    initial_conditions = {'rx': 0.082281058280732,
                          'ry': 0.073695728748285,
                          'rz':-1.001866133999734,
                          'Vx':-0.523958196890501,
                          'Vy':-0.468809660735925,
                          'Vz': -0.115274320782175,
                          'm': 12431.832318722376/m_scale}

# Case 1355 - Abort Time: 240
if 0:
    initial_conditions = {'rx': 0.077503130514565,
                          'ry': 0.069421037120208,
                          'rz':-1.002476421274584,
                          'Vx':-0.511501447627683,
                          'Vy':-0.457651837439726,
                          'Vz':-0.107944417125267,
                          'm': 12304.513603414716/m_scale}

# Case 318 - Abort Time: 250
if 0:
    initial_conditions = {'rx': 0.072901427722291,
                          'ry': 0.065304005486362,
                          'rz':-1.003011448554121,
                          'Vx':-0.498898942171510,
                          'Vy':-0.446368123947842,
                          'Vz':-0.100926847596432,
                          'm': 12177.478228846176/m_scale}


# Case 2580 - Abort Time: 260
if 0:
    initial_conditions = {'rx': 0.068195292157576,
                          'ry': 0.061093494702373,
                          'rz':-1.003525558612518,
                          'Vx':-0.485974313269779,
                          'Vy':-0.434800318165984,
                          'Vz':-0.094781491141546,
                          'm': 12042.234964143003/m_scale}

# Case 480 - Abort Time: 270
if 0:
    initial_conditions = {'rx': 0.063863102328262,
                          'ry': 0.057217475365984,
                          'rz':-1.003926570467992,
                          'Vx':-0.472877609855235,
                          'Vy':-0.423082550965536,
                          'Vz':-0.088264217231898,
                          'm': 11913.677457386162/m_scale}

# Case 2447 - Abort Time: 280
if 0:
    initial_conditions = {'rx': 0.059665656461338,
                          'ry': 0.053461900733260,
                          'rz':-1.004265779854917,
                          'Vx':-0.459553208872069,
                          'Vy':-0.411164780252120,
                          'Vz':-0.081996657033966,
                          'm': 11784.564729779073/m_scale}

# Case 2077 - Abort Time: 290
if 0:
    initial_conditions = {'rx': 0.055429044700880,
                          'ry': 0.049671143603471,
                          'rz':-1.004579989181301,
                          'Vx':-0.446088276912695,
                          'Vy':-0.399124616889840,
                          'Vz':-0.076709711725079,
                          'm': 11648.950081917923/m_scale}

# Case 273 - Abort Time: 300
if 1:
    initial_conditions = {'rx': 0.051473261347652,
                          'ry': 0.046131470657302,
                          'rz':-1.004802504773337,
                          'Vx':-0.432215441070972,
                          'Vy':-0.386722961483122,
                          'Vz':-0.070907047408024,
                          'm': 11517.828034303415/m_scale}

integrator = ODEIntegrator(ode_function, formulation, method_name,
    initial_time=initial_time, normalized_times=normalized_times,
    initial_conditions=initial_conditions)

# ---------------------------------------------------------------------------------------------------------
# passing in constraints

constraint_one = OrbitalConditionOne(a=1.04603,ecc=0.0385)
constraint_two = OrbitalConditionTwo(a=1.04603)
constraint_three = OrbitalConditionThree(a=1.04603,ecc=0.0385,inc=np.pi/2)
constraint_unit = UnitVectorConstraint(num_nodes=num)

# ---------------------------------------------------------------------------------------------------------
# problem formulation 1 - Using connections

prob = Problem()

# Define independent variable components
prob.model.add_subsystem('final_time_comp', IndepVarComp('final_time', val=0.1))
prob.model.add_subsystem('ux_comp', IndepVarComp('ux', val=-0.1, shape=(num,1)))
prob.model.add_subsystem('uy_comp', IndepVarComp('uy', val=-0.1, shape=(num,1)))
prob.model.add_subsystem('uz_comp', IndepVarComp('uz', val=-0.1, shape=(num,1)))

# Add integrator group
prob.model.add_subsystem('integrator_group', integrator)

# prob.model.add_subsystem('mass_objective', ExecComp('m_obj = 1. - m'))
# prob.model.connect('integrator_group.state:m', 'mass_objective.m', src_indices=-1)

# Issue connections of independent components
prob.model.connect('final_time_comp.final_time', 'integrator_group.final_time')
prob.model.connect('ux_comp.ux', 'integrator_group.dynamic_parameter:ux')
prob.model.connect('uy_comp.uy', 'integrator_group.dynamic_parameter:uy')
prob.model.connect('uz_comp.uz', 'integrator_group.dynamic_parameter:uz')

# Issue connections of constraint 1
prob.model.add_subsystem('ConstraintOne', constraint_one)
prob.model.connect('integrator_group.state:rx', 'ConstraintOne.rx', src_indices=-1)
prob.model.connect('integrator_group.state:ry', 'ConstraintOne.ry', src_indices=-1)
prob.model.connect('integrator_group.state:rz', 'ConstraintOne.rz', src_indices=-1)
prob.model.connect('integrator_group.state:Vx', 'ConstraintOne.Vx', src_indices=-1)
prob.model.connect('integrator_group.state:Vy', 'ConstraintOne.Vy', src_indices=-1)
prob.model.connect('integrator_group.state:Vz', 'ConstraintOne.Vz', src_indices=-1)

# Issue connections of constraint 2
prob.model.add_subsystem('ConstraintTwo', constraint_two)
prob.model.connect('integrator_group.state:rx', 'ConstraintTwo.rx', src_indices=-1)
prob.model.connect('integrator_group.state:ry', 'ConstraintTwo.ry', src_indices=-1)
prob.model.connect('integrator_group.state:rz', 'ConstraintTwo.rz', src_indices=-1)
prob.model.connect('integrator_group.state:Vx', 'ConstraintTwo.Vx', src_indices=-1)
prob.model.connect('integrator_group.state:Vy', 'ConstraintTwo.Vy', src_indices=-1)
prob.model.connect('integrator_group.state:Vz', 'ConstraintTwo.Vz', src_indices=-1)

# Issue connections of constraint 3
prob.model.add_subsystem('ConstraintThree', constraint_three)
prob.model.connect('integrator_group.state:rx', 'ConstraintThree.rx', src_indices=-1)
prob.model.connect('integrator_group.state:ry', 'ConstraintThree.ry', src_indices=-1)
prob.model.connect('integrator_group.state:rz', 'ConstraintThree.rz', src_indices=-1)
prob.model.connect('integrator_group.state:Vx', 'ConstraintThree.Vx', src_indices=-1)
prob.model.connect('integrator_group.state:Vy', 'ConstraintThree.Vy', src_indices=-1)
prob.model.connect('integrator_group.state:Vz', 'ConstraintThree.Vz', src_indices=-1)


prob.model.add_subsystem('ConstraintUnit', constraint_unit)
prob.model.connect('ux_comp.ux', 'ConstraintUnit.ux')
prob.model.connect('uy_comp.uy', 'ConstraintUnit.uy')
prob.model.connect('uz_comp.uz', 'ConstraintUnit.uz')

# Design Variables
prob.model.add_design_var('final_time_comp.final_time', lower=1./1034.2)
prob.model.add_design_var('ux_comp.ux',lower=-1.,upper=1.)
prob.model.add_design_var('uy_comp.uy',lower=-1.,upper=1.)
prob.model.add_design_var('uz_comp.uz',lower=-1.,upper=1.)

# Objective
prob.model.add_objective('final_time_comp.final_time')
# prob.model.add_objective('mass_objective.m_obj')

# Constraints
prob.model.add_constraint('ConstraintOne.C1', equals=0.)
prob.model.add_constraint('ConstraintTwo.C2', equals=0.)
prob.model.add_constraint('ConstraintThree.C3', equals=0.)
prob.model.add_constraint('ConstraintUnit.CU', equals=0.)
prob.model.add_constraint('integrator_group.state:m', lower=1./m_scale, indices=[-1])

# ---------------------------------------------------------------------------------------------------------
# problem optimizer

prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'

# ---------------------------------------------------------------------------------------------------------
# problem setup
prob.setup(check=False)

if 0:
    prob.run_model()
    prob.check_partials(compact_print=True)
    exit()

prob.run()

t = prob['integrator_group.times']
rx = prob['integrator_group.state:rx']
ry = prob['integrator_group.state:ry']
rz = prob['integrator_group.state:rz']
Vx = prob['integrator_group.state:Vx']
Vy = prob['integrator_group.state:Vy']
Vz = prob['integrator_group.state:Vz']
m = prob['integrator_group.state:m']
ux = prob['integrator_group.dynamic_parameter:ux']
uy = prob['integrator_group.dynamic_parameter:uy']
uz = prob['integrator_group.dynamic_parameter:uz']

print('a', 1.04603)
print('ecc', 0.0385)
print('inc', np.pi/2)
print('t', t[-1]) #*t_scale)
print('rx', rx[-1]) #*r_scale)
print('ry', ry[-1]) #*r_scale)
print('rz', rz[-1]) #*r_scale)
print('Vx', Vx[-1]) #*v_scale)
print('Vy', Vy[-1]) #*v_scale)
print('Vz', Vz[-1]) #*v_scale)
print('m', m[-1]) #*m_scale)


if 0:
    r = np.zeros(num)
    for i in range(0,num):
        r[i] = linalg.norm([rx[i],ry[i],rz[i]],2)
    plt.plot(1034.2*t, (1738100.*r - 1738100.))
    plt.xlabel('time')
    plt.ylabel('altitude')
    plt.grid()
    plt.show()

if 0:
    V = np.zeros(num)
    for i in range(0,num):
        V[i] = linalg.norm([Vx[i],Vy[i],Vz[i]],2)
    plt.plot(1034.2*t, 1680.6*V)
    plt.xlabel('time')
    plt.ylabel('velocity')
    plt.grid()
    plt.show()

u = np.zeros(num)
for i in range(0,num):
    u[i] = linalg.norm([ux[i],uy[i],uz[i]],2)

if 0:
    plt.plot(1034.2*t, ux)
    plt.xlabel('time')
    plt.ylabel('ux')
    plt.ylim(-1.2,1.2)
    plt.grid()
    plt.show()

if 0:
    plt.plot(1034.2*t, uy)
    plt.xlabel('time')
    plt.ylabel('uy')
    plt.ylim(-1.2,1.2)
    plt.grid()
    plt.show()

if 0:
    plt.plot(1034.2*t, uz)
    plt.xlabel('time')
    plt.ylabel('uz')
    plt.ylim(-1.2,1.2)
    plt.grid()
    plt.show()

if 0:
    plt.plot(1034.2*t, m*m_scale)
    plt.xlabel('time')
    plt.ylabel('mass')
    plt.grid()
    plt.show()

if 0:
    plt.plot(1034.2*prob['integrator_group.times'], aT/1.833563378)
    plt.xlabel('time')
    plt.ylabel('throttle')
    plt.grid()
    plt.show()

# ------------------------------------------------------------------------------
#                                SAVE DATA
# ------------------------------------------------------------------------------
# print("size Check")
# print(np.shape(r))
# print(np.shape(V))
# print(np.shape(m))
# print(np.shape(u))
# print(np.shape(t))

# data min time
# data = [r, V, u, t]
# pickle.dump(data,open("data_timeobj.dat","wb"))
# data = [m]
# pickle.dump(data,open("data_timeobj_mass.dat","wb"))
# data = [ux, uy, uz]
# pickle.dump(data,open("data_timeobj_control.dat","wb"))

# data max prop
# data = [r, V, u, t]
# pickle.dump(data,open("data_massobj.dat","wb"))
# data = [m]
# pickle.dump(data,open("data_massobj_mass.dat","wb"))
# data = [ux, uy, uz]
# pickle.dump(data,open("data_massobj_control.dat","wb"))

scipy.io.savemat('mydata.mat', mdict={'OMt':t,'OMrx':rx,'OMry':ry,'OMrz':rz,'OMvx':Vx,'OMvy':Vy,'OMvz':Vz,'OMm':m,'OMux':ux,'OMuy':uy,'OMuz':uz})
