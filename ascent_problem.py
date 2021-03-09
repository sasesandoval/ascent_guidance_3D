import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg
from openmdao.api import Problem, IndepVarComp, ExecComp, ScipyOptimizer, pyOptSparseDriver
from ozone.api import ODEIntegrator
from ascent_function import AscentFunction
from orbital_condition_one import OrbitalConditionOne
from orbital_condition_two import OrbitalConditionTwo
from orbital_condition_three import OrbitalConditionThree
from thrust_vector_constraint import ThrustVectorConstraint

num = 21 # number of nodes

# ---------------------------------------------------------------------------------------------------------
# integrator group

ode_function = AscentFunction(system_init_kwargs={'g0': 1.625,'R0':1738100.,'vex':3047.80,'w2':1.,'T':45000.})
formulation = 'solver-based'
method_name = 'RK4'
initial_time = 0.
normalized_times = np.linspace(0., 1., num)
initial_conditions = {'rx': 0., 'ry': 0., 'rz': 1., 'Vx': 0., 'Vy': 0., 'Vz': 0., 'm': 11414.}

integrator = ODEIntegrator(ode_function, formulation, method_name,
    initial_time=initial_time, normalized_times=normalized_times,
    initial_conditions=initial_conditions)

# ---------------------------------------------------------------------------------------------------------
# passing in constraints

constraint_one = OrbitalConditionOne(a=1.0460253,ecc=0.0385)
constraint_two = OrbitalConditionTwo(a=1.0460253)
constraint_three = OrbitalConditionThree(inc=np.pi/2)
constraint_thrust = ThrustVectorConstraint()

# ---------------------------------------------------------------------------------------------------------
# problem formulation 1

# prob = Problem()
#
# # Define independent variable components
# prob.model.add_subsystem('final_time_comp', IndepVarComp('final_time', val=1.0))
# prob.model.add_subsystem('ux_comp', IndepVarComp('ux', shape=(num,1)))
# prob.model.add_subsystem('uy_comp', IndepVarComp('uy', shape=(num,1)))
# prob.model.add_subsystem('uz_comp', IndepVarComp('uz', shape=(num,1)))
# # prob.model.add_subsystem('T_comp', IndepVarComp('T', shape=(num,1))) # Add thrust as indep. var.
#
# # Add integrator group
# prob.model.add_subsystem('integrator_group', integrator)
#
# # Add constraint groups
# prob.model.add_subsystem('ConstraintOne', constraint_one, promotes=['rx','ry','rz','Vx','Vy','Vz','C1'])
# prob.model.add_subsystem('ConstraintTwo', constraint_two, promotes=['rx','ry','rz','Vx','Vy','Vz','C2'])
# prob.model.add_subsystem('ConstraintThree', constraint_three, promotes=['rx','ry','rz','Vx','Vy','Vz','C3'])
# prob.model.add_subsystem('ConstraintThrust', constraint_thrust, promotes=['ux','uy','uz',CT'])
#
# # Issue connections of independent components
# prob.model.connect('final_time_comp.final_time', 'integrator_group.final_time')
# prob.model.connect('ux_comp.ux', 'integrator_group.dynamic_parameter:ux')
# prob.model.connect('ux_comp.uy', 'integrator_group.dynamic_parameter:uy')
# prob.model.connect('ux_comp.uz', 'integrator_group.dynamic_parameter:uz')
#
# prob.model.add_design_var('final_time_comp.final_time', lower=1./806.)
# prob.model.add_design_var('ux_comp.ux')
# prob.model.add_design_var('uy_comp.uy')
# prob.model.add_design_var('uz_comp.uz')
# prob.model.add_objective('final_time_comp.final_time')
# prob.model.add_constraint('ConstraintOne.C1', equals=0.)
# prob.model.add_constraint('ConstraintOne.C2', equals=0.)
# prob.model.add_constraint('ConstraintOne.C3', equals=0.)
# prob.model.add_constraint('ConstraintOne.CT', equals=0.)

# ---------------------------------------------------------------------------------------------------------
# problem formulation 2

prob = Problem()

# Define independent variable components
prob.model.add_subsystem('final_time_comp', IndepVarComp('final_time', val=1.0))
prob.model.add_subsystem('ux_comp', IndepVarComp('ux', shape=(num,1)))
prob.model.add_subsystem('uy_comp', IndepVarComp('uy', shape=(num,1)))
prob.model.add_subsystem('uz_comp', IndepVarComp('uz', shape=(num,1)))
# prob.model.add_subsystem('rx_comp', IndepVarComp('rx', shape=(num,1)))
# prob.model.add_subsystem('T_comp', IndepVarComp('T', shape=(num,1))) # Add thrust as indep. var.

# Add integrator group
prob.model.add_subsystem('integrator_group', integrator)

# Add constraint groups
prob.model.add_subsystem('ConstraintOne', constraint_one)
prob.model.add_subsystem('ConstraintTwo', constraint_two)
prob.model.add_subsystem('ConstraintThree', constraint_three)
prob.model.add_subsystem('ConstraintThrust', constraint_thrust)

# Issue connections of independent components
prob.model.connect('final_time_comp.final_time', 'integrator_group.final_time')
prob.model.connect('ux_comp.ux', 'integrator_group.dynamic_parameter:ux')
prob.model.connect('uy_comp.uy', 'integrator_group.dynamic_parameter:uy')
prob.model.connect('uz_comp.uz', 'integrator_group.dynamic_parameter:uz')

# Issue connections of constraint 1
prob.model.connect('integrator_group.state:rx', 'ConstraintOne.rx', src_indices=-1) # src_indices=-1
prob.model.connect('integrator_group.state:ry', 'ConstraintOne.ry', src_indices=-1)
prob.model.connect('integrator_group.state:rz', 'ConstraintOne.rz', src_indices=-1)
prob.model.connect('integrator_group.state:Vx', 'ConstraintOne.Vx', src_indices=-1)
prob.model.connect('integrator_group.state:Vy', 'ConstraintOne.Vy', src_indices=-1)
prob.model.connect('integrator_group.state:Vz', 'ConstraintOne.Vz', src_indices=-1)

# Issue connections of constraint 2
prob.model.connect('integrator_group.state:rx', 'ConstraintTwo.rx', src_indices=-1)
prob.model.connect('integrator_group.state:ry', 'ConstraintTwo.ry', src_indices=-1)
prob.model.connect('integrator_group.state:rz', 'ConstraintTwo.rz', src_indices=-1)
prob.model.connect('integrator_group.state:Vx', 'ConstraintTwo.Vx', src_indices=-1)
prob.model.connect('integrator_group.state:Vy', 'ConstraintTwo.Vy', src_indices=-1)
prob.model.connect('integrator_group.state:Vz', 'ConstraintTwo.Vz', src_indices=-1)

# Issue connections of constraint 3
prob.model.connect('integrator_group.state:rx', 'ConstraintThree.rx', src_indices=-1)
prob.model.connect('integrator_group.state:ry', 'ConstraintThree.ry', src_indices=-1)
prob.model.connect('integrator_group.state:rz', 'ConstraintThree.rz', src_indices=-1)
prob.model.connect('integrator_group.state:Vx', 'ConstraintThree.Vx', src_indices=-1)
prob.model.connect('integrator_group.state:Vy', 'ConstraintThree.Vy', src_indices=-1)
prob.model.connect('integrator_group.state:Vz', 'ConstraintThree.Vz', src_indices=-1)

# Issue connections of constraint 4
prob.model.connect('ux_comp.ux', 'ConstraintThrust.ux', src_indices=-1)
prob.model.connect('uy_comp.uy', 'ConstraintThrust.uy', src_indices=-1)
prob.model.connect('uz_comp.uz', 'ConstraintThrust.uz', src_indices=-1)

prob.model.add_design_var('final_time_comp.final_time', lower=1./1034.2)
prob.model.add_design_var('ux_comp.ux')
prob.model.add_design_var('uy_comp.uy')
prob.model.add_design_var('uz_comp.uz')
prob.model.add_objective('final_time_comp.final_time')
prob.model.add_constraint('ConstraintOne.C1', equals=0.)
prob.model.add_constraint('ConstraintTwo.C2', equals=0.)
prob.model.add_constraint('ConstraintThree.C3', equals=0.)
prob.model.add_constraint('ConstraintThrust.CT', equals=0.)

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

prob.run_driver()

t = []
rx = prob['integrator_group.state:rx']
ry = prob['integrator_group.state:ry']
rz = prob['integrator_group.state:rz']
Vx = prob['integrator_group.state:Vx']
Vy = prob['integrator_group.state:Vy']
Vz = prob['integrator_group.state:Vz']
ux = prob['integrator_group.dynamic_parameter:ux']
uy = prob['integrator_group.dynamic_parameter:uy']
uz = prob['integrator_group.dynamic_parameter:uz']

if 1:
    r = np.zeros(num)
    for i in range(0,num):
        r[i] = linalg.norm([rx[i],ry[i],rz[i]],2)

    print(r)
    plt.plot(1034.2*prob['integrator_group.times'], (1738100.*r - 1738100.))
    plt.xlabel('time')
    plt.ylabel('altitude')
    plt.grid()
    plt.show()
print(' ')
if 0:
    V = np.zeros(num)
    for i in range(0,num):
        V[i] = linalg.norm([Vx[i],Vy[i],Vz[i]],2)

    print(V)
    plt.plot(1034.2*prob['integrator_group.times'], 1680.6*V)
    plt.xlabel('time')
    plt.ylabel('velocity')
    plt.grid()
    plt.show()
print(' ')
if 1:
    u = np.zeros(num)
    for i in range(0,num):
        u[i] = linalg.norm([ux[i],uy[i],uz[i]],2)

    print(u)
    plt.plot(1034.2*prob['integrator_group.times'], u)
    plt.xlabel('time')
    plt.ylabel('control')
    plt.grid()
    plt.show()
print(' ')
if 0:
    plt.plot(1034.2*prob['integrator_group.times'], ux)
    plt.xlabel('time')
    plt.ylabel('control x')
    plt.grid()
    plt.show()
print(' ')
if 0:
    plt.plot(1034.2*prob['integrator_group.times'], uy)
    plt.xlabel('time')
    plt.ylabel('control x')
    plt.grid()
    plt.show()
    print(' ')
if 0:
    plt.plot(1034.2*prob['integrator_group.times'], uz)
    plt.xlabel('time')
    plt.ylabel('control x')
    plt.grid()
    plt.show()
