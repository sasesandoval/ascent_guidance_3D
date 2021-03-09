import numpy as np
from numpy import linalg
from openmdao.api import ExplicitComponent

class OrbitalConditionTwo(ExplicitComponent):

    def initialize(self):
        #self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('a', default=1., types=float)

    def setup(self):
        #num = self.options['num_nodes']
        a = self.options['a']

        self.add_input('rx', shape=1)
        self.add_input('ry', shape=1)
        self.add_input('rz', shape=1)
        self.add_input('Vx', shape=1)
        self.add_input('Vy', shape=1)
        self.add_input('Vz', shape=1)

        self.add_output('C2', shape=1)

        self.declare_partials('*','*',dependent=False)
        self.declare_partials('C2','rx')
        self.declare_partials('C2','ry')
        self.declare_partials('C2','rz')
        self.declare_partials('C2','Vx')
        self.declare_partials('C2','Vy')
        self.declare_partials('C2','Vz')

    def compute(self,inputs,outputs):
        a = self.options['a']

        rx = inputs['rx'][0]
        ry = inputs['ry'][0]
        rz = inputs['rz'][0]
        Vx = inputs['Vx'][0]
        Vy = inputs['Vy'][0]
        Vz = inputs['Vz'][0]

        r = [rx, ry, rz]
        V = [Vx, Vy, Vz]

        outputs['C2'] = linalg.norm(V,2)/2 - 1/linalg.norm(r,2) + 1/(2*a)

    def compute_partials(self,inputs,partials):

        rx = inputs['rx'][0]
        ry = inputs['ry'][0]
        rz = inputs['rz'][0]
        Vx = inputs['Vx'][0]
        Vy = inputs['Vy'][0]
        Vz = inputs['Vz'][0]

        r = [rx, ry, rz]
        V = [Vx, Vy, Vz]
        norm_r = linalg.norm(r)
        norm_V = linalg.norm(V)

        dnorm_r_drx = rx / norm_r
        dnorm_r_dry = ry / norm_r
        dnorm_r_drz = rz / norm_r
        dnorm_V_dVx = Vx / norm_V
        dnorm_V_dVy = Vy / norm_V
        dnorm_V_dVz = Vz / norm_V

        partials['C2','rx'] = dnorm_r_drx / norm_r ** 2
        partials['C2','ry'] = dnorm_r_dry / norm_r ** 2
        partials['C2','rz'] = dnorm_r_drz / norm_r ** 2
        partials['C2','Vx'] = dnorm_V_dVx / 2
        partials['C2','Vy'] = dnorm_V_dVy / 2
        partials['C2','Vz'] = dnorm_V_dVz / 2