import numpy as np
from openmdao.api import ExplicitComponent

class OrbitalConditionOne(ExplicitComponent):

    def initialize(self):
        #self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('a', default=1., types=float)
        self.options.declare('ecc', default=1., types=float)

    def setup(self):
        #num = self.options['num_nodes']
        a = self.options['a']
        ecc = self.options['ecc']

        self.add_input('rx', shape=1)
        self.add_input('ry', shape=1)
        self.add_input('rz', shape=1)
        self.add_input('Vx', shape=1)
        self.add_input('Vy', shape=1)
        self.add_input('Vz', shape=1)

        self.add_output('C1', shape=1)

        self.declare_partials('*','*',dependent=False)
        self.declare_partials('C1','rx')
        self.declare_partials('C1','ry')
        self.declare_partials('C1','rz')
        self.declare_partials('C1','Vx')
        self.declare_partials('C1','Vy')
        self.declare_partials('C1','Vz')

    def compute(self,inputs,outputs):
        a = self.options['a']
        ecc = self.options['ecc']

        rx = inputs['rx'][0]
        ry = inputs['ry'][0]
        rz = inputs['rz'][0]
        Vx = inputs['Vx'][0]
        Vy = inputs['Vy'][0]
        Vz = inputs['Vz'][0]

        r = [rx, ry, rz]
        V = [Vx, Vy, Vz]
        rxV = np.cross(r,V)

        outputs['C1'] = np.dot(rxV,rxV) - a*(1 - ecc**2)

    def compute_partials(self,inputs,partials):

        rx = inputs['rx'][0]
        ry = inputs['ry'][0]
        rz = inputs['rz'][0]
        Vx = inputs['Vx'][0]
        Vy = inputs['Vy'][0]
        Vz = inputs['Vz'][0]

        partials['C1','rx'] = 2*(rx*Vz-rz*Vx)*Vz + 2*(rx*Vy-ry*Vx)*Vy
        partials['C1','ry'] = 2*(ry*Vz-rz*Vy)*Vz - 2*(rx*Vy-ry*Vx)*Vx
        partials['C1','rz'] =-2*(ry*Vz-rz*Vy)*Vy - 2*(rx*Vz-rz*Vx)*Vx
        partials['C1','Vx'] =-2*(rx*Vz-rz*Vx)*rz - 2*(rx*Vy-ry*Vx)*ry
        partials['C1','Vy'] =-2*(ry*Vz-rz*Vy)*rz + 2*(rx*Vy-ry*Vx)*rx
        partials['C1','Vz'] = 2*(ry*Vz-rz*Vy)*ry + 2*(rx*Vz-rz*Vx)*rx
