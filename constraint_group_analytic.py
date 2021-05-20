import numpy as np
from numpy import linalg
from openmdao.api import ExplicitComponent

class OrbitalConditionOne(ExplicitComponent):

    def initialize(self):
        self.options.declare('a', default=1., types=float)
        self.options.declare('ecc', default=1., types=float)

    def setup(self):
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





class OrbitalConditionTwo(ExplicitComponent):

    def initialize(self):
        self.options.declare('a', default=1., types=float)

    def setup(self):
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

        outputs['C2'] = linalg.norm(V,2)**2/2. - 1./linalg.norm(r,2) + 1./(2*a)

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
        partials['C2','Vx'] = Vx
        partials['C2','Vy'] = Vy
        partials['C2','Vz'] = Vz





class OrbitalConditionThree(ExplicitComponent):

    def initialize(self):
        self.options.declare('a', default=1., types=float)
        self.options.declare('ecc', default=1., types=float)
        self.options.declare('inc', default=1., types=float)

    def setup(self):
        a = self.options['a']
        ecc = self.options['ecc']
        inc = self.options['inc']

        self.add_input('rx', shape=1)
        self.add_input('ry', shape=1)
        self.add_input('rz', shape=1)
        self.add_input('Vx', shape=1)
        self.add_input('Vy', shape=1)
        self.add_input('Vz', shape=1)

        self.add_output('C3', shape=1)

        self.declare_partials('*','*',dependent=False)
        self.declare_partials('C3','rx')
        self.declare_partials('C3','ry')
        self.declare_partials('C3','rz')
        self.declare_partials('C3','Vx')
        self.declare_partials('C3','Vy')
        self.declare_partials('C3','Vz')

    def compute(self,inputs,outputs):
        a = self.options['a']
        ecc = self.options['ecc']
        inc = self.options['inc']

        rx = inputs['rx'][0]
        ry = inputs['ry'][0]
        rz = inputs['rz'][0]
        Vx = inputs['Vx'][0]
        Vy = inputs['Vy'][0]
        Vz = inputs['Vz'][0]

        r = [rx, ry, rz]
        V = [Vx, Vy, Vz]
        rxV = np.cross(r,V)
        z = [0, 0, 1]
        h = np.sqrt(a*(1-ecc**2))

        outputs['C3'] = np.dot(z,rxV) - h*np.cos(inc)

    def compute_partials(self,inputs,partials):

        rx = inputs['rx'][0]
        ry = inputs['ry'][0]
        rz = inputs['rz'][0]
        Vx = inputs['Vx'][0]
        Vy = inputs['Vy'][0]
        Vz = inputs['Vz'][0]

        r = [rx, ry, rz]
        V = [Vx, Vy, Vz]
        rxV = np.cross(r,V)

        partials['C3','rx'] = Vy
        partials['C3','ry'] =-Vx
        partials['C3','Vx'] =-ry
        partials['C3','Vy'] = rx



class UnitVectorConstraint(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        num = self.options['num_nodes']

        self.add_input('ux', shape=(num,1))
        self.add_input('uy', shape=(num,1))
        self.add_input('uz', shape=(num,1))

        self.add_output('CU', shape=(num,1))

        arange = np.arange(num)
        self.declare_partials('CU','ux', rows=arange, cols=arange)
        self.declare_partials('CU','uy', rows=arange, cols=arange)
        self.declare_partials('CU','uz', rows=arange, cols=arange)

    def compute(self,inputs,outputs):

        ux = inputs['ux']
        uy = inputs['uy']
        uz = inputs['uz']

        u = [ux, uy, uz]

        outputs['CU'] = ux**2 + uy**2 + uz**2 - 1.

    def compute_partials(self,inputs,partials):

        ux = inputs['ux'].flatten()
        uy = inputs['uy'].flatten()
        uz = inputs['uz'].flatten()

        partials['CU','ux'] = 2 * ux
        partials['CU','uy'] = 2 * uy
        partials['CU','uz'] = 2 * uz
