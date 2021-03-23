import numpy as np
from numpy import linalg
from openmdao.api import ExplicitComponent

class OrbitalConditionThree(ExplicitComponent):

    def initialize(self):
        self.options.declare('inc', default=1., types=float)

    def setup(self):
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

        #outputs['C3'] = np.dot(z,rxV) - linalg.norm(rxV,2)*np.cos(inc)
        outputs['C3'] = rx*Vy - ry*Vx - np.cos(inc)*((rx*Vy - ry*Vx)**2 \
            + (rx*Vz - rz*Vx)**2 + (ry*Vz - rz*Vy)**2)**(0.5)

    def compute_partials(self,inputs,partials):
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

        norm_rxV = np.sqrt((ry*Vz-rz*Vy)**2 + (rz*Vx-rx*Vz)**2 + (rx*Vy-ry*Vx)**2)

        C = np.cos(inc)

        du_drx =-C*(2*Vy*(rx*Vy-ry*Vx) + 2*Vz*(rx*Vz-rz*Vx))
        du_dry = C*(2*Vx*(rx*Vy-ry*Vx) - 2*Vz*(ry*Vz-rz*Vy))
        du_drz = C*(2*Vx*(rx*Vz-rz*Vx) + 2*Vy*(ry*Vz-rz*Vy))
        du_dVx = C*(2*ry*(rx*Vy-ry*Vx) + 2*rz*(rx*Vz-rz*Vx))
        du_dVy =-C*(2*rx*(rx*Vy-ry*Vx) - 2*rz*(ry*Vz-rz*Vy))
        du_dVz =-C*(2*rx*(rx*Vz-rz*Vx) + 2*ry*(ry*Vz-rz*Vy))

        den = 2*(((rx*Vy - ry*Vx)**2 + (rx*Vz - rz*Vx)**2 + (ry*Vz - rz*Vy)**2)**(1/2))

        partials['C3','rx'] = du_drx/den + Vy
        partials['C3','ry'] = du_dry/den - Vx
        partials['C3','rz'] = du_drz/den
        partials['C3','Vx'] = du_dVx/den - ry
        partials['C3','Vy'] = du_dVy/den + rx
        partials['C3','Vz'] = du_dVz/den
