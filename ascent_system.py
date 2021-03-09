import numpy as np
from openmdao.api import ExplicitComponent

class AscentSystem(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('R0', default=1., types=(int, float))
        self.options.declare('g0', default=1., types=(int, float))
        self.options.declare('vex', default=1., types=(int, float))
        self.options.declare('w2', default=1., types=(int, float))
        self.options.declare('T', default=1., types=(int, float))

    def setup(self):
        num = self.options['num_nodes']
        g0 = self.options['g0']
        R0 = self.options['R0']
        vex = self.options['vex']
        w2 = self.options['w2']
        T = self.options['T']

        self.add_input('ux', shape=(num,1))
        self.add_input('uy', shape=(num,1))
        self.add_input('uz', shape=(num,1))
        self.add_input('rx', shape=(num,1))
        self.add_input('ry', shape=(num,1))
        self.add_input('rz', shape=(num,1))
        self.add_input('Vx', shape=(num,1))
        self.add_input('Vy', shape=(num,1))
        self.add_input('Vz', shape=(num,1))
        self.add_input('m', shape=(num,1))

        self.add_output('drx_dt', shape=(num,1))
        self.add_output('dry_dt', shape=(num,1))
        self.add_output('drz_dt', shape=(num,1))
        self.add_output('dVx_dt', shape=(num,1))
        self.add_output('dVy_dt', shape=(num,1))
        self.add_output('dVz_dt', shape=(num,1))
        self.add_output('dm_dt', shape=(num,1))

        self.declare_partials('*','*', dependent=False)

        self.declare_partials('drx_dt', 'Vx', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dry_dt', 'Vy', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('drz_dt', 'Vz', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVx_dt', 'rx', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVy_dt', 'ry', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVz_dt', 'rz', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVx_dt', 'ux', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVy_dt', 'uy', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVz_dt', 'uz', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVx_dt', 'm', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVy_dt', 'm', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dVz_dt', 'm', rows=np.arange(num), cols=np.arange(num))

    def compute(self, inputs, outputs):
        g0 = self.options['g0']
        R0 = self.options['R0']
        vex = self.options['vex']
        w2 = self.options['w2']
        T = self.options['T']

        outputs['drx_dt'] = inputs['Vx']
        outputs['dry_dt'] = inputs['Vy']
        outputs['drz_dt'] = inputs['Vz']
        outputs['dVx_dt'] = - w2*inputs['rx'] + (T * inputs['ux']) / (inputs['m'] * g0)
        outputs['dVy_dt'] = - w2*inputs['ry'] + (T * inputs['uy']) / (inputs['m'] * g0)
        outputs['dVz_dt'] = - w2*inputs['rz'] + (T * inputs['uz']) / (inputs['m'] * g0)
        outputs['dm_dt'] = - (T / vex) * np.sqrt(R0/g0)

    def compute_partials(self, inputs, partials):
        g0 = self.options['g0']
        T = self.options['T']
        w2 = self.options['w2']

        ux = inputs['ux'][:, 0]
        uy = inputs['uy'][:, 0]
        uz = inputs['uz'][:, 0]
        m = inputs['m'][:, 0]

        # specify values of the Jacobian
        partials['drx_dt', 'Vx'] = 1.
        partials['dry_dt', 'Vy'] = 1.
        partials['drz_dt', 'Vz'] = 1.
        partials['dVx_dt', 'rx'] = - w2
        partials['dVy_dt', 'ry'] = - w2
        partials['dVz_dt', 'rz'] = - w2
        partials['dVx_dt', 'ux'] = T / (m * g0)
        partials['dVy_dt', 'uy'] = T / (m * g0)
        partials['dVz_dt', 'uz'] = T / (m * g0)
        partials['dVx_dt', 'm'] = - (T * ux) / (m**2 * g0)
        partials['dVy_dt', 'm'] = - (T * uy) / (m**2 * g0)
        partials['dVz_dt', 'm'] = - (T * uz) / (m**2 * g0)
