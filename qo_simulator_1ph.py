import copy
import numpy as np
from numpy import sqrt, pi, exp, cos, sin
import matplotlib.pyplot as plt
##
from lib._qo_simulator import _QOsimulator
from qstates.qstates_1ph1a import QStates_1ph1A

class QOsimulator_1ph():
    '''
    One photon simulations
    '''
    def __init__(self,N, L, dt, beam, objects=[], calc_pol=False):
        '''
        Parameters
        ----------
        N: tuple of int
            N[0] and N[1] indicate the number of grid in the x and y directions, respectively.
        L: tuple of float
            Size of space in the x and y dicrections
        dt: float
            time step
        beam: QObeams object
        objects: list or QOobjects object
            Its element is QOobjects object
        calc_pol: bool
            True for calculation of polarization dof.
        '''
        if isinstance(objects, list):
            pass
        else:
            objects = [objects]

        self.N = N
        self.L = L
        self.dt = dt
        self.beam = beam
        self.objects=objects
        self.calc_pol = calc_pol
        self.qosim_1ph=_QOsimulator(N, L, dt, beam=beam, objects=objects, calc_pol=calc_pol)
        self.fig = self.qosim_1ph.fig
        return

    def gen_initial_state(self):
        '''
        Returns
        -------
        init_phi: QStates_1ph1A object
        '''        
        init_phi = QStates_1ph1A(self.qosim_1ph)
        return init_phi

    def suzuki_trotter_step(self, phi):
        '''
        Parameters
        ----------
        phi: QStates_1ph1A object
        
        Returns
        -------
        phi: QStates_1ph1A object
        '''        
        cr = phi.ph.states[0].r["1"]
        cj = phi.A.states[0].r["1"]
        cr, cj, _ = self.qosim_1ph.suzuki_trotter_step(cr, cj, degree=2, normalization=False)
        ## Update phi
        phi.ph.states[0].set({"1": cr}, _in = "r")
        phi.A.states[0].set({"1": cj}, _in = "r")
        return phi

    def calc_EF(self, phi):
        ck = phi.ph.states[0].k["1"]
        ef = self.qosim_1ph.calc_EF(ck)
        return ef

    def calc_EA(self, phi):
        cj = phi.A.states[0].r["1"]
        ea = self.qosim_1ph.calc_EA(cj)
        return ea

    def calc_EI(self, phi):
        cr = phi.ph.states[0].r["1"]
        cj = phi.A.states[0].r["1"]
        ei = self.qosim_1ph.calc_EI(cr, cj)
        return ei

    def calc_Econst(self):
        econst = self.qosim_1ph.calc_Econst()
        return econst

    def _set_permutation_term(self, phi):
        phi.ph.states = [phi.ph.states[0]]
        phi.A.states = [phi.A.states[0]]
        return

    def show(self, field, t, **kwargs):
        return self.qosim_1ph.show(field, t, **kwargs)
