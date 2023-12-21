import copy
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt
##
from lib._qo_simulator import _QOsimulator
from qstates.qstates_2ph2a import QStates_2ph2A

class QOsimulator_2ph():
    '''
    Two photon simulations in fast version (variable separation)
    '''
    def __init__(self,N, L, dt, beams, objects=[], calc_pol=False, entanglement=None):
        '''
        Parameters
        ----------
        N: tuple of int
            N[0] and N[1] indicate the number of grid in the x and y directions, respectively.
        L: tuple of float
            Size of space in the x and y dicrections
        dt: float
            time step
        beams: list or QObeams object
        objects: list or QOobjects object
            Its element is QOobjects object
        calc_pol: bool
            True for calculation of polarization dof.
        entanglement: str or None
            None: Independent two photons
            "Phi+": (|theta_pol_a, theta_pol_a> + |theta_pol_b, theta_pol_b>)/sqrt(2)
            "Phi-": (|theta_pol_a, theta_pol_a> - |theta_pol_b, theta_pol_b>)/sqrt(2)
            "Psi+": (|theta_pol_a, theta_pol_b> + |theta_pol_b, theta_pol_a>)/sqrt(2)
            "Psi-": (|theta_pol_a, theta_pol_b> - |theta_pol_b, theta_pol_a>)/sqrt(2)
        '''
        if isinstance(objects, list):
            pass
        else:
            objects = [objects]
        
        if len(beams) != 2:
            raise ValueError("Number of beams should be 2 for 2photons system.")
        
        self.N = N
        self.L = L
        self.dt = dt
        self.beams = beams
        self.objects=objects
        self.calc_pol = calc_pol
        if entanglement is None:
            pass
        else:
            if  self.calc_pol == False:
                raise ValueError("Error: calc_pol should be True, if entanglement is turned on.")
            print("entanglement=",entanglement)
        self.entanglement = entanglement
        beam_a = self.beams[0]
        beam_b = self.beams[1]
        self.qosim_1ph_a=_QOsimulator(N, L, dt, beam = beam_a, objects=objects, calc_pol=calc_pol)
        self.qosim_1ph_b=_QOsimulator(N, L, dt, beam = beam_b, objects=objects, calc_pol=calc_pol)
        self.fig = self.qosim_1ph_b.fig
        return
    
    def gen_initial_state(self):
        '''
        Returns
        -------
        init_phi: QStates_1ph1A object
        '''        
        init_phi = QStates_2ph2A(self.qosim_1ph_a, self.qosim_1ph_b, entanglement=self.entanglement)
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
        for ii in self._get_list_range(phi):  
            phi = self._suzuki_trotter_step(phi, ii)
        phi = self._set_permutation_term(phi)
        return phi

    def _suzuki_trotter_step(self, phi, ii):
        cr_a = phi.phph.states[ii].rr["1"]
        cj_a = phi.AA.states[ii].rr["1"]
        cr_a, cj_a, _ = self.qosim_1ph_a.suzuki_trotter_step(cr_a, cj_a, degree=2, normalization=False)
        ##
        cr_b = phi.phph.states[ii].rr["2"]
        cj_b = phi.AA.states[ii].rr["2"]
        cr_b, cj_b, _ = self.qosim_1ph_b.suzuki_trotter_step(cr_b, cj_b, degree=2, normalization=False)
        ## Update phi
        phi.phph.states[ii].set({"1": cr_a, "2": cr_b}, _in = "rr")
        phi.Aph.states[ii].set( {"1": cj_a, "2": cr_b}, _in = "rr")
        phi.phA.states[ii].set( {"1": cr_a, "2": cj_b}, _in = "rr")
        phi.AA.states[ii].set(  {"1": cj_a, "2": cj_b}, _in = "rr")
        return phi
    
    def calc_Econst(self):
        econst = self.qosim_1ph_a.calc_Econst() + self.qosim_1ph_b.calc_Econst()
        return econst

    def calc_EF(self, phi):
        ef = self._calc_energy(phi, self._calc_EF)
        return ef

    def calc_EA(self, phi):
        ea = self._calc_energy(phi, self._calc_EA)
        return ea

    def calc_EI(self, phi):
        ei = self._calc_energy(phi, self._calc_EI)
        return ei

    def calc_EP(self, phi):
        ep = self._calc_energy(phi, self._calc_EP)
        return ep

    def _calc_energy(self, phi, func):
        energy = 0.0
        for ii in self._get_list_range(phi):
            energy += func(phi, ii)
        energy = 2.0*energy # <= double is caused by permutated state
        return energy
        
    def _calc_EF(self, phi, ii):
        qstate = phi.phph.states[ii]
        probs=qstate.probabilites()
        ef_a = self.qosim_1ph_a.calc_EF(qstate.kk["1"])*probs["2"]
        ef_b = self.qosim_1ph_b.calc_EF(qstate.kk["2"])*probs["1"]
        return ef_a + ef_b
                
    def _calc_EA(self, phi, ii):
        qstate = phi.AA.states[ii]
        probs=qstate.probabilites()
        ea_a = self.qosim_1ph_a.calc_EA(qstate.rr["1"])*probs["2"]
        ea_b = self.qosim_1ph_b.calc_EA(qstate.rr["2"])*probs["1"]
        return ea_a + ea_b

    def _calc_EI(self, phi, ii):
        qstate_phph = phi.phph.states[ii]
        qstate_AA = phi.AA.states[ii]
        probs_phph=qstate_phph.probabilites()
        probs_AA=qstate_AA.probabilites()
        tmp1 = self.qosim_1ph_a.calc_EI(qstate_phph.rr["1"], qstate_AA.rr["1"])
        tmp2 = self.qosim_1ph_a.calc_EI(qstate_phph.rr["2"], qstate_AA.rr["2"])
        ei = 0.0
        ei += tmp1*(probs_phph["2"]+probs_AA["2"]) 
        ei += (probs_phph["1"]+probs_AA["1"])*tmp2
        return ei
               
    def _set_permutation_term(self, phi):
        list_range = self._get_list_range(phi)
        def _exec(states):
            _states = [ states[ii] for ii in list_range]
            _states = _states + [ _state.permutated_state() for _state in _states]
            return _states
        phi.phph.states = _exec(phi.phph.states)
        phi.Aph.states = _exec(phi.Aph.states)
        phi.phA.states = _exec(phi.phA.states)
        phi.AA.states = _exec(phi.AA.states)
        return phi
    
    def _get_list_range(self, phi):
        num_linear_combinations = int(len(phi.phph.states)) # entangled case: 4, product state case: 2
        num_linear_combinations = int(num_linear_combinations/2) # exclude permutation part
        list_range = list(range(num_linear_combinations)) # entangled case: [0,1], product state case: [0]
        return list_range

        
    def show(self, field, t, **kwargs):
        return self.qosim_1ph_a.show(field, t, **kwargs)

        

        
