import copy
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt

class QStates_2ph2A():
    '''
    Quantum states of two photon system with corresponding two-level atoms.
    '''

    def __init__(self, qosim_1ph_a, qosim_1ph_b, entanglement):
        '''
        entanglement: str or None
            None: Independent two photons
            "Phi+": (|theta_pol_a, theta_pol_a> + |theta_pol_b, theta_pol_b>)/sqrt(2)
            "Phi-": (|theta_pol_a, theta_pol_a> - |theta_pol_b, theta_pol_b>)/sqrt(2)
            "Psi+": (|theta_pol_a, theta_pol_b> + |theta_pol_b, theta_pol_a>)/sqrt(2)
            "Psi-": (|theta_pol_a, theta_pol_b> - |theta_pol_b, theta_pol_a>)/sqrt(2)
        '''
        self.N = qosim_1ph_a.N
        ## check
        correct_entanglemnts = [None, "Phi+", "Phi-", "Psi+", "Psi-"]
        if entanglement in correct_entanglemnts:
            pass
        else:
            raise ValueError("entanglement", entanglement, "is not in", correct_entanglemnts)
        self.entanglement=entanglement

        self.phph  = Qstates("2ph", self.N, qosim_1ph_a, qosim_1ph_b, entanglement)
        self.Aph = Qstates("Aph", self.N, qosim_1ph_a, qosim_1ph_b, entanglement)
        self.phA = Qstates("phA", self.N, qosim_1ph_a, qosim_1ph_b, entanglement)
        self.AA   = Qstates("AA", self.N, qosim_1ph_a, qosim_1ph_b, entanglement)
        
    def norm(self):
        prob_2ph = self.phph.probability()
        prob_Aph = self.Aph.probability()
        prob_phA = self.phA.probability()
        prob_AA = self.AA.probability()
        return prob_2ph+prob_Aph+prob_phA+prob_AA
    
    
class Qstates():
    '''
    Attributes
    ----------
    states: list
        It indicates a linear compination of states.
        For example states=[state1, state2] referes to state1 + state2
    '''
    
    @classmethod
    def innerproduct(cls, qstates1, qstates2):
        value = 0.0
        for state1 in qstates1.states:
            for state2 in qstates2.states:
                _value  = 1.0
                for key in state2.rr.keys():
                    c1  =state1.rr[key]
                    c2  =state2.rr[key]
                    _value = _value * np.sum(c1.conjugate()*c2)
                value += _value
        return value
    
    def __init__(self, name, N, *args):
        self.name = name
        self.N = N
        self.states = []
        ### Put permutation symmetry
        if name == "manual":
            dict_cs=args[0]
            _in = args[1]
            for dict_c in dict_cs:
                state = QState(name, N, dict_c, _in)
                self.states.append(state)
        else:
            qosim_1ph_a, qosim_1ph_b = args[0], args[1]
            coef = 1.0
            self.entanglement = args[2]
            if self.entanglement is None:
                state = QState(name, N, qosim_1ph_a, qosim_1ph_b, coef)
                self.states = [state, state.perumuatated_state()]
            else:
                norm_coef = 2.0**(-1/4)
                coef1 = norm_coef
                if "-" in self.entanglement:
                    coef2 = -1.0*norm_coef
                else:
                    coef2=norm_coef
                if "Phi" in self.entanglement:
                    _qosim_1ph_b = copy.deepcopy(qosim_1ph_b)
                    _qosim_1ph_b.beam.theta_pol = qosim_1ph_a.beam.theta_pol
                    state1 = QState(name, N, qosim_1ph_a, _qosim_1ph_b, coef1)
                    _qosim_1ph_a = copy.deepcopy(qosim_1ph_a)
                    _qosim_1ph_a.beam.theta_pol = qosim_1ph_b.beam.theta_pol
                    state2 = QState(name, N, _qosim_1ph_a, qosim_1ph_b, coef2)
                elif "Psi" in self.entanglement:
                    state1 = QState(name, N, qosim_1ph_a, qosim_1ph_b, coef1)
                    _qosim_1ph_b = copy.deepcopy(qosim_1ph_b)
                    _qosim_1ph_b.beam.theta_pol = qosim_1ph_a.beam.theta_pol
                    _qosim_1ph_a = copy.deepcopy(qosim_1ph_a)
                    _qosim_1ph_a.beam.theta_pol = qosim_1ph_b.beam.theta_pol
                    state2 = QState(name, N, _qosim_1ph_a, _qosim_1ph_b, coef2)                    
                self.states = [state1, state2, state1.permutated_state(), state2.permutated_state()]
        self.num_linear_combinations = len(self.states)
        return
       
    def probability(self):
        prob = Qstates.innerproduct(self, self).real
        return prob
    
    def normalize(self):
        norm = sqrt(self.probability()) + 1.e-15
        coef = 1/norm
        normalized_states = []
        for state in self.states:
            keys = state.rr.keys()
            dict_basis = copy.deepcopy(state.rr)
            for key in keys:
                coef_normalize = coef**(1/len(keys))
                dict_basis[key] = coef_normalize*dict_basis[key]
            state.set(dict_basis, "rr")
            normalized_states.append(state)
        qstates = copy.deepcopy(self)
        qstates.states = normalized_states
        return qstates
    
    def partial_density(self, _in):
        '''
        |a>|b> + |b>|a> -> |a|^2 + |b|^2

        Parameters
        ----------
        _in: str
            "kk", "rk", "kr", "rr"
            
        Returns
        -------
        density: np array
            if calc_pol == True,  density=np.zeros((2,self.N),  dtype = 'complex_')
            if calc_pol == False, density=np.zeros(self.N,  dtype = 'complex_')

        '''
        target_key = "1" # dummy parameter
        density = np.zeros(self.states[0].rr[target_key].shape,  dtype = 'complex_')
        for state in self.states:
            cc = self._marginalization(state, target_key, _in)
            density += np.abs(cc)**2
        density = density / (self.num_linear_combinations/2.)
        return density
    
    def _marginalization(self, state, target_key, _in) :
        if _in == "kk":
            dict_c = copy.deepcopy(state.kk)
        elif _in == "rk":
            dict_c = copy.deepcopy(state.rk)
        elif _in == "kr":
            dict_c = copy.deepcopy(state.kr)
        elif _in == "rr":
            dict_c = copy.deepcopy(state.rr)
        else:
            raise Exception("Error: invalid input parameter _in", _in)
        keys = list(dict_c.keys()) ## ["1", "2"]
        # basis
        for key in keys:
            dict_c[key] = dict_c[key]*(self.num_linear_combinations**(1/4))
        # erase target_key from keys
        keys = [item for item in keys if item != target_key]  
        coef = 1.0
        for _key in keys:
            cc = dict_c[_key]
            coef = coef*np.sum(np.abs(cc)**2)
        cc = coef*dict_c[target_key]
        return cc
        

    
###------------------------------------STATE------------------------------###
class QState():
    '''
    
    Attributes
    ----------
    kk: dict of numpy array
        example: {"1":c(k), "2":c(k)} that indidicates c1(k)c2(k)
    rk: dict of numpy array
        {"1":c(r), "2":c(k)}
    kr: dict of numpy array
        {"1":c(k), "2":c(r)}
    rr: dict of numpy array
        {"1":c(r), "2":c(r)}
        , where c is an array of the coefficients
    Methods
    -------
    set(dict of numpy array, _in)
        set "dict of numpy array" as same as the attributes, and updates other those in bases
    probabilites()
        return the probailites of the state.
    perumuatated_state()
        return the permutated state
    '''
    def __init__(self, name, N, *args):
        self.name = name
        self.N = N
        if self.name == "2ph":
            self.set(self._calc_2phk_t0(*args), _in="kk")
        elif self.name=="Aph": 
            self.set(self._calc_Aphk_t0(*args), _in="rk")
        elif self.name == "phA":
            self.set(self._calc_phkA_t0(*args), _in="kr")
        elif self.name == "AA":
            self.set(self._calc_2A_t0(*args), _in="rr")
        elif self.name == "manual":
            dict_basis = args[0]
            _in = args[1]
            self.set(dict_basis, _in)
        return    
    
    def set(self, dict_basis, _in):
        '''
        _in: str
            Basis of dict_bases. "kk", "rk", "kr", "rr" 
        '''
        if _in == "kk":
            self.kk = dict_basis
            c_r1 = self.to_rbases(self.kk["1"])
            c_r2 = self.to_rbases(self.kk["2"])
            self.rr = {"1":c_r1, "2": c_r2}
            self.kr = {"1":self.kk["1"], "2": c_r2}
            self.rk = {"1":c_r1, "2": self.kk["2"]}
        elif _in == "rk":
            self.rk = dict_basis
            c_k1 = self.to_kbases(self.rk["1"])
            c_r2 = self.to_rbases(self.rk["2"])
            self.rr = {"1":self.rk["1"], "2": c_r2}
            self.kr = {"1":c_k1, "2": c_r2}
            self.kk = {"1":c_k1, "2": self.rk["2"]}
        elif _in == "kr":
            self.kr = dict_basis
            c_r1 = self.to_rbases(self.kr["1"])
            c_k2 = self.to_kbases(self.kr["2"])
            self.rr = {"1":c_r1, "2": self.kr["2"]}
            self.rk = {"1":c_r1, "2": c_k2}
            self.kk = {"1":self.kr["1"], "2": c_k2}
        elif _in == "rr":
            self.rr = dict_basis
            c_k1 = self.to_kbases(self.rr["1"])
            c_k2 = self.to_kbases(self.rr["2"])
            self.kk = {"1":c_k1, "2": c_k2}
            self.kr = {"1":c_k1, "2": self.rr["2"]}
            self.rk = {"1":self.rr["1"], "2": c_k2}
    
    def probabilites(self):
        '''
        Returns
        -------
        probs: dict
            example, probs["1"] is sum of c1
        '''
        probs = {}
        for key, cc in self.kk.items():
            probs[key] = np.sum(np.abs(cc.flatten())**2)
        return probs
    
    def permutated_state(self):
        _state = copy.deepcopy(self)
        _state.kk = {"1":self.kk["2"], "2":self.kk["1"]}
        _state.rr = {"1":self.rr["2"], "2":self.rr["1"]}
        _state.kr = {"1":self.kr["2"], "2":self.kr["1"]}
        _state.rk = {"1":self.rk["2"], "2":self.rk["1"]}
        return _state
    
    def to_rbases(self,ck):
        '''
        2D inverse FFT: ck -> cr
        '''
        cr= np.fft.ifft2(ck, norm="ortho")
        return cr

    def to_kbases(self,cr):
        '''
        2D FFT: cr -> ck
        '''
        ck = np.fft.fft2(cr, norm="ortho")
        return ck

    def _calc_2phk_t0(self, qosim_1ph_a, qosim_1ph_b, coef=1.0):
        '''
        '''
        cak_t0 = qosim_1ph_a.calc_ck_t0()
        cbk_t0 = qosim_1ph_b.calc_ck_t0()
        coef = coef*2.0**(-1/4)
        dict_basis = {"1":coef*cak_t0, "2":np.abs(coef)*cbk_t0} # |phi_{1ph}^a>|phi_{1ph}^b>
        return dict_basis

    def _calc_Aphk_t0(self, qosim_1ph_a, qosim_1ph_b, coef=1.0):
        '''
        '''
        cj_t0=qosim_1ph_a.calc_cj_t0() # np.zeros(self.N,  dtype = 'complex_')
        ck_t0=qosim_1ph_b.calc_ck_t0()
        dict_basis = {"1":coef*cj_t0, "2":coef*ck_t0} # |A>|phi_1ph>
        return dict_basis


    def _calc_phkA_t0(self, qosim_1ph_a, qosim_1ph_b, coef=1.0):
        '''
        '''
        ck_t0=qosim_1ph_a.calc_ck_t0()
        cj_t0=qosim_1ph_b.calc_cj_t0() #np.zeros(self.N,  dtype = 'complex_')
        dict_basis = {"1":coef*ck_t0, "2":coef*cj_t0}  # |phi_1ph>|A>
        return dict_basis

    def _calc_2A_t0(self, qosim_1ph_a, qosim_1ph_b, coef=1.0):
        '''
        '''
        cj_t0_a= qosim_1ph_a.calc_cj_t0() #np.zeros(self.N,  dtype = 'complex_')
        cj_t0_b= qosim_1ph_b.calc_cj_t0() #np.zeros(self.N,  dtype = 'complex_')
        dict_basis = {"1":coef*cj_t0_a, "2":coef*cj_t0_b}  # |A>|A>
        return dict_basis

    



