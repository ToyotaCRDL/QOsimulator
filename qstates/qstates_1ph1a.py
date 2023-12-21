import copy
import numpy as np
from numpy import sqrt, pi, exp, cos, sin

class QStates_1ph1A():
    '''
    Quantum states of single photon system with corresponding two-level atoms.
    '''

    def __init__(self, qosim_1ph):
        self.N = qosim_1ph.N
        self.ph  = Qstates("ph", self.N, qosim_1ph)
        self.A   = Qstates("A", self.N, qosim_1ph )

    def norm(self):
        prob_ph = self.ph.probability()
        prob_A = self.A.probability()
        return prob_ph+prob_A

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
                for key in state2.r.keys():
                    c1  =state1.r[key]
                    c2  =state2.r[key]
                    _value = _value * np.sum(c1.conjugate()*c2)
                value += _value
        return value

    def __init__(self, name, N, *args):
        self.name = name
        self.N = N
        self.states = []
        state = QState(name, N, *args)
        self.states = [state]
        return

    def probability(self):
        prob = Qstates.innerproduct(self, self).real
        return prob

    def normalize(self):
        norm = sqrt(self.probability())
        coef = 1/norm
        normalized_states = []
        for state in self.states:
            keys = state.r.keys()
            dict_basis = copy.deepcopy(state.r)
            for key in keys:
                coef_normalize = coef**(1/len(keys))
                dict_basis[key] = coef_normalize*dict_basis[key]
            state.set(dict_basis, "r")
            normalized_states.append(state)
        qstates = copy.deepcopy(self)
        qstates.states = normalized_states
        return qstates

    def partial_density(self, _in):
        '''

        Parameters
        ----------
        _in: str
            "k","r"
        
        Returns
        -------
        density: np array
            if calc_pol == True,  density=np.zeros((2,self.N),  dtype = 'complex_')
            if calc_pol == False, density=np.zeros(self.N,  dtype = 'complex_')
        '''
        state = self.states[0]
        if _in == "r":
            cc = state.r["1"]
        elif _in == "k":
            cc = state.k["1"]
        else:
            raise Exception("Error: invalid input parameter _in", _in)
        density = np.abs(cc)**2
        return density

###------------------------------------STATE------------------------------###
class QState():
    '''

    Attributes
    ----------
    k: dict of numpy array
        example: {"1":c(k)}
    r: dict of numpy array
        {"1":c(r)}
        , where c is an array of the coefficients
    Methods
    -------
    set(dict of numpy array, _in)
        set "dict of numpy array" as same as the attributes, and updates other those in bases
    probabilites()
        return the probailites of the state.
    '''
    def __init__(self, name, N, *args):
        self.name = name
        self.N = N
        self.angle = 0.0
        if self.name == "ph":
            self.set(self._calc_phk_t0(*args), _in="k")
        elif self.name == "A":
            self.set(self._calc_A_t0(*args), _in="r")
        elif self.name == "manual":
            dict_basis = args[0]
            _in = args[1]
            self.set(dict_basis, _in)
        return

    def set(self, dict_basis, _in):
        '''
        _in: str
            Basis of dict_bases. "k", "r"
        '''
        if _in == "k":
            self.k = dict_basis
            c_r = self.to_rbases(self.k["1"])
            self.r = {"1":c_r}
        elif _in == "r":
            self.r = dict_basis
            c_k = self.to_kbases(self.r["1"])
            self.k = {"1":c_k}

    def probabilites(self):
        '''
        Returns
        -------
        probs: dict
            example, probs["1"] is sum of c1
        '''
        probs = {}
        for key, cc in self.k.items():
            probs[key] = np.sum(np.abs(cc.flatten())**2)
        return probs

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

    def _calc_phk_t0(self, qosim_1ph):
        '''
        '''
        ck_t0 = qosim_1ph.calc_ck_t0()
        dict_basis = {"1":ck_t0}
        return dict_basis

    def _calc_A_t0(self, qosim_1ph):
        '''
        '''
        cj_t0 = qosim_1ph.calc_cj_t0() # np.zeros(self.N,  dtype = 'complex_')
        dict_basis = {"1":cj_t0}
        return dict_basis
