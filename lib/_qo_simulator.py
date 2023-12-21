import copy
import numpy as np
from numpy import sqrt, pi, exp, cos, sin
import matplotlib.pyplot as plt

class _QOsimulator():
    def __init__(self,N, L, dt, beam, objects=[], calc_pol=False, make_fig=True):
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
        objects: list of QOobjects object
            Its element is QOobjects object
        calc_pol: bool
            If True, the calculation includes polarization degree of freedom.
        make_fig: bool
            If False, no figure is plotted.
        '''
        if isinstance(objects, list):
            pass
        else:
            objects = [objects]

        self.N = N
        self.L = L
        self.dt = dt
        self.beam = beam
        ## check duplicated atomic locations in objects
        ll =[]
        for _obj in objects:
            ll +=  [list(elem) for elem in list(_obj.atom_pos_indices)]
        def has_duplicates(ll):
            seen = []
            unique_list = [x for x in ll if x not in seen and not seen.append(x)]
            return len(ll) != len(unique_list)
        if has_duplicates(ll):
            raise ValueError("Duplicated atomic locations in objects.")
        ##
        self.objects=objects
        self.calc_pol=calc_pol
        ##
        def set_kvec(num, length):
            kmax = num*pi/length
            kk = np.linspace(0, 2*pi*(num-1)/length, num) ## => [0.  0.2 0.4] if N=4
            newkk = []
            for elem in kk: # Impose periodicity
                if elem <= kmax:
                    newkk.append(elem)
                else:
                    newkk.append(elem-2*kmax)
            return newkk ## => [0.0, 0.2, -0.2]
        self.kx = set_kvec(N[0], L[0])
        self.ky = set_kvec(N[1], L[1])
        self.x = np.linspace(0, L[0], N[0])
        self.y = np.linspace(0, L[1], N[1])
        self.omega_k = np.zeros(N)
        for i in range(N[0]):
            for j in range(N[1]):
                self.omega_k[i,j] = np.linalg.norm(np.array([self.kx[i], self.ky[j]]))
        self.HF = self.get_HF_k()
        self.HA = self.get_HA()
        ## show
        if make_fig==True:
            self.fig_coef = self.N[0]/self.N[1]
            fig = plt.figure(figsize=(int(5*self.fig_coef),5))
            ax1 = fig.add_subplot(1, 1, 1)
            self.fig = fig
            self.ax1 = ax1

        return

    def calc_ck_t0(self):
        '''
        Get ck(t=0) of the photon beam

        Returns
        -----------
        ck_t0: 2d numpy array or (2, N[0], N[1]) numpy array
            c(kx, ky). If calc_pol==True, [c_H(kx, ky), c_V(kx, ky)]
        '''
        ck_t0 = self.beam.gen_photon_ck(self.kx, self.ky)
        if self.calc_pol==True:
            theta = self.beam.theta_pol
            cH = cos(theta) # coefficient of the Horizontal polarization
            cV = sin(theta) # coefficient of the Vertical polarization
            tmp = copy.deepcopy(ck_t0)
            ck_t0 = np.array([cH*tmp, cV*tmp])
        return ck_t0

    def calc_cj_t0(self):
        '''
        Get cj(t=0) in which the constituent spins are all down.

        Returns
        -----------
        cj_t0: 2d numpy array or (2, N[0], N[1]) numpy array
            c(x, y). If calc_pol==True, [c(x, y), c(x, y)].
        '''
        N = self.N
        cj_t0 = np.zeros(N,dtype = 'complex_')
        for _obj in self.objects:
            for idx in _obj.atom_pos_indices:
                cj_t0[idx[0], idx[1]] = 0.0
        if self.calc_pol==True:
            cj_t0 = np.array([cj_t0, cj_t0])
        return cj_t0

    def get_HF_k(self,):
        '''
        Get Hamiltonian of photon in free space, represented in k space

        Returns
        -----------
        HF: 2d numpy array
            H_F(kx, ky)
        '''
        N = self.N
        HF = np.zeros(N,dtype = 'complex_')
        for i in range(N[0]):
            for j in range(N[1]):
                HF[i, j] = self.omega_k[i, j]
        return HF
    
    def get_HA(self,):
        '''
        Get Hamiltonian of atom, represented in r space

        Returns
        -----------
        HA: 2d numpy array
            H_A(x, y)
        '''
        HA = np.zeros(self.N, dtype = 'complex_')
        for _obj in self.objects:
            omega_j = _obj.omega_j
            for idx in _obj.atom_pos_indices:
                HA[idx[0],idx[1]] = 2*omega_j
        return HA
    

    def by_H0_k(self, ck, cj):
        '''
        Multiply H0 (ck, cj), where H0 = HF + HA. HA is atomic Hamiltonian.

        Parameters
        -----------
        ck, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True

        Returns
        -----------
        _ck, _cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        '''
        _ck = self.HF *ck
        _cj = self.HA *cj
        return _ck, _cj

    def move_by_H0_k(self, ck, cj, dt):
        '''
        Time developement by exp(-i H0 dt) (ck, cj)

        Parameters
        -----------
        ck, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        dt: float

        Returns
        -----------
        _ck, _cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        '''
        _ck = exp(-1.0j * self.HF * dt) * ck
        _cj = exp(-1.0j * self.HA * dt) * cj
        return _ck, _cj

    def by_HI_r(self, cr, cj):
        '''
        Multiply HI (cr, cj)

        Parameters
        -----------
        cr, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True

        Returns
        -----------
        _cr, _cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        '''
        wjs = self._get_wjs()
        if self.calc_pol==True:
            _cr = np.zeros((2,*self.N),dtype = 'complex_')
            _cj = np.zeros((2,*self.N),dtype = 'complex_')
        else:
            _cr = np.zeros(self.N,dtype = 'complex_')
            _cj = np.zeros(self.N,dtype = 'complex_')                
        for iobj, _obj in enumerate(self.objects):
            wj = wjs[iobj]
            cwj = wj.conjugate()
            if self.calc_pol==True:
                _obj_name = _obj.__class__. __name__
                ## If "Poalrization rotator", basis transformation is performed depending on the rotation angle.
                if _obj_name == "PolarizationRotator" or hasattr(_obj, "opt_pr")==True:
                    cr, cj = self.rotate_pbases(cr,cj,_obj)
                ##
                for ipol in _obj.ipols:
                    for idx in _obj.atom_pos_indices:
                        _cj[ipol][idx[0], idx[1]] = wj *cr[ipol][idx[0], idx[1]]
                        _cr[ipol][idx[0], idx[1]] = cwj*cj[ipol][idx[0], idx[1]]
                ## Inverse transformation of the polarization basis
                if _obj_name == "PolarizationRotator" or hasattr(_obj, "opt_pr")==True:
                    cr, cj = self.rotate_pbases(cr,cj,_obj,inverse=True)
            else:
                for idx in _obj.atom_pos_indices:
                    _cj[idx[0], idx[1]] = wj *cr[idx[0], idx[1]]
                    _cr[idx[0], idx[1]] = cwj*cj[idx[0], idx[1]]
        return _cr, _cj

    def by_HI_k(self, ck, cj):
        '''
        Multiply HI (ck, cj)

        Parameters
        -----------
        ck, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True

        Returns
        -----------
        _ck, _cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        '''
        wjs = self._get_wjs()
        N=self.N
        ##----------------------------------------------
        T_r = np.fft.ifft2(ck, norm="forward")
        _cj = np.zeros(N,dtype = 'complex_')
        for iobj, _obj in enumerate(self.objects):
            fac = wjs[iobj]/sqrt(self.N[0]*self.N[1])
            for idx in _obj.atom_pos_indices:
                 _cj[idx[0],idx[1]] = fac * T_r[idx[0],idx[1]]
        ##---------------------------------------------
        _ck = fac.conjugate() * np.fft.fft2(cj)
        ##---------------------------------------------
        return _ck, _cj

    def move_by_HI_r(self, cr, cj, dt):
        '''
        Time developement by exp(-i HI dt) (cr, cj)

        Parameters
        -----------
        cr, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        dt: float

        Returns
        -----------
        cr, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        '''
        wjs = self._get_wjs()
        UI = np.array([[1.0, -1.0j], [1.0, 1.0j]])/(2.0)**0.5
        UI_inv = UI.transpose().conj()
        for iobj, _obj in enumerate(self.objects):
            wj = wjs[iobj]
            HI_eig = np.array([[-1.0j*wj, 0.0], [0.0, 1.0j*wj]])
            T_HI_eig = np.array([[exp(-1.0j*dt*HI_eig[0,0]), 0.0], [0.0, exp(-1.0j*dt*HI_eig[1,1]) ]])
            if self.calc_pol==True:
                _obj_name = _obj.__class__. __name__
                ## If "Poalrization rotator", basis transformation is performed depending on the rotation angle.
                if _obj_name == "PolarizationRotator" or hasattr(_obj, "opt_pr")==True:
                    cr, cj = self.rotate_pbases(cr,cj,_obj)
                ##
                for ipol in _obj.ipols:
                    cr[ipol], cj[ipol] = self._move_by_HI_r(cr[ipol], cj[ipol], _obj, T_HI_eig, UI, UI_inv)
                ## Inverse transformation of the polarization basis
                if _obj_name == "PolarizationRotator" or hasattr(_obj, "opt_pr")==True:
                    cr, cj = self.rotate_pbases(cr,cj,_obj,inverse=True)
                ####
            else:
                cr, cj = self._move_by_HI_r(cr, cj, _obj, T_HI_eig, UI, UI_inv)
        return cr, cj

    def _move_by_HI_r(self, cr, cj, _obj, T_HI_eig, UI, UI_inv):
        matrix = np.dot(T_HI_eig, UI)
        matrix = np.dot(UI_inv, matrix)
        for idx in _obj.atom_pos_indices:
            cvec = np.array( [[cr[idx[0], idx[1]]], [cj[idx[0], idx[1]]]] )
            cvec = np.dot(matrix, cvec)
            cr[idx[0], idx[1]] =  cvec[0][0]
            cj[idx[0], idx[1]]=  cvec[1][0]
        return cr, cj

    def move_by_HI_k_interaction_picture(self, ck, cj, dt):
        '''
        Time developement by exp(-i HI dt) (ck, cj) in the interaction picture.

        Parameters
        -----------
        ck, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        dt: float

        Returns
        -----------
        ck_new, cj_new: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        '''
        omega_k=self.omega_k
        wjs = self._get_wjs()
        N = self.N
        ##----------------------------------------------
        cj_new = np.zeros(N,dtype = 'complex_')
        ck_tmp = ck * exp(-1.0j * omega_k * dt)
        T_r = np.fft.ifft2(ck_tmp, norm="forward")
        omega_j = self.objects[0].omega_j
        for iobj, _obj in enumerate(self.objects):
            fac = wjs[iobj]/sqrt(self.N[0]*self.N[1])
            omega_j = _obj.omega_j
            for idx in _obj.atom_pos_indices:
                 cj_new[idx[0],idx[1]] = fac * T_r[idx[0],idx[1]] * exp(2.0j*omega_j*dt)
        ##---------------------------------------------
        cj_tmp = np.zeros(N,dtype = 'complex_')
        for iobj, _obj in enumerate(self.objects):
            fac = wjs[iobj]/sqrt(self.N[0]*self.N[1])
            omega_j = _obj.omega_j
            for idx in _obj.atom_pos_indices:
                cj_tmp[idx[0],idx[1]] =  fac.conjugate()*cj[idx[0],idx[1]]*exp(-2.0j*omega_j*dt)
        U_k = np.fft.fft2(cj_tmp)
        ck_new = U_k * exp(1.0j*omega_k*dt)
        return ck_new, cj_new

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
    
    def rotate_pbases(self, cr, cj, obj, inverse=False):
        '''
        Rotate polarization bases
        '''
        theta = 0.5 * obj.theta_rot + self.beam.theta_pol
        if inverse==True:
            theta = -1.0 * theta
        _cr = np.zeros_like(cr)
        _cj = np.zeros_like(cj)
        _cr[0] =  cos(theta)*cr[0] + sin(theta)*cr[1]
        _cr[1] = -sin(theta)*cr[0] + cos(theta)*cr[1]
        _cj[0] =  cos(theta)*cj[0] + sin(theta)*cj[1]
        _cj[1] = -sin(theta)*cj[0] + cos(theta)*cj[1]

        return _cr, _cj

    ##--------------------------------------------------------------------------
    def suzuki_trotter_step(self, cr, cj, degree=2, normalization=False):
        '''
        Time evolution by Suzuki-Trotter decomposition. HI is solved by r bases.

        Paramters
        ---------
        cr, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        degree: int
            Degree of method. 1, 2, 4 and 6 are allowed.
        normalization: bool
            If true, normalization of probability amplitude is performed.

        Returns
        -----------
        cr, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        "r": str
        '''

        if degree in [1,2,4,6]:
            pass
        else:
            raise ValueError("Invalid degree in symplectic_step. degree = ", degree)
        dt = self.dt
        if degree == 1:
            return self.trotter_step(cr, cj, degree=1)
        ##
        def _s2(cr, cj, tau):
            ###
            cr, cj = self.move_by_HI_r(cr, cj, 0.5*tau)
            ck = self.to_kbases(cr)
            ck, cj = self.move_by_H0_k(ck, cj, tau)
            cr = self.to_rbases(ck)
            cr, cj = self.move_by_HI_r(cr, cj, 0.5*tau)
            return cr, cj
        if degree == 2:
            cr, cj = _s2(cr, cj, dt)
        elif degree == 4:
            xx=1.0/(2.0-2.0**(1/3))
            cr, cj = _s2(cr, cj, xx*dt)
            cr, cj = _s2(cr, cj, (1.0-2*xx)*dt)
            cr, cj = _s2(cr, cj, xx*dt)
        elif degree == 6:
            y1=-1.17767998417887
            y2=0.235573213359357
            y3=0.784513610477560
            y0=1.0-2.0*(y1+y2+y3)
            cr, cj = _s2(cr, cj, y3*dt)
            cr, cj = _s2(cr, cj, y2*dt)
            cr, cj = _s2(cr, cj, y1*dt)
            cr, cj = _s2(cr, cj, y0*dt)
            cr, cj = _s2(cr, cj, y1*dt)
            cr, cj = _s2(cr, cj, y2*dt)
            cr, cj = _s2(cr, cj, y3*dt)
        if normalization == True:
            norm = np.sum(np.abs(cr.flatten())**2)
            norm += np.sum(np.abs(cj.flatten())**2)
            cr = cr/sqrt(norm)
            cj = cj/sqrt(norm)
        return cr, cj, "r"

    def runge_kutta_step_interaction_picture(self, ck, cj, t, degree=4, normalization=False):
        '''
        Time evolution by Runge-Kutta method. HI is written in the interaction picture.

        Paramters
        ---------
        cr, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        degree: int
            Degree of method. 1 and 4 are allowed.
        normalization: bool
            If true, normalization of probability amplitude is performed.

        Returns
        -----------
        ck, cj: 2d numpy array, or (2, N[0], N[1]) numpy array if calc_pol==True
        "k": str
        '''
        if degree in [1,4]:
            pass
        else:
            raise ValueError("Invalid degree in runge_kutta_step. degree = ", degree)

        if self.calc_pol==True:
            raise ValueError("Polarization calculation is not imeplemented in runge_kutta_step_interaction_picture.")

        dt = self.dt

        ck, cj = self.move_by_H0_k(ck, cj, -t) ## => To the interaction picrure

        def _rk_step(ck, cj, dt):
            _ck, _cj = self.move_by_HI_k_interaction_picture(ck, cj, t)
            _ck = -1.0j * dt * _ck
            _cj = -1.0j * dt * _cj
            return _ck, _cj

        k1_ck, k1_cj = _rk_step(ck, cj, dt)
        if degree ==1:
            ck = ck + k1_ck
            cj = cj + k1_cj
        elif degree == 4:
            k2_ck, k2_cj = _rk_step(ck+ 0.5*k1_ck, cj+ 0.5*k1_cj, dt)
            k3_ck, k3_cj = _rk_step(ck+ 0.5*k2_ck, cj+ 0.5*k2_cj, dt)
            k4_ck, k4_cj = _rk_step(ck+ 0.5*k3_ck, cj+ 0.5*k3_cj, dt)

            ck = ck + 1.0/6.0 * (k1_ck + 2.0*k2_ck + 2.0*k3_ck + k4_ck)
            cj = cj + 1.0/6.0 * (k1_cj + 2.0*k2_cj + 2.0*k3_cj + k4_cj)
        ck, cj = self.move_by_H0_k(ck, cj, t+self.dt) ## => Back from the interaction picture
        if normalization == True:
            norm = np.sum(np.abs(ck.flatten())**2)
            norm += np.sum(np.abs(cj.flatten())**2)
            ck = ck/sqrt(norm)
            cj = cj/sqrt(norm)
        return ck, cj, "k"

    ##----- info
    def show(self, field, t, **kwargs):
        ## Plot result
        x = np.linspace(0, self.L[0], self.N[0])
        y = np.linspace(0, self.L[1], self.N[1])
        plt.figure(figsize=(int(5*self.fig_coef),5))
        plt.xlabel("x", fontsize=18)
        plt.ylabel("y", fontsize=18)
        img=plt.contourf(x, y, field.T,**kwargs)
        # img=plt.contourf(x, y, field.T)
        plt.colorbar(img)
        plt.title("t = %.2f" % t, fontsize=18)
        ## for animation
        im = self.ax1.contourf(x, y, field.T,**kwargs)
        title = self.ax1.text(0.5, 1.01, 't = %.2f' % t,
                         ha='center', va='bottom',
                         transform=self.ax1.transAxes, fontsize=18)
        return im, title

    def calc_EF(self, ck):
        eF = self._innerproduct_coef(ck, self.HF *ck).real
        return eF

    def calc_EA(self,cj):
        eA = self._innerproduct_coef(cj, self.HA *cj).real
        return eA

    def calc_EI(self,cr, cj):
        _hI_cr, _hI_cj = self.by_HI_r(cr, cj)
        eI = self._innerproduct_coef(cr, _hI_cr) + self._innerproduct_coef(cj, _hI_cj)
        eI = eI.real
        return eI

    def calc_Econst(self):
        sum_omega_j = 0.0
        for _obj in self.objects:
            sum_omega_j += _obj.omega_j*_obj.NA
        econst = - np.sum(sum_omega_j)
        return econst

    ##--------- local functions
    def _innerproduct_coef(self, c1, c2):
        value = np.sum(c1.conjugate()*c2)
        return value

    def _get_wjs(self):
        num_objects = len(self.objects)
        wjs = np.zeros(num_objects, dtype = 'complex_')
        for iobj, _obj in enumerate(self.objects):
            Dj = _obj.Dj
            omega_j = _obj.omega_j
            wj = -1.0j/sqrt(2.0*self.L[0]*self.L[1]) * Dj *sqrt(omega_j)*sqrt(self.N[0]*self.N[1])
            wjs[iobj]=wj
        return wjs
