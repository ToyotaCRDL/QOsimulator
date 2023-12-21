import os, sys
import copy
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt

class QOobject():
    '''
    Object of ensamble of two-level atoms.
    '''
    def __init__(self, NA, layer, Dj, omega_j, N, L, center_nx, center_ny, angle=45, which_pol=False):
        '''
        Parameters
        ----------
        NA: int
            Number of atom.
        layer: int
            Number of layer.
        Dj: float
            Dipole constant of the j-th atom.
        omega_j: float
            Transition frequency of the j-th atom.
        N: tuple of int
            N[0] and N[1] indicate the number of grid in the x and y directions, respectively.
        L: tuple of float
            Size of space in the x and y dicrections.
        center_nx: int
            Center position of the object in real-space index toward x-axis.
        center_ny: int
            Center position of the object in real-space index toward y-axis.
        angle: int
            Angel against x-axis.
        which_pol: int or str or False
            Polarization with which the object interacts. 
            0 corresponds "H", 1 corresponds "V".
            If False, it interacts with both.
        '''
        self.NA = NA
        self.layer = layer
        self.Dj = Dj
        self.omega_j = omega_j
        self.N = N
        self.L = L
        self.center_nx = center_nx
        self.center_ny = center_ny
        self.angle = angle
        ### set indices polarization
        if which_pol == False:
            self.ipols = [0,1]
        elif which_pol == True:
            self.ipols = [0] # default value
        elif type(which_pol)==str:
            if which_pol=="H":
                self.ipols=[0]
            elif which_pol=="V":
                self.ipols=[1]
            else:
                raise Exception("Error: invalid value for which_pol", which_pol)
        # self.which_pol = which_pol
        self.atom_pos_indices = self.set_atom_pos(NA, layer, N, center_nx, center_ny, angle)

    def set_atom_pos(self, NA, layer, N, center_nx, center_ny, angle):
        '''
        Make a list of atom position.

        Parameters
        ----------
        NA: int
            Number of atom.
        layer: int
            Number of layer.
        N: tuple of int
            N[0] and N[1] indicate the number of grid in the x and y directions, respectively.
        center_nx: int
            Center position of the object in real-space index toward x-axis.
        center_ny: int
            Center position of the object in real-space index toward y-axis.
        angle: int
            Angel against x-axis.

        Returns
        -------
        idx_list: (NA, 2) numpy array of int
            Index list of atoms
        '''
        idx_list = np.zeros((NA,2),dtype = 'int')
        na_per_layer = int(NA/layer)
        idx_list = np.zeros((int(layer*na_per_layer),2),dtype = 'int')
        if angle==0:
            arr_x = np.arange(int(-1.0*na_per_layer/2 + center_nx), int(na_per_layer/2 + center_nx))
            arr_y = np.array([0]*na_per_layer) + center_ny
            count_even = 0
            count_odd = 1
            for i in range(int(layer)):
                if i%2 == 0: # even
                    _layer = np.stack([arr_x, arr_y-count_even], 1)
                    count_even += 1
                if i%2 == 1: # odd
                    _layer = np.stack([arr_x, arr_y+count_odd], 1)
                    count_odd += 1
                idx_list[ i*na_per_layer :(i+1)*na_per_layer] = _layer
        elif angle==90:
            arr_x = np.array([0]*na_per_layer) + center_nx
            arr_y = np.arange(int(-1.0*na_per_layer/2 + center_ny), int(na_per_layer/2 + center_ny))
            count_even = 0
            count_odd = 1
            for i in range(int(layer)):
                if i%2 == 0: # even
                    _layer = np.stack([arr_x+count_even, arr_y], 1)
                    count_even += 1
                if i%2 == 1: # odd
                    _layer = np.stack([arr_x-count_odd, arr_y], 1)
                    count_odd += 1
                idx_list[ i*na_per_layer :(i+1)*na_per_layer] = _layer
        elif angle==45:
            arr_x = np.arange(int(-1.0*na_per_layer/2 + center_nx), int(na_per_layer/2 + center_nx))
            arr_y = np.arange(int(-1.0*na_per_layer/2 + center_ny), int(na_per_layer/2 + center_ny))
            count_even = 0
            count_odd = 1
            for i in range(int(layer)):
                if i%2 == 0: # even
                    _layer = np.stack([arr_x+count_even, arr_y-count_even], 1)
                    count_even += 1
                if i%2 == 1: # odd
                    _layer = np.stack([arr_x-count_odd, arr_y+count_odd], 1)
                    count_odd += 1
                idx_list[ i*na_per_layer :(i+1)*na_per_layer] = _layer
        elif angle==135:
            arr_x = np.arange(int(-1.0*na_per_layer/2 + center_nx), int(na_per_layer/2 + center_nx))
            arr_y = np.arange(int(-1.0*na_per_layer/2 + center_ny), int(na_per_layer/2 + center_ny))[::-1]
            count_even = 0
            count_odd = 1
            for i in range(int(layer)):
                if i%2 == 0: # even
                    _layer = np.stack([arr_x-count_even, arr_y-count_even], 1)
                    count_even += 1
                if i%2 == 1: # odd
                    _layer = np.stack([arr_x+count_odd, arr_y+count_odd], 1)
                    count_odd += 1
                idx_list[ i*na_per_layer :(i+1)*na_per_layer] = _layer
        else:
            raise ValueError("angle = " + str(angle) + " is not implemented. Please set angle in (0, 45, 90, 135).")

        ## apply the periodic boundary condition
        tmp = copy.deepcopy(idx_list)
        for j, idx in enumerate(idx_list):
            if np.abs(idx[0]) >= N[0]:
                tmp[j,0] = np.sign(idx[0]) * np.abs(idx[0]) % N[0]
            if np.abs(idx[1]) >= N[1]:
                tmp[j,1] = np.sign(idx[0]) * np.abs(idx[1]) % N[1]
        idx_list = np.unique(tmp, axis=0)

        return idx_list

    def resize(self, size=1):
        '''
        Resize QOobject

        Parameters
        ----------
        size: float
            Size of QOobject.
        '''
        NA=self.NA
        if hasattr(self, 'size'):
            NA = NA/self.size
        NA = int(size*NA)
        self.atom_pos_indices = self.set_atom_pos(NA, self.layer, self.N, self.center_nx, self.center_ny, self.angle)
        self.NA=NA
        self.size=size

    def move(self, delta_nx=0, delta_ny=0):
        '''
        Move QOobject

        Parameters
        ----------
        delta_nx: int
            Num of index toward x-axis.
        delta_ny: int
            Num of index toward y-axis.
        '''
        center_nx = int(self.center_nx + delta_nx)
        center_ny = int(self.center_ny + delta_ny)
        self.atom_pos_indices = self.set_atom_pos(self.NA, self.layer, self.N, center_nx, center_ny, self.angle)
        self.center_nx=center_nx
        self.center_ny=center_ny

    def rotate(self, delta_angle=45):
        '''
        Rotate QOobject

        Parameters
        ----------
        delta_angle: int
            Angel against x-axis.
        '''
        angle = int((self.angle + delta_angle)%180)
        self.atom_pos_indices = self.set_atom_pos(self.NA, self.layer, self.N, self.center_nx, self.center_ny, angle)
        self.angle=angle

    ##----- info
    def show(self):
        ## Plot result
        mat = np.zeros((self.N[0], self.N[1]))
        for idx in self.atom_pos_indices:
            mat[idx[0],idx[1]] = 1
        coef = self.N[0]/self.N[1]
        plt.figure(figsize=(int(5*coef),5))
        x =  np.linspace(0, self.L[0], self.N[0])
        y = np.linspace(0, self.L[1], self.N[1])
        plt.contourf(x, y, mat.T)
        plt.xlabel("x", fontsize=18)
        plt.ylabel("y", fontsize=18)

    @classmethod
    def show_all(cls, objects=[]):
        ## Plot result
        Nx = objects[0].N[0]
        Ny = objects[0].N[1]
        Lx = objects[0].L[0]
        Ly = objects[0].L[1]
        mat = np.zeros((Nx, Ny))
        for obj in objects:
            for idx in obj.atom_pos_indices:
                mat[idx[0],idx[1]] = 1
        coef = Nx / Ny
        plt.figure(figsize=(int(5*coef),5))
        x =  np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        plt.contourf(x, y, mat.T)
        plt.xlabel("x", fontsize=18)
        plt.ylabel("y", fontsize=18)

    @classmethod
    def check_param(cls, N, L, beam):
        sys.path.append(os.path.join(os.path.dirname(__file__), "../lib/"))
        from _qo_simulator import _QOsimulator

        ## Check grid_density is high enough for the mirror object.
        grid_density = sqrt(N[0]*N[1])*pi/sqrt(L[0]*L[1])
        if grid_density > 6.0:
            pass
        else:
            raise ValueError("Grid density is too small... Set larger N or smaller L.")

        ## Check kx0 and ky0 is NOT too high enough.
        qosim = _QOsimulator(N, L, 0.1, beam, objects=[], make_fig=False)
        kx_max = max(qosim.kx)
        ky_max = max(qosim.ky)
        if (beam.kx0 < 0.7*kx_max):
            pass
        else:
            raise ValueError("Photon wave number is too large... Set smaller kx0.")
        if (beam.ky0 < 0.7*ky_max):
            pass
        else:
            raise ValueError("Photon wave number is too large... Set smaller ky0.")


class Mirror(QOobject):
    '''
    Mirror object
    '''
    def __init__(self, N, L, center_nx, center_ny, size=1.0, angle=45, beam=None, which_pol=False):
        '''
        Parameters
        ----------
        N: tuple of int
            N[0] and N[1] indicate the number of grid in the x and y directions, respectively.
        L: tuple of float
            Size of space in the x and y dicrections.
        center_nx: int
            Center position of the object in real-space index toward x-axis.
        center_ny: int
            Center position of the object in real-space index toward y-axis.
        size: float
            Size of the object. (default 1.0)
        angle: int
            Angel against x-axis. (default 45)
        beam: QObeams object
        which_pol: int or str or False
            Polarization with which the object interacts.
        '''
        from qo_object import QOobject

        ## check grid density and beam parameters
        QOobject.check_param(N,L,beam)

        rat   = min(N)/256
        NA    = int(size*792*rat**2) # number of atom
        layer = int(8*rat)           # number of layer
        self.size=size

        ## Dipole constant using fitting equation
        Dj = 7.22163001*(sqrt(L[0]/pi*L[1]/pi)/sqrt(N[0]*N[1])) + 0.28201082

        ## omega_j = 0.5*omega_photon
        omega_j = 0.5*np.linalg.norm([beam.kx0,beam.ky0])

        super().__init__(NA, layer, Dj, omega_j, N, L, center_nx, center_ny, angle, which_pol=which_pol)

class BeamSplitter(QOobject):
    '''
    Beamsplitter object
    '''
    def __init__(self, N, L, center_nx, center_ny, size=1.0, angle=45, beam=None, Lexp=10*pi, which_pol=False):
        '''
        Parameters
        ----------
        N: tuple of int
            N[0] and N[1] indicate the number of grid in the x and y directions, respectively.
        L: tuple of float
            Size of space in the x and y dicrections.
        center_nx: int
            Center position of the object in real-space index toward x-axis.
        center_ny: int
            Center position of the object in real-space index toward y-axis.
        size: float
            Size of the object. (default 1.0)
        angle: int
            Angel against x-axis. (default 45)
        beam: QObeams object
        Lexp: float
            Size of space used in optimization of parameter of beamsplitter.
        which_pol: int or str or False
            Polarization with which the object interacts.
        '''
        from qo_object import QOobject
        sys.path.append(os.path.join(os.path.dirname(__file__), "../lib/"))
        from _qo_simulator import _QOsimulator
        sys.path.append(os.path.join(os.path.dirname(__file__), "../qo_beam/"))
        from qo_beam import QObeam

        ## check grid density and beam parameters
        QOobject.check_param(N,L,beam)

        rat   = min(N)/256
        NA    = int(size*88*rat)  # number of atom
        layer = 1               # number of layer
        self.size=size

        ## Dipole constant using fitting equation
        Dj = 7.22163001*(sqrt(L[0]/pi*L[1]/pi)/sqrt(N[0]*N[1])) + 0.28201082
        Dj = 5.0 * Dj # to get high fidelity (heuristic)

        ## local function for optimizing omega_j
        def fun(omj):
            fac=Lexp/min(L)
            ## Space parameters
            _L = (Lexp,Lexp)                                     # Size of space in the x and y dicrections
            _N = (int(_L[0]/(L[0]/N[0])),int(_L[1]/(L[1]/N[1]))) # number of grid in the x and y directions
            ## Object parameters
            _NA       = min(_N)
            omega_j   = omj   # transition frequency of the atoms
            center_nx = int(_N[0]/2)
            center_ny = int(_N[1]/2)
            ## Time evolution parameter
            dt      = 0.1       # timestep
            t_max   = 25 * (min(_L)/(10*pi))
            ### calc beamsplitter ###
            x0      = fac*2.0
            y0      = _L[1]/2
            kx0     = beam.kx0
            ky0     = beam.ky0
            sigma_x = fac*beam.sigma_x
            sigma_y = fac*beam.sigma_y
            _beam = QObeam(_N, _L, x0, y0, kx0, ky0, sigma_x, sigma_y)
            beamsplitter = QOobject(_NA, layer, Dj, omega_j, _N, _L, center_nx, center_ny, angle=45)
            qosim = _QOsimulator(_N, _L, dt, _beam, objects=[beamsplitter], make_fig=False)
            ck_t0 = qosim.calc_ck_t0()
            cr_t0 = qosim.to_rbases(ck_t0)
            cj_t0 = qosim.calc_cj_t0()
            for i in range(int(t_max/dt)+1):
                t = i*dt
                if i == 0:
                    ck = copy.deepcopy(ck_t0)
                    cr = copy.deepcopy(cr_t0)
                    cj = copy.deepcopy(cj_t0)
                    flag = "k"
                cr, cj, flag = qosim.suzuki_trotter_step(cr, cj, degree=2, normalization=False)

            ### trans, refrec prob ###
            def _calc_split_part(cr):
                cr_xdirection = np.zeros(_N, dtype='complex_')
                for i in range(_N[0]):
                    for j in range(_N[1]):
                        if i>j:
                            cr_xdirection[i,j] = cr[i,j]
                cr_ydirection = np.zeros(_N, dtype='complex_')
                for i in range(_N[0]):
                    for j in range(_N[1]):
                        if i<j:
                            cr_ydirection[i,j] = cr[i,j]
                return cr_xdirection, cr_ydirection
            ##=========================

            ## Calculate prob.
            cr_xdirection, cr_ydirection = _calc_split_part(cr)
            px = np.sum(np.abs(cr_xdirection)**2)
            py = np.sum(np.abs(cr_ydirection)**2)

            return np.abs(px-py)**2

        print("----> optimizing omega_j for Beamsplitter...")
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(fun)
        print("----> Result: omega_j=%.6f, diff_prob=%.6f" % (res.x, res.fun))
        print("")

        omega_j = res.x

        super().__init__(NA, layer, Dj, omega_j, N, L, center_nx, center_ny, angle, which_pol=which_pol)

class PhaseShifter(QOobject):
    '''
    Phase shifter object
    '''
    def __init__(self, N, L, center_nx, center_ny, angle=0, beam=None, which_pol=False):
        '''
        Parameters
        ----------
        N: tuple of int
            N[0] and N[1] indicate the number of grid in the x and y directions, respectively.
        L: tuple of float
            Size of space in the x and y dicrections.
        center_nx: int
            Center position of the object in real-space index toward x-axis.
        center_ny: int
            Center position of the object in real-space index toward y-axis.
        angle: int
            Angel against x-axis.
        beam: QObeams object
        which_pol: int or str or False
            Polarization with which the object interacts.
        '''
        NA=960
        layer=8
        Dj=0.5
        omega_j=0.4*0.5*np.linalg.norm([beam.kx0,beam.ky0])
        super().__init__(NA, layer, Dj, omega_j, N, L, center_nx, center_ny, angle, which_pol=which_pol)

class PolarizationRotator(QOobject):
    '''
    Polarization rotator object
    '''
    def __init__(self, N, L, center_nx, center_ny, size=1.0, angle=90, beam=None, omega_j=None, theta_rot=0, Lexp=10*pi):
        '''
        Parameters
        ----------
        N: tuple of int
            N[0] and N[1] indicate the number of grid in the x and y directions, respectively.
        L: tuple of float
            Size of space in the x and y dicrections.
        center_nx: int
            Center position of the object in real-space index toward x-axis.
        center_ny: int
            Center position of the object in real-space index toward y-axis.
        size: float
            Size of the object. (default 1.0)
        angle: int
            Angel against x-axis. (default 45)
        beam: QObeams object
        omega_j: float
            Transition frequency of the j-th atom. If this is not None, optimization step will be skipped.
        theta_rot: float
            Polarization rotation angle.
        Lexp: float
            Size of space used in optimization of parameter of beamsplitter.
        which_pol: int or str or False
            Polarization with which the object interacts.
        '''
        from qo_object import QOobject
        sys.path.append(os.path.join(os.path.dirname(__file__), "../lib/"))
        from _qo_simulator import _QOsimulator
        sys.path.append(os.path.join(os.path.dirname(__file__), "../qo_beam/"))
        from qo_beam import QObeam

        ## check grid density and beam parameters
        QOobject.check_param(N,L,beam)

        rat   = min(N)/256
        if np.abs(beam.kx0)>0:
            rat_layer = N[0]/256 * (10*pi/L[0])
        else:
            rat_layer = N[1]/256 * (10*pi/L[1])
        NA    = int(size*2048*rat)    # number of atom
        layer = int(16*rat_layer)     # number of layer
        self.size=size
        self.theta_rot = theta_rot

        ## Dipole constant using fitting equation
        Dj = 7.22163001*(sqrt(L[0]/pi*L[1]/pi)/sqrt(N[0]*N[1])) + 0.28201082

        ## local function for optimizing omega_j
        def fun(omj):
            fac=Lexp/min(L)
            ## Space parameters
            _L = (Lexp,Lexp)                                     # Size of space in the x and y dicrections
            _N = (int(_L[0]/(L[0]/N[0])),int(_L[1]/(L[1]/N[1]))) # number of grid in the x and y directions
            ## Object parameters
            _NA       = int(fac*NA)
            _layer    = int(layer)
            omega_j   = omj   # transition frequency of the atoms
            center_nx = int(_N[0]/2)
            center_ny = int(_N[1]/2)
            ## Time evolution parameter
            dt      = 0.1       # timestep
            t_max   = 25 * (min(_L)/(10*pi))
            ### calc beamsplitter ###
            x0      = fac*2.0
            y0      = _L[1]/2
            kx0     = beam.kx0
            ky0     = beam.ky0
            sigma_x = fac*beam.sigma_x
            sigma_y = fac*beam.sigma_y
            _beam = QObeam(_N, _L, x0, y0, kx0, ky0, sigma_x, sigma_y, theta_pol=0.)
            polarizationrotator = QOobject(_NA, _layer, Dj, omega_j, _N, _L, center_nx, center_ny, angle=90, which_pol="V")

            polarizationrotator.opt_pr=True
            polarizationrotator.theta_rot=pi/2.

            qosim = _QOsimulator(_N, _L, dt, _beam, objects=[polarizationrotator], calc_pol=True, make_fig=False)
            ck_t0 = qosim.calc_ck_t0()
            cr_t0 = qosim.to_rbases(ck_t0)
            cj_t0 = qosim.calc_cj_t0()
            for i in range(int(t_max/dt)+1):
                t = i*dt
                if i == 0:
                    ck = copy.deepcopy(ck_t0)
                    cr = copy.deepcopy(cr_t0)
                    cj = copy.deepcopy(cj_t0)
                    flag = "k"
                cr, cj, flag = qosim.suzuki_trotter_step(cr, cj, degree=2, normalization=False)

            ### Horizontal, Vertical prob ###
            cr_H = cr[0]
            pH = np.sum(np.abs(cr_H)**2)
            cr_V = cr[1]
            pV = np.sum(np.abs(cr_V)**2)
            # print(omj, pH, pV)

            return (1-pV)**2.

        if omega_j==None:
            print("----> optimizing omega_j for Polarization rotator...")
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(fun)
            print("----> Result: omega_j=%.6f, diff_prob=%.6f" % (res.x, res.fun))
            print("")
            omega_j = res.x

        super().__init__(NA, layer, Dj, omega_j, N, L, center_nx, center_ny, angle, which_pol="V")
