import rebound
import numpy as np
import theano
import theano.tensor as T
from exoplanet.theano_ops.kepler import KeplerOp
import matplotlib.pyplot as plt
from celmech.disturbing_function import get_fg_coeffs
from IntegrableModelUtils import getOmegaMatrix, calc_DisturbingFunction_with_sinf_cosf
from IntegrableModelUtils import get_secular_f2_and_f10
from scipy.optimize import root_scalar,root
from warnings import warn
DEBUG = False

def get_compiled_theano_functions(N_QUAD_PTS):
        # resonance j and k
        j,k = T.lscalars('jk')

        # Planet masses: m1,m2
        m1,m2 = T.dscalars(2)

        # resonance f and g coefficients
        f,g = T.dscalars(2)
        
        # Planet and star mass variables
        Mstar = 1
        mu1 = m1 / (Mstar + m1)
        mu2 = m2 / (Mstar + m2)
        eps = m1 * mu2 / (mu1 + mu2) / Mstar
        
        # Resonant semi-major axis ratio
        alpha = ((j-k)/j)**(2/3) * ((Mstar + m1) / (Mstar+m2))**(1/3)
        
        # Constants in Eq. (15)
        fTilde = T.sqrt((mu1+mu2) / (mu1 * T.sqrt(alpha))) * f
        gTilde = T.sqrt((mu1+mu2) / mu2 ) * g
        
        # Constant in Eq. (8)
        A = 1.5 * j * (mu1 + mu2) * (j / mu2 + (j-k) / mu1 / T.sqrt(alpha) )
        
        # Dynamical variables:
        dyvars = T.vector()
        theta,theta_star,J,J_star = [dyvars[i] for i in range(4)]
        
        # Angle variable to average disturbing function over
        kappa = T.dvector()
        
        # Quadrature weights
        quad_weights = T.dvector('w')
        
        # Convert dynamical variables to eccentricities and angles:
        # Note:
        #   Q is set to zero since it does not
        #   enter disturbing function except in combinations
        #   with z and w.
        Q = T.as_tensor(0)
        z = Q / k - theta
        
        # See Eq. 20
        Zsq = J * (fTilde*fTilde + gTilde*gTilde) / (f*f+g*g)
        Z = T.sqrt(Zsq)
        
        # Set W to zero
        Wsinw,Wcosw = 0, 0
        Zsinz,Zcosz = Z * T.sin(z), Z * T.cos(z)
        
        # Convert Z and W to planet eccentricities
        atan_f_g = T.arctan2(g , f)
        c,s = T.cos(atan_f_g),T.sin(atan_f_g)
        
        e1cos = c * Zcosz - s * Wcosw
        e1sin = c * Zsinz - s * Wsinw
        
        e2cos = s * Zcosz + c * Wcosw
        e2sin = s * Zsinz + c * Wsinw
        
        w1 = T.arctan2(e1sin,e1cos)
        w2 = T.arctan2(e2sin,e2cos)
        
        e1 = T.sqrt(e1sin*e1sin + e1cos*e1cos)
        e2 = T.sqrt(e2sin*e2sin + e2cos*e2cos)
        
        # Planets' mean longitudes
        l1 = Q / k - j * kappa
        l2 = Q / k + (k-j) * kappa
        
        # Planets mean anomalies
        M1 = l1 - w1
        M2 = l2 - w2
        
        # Convert mean to true anomalies using
        # function 'exoplanet.theano_ops.kepler.KeplerOp'
        ko = KeplerOp()
        sinf1,cosf1 =  ko( M1, e1 + T.zeros_like(M1) )
        sinf2,cosf2 =  ko( M2, e2 + T.zeros_like(M2) )
        
        # Vector of distrubing function values with same dimension as kappa vector
        DFfull = calc_DisturbingFunction_with_sinf_cosf(alpha,e1,e2,w1,w2,sinf1,cosf1,sinf2,cosf2)
        
        # Average distrubing function by weighting values with user-specified
        # quadrature weights.
        DFav = DFfull.dot(quad_weights)
        
        # Hamiltonian
        Hkep = -0.5 * A/k/k * (J - J_star) * (J - J_star)
        Hres = -2 * eps * DFav
        # ******************IMPORTANT NOTE*************************
        # I have *NOT* subtraced off the secular component of
        # the disturbing function. This means that the Hamiltonian
        # differs slightly from the one defined in the paper.
        # This is generally of little consequence to the resonant
        # dynamics but should be borne in mind when exploring
        # secular dynamics.
        # *********************************************************
        H = Hkep + Hres
        
        # Gradient and hessian of Hamiltonian w.r.t. phase space variables
        gradHtot = T.grad(H,wrt=dyvars)
        hessHtot = theano.gradient.hessian(H,wrt=dyvars)
        
        # Flow vector and Jacobian for equations of motion
        OmegaTens = T.as_tensor(getOmegaMatrix(2))
        H_flow_vec = OmegaTens.dot(gradHtot)
        H_flow_jac = OmegaTens.dot(hessHtot)

        #####################################################
        # Set parameters for compiling functions with Theano
        #####################################################
        
        # Get numerical quadrature nodes and weights
        nodes,weights = np.polynomial.legendre.leggauss(N_QUAD_PTS)
        
        # Rescale for integration interval from [-1,1] to [-pi,pi]
        nodes = nodes * np.pi
        weights = weights * 0.5
        
        
        # 'ins' will set the inputs of Theano functions compiled below
        extra_ins = [m1,m2,j,k,f,g]
        ins = [dyvars] + extra_ins
        
        # 'givens' will fix some parameters of Theano functions compiled below
        givens=[
                (kappa,nodes),
                (quad_weights,weights)
               ]

        ##########################
        # Compile Theano functions
        ##########################
        
        if not DEBUG:
            # Note that compiling can take a while
            #  so I've put a debugging switch here 
            #  to skip evaluating these functions when
            #  desired.
            H_fn = theano.function(
                inputs=ins,
                outputs=H,
                givens=givens
            )
            
            H_flow_vec_fn = theano.function(
                inputs=ins,
                outputs=H_flow_vec,
                givens=givens
            )
            
            H_flow_jac_fn = theano.function(
                inputs=ins,
                outputs=H_flow_jac,
                givens=givens
            )
        else:
            H_fn,H_flow_vec_fn,H_flow_jac_fn = [lambda x: x for _ in range(3)]
        
        # Some convenience functions...
        Zsq_to_J_Eq20 = (f*f + g*g) / (fTilde*fTilde + gTilde*gTilde)
        dJ_to_Delta_Eq21 = 1.5 * (mu1+mu2) * (j * mu1*T.sqrt(alpha) + (j-k) * mu2) / (k * T.sqrt(alpha) * mu1 * mu2)
        ecc_vars_fn = theano.function(inputs=ins, outputs=[e1,w1,e2,w2], on_unused_input='ignore')
        Zsq_to_J_Eq20_fn =theano.function(inputs=extra_ins,outputs = Zsq_to_J_Eq20, on_unused_input='ignore')
        dJ_to_Delta_Eq21_fn = theano.function(inputs=extra_ins, outputs = dJ_to_Delta_Eq21, on_unused_input='ignore')
        return (H_fn,
                H_flow_vec_fn,
                H_flow_jac_fn,
                Zsq_to_J_Eq20_fn,
                dJ_to_Delta_Eq21_fn,
                ecc_vars_fn
                )

class IntegrableResonanceModel():
    """
    A class for the Hamiltonian model describing
    the dynamics of a pair of planar planets.

    Attributes
    ----------
    j : int
        Together with k specifies j:j-k resonance
    
    k : int
        Order of resonance.
    
    f : float
        Coefficient appearing in Equation 11

    g : float
        Coefficient appearing in Equation 11

    alpha : float
        Semi-major axis ratio a_1/a_2

    eps : float
        Mass parameter m1*mu2 / (mu1+mu2)

    m1 : float
        Inner planet mass

    m2 : float
        Outer planet mass

    Zsq_to_J : float
        Conversion factor from Z^2 to canonical variable J.
        See Equation 20.
    
    dJ_to_Delta : float
        Conversion factor from (J-J^*) to Delta = (j-k)P_2 /j P_1 - 1.
        See Equation 21.

    """
    def __init__(self,j,k, n_quad_pts = 40, m1 = 1e-5 , m2 = 1e-5):
        f,g = get_fg_coeffs(j,k)
        self._j = j
        self._k = k
        self._f = f
        self._g = g
        self._m1 = m1
        self._m2 = m2

        compiled_functions = get_compiled_theano_functions(N_QUAD_PTS=n_quad_pts)

        H_fn,H_flow_vec_fn,H_flow_jac_fn,\
                Zsq_to_J_Eq20_fn,dJ_to_Delta_Eq21_fn,ecc_vars_fn=compiled_functions

        self._H_fn = H_fn
        self._H_flow_vec_fn = H_flow_vec_fn
        self._H_flow_jac_fn = H_flow_jac_fn
        self._Zsq_to_J_Eq20_fn = Zsq_to_J_Eq20_fn 
        self._dJ_to_Delta_Eq21_fn = dJ_to_Delta_Eq21_fn
        self._ecc_vars_fn = ecc_vars_fn


    @property
    def extra_args(self):
        return self.m1,self.m2,self.j,self.k,self.f,self.g
    
    @property
    def j(self):
        return self._j
    @j.setter
    def j(self,val):
        self._f,self._g = get_fg_coeffs(val,self.k)
        self._j = val

    @property
    def k(self):
        return self._k
    @k.setter
    def k(self,val):
        self._f,self._g = get_fg_coeffs(self.j,val)
        self._k = val

    @property
    def f(self):
        return self._f
    @property
    def g(self):
        return self._g

    @property
    def m1(self):
        return self._m1
    @m1.setter
    def m1(self,val):
        self._m1 = val

    @property
    def m2(self):
        return self._m2
    @m2.setter
    def m2(self,val):
        self._m2 = val

    @property
    def fTilde(self):
        mu1 = self.m1 / (1+self.m1)
        mu2 = self.m2 / (1+self.m2)
        f = self.f
        return f * np.sqrt((mu1+mu2) / (np.sqrt(self.alpha)*mu1))

    @property
    def gTilde(self):
        mu1 = self.m1 / (1+self.m1)
        mu2 = self.m2 / (1+self.m2)
        g = self.g
        return g * np.sqrt((mu1+mu2)/mu2)

    @property
    def Zsq_to_J(self):
        args = self.extra_args
        return self._Zsq_to_J_Eq20_fn(*args)

    @property
    def dJ_to_Delta(self):
        args = self.extra_args
        return self._dJ_to_Delta_Eq21_fn(*args)

    def H(self,y):
        """
        Calculate the value of the Hamiltonian

        Arguments
        ---------
        y : array_like
            Dynamical variables {theta,theta*,J,J*}

        Returns
        -------
        float
        """
        return self._H_fn(y,*self.extra_args)

    def flow_vec(self,y):
        """
        Calculate the flow vector of the equations of motion
        generated by the the Hamiltonian.

        Arguments
        ---------
        y : array_like
            Dynamical variables {theta,theta*,J,J*}

        Returns
        -------
        ndarray, shape (4,)
        """
        return self._H_flow_vec_fn(y,*self.extra_args)

    def flow_jac(self,y):
        """
        Calculate the Jacobian of the equations of motion
        generated by the the Hamiltonian with respect to the 
        dynamical variables.

        Arguments
        ---------
        y : array_like
            Dynamical variables {theta,theta*,J,J*}

        Returns
        -------
        ndarray, shape (4,4)
        """
        return  self._H_flow_jac_fn(y,*self.extra_args)

    def _elliptic_fp_root_rn(self,J,Jstar):
        """
        Convenience function to look for the elliptic fixed point by finding
            dtheta/dt = 0
        searching in J with theta=\pi.
        Desinged for use with scipy.optimize.root_scalar
        """
        if self.k%2:
            # odd
            theta_ell = np.pi
        else:
            # even
            theta_ell = np.pi/self.k
        y = np.array([theta_ell ,0,J,Jstar])
        f = self.flow_vec(y)[0]
        df = self.flow_jac(y)[0,2]
        return f,df

    def _unstable_fp_root_rn(self,J,Jstar):
        """
        Convenience function to look for the unstable root by finding
            dtheta/dt = 0
        searching in J with theta=0.
    
        Desinged for use with scipy.optimize.root_scalar
        """
        y = np.array([0,0,J,Jstar])
        f = self.flow_vec(y)[0]
        df = self.flow_jac(y)[0,2]
        return f,df

    def elliptic_fixed_point(self,Jstar):
        """
        Locate the elliptic fixed point of the system for a given value of J^*.

        Arguments
        ---------
        Jstar : float
            Value of J^*

        Returns
        -------
        ndarray, shape (4,)
            Vector of the full phase space variables (theta,theta*,J,J*) at
            the fixed point.  theta* is set to 0.
        """
        rt_st = root_scalar(self._elliptic_fp_root_rn,x0=Jstar,args=(Jstar),fprime=True)
        if not rt_st.converged:
            warn( RuntimeWarning("Search for elliptic fixed point did not converge!") )
        if self.k%2:
            # odd
            theta_ell = np.pi
        else:
            # even
            theta_ell = np.pi/self.k
        return np.array([theta_ell, 0,rt_st.root,Jstar])   

    def unstable_fixed_point(self,Jstar):
        """
        Locate the unstable fixed point of the systemfor a given value of J^*.

        Arguments
        ---------
        Jstar : float
            Value of J^*

        Returns
        -------
        ndarray, shape (4,)
            Vector of the full phase space variables (theta,theta*,J,J*) at
            the fixed point.  theta* is set to 0.
        """
        rt_unst = root_scalar(self._unstable_fp_root_rn,x0=Jstar,args=(Jstar),fprime=True)
        if not rt_unst.converged:
            warn( RuntimeWarning("Search for elliptic fixed point did not converge!") )
        return np.array([0,0,rt_unst.root,Jstar])   

    @property
    def alpha(self):
        return ((self.j-self.k)/self.j)**(2/3)

    @property
    def eps(self):
        Mstar = 1
        mu1 = self.m1 / (Mstar + self.m1)
        mu2 = self.m2 / (Mstar + self.m2)
        return self.m1 * mu2 / (mu1 + mu2) / Mstar


    def dyvars_to_orbels(self, dyvars, P2=1, Q=0,l1 = 0, W = None, Psi = None, calA = None ):
        """
        Convert dynamical variables to orbital elements.
        User can speficy the conserved quantity ${\cal A}$ directly through
        the keyword argument `calA` or inderectly by setting either `W` or
        `Psi`. If none are specified, W=0 is assumed by default.

        Arguemnts
        ---------
        dyvars : ndarray, shape (4,)
            Vector of the full phase space variables:
                dyvars = [theta,theta*,J,J*]

        P2 : float, optional
            Period of outer planet. 
            Default is P2=1

        Q : float, optional
            Value of angular variable Q=j*l2 - (j-k) l1. 
            Default is Q=0

        l1 : float, optional
            Value of inner planet's mean longitude
            Default is l1=0

        W : float, optional
            Value of eccentricity-like variable W.
            See Equation 13.

        Psi : float, optional
            Value of action-like variable Psi.
            See Equation 15.

        calA : float, optional
            Value of conserved AMD-like quantity ${\cal A}$
            See Equation 18.

        Returns
        -------
        ndarray, shape (2,4)
            Orbital elements for the pair of resonant planets.
            Format is:
                [ [P1,e1,l1,w1] , [P2,e2,l2,w2] ]
        """
        theta,theta_star,J,Jstar = dyvars
        Delta = self.dJ_to_Delta * (J-Jstar)
        P1 = (self.j-self.k) * P2  / ( self.j *  (1+Delta) )
        l2 = np.mod( (Q + (self.j-self.k) * l1) / self.j  ,2*np.pi) 
        e1,w1,e2,w2 = self._ecc_vars_fn(dyvars,*self.extra_args)

        if W is not None and not np.isclose(W,0):
            assert calA is None and Psi is None, "Can only specify one of 'W', 'calA', or 'Psi'"

            phi = theta - Q/self.k
            psi = -1 * theta_star - Q/self.k
            Phi = J 
            f = self.f
            g = self.g
            Z = np.sqrt(J / self.Zsq_to_J)
            z = -1*phi

            # (Phi,Psi) to z1,z2
            M = self._get_M_matrix()

            # (z1,z2) to (Z,W)
            z1z2_to_ZW = np.array([[f,g],[-g,f]]) / np.sqrt(f*f+g*g)

            # (Phi,Psi) to (Z,W)
            PhiPsi_to_ZW = z1z2_to_ZW.dot(M)

            # (Z,W) to (Phi,Psi)
            ZW_to_PhiPsi = np.linalg.inv(PhiPsi_to_ZW)

            # sqrt(Psi) * exp[i psi] = a * Ze^{-iz} + b * We^{-iw}
            a,b = ZW_to_PhiPsi[1]
            coszpsi = np.cos(z+psi)
            cos2zpsi = np.cos(2*(z+psi))
            disc = 4*b*b*W*W - 2*a*a*Z*Z * ( 1 - cos2zpsi )
            disc = np.abs(disc)
            Psi = b*b*W*W + a*a*Z*Z*cos2zpsi + a*Z*coszpsi*np.sqrt(disc)
            if Psi>0:
                w = -1*np.angle(( np.sqrt(Psi)*np.exp(1j*psi) - a * Z *np.exp(-1j*z))/b )
            else:
                Psi = -1*Psi
                w = -1*np.angle(( 1j*np.sqrt(Psi)*np.exp(1j*psi) - a * Z *np.exp(-1j*z))/b)
                
            WexpIw = W * np.exp(1j * w)
            z1 = e1 * np.exp(1j * w1) - self.g * WexpIw  / np.sqrt(self.f**2 + self.g**2)
            z2 = e2 * np.exp(1j * w2) + self.f * WexpIw  / np.sqrt(self.f**2 + self.g**2)
            e1,w1 = np.abs(z1),np.angle(z1)
            e2,w2 = np.abs(z2),np.angle(z2)

        elif Psi is not None and not np.isclose(Psi,0):
            assert calA is None and W is None, "Can only specify one of 'W', 'calA', or 'Psi'"
            phi = theta - Q/self.k
            psi = -1 * theta_star - Q/self.k
            Phi = J 
            # (Phi,Psi) to z1,z2
            M = self._get_M_matrix()
            z1,z2 = M.dot([np.sqrt(Phi) * np.exp(-1j * phi) , np.sqrt(Psi) * np.exp(-1j * psi)])
            e1,w1 = np.abs(z1),np.angle(z1)
            e2,w2 = np.abs(z2),np.angle(z2)

        elif calA is not None and not np.isclose(calA,Jstar):
            assert Psi is None and W is None, "Can only specify one of 'W', 'calA', or 'Psi'"
            phi = theta - Q/self.k
            psi = -1 * theta_star - Q/self.k
            Phi = J 
            Psi = calA - Jstar
            # (Phi,Psi) to z1,z2
            M = self._get_M_matrix()
            z1,z2 = M.dot([np.sqrt(Phi) * np.exp(-1j * phi) , np.sqrt(Psi) * np.exp(-1j * psi)])
            e1,w1 = np.abs(z1),np.angle(z1)
            e2,w2 = np.abs(z2),np.angle(z2)

        return np.array( [ [P1,e1,l1,w1] , [P2,e2,l2,w2] ])

    def dyvars_to_rebound_sim(self, dyvars, P2=1, Q=0,l1 = 0, W = None, Psi = None, calA = None ):
        """
        Initialize a rebound simulation from a set of dynamical variables.

        Arguemnts
        ---------
        dyvars : ndarray, shape (4,)
            Vector of the full phase space variables:
                dyvars = [theta,theta*,J,J*]
        P2 : float, optional
            Period of outer planet. 
            Default is P2=1
        Q : float, optional
            Value of angular variable Q=j*l2 - (j-k) l1. 
            Default is Q=0
        l1 : float, optional
            Value of inner planet's mean longitude
            Default is l1=0

        W : float, optional
            Value of eccentricity-like variable W.
            See Equation 13.

        Psi : float, optional
            Value of action-like variable Psi.
            See Equation 15.

        calA : float, optional
            Value of conserved AMD-like quantity ${\cal A}$
            See Equation 18.

        Returns
        -------
        rebound.Simulation object
        """
        m1 = self.m1
        m2 = self.m2
        sim = rebound.Simulation()
        sim.add(m=1,hash="star")
        orbels = self.dyvars_to_orbels(dyvars, P2 = P2, Q = Q, l1 = l1, W = W,  Psi = Psi, calA = calA )
        for i,els in enumerate(orbels):
            P,e,l,w = els
            sim.add(m=[m1,m2][i],P=P,e=e,l=l,pomega=w,hash="planet{}".format(i))
        return sim
    def _get_M_matrix(self):
        """
        Convenience function that returns the 'M' matrix defined in Appendix B.
        .. math::
            (z1^*,z2^*) = M \cdot (\sqrt{\Phi}e^{i\phi}, \sqrt{\Psi}e^{i\psi})
        """
        fTilde = self.fTilde
        gTilde = self.gTilde
        mu1 = self.m1 / (1+self.m1)
        mu2 = self.m2 / (1+self.m2)
        ang = np.arctan2(gTilde,fTilde)
        c,s = np.cos(ang),np.sin(ang)
        M2 = np.array([[c,-s],[s,c]])
        M1 = np.array([
            [np.sqrt((mu1+mu2)/(mu1*np.sqrt(self.alpha))), 0 ],
            [0,np.sqrt((mu1+mu2)  / mu2)]
        ])
        M = M1.dot(M2)
        return M
    def secular_coeffs(self):
        """
        Return coefficients a_s,b_s, and c_s appearing in the
        secular Hamiltonian (Eq 25).
        """
        # Calculate matrices appearing in B7-B9
        M = self._get_M_matrix()
        f2,f10 = get_secular_f2_and_f10(self.alpha)
        Smtrx = np.array([[f2,f10/2],[f10/2,f2]])
        # Coefficients a_s,b_s, and c_s defined in Eq. B11
        Q = 2 * np.transpose(M).dot( Smtrx.dot(M) )
        a_s = Q[0,0]
        b_s = Q[1,1]
        c_s = Q[0,1] + Q[1,0]
        return a_s,b_s,c_s
    def get_z1z2Z_from_zfrac(self,zfrac,W=0):
        """
        Get 'Z' and planets' complex eccentricities by 
        specifiying Z/Zcross (and W).

        Arguments
        ---------
        zfrac : float
            Set Z = Z_cross * zfrac
        W : float, optional
            Defaults to 0

        Returns
        -------
        z1,z2,Z : tuple of floats
        """
        
        alpha = self.alpha
        f,g=self.f,self.g
        X = np.sqrt(f*f+g*g)
        Zcross =  (X - f * W - alpha * X - g * W * alpha) / (g-f*alpha)
        Z = zfrac * Zcross
        z1 = (f * Z - g * W) / X
        z2 = (f * W + g * Z) / X
        return z1,z2,Z

def _solve_Psi_w_root_fn(pars, Z, W, z, psi, ZW_to_PhiPsi):
    """
    Convenience function for converting W and psi to Psi and w
    by root-finding when solving for orbital elements. 
    For used with scipy.root
    """

    Psi,w = pars
    rtPsi = np.sqrt(Psi)
    xcoord_root = rtPsi * np.cos(psi) - ZW_to_PhiPsi[1,0] * Z*np.cos(-1*z) - ZW_to_PhiPsi[1,1] * W * np.cos(-1*w)
    ycoord_root = rtPsi * np.sin(psi) - ZW_to_PhiPsi[1,0] * Z*np.sin(-1*z) - ZW_to_PhiPsi[1,1] * W * np.sin(-1*w)
    
    dxdPsi = 0.5 * np.cos(psi) / rtPsi
    dxdw = ZW_to_PhiPsi[1,1] * W * np.sin(w)
    dydPsi = 0.5 * np.sin(psi) / rtPsi
    dydw = ZW_to_PhiPsi[1,1] * W * np.cos(w)
    return np.array([xcoord_root,ycoord_root]),np.array([[dxdPsi,dxdw],[dydPsi,dydw]])
