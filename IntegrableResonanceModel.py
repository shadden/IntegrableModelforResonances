import rebound
import numpy as np
import theano
import theano.tensor as T
from exoplanet.theano_ops.kepler import KeplerOp
import matplotlib.pyplot as plt
from celmech.disturbing_function import get_fg_coeffs
from IntegrableModelUtils import getOmegaMatrix, calc_DisturbingFunction_with_sinf_cosf
from scipy.optimize import root_scalar
from warnings import warn

def get_compiled_theano_functions(j,k,f,g,N_QUAD_PTS,DEFAULT_MASS):
        # Planet masses: m1,m2
        m1,m2 = T.dscalars(2)
        
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
        ins = [
                dyvars,
                theano.In(m1,name="m1",value=DEFAULT_MASS),
                theano.In(m2,name="m2",value=DEFAULT_MASS)
              ]
        
        # 'givens' will fix some parameters of Theano functions compiled below
        givens=[
                (kappa,nodes),
                (quad_weights,weights)
               ]

        ##########################
        # Compile Theano functions
        ##########################
        
        # Note that this may take a while...
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

        # Some convenience functions...
        Zsq_to_J_Eq20 = (f*f + g*g) / (fTilde*fTilde + gTilde*gTilde)
        dJ_to_Delta_Eq21 = 1.5 * (mu1+mu2) * (j * mu1*T.sqrt(alpha) + (j-k) * mu2) / (k * T.sqrt(alpha) * mu1 * mu2)
        # canonical_variable_rotation_Eq15 = T.stacklists([[fTilde,gTilde],[-gTilde,fTilde]]) / T.sqrt(fTilde*fTilde + gTilde*gTilde)
        ecc_vars_fn = theano.function(inputs=[dyvars,m1,m2],outputs=[e1,w1,e2,w2])
        
        Zsq_to_J_Eq20_fn =theano.function(inputs=[m1,m2],outputs = Zsq_to_J_Eq20 )
        dJ_to_Delta_Eq21_fn = theano.function(inputs=[m1,m2],outputs = dJ_to_Delta_Eq21 )
        #canonical_variable_rotation_Eq15_fn =theano.function(inputs=[m1,m2],outputs = canonical_variable_rotation_Eq15 )
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
    f : float
        Coefficient appearing in Equation 11
    g : float
        Coefficient appearing in Equation 11
    default_mass  : float
        Default value of planet mass to use when
        not specified.
    """
    def __init__(self,j,k, n_quad_pts = 40, default_mass = 1e-5):
        f,g = get_fg_coeffs(j,k)
        self.j = j
        self.k = k
        self.f = f
        self.g = g
        self.default_mass = default_mass

        H_fn,H_flow_vec_fn,H_flow_jac_fn,Zsq_to_J_Eq20_fn,dJ_to_Delta_Eq21_fn,ecc_vars_fn = get_compiled_theano_functions(
                j,k,f,g,
                N_QUAD_PTS=n_quad_pts,
                DEFAULT_MASS = default_mass
                )
        self._H_fn = H_fn
        self._H_flow_vec_fn = H_flow_vec_fn
        self._H_flow_jac_fn = H_flow_jac_fn
        self._Zsq_to_J_Eq20_fn = Zsq_to_J_Eq20_fn 
        self._dJ_to_Delta_Eq21_fn = dJ_to_Delta_Eq21_fn
        #self._canonical_variable_rotation_Eq15_fn = canonical_variable_rotation_Eq15_fn
        self._ecc_vars_fn = ecc_vars_fn

    def Zsq_to_J(self,**kwargs):
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        return self._Zsq_to_J_Eq20_fn(m1,m2)
    def dJ_to_Delta(self,**kwargs):
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        return self._dJ_to_Delta_Eq21_fn(m1,m2)

    def H(self,y,**kwargs):
        """
        Calculate the value of the Hamiltonian

        Arguments
        ---------
        y : array_like
            Dynamical variables {theta,theta*,J,J*}
        m1 : float, optional
            Inner planet mass
        m2 : float, optional
            Outer planet mass

        Returns
        -------
        float
        """
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        return self._H_fn(y,m1,m2)

    def flow_vec(self,y,**kwargs):
        """
        Calculate the flow vector of the equations of motion
        generated by the the Hamiltonian.

        Arguments
        ---------
        y : array_like
            Dynamical variables {theta,theta*,J,J*}
        m1 : float, optional
            Inner planet mass
        m2 : float, optional
            Outer planet mass

        Returns
        -------
        ndarray, shape (4,)
        """
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        return self._H_flow_vec_fn(y,m1,m2)

    def flow_jac(self,y,**kwargs):
        """
        Calculate the Jacobian of the equations of motion
        generated by the the Hamiltonian with respect to the 
        dynamical variables.

        Arguments
        ---------
        y : array_like
            Dynamical variables {theta,theta*,J,J*}
        m1 : float, optional
            Inner planet mass
        m2 : float, optional
            Outer planet mass

        Returns
        -------
        ndarray, shape (4,4)
        """
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        return  self._H_flow_jac_fn(y,m1,m2)

    def _elliptic_fp_root_rn(self,J,Jstar,m1,m2):
        """
        Convenience function to look for the elliptic fixed point by finding
            dtheta/dt = 0
        searching in J with theta=\pi.
        Desinged for use with scipy.optimize.root_scalar
        """
        y = np.array([np.pi / self.k ,0,J,Jstar])
        f = self.flow_vec(y,m1=m1,m2=m2)[0]
        df = self.flow_jac(y,m1=m1,m2=m2)[0,2]
        return f,df

    def _unstable_fp_root_rn(self,J,Jstar,m1,m2):
        """
        Convenience function to look for the unstable root by finding
            dtheta/dt = 0
        searching in J with theta=0.
    
        Desinged for use with scipy.optimize.root_scalar
        """
        y = np.array([0,0,J,Jstar])
        f = self.flow_vec(y,m1=m1,m2=m2)[0]
        df = self.flow_jac(y,m1=m1,m2=m2)[0,2]
        return f,df

    def elliptic_fixed_point(self,Jstar,**kwargs):
        """
        Locate the elliptic fixed point of the system for a given value of J^*.

        Arguments
        ---------
        Jstar : float
            Value of J^*
        m1 : float, optional
            inner planet mass
        m2 : float, optional
            inner planet mass

        Returns
        -------
        ndarray, shape (4,)
            Vector of the full phase space variables (theta,theta*,J,J*) at
            the fixed point.  theta* is set to 0.
        """
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        rt_st = root_scalar(self._elliptic_fp_root_rn,x0=Jstar,args=(Jstar,m1,m2),fprime=True)
        if not rt_st.converged:
            warn( RuntimeWarning("Search for elliptic fixed point did not converge!") )
        return np.array([np.pi / self.k ,0,rt_st.root,Jstar])   

    def unstable_fixed_point(self,Jstar,**kwargs):
        """
        Locate the unstable fixed point of the systemfor a given value of J^*.

        Arguments
        ---------
        Jstar : float
            Value of J^*
        m1 : float, optional
            inner planet mass
        m2 : float, optional
            inner planet mass

        Returns
        -------
        ndarray, shape (4,)
            Vector of the full phase space variables (theta,theta*,J,J*) at
            the fixed point.  theta* is set to 0.
        """
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        rt_unst = root_scalar(self._unstable_fp_root_rn,x0=Jstar,args=(Jstar,m1,m2),fprime=True)
        if not rt_unst.converged:
            warn( RuntimeWarning("Search for elliptic fixed point did not converge!") )
        return np.array([0,0,rt_unst.root,Jstar])   

    @property
    def alpha(self):
        return ((self.j-self.k)/self.j)**(2/3)

    def dyvars_to_orbels(self, dyvars, P2=1, Q=0,l1 = 0,  **kwargs):
        """
        Convert dynamical variables to orbital elements.

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
        m1 : float, optional
            inner planet mass
        m2 : float, optional
            inner planet mass

        Returns
        -------
        ndarray, shape (2,4)
            Orbital elements for the pair of resonant planets.
            Format is:
                [ [P1,e1,l1,w1] , [P2,e2,l2,w2] ]
        Note
        ----
        Orbital elements are computed assuming W=0. 
        """
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        theta,theta_star,J,Jstar = dyvars
        e1,w1,e2,w2 = self._ecc_vars_fn(dyvars,m1,m2)
        Delta = self.dJ_to_Delta(m1=m1,m2=m2) * (J-Jstar)
        P1 = (self.j-self.k) * P2  / ( self.j *  (1+Delta) )
        l2 = np.mod( (Q + (self.j-self.k) * l1) / self.j  ,2*np.pi) 
        return np.array( [ [P1,e1,l1,w1] , [P2,e2,l2,w2] ])

    def dyvars_to_rebound_sim(self, dyvars, P2=1, Q=0,l1 = 0,  **kwargs):
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
        m1 : float, optional
            inner planet mass
        m2 : float, optional
            inner planet mass

        Returns
        -------
        rebound.Simulation object
        """
        m1 = kwargs.get('m1',self.default_mass)
        m2 = kwargs.get('m2',self.default_mass)
        sim = rebound.Simulation()
        sim.add(m=1,hash="star")
        orbels = self.dyvars_to_orbels(dyvars, P2=1, Q=0,l1 = 0,  **kwargs)
        for i,els in enumerate(orbels):
            P,e,l,w = els
            sim.add(m=[m1,m2][i],P=P,e=e,l=l,pomega=w,hash="planet{}".format(i))
        return sim
