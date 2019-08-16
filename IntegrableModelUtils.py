import numpy as np
import theano
import theano.tensor as T
from scipy.integrate import solve_ivp


def getOmegaMatrix(n):
    """
    Get the 2n x 2n skew-symmetric block matrix:
          [0 , I_n]
          [-I_n, 0 ]
    that appears in Hamilton's equations.

    Arguments
    ---------
    n : int
        Determines matrix dimension

    Returns
    -------
    numpy.array
    """
    return np.vstack(
        (
        np.concatenate([np.zeros((n,n)),np.eye(n)]).T,
        np.concatenate([-np.eye(n),np.zeros((n,n))]).T
    )
    )


def calc_DisturbingFunction_with_sinf_cosf(alpha,e1,e2,w1,w2,sinf1,cosf1,sinf2,cosf2):
    """
    Compute the value of the disturbing function
    .. math::
        \frac{a'}{|r-r'|} - a'\frac{r.r'}{|r'^3|}
    from a set of input orbital elements for coplanar planets.

    Arguments
    ---------
    alpha : float
        semi-major axis ratio
    e1 : float
        inner eccentricity
    e2 : float
        outer eccentricity
    w1 : float
        inner long. of peri
    w2 : float
        outer long. of peri
    sinf1 : float
        sine of inner planet true anomaly
    cosf1 : float
        cosine of inner planet true anomaly
    sinf2 : float
        sine of outer planet true anomaly
    cosf2 : float
        cosine of outer planet true anomaly

    Returns
    -------
    float :
        Disturbing function value
    """
    r1 = alpha * (1-e1*e1) /(1 + e1 * cosf1)
    _x1 = r1 * cosf1
    _y1 = r1 * sinf1
    Cw1 = T.cos(w1)
    Sw1 = T.sin(w1)
    x1 = Cw1 * _x1  - Sw1 * _y1
    y1 = Sw1 * _x1  + Cw1 * _y1

    r2 = (1-e2*e2) /(1 + e2 * cosf2)
    _x2 = r2 * cosf2
    _y2 = r2 * sinf2
    Cw2 = T.cos(w2)
    Sw2 = T.sin(w2)
    x2 = Cw2 * _x2  - Sw2 * _y2
    y2 = Sw2 * _x2  + Cw2 * _y2

    # direct term
    dx = (x2 - x1)
    dy = (y2 - y1)
    dr2 = dx*dx + dy*dy
    direct = 1 / T.sqrt(dr2)

    # indirect term
    r1dotr = (x2 * x1 + y2 * y1)
    r1sq = x2*x2 + y2*y2
    r1_3 = 1 / r1sq / T.sqrt(r1sq)
    indirect = -1 * r1dotr * r1_3

    return direct+indirect

# Secular Hamiltonian coefficients evaluated at alpha=a1/a2
from celmech.disturbing_function import laplace_B
def get_secular_f2_and_f10(alpha):
    """
    Calculate f_{2} and f_{10}, the combinations
    Laplace coefficients that appear in the secular
    disturbing function at second order in eccentricity:
    ..math::
        R_\text{sec} = f_2\left(e_1^2 + e_2^2\right) +
                        f_{10} e_1 e_2\cos(\varpi_2-\varpi_1)
    
    Arguments
    ---------
    alpha : float
        The semi-major axis ratio.
    
    Returns
    -------
    f2 : float
    f10 : float
    """
    b01 = laplace_B.eval(1/2,0,1,alpha)
    b02 = laplace_B.eval(1/2,0,2,alpha)
    
    b10 = laplace_B.eval(1/2,1,0,alpha)
    b11 = laplace_B.eval(1/2,1,1,alpha)
    b12 = laplace_B.eval(1/2,1,2,alpha)
    
    
    
    f2 = 0.25 * alpha * b01 + 0.125 * alpha * alpha * b02
    f10 = 0.5 * b10 - 0.5 * alpha * b11 - 0.25 * alpha * alpha * b12
    
    return f2,f10

"""
For calls to solve_ivp variable order should be:
    y[0] = theta
    y[1] = theta*
    y[2] = J
    y[3] = J*
    y[4] = action ({\cal J})
"""

def _action_ydot(t,y,res_model):
    """
    dy/dt for use with scipy.integrate.solve_ivp
    in calc_action_and_frequencies below
    """
    ymost = y[:-1]
    flow = res_model.flow_vec(ymost)
    # Symmetric form of action, see Henrard 1990, Eq. 15
    action_dot = (y[2] * flow[0] - y[0] * flow[2]) / (4*np.pi)
    return np.append(flow,action_dot)

def _action_ydot_jac(t,y,res_model):
    """
    grad(dy/dt) w.r.t. y for use with 
    scipy.integrate.solve_ivp in 
    calc_action_and_frequencies below.
    """
    ymost = y[:-1]
    flow = res_model.flow_vec(ymost)
    jacmost = res_model.flow_jac(ymost)
    jac = np.pad(jacmost,(0,1),mode='constant')
    jac[4,:4] = (y[0] * jacmost[2] - y[2] * jacmost[0]) / (4*np.pi)
    jac[4,0] += flow[2] / (2*np.pi)
    jac[4,2] -= flow[0] / (2*np.pi)
    return jac

def _action_event(t,y,J0):
    """
    Determine when J returns to initial
    value, J0
    """
    if np.isclose(t,0):
        return 1
    J = y[2]
    return J - J0


def calc_action_and_frequencies(y0, res_model, return_solution = False):
    """
    Determine the dynamical frequencies associated with 
    initial libration amplitude delta_theta and conserved 
    quantity Jstar.
    
    Based on Henrard (1990, doi: 10.1007/BF00048581).
    
    The equations of motion are integrated until the trajectory
    returns to it's initial value of J  (with the additional
    condtion that dJ/dt has the same sign as it does initially).
    The time it takes for J to return to its initial value is
    the simply libration period. 
    
    The secular frequency is determined as follows:
    the angle $\theta^*$ is related to the canonical 
    action-angle variables $({\cal J},\zeta,I^*,\phi^*)$ by: 
    
    $$
        \theta^* = \phi^* + \rho(\zeta; I^*,{\cal J})
    $$
    
    where $\rho$ is a 2pi periodic function of $\zeta$.
    Therefore, after one libration period,
    $$
     \theta^* = \phi^*(0) + \frac{\Omega_{sec}}\times T
    $$
    where T is the libration period. The secular frequency
    can then be determined from the change in \theta^* over
    one libration period.
    
    Arguments
    ---------
    y0 : ndarray, shape (4,)
        Intial condtions of trajectory in coordinates
            y0 = [theta,theta*,J,J*]
    
    res_model : IntegrableResonanceModel object
        Resonance model used for equations of motion.
        
    Returns
    -------
    Action : float
        Canonical action variable
        .. math::
            {\cal J} = \frac{1}{2\pi} \oint J d\theta
            
    Omega_res : float
        Frequency of (J,theta) degree of freedom.
        
    Omega_sec : float
        Frequency of (J*,theta*) degree of freedom.
        This is the mean precession rate of theta*.
        Note that this Omega_sec differs from the one
        defined in the paper by eps*b_s
        
    Notes
    -----
    Since the system is periodic in the (J,theta) degree of freedom,
    the integration proceeds until J repeats its initial values
    with its time derivative having the same sign as it does intially.
    
    If y0 is close to the elliptic fixed point, the function
    returns Action=0 and frequencies based on linearized equations
    of motion.
    """
    J0 = y0[2]
    Jstar = y0[3]
    
    # If ICs are close to elliptic fixed point...
    yell = res_model.elliptic_fixed_point(Jstar)
    if np.alltrue(np.isclose(y0,yell)):
        fvec = res_model.flow_vec(y0)
        jac = res_model.flow_jac(y0)
        Omega_res = np.max(np.imag(np.linalg.eigvals(jac)))
        Omega_sec = fvec[1]
        action = 0        
    # ... otherwise do integration
    else:
        # Stop when J returns to J0 with the proper dJ/dt direction
        eventfn = lambda t,y: _action_event(t,y,J0)
        eventfn.terminal=True
        eventfn.direction = np.sign(res_model.flow_vec(y0)[2])
        sol = solve_ivp(lambda t,y: _action_ydot(t,y,res_model),
                        t_span=(0,np.inf),
                        y0=np.append(y0,0),
                        jac=lambda t,y: _action_ydot_jac(t,y,res_model),
                        method='Radau',
                        events=[eventfn],
                        dense_output=True
                       )

        # Resonant libration period/freq. 
        T = sol.t_events[0][0]
        Omega_res = 2 * np.pi / T 
        yfinal = sol.sol(T)
        theta_star_final = yfinal[1]
        Omega_sec = theta_star_final / T
        action = yfinal[-1]
    if return_solution:
        return Omega_res,Omega_sec,action,sol
    return action,Omega_res,Omega_sec
