import numpy as np
import theano
import theano.tensor as T


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


 
