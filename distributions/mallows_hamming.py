import numpy as np
import itertools as it
from scipy.optimize import linear_sum_assignment
import mallows_model as mm

#************* Distance **************#


def distance(A, B=None):
    """
    This function computes Hamming distance between two permutations.
    If only one permutation is given, the distance will be computed with the
    identity permutation as the second permutation.
   Parameters
   ----------
   A: ndarray
        The first permutation
   B: ndarray, optional
        The second permutation (default is None)
   Returns
   -------
   int
        Hamming distance between A and B
    """
    if B is None: B = np.arange(len(A))

    return sum(A != B)


def dist_at_uniform(n):
    return n


#************ Sampling ************#


def sample(m, n, *, theta=None, phi=None, s0=None):
    """This function generates m permutations (rankings) according to Mallows Models.
        Parameters
        ----------
        m: int
            Number of rankings to generate
        n: int
            Length of rankings
        theta: float, optional (if phi given)
            Dispersion parameter theta
        phi: float, optional (if theta given)
            Dispersion parameter phi
        s0: ndarray
            Consensus ranking
        Returns
        -------
        ndarray
            The rankings generated.
    """
    sample = np.zeros((m, n))
    theta, phi = mm.check_theta_phi(theta, phi)

    facts_ = np.array([1, 1] + [0] * (n - 1), dtype=np.float)
    deran_num_ = np.array([1, 0] + [0] * (n - 1), dtype=np.float)
    for i in range(2, n + 1):
        facts_[i] = facts_[i - 1] * i
        deran_num_[i] = deran_num_[i - 1] * (i - 1) + deran_num_[i - 2] * (i -
                                                                           1)
    hamm_count_ = np.array([
        deran_num_[d] * facts_[n] / (facts_[d] * facts_[n - d])
        for d in range(n + 1)
    ],
                           dtype=np.float)
    probsd = np.array(
        [hamm_count_[d] * np.exp(-theta * d) for d in range(n + 1)],
        dtype=np.float)

    for m_ in range(m):
        target_distance = np.random.choice(n + 1, p=probsd / probsd.sum())
        sample[m_, :] = sample_at_dist(n, target_distance, s0)

    return sample


def sample_at_dist(n, dist, sigma0=None):
    """This function randomly generates a permutation with length n at distance
    dist to a given permutation sigma0.
        Parameters
        ----------
        n: int
            Length of the permutations
        dist: int
            Distance between the permutation generated randomly and a known
            permutation sigma0
        sigma0: ndarray, optional
            A known permutation (If not given, then it equals the identity)
        Returns
        -------
        ndarray
            A random permutation at distance dist to sigma0.
    """
    if sigma0 is None: sigma0 = np.arange(n)
    sigma = np.zeros(n) - 1
    fixed_points = np.random.choice(n, n - dist, replace=False)
    sigma[fixed_points] = fixed_points
    unfix = np.setdiff1d(np.arange(n), fixed_points)
    unfix = np.random.permutation(unfix)
    for i in range(len(unfix) - 1):
        sigma[unfix[i]] = unfix[i + 1]
    if len(unfix) > 0: sigma[unfix[-1]] = unfix[0]
    return sigma[sigma0].astype(int)


#********* Expected distance *********#


def expected_dist_mm(n, theta=None, phi=None):
    """The function computes the expected value of Hamming distance under Mallows Models (MMs).
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter, optional (if phi is given)
        phi: float
            Real dispersion parameter, optional (if theta is given)
        Returns
        -------
        float
            The expected distance under MMs.
    """
    theta, phi = mm.check_theta_phi(theta, phi)

    facts_ = np.array([1, 1] + [0] * (n - 1), dtype=np.float)
    for i in range(2, n + 1):
        facts_[i] = facts_[i - 1] * i
    x_n_1, x_n = 0, 0

    for k in range(n + 1):
        aux = (np.exp(theta) - 1)**k / facts_[k]
        x_n += aux
        if k < n: x_n_1 += aux
    return (n * x_n - x_n_1 * np.exp(theta)) / x_n


#************ Learning ************#


def median(sample, ws=1):
    """This function computes the central permutation (consensus ranking) given
    several permutations using Hungarian algorithm.
        Parameters
        ----------
        sample: ndarray
            Matrix of several permutations
        ws: float optional
            weight (not weighted by default)
        Returns
        -------
        ndarray
            The central permutation of permutations given
    """
    m, n = sample.shape
    wmarg = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            freqs = (sample[:, i] == j)
            wmarg[i, j] = (freqs * ws).sum()
    row_ind, col_ind = linear_sum_assignment(-wmarg)

    return col_ind


def prob(sigma, sigma0, theta=None, phi=None):
    """ Probability mass function of a MM with central ranking sigma0 and
    dispersion parameter theta/phi.
    Parameters
    ----------
    sigma: ndarray
        A pemutation
    sigma0: ndarray
        central permutation
    theta: float
        Dispersion parameter (optional, if phi is given)
    phi: float
        Dispersion parameter (optional, if theta is given)
    Returns
    -------
    float
        Probability mass function.
    """
    theta, phi = mm.check_theta_phi(theta, phi)
    d = distance(sigma, sigma0)
    n = len(sigma)
    facts_ = np.array([1, 1] + [0] * (n - 1), dtype=np.float)

    for i in range(2, n + 1):
        facts_[i] = facts_[i - 1] * i
    sum = 0
    for i in range(n + 1):
        sum += (((np.exp(theta) - 1)**i) / facts_[i])
    psi = sum * np.exp(-n * theta) * facts_[n]
    return np.exp(-d * theta) / psi


def find_phi(n, dmin, dmax):
    """ Find the dispersion parameter phi that gives an expected distance between
    dmin and dmax where the length of rankings is n.
    Parameters
    ----------
    n: int
        Length of permutations
    dmin: int
        Minimum of expected distance
    dmax: int
        Maximum of expected distance
    Returns
    -------
    float
        The value of phi.
    """
    assert dmin < dmax
    imin, imax = 0.0, 1.0
    iterat = 0
    while iterat < 500:
        med = (imax + imin) / 2
        d = expected_dist_mm(n, phi=med)

        if d < dmin: imin = med
        elif d > dmax: imax = med
        else: return med
        iterat += 1

    assert False, "Max iterations reached"

    # end
