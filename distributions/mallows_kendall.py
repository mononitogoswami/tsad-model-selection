import numpy as np
import itertools as it
import scipy.optimize as sp_opt
import permutil as pu
import mallows_model as mm

#******** Complete rankings **********#
#*************************************#


#************* Distance **************#

def merge(left, right):
    """
    This function uses Merge sort algorithm to count the number of
    inversions in a permutation of two parts (left, right).
    Parameters
    ----------
    left: ndarray
        The first part of the permutation
    right: ndarray
        The second part of the permutation
    Returns
    -------
    result: ndarray
        The sorted permutation of the two parts
    count: int
        The number of inversions in these two parts.
    """
    result = []
    count = 0
    i, j = 0, 0
    left_len = len(left)
    while i < left_len and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            count += left_len - i
            j += 1
    result += left[i:]
    result += right[j:]

    return result, count

def mergeSort_rec(lst):
    """
    This function splits recursively lst into sublists until sublist size is 1. Then, it calls the function merge()
    to merge all those sublists to a sorted list and compute the number of inversions used to get that sorted list.
    Finally, it returns the number of inversions in lst.
    Parameters
    ----------
    lst: ndarray
        The permutation
    Returns
    -------
    result: ndarray
        The sorted permutation
    d: int
        The number of inversions.
    """
    lst = list(lst)
    if len(lst) <= 1:
        return lst, 0
    middle = int( len(lst) / 2 )
    left, a   = mergeSort_rec(lst[:middle])
    right, b  = mergeSort_rec(lst[middle:])
    sorted_, c = merge(left, right)
    d = (a + b + c)
    return sorted_, d



def distance(A, B=None):
    """
    This function computes Kendall's-tau distance between two permutations
    using Merge sort algorithm.
    If only one permutation is given, the distance will be computed with the
    identity permutation as the second permutation
   Parameters
   ----------
   A: ndarray
        The first permutation
   B: ndarray, optional
        The second permutation (default is None)
   Returns
   -------
   int
        Kendall's-tau distance between both permutations (equal to the number of inversions in their composition).
    """
    if B is None : B = list(range(len(A)))

    A = np.asarray(A).copy()
    B = np.asarray(B).copy()
    n = len(A)

    # check if A contains NaNs
    msk = np.isnan(A)
    indexes = np.array(range(n))[msk]

    if indexes.size:
        A[indexes] = n#np.nanmax(A)+1

    # check if B contains NaNs
    msk = np.isnan(B)
    indexes = np.array(range(n))[msk]

    if indexes.size:
        B[indexes] = n#np.nanmax(B)+1

    # print(A,B,n)
    inverse = np.argsort(B)
    compose = A[inverse]
    _, distance = mergeSort_rec(compose)
    return distance


def max_dist(n):
    """ This function computes the maximum distance between two permutations of n length.
        Parameters
        ----------
        n: int
            Length of permutations
        Returns
        -------
        int
            Maximum distance between permutations of given n length.
    """
    return int(n*(n-1)/2)


#************ Vector/Rankings **************#

def v_to_ranking(v, n):
    """This function computes the corresponding permutation given a decomposition vector.
    The O(n log n) version in 10.1.1 of
    Arndt, J. (2010). Matters Computational: ideas, algorithms, source code.
    Springer Science & Business Media.
        Parameters
        ----------
        v: ndarray
            Decomposition vector, same length as the permutation, last item must be 0
        n: int
            Length of the permutation
        Returns
        -------
        ndarray
            The permutation corresponding to the decomposition vectors.
    """
    rem = list(range(n))
    rank = np.full(n, np.nan)
    for i in range(len(v)):
        rank[i] = rem[v[i]]
        rem.pop(v[i])
    return rank.astype(int)

def ranking_to_v(sigma, k=None):
    """This function computes the corresponding decomposition vector given a permutation
    The O(n log n) version in 10.1.1 of
    Arndt, J. (2010). Matters Computational: ideas, algorithms, source code.
    Springer Science & Business Media.
        Parameters
        ----------
        sigma: ndarray
            A permutation
        k: int, optional
            The index to perform the conversion for a partial
            top-k list
        Returns
        -------
        ndarray
            The decomposition vector corresponding to the permutation. Will be
            of length n and finish with 0.
    """
    n = len(sigma)
    if k is not None:
        sigma = sigma[:k]
        sigma = np.concatenate((sigma, np.array([np.float(i) for i in range(n) if i not in sigma])))
    V = []
    for j, sigma_j in enumerate(sigma):
        V_j = 0
        for i in range(j+1, n):
            if sigma_j > sigma[i]:
                V_j += 1
        V.append(V_j)
    return np.array(V)


#************ Sampling ************#

def sample(m, n, *, k=None, theta=None, phi=None, s0=None):
    """This function generates m (rankings) according to Mallows Models (if the given parameters
    are m, n, k/None, theta/phi: float, s0/None) or Generalized Mallows Models (if the given
    parameters are m, n, theta/phi: ndarray, s0/None). Moreover, the parameter k allows the
    function to generate top-k rankings only.
        Parameters
        ----------
        m: int
            Number of rankings to generate
        n: int
            Length of rankings
        theta: float or ndarray, optional (if phi given)
            The dispersion parameter theta
        phi: float or ndarray, optional (if theta given)
            Dispersion parameter phi
        k: int
            Length of partial permutations (only top items)
        s0: ndarray
            Consensus ranking
        Returns
        -------
        ndarray
            The rankings generated
    """

    theta, phi = mm.check_theta_phi(theta, phi)

    theta = np.full(n-1, theta)

    if s0 is None:
        s0 = np.array(range(n))

    rnge = np.array(range(n-1))

    psi = (1 - np.exp(( - n + rnge )*(theta[ rnge ])))/(1 - np.exp( -theta[rnge]))
    vprobs = np.zeros((n, n))
    for j in range(n-1):
        vprobs[j][0] = 1.0/psi[j]
        for r in range(1, n-j):
            vprobs[j][r] = np.exp( -theta[j] * r ) / psi[j]
    sample = []
    vs = []
    for samp in range(m):
        v = [np.random.choice(n, p=vprobs[i, :]) for i in range(n-1)]
        v += [0]
        ranking = v_to_ranking(v, n)
        sample.append(ranking)

    sample = np.array([s[s0] for s in sample])

    if k is not None:
        sample_rankings = np.array([pu.inverse(ordering) for ordering in sample])
        sample_rankings = np.array([ran[s0] for ran in sample_rankings])
        sample = np.array([[i if i in range(k) else np.nan for i in ranking] for
                        ranking in sample_rankings])
    return sample

def num_perms_at_dist(n):
    """This function computes the number of permutations of length 1 to n for
    each possible Kendall's-tau distance d. See the online Encyclopedia of
    Integer Sequences, OEIS-A008302
        Parameters
        ----------
        n: int
            Length of the permutations
        Returns
        -------
        ndarray
            The number of permutations of length 1 to n for each possible
            Kendall's-tau distance d
    """
    sk = np.zeros((n+1, int(n*(n-1)/2+1)))
    for i in range(n+1):
        sk[i, 0] = 1
    for i in range(1, 1+n):
        for j in range(1,int(i*(i-1)/2+1)):
            if j - i >= 0 :
                sk[i, j] = sk[i,j-1]+ sk[i-1,j] - sk[i-1, j-i]
            else:
                sk[i, j] = sk[i, j-1]+ sk[i-1, j]
    return sk.astype(np.uint64)

def sample_at_dist(n, dist, sk=None, sigma0=None):
    """This function randomly generates a permutation with length n at distance
    dist to a given permutation sigma0.
        Parameters
        ----------
        n: int
            Length of the permutations
        dist: int
            Distance between the permutation generated randomly and a known
            permutation sigma0
        sk: matrix
            matrix returned by the function mallows_kendall::num_perms_at_dist(n)
            if this function is to be called many times, to avoid recomputation,
            sk can be provided in the input. Otherwise, the function is called here
        sigma0: ndarray, optional
            A known permutation (If not given, then it equals the identity)
        Returns
        -------
        ndarray
            A random permutation at distance dist to sigma0.
    """
    i = 0
    probs = np.zeros(n+1)
    v = np.zeros(n, dtype=int)
    if sk is None: sk = num_perms_at_dist(n)

    while i<n and dist > 0 :
        rest_max_dist = (n - i - 1 ) * ( n - i - 2 ) / 2
        if rest_max_dist  >= dist:
            probs[0] = sk[n-i-1, dist]
        else:
            probs[0] = 0
        mi = min(dist + 1, n - i )
        for j in range(1, mi):
            if rest_max_dist + j >= dist: probs[j] = sk[n-i-1, dist-j]
            else: probs[ j ] = 0
        v[i] = np.random.choice(mi, 1, p=probs[:mi]/probs[:mi].sum())
        dist -= v[i]
        i += 1
    random_perm = v_to_ranking(v, n)

    return random_perm[sigma0].reshape(-1)

#********* Expected distance *********#

def expected_dist_mm(n, theta=None, phi=None):
    """The function computes the expected distance of Kendall's-tau distance under Mallows models (MMs).
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
    rnge = np.array(range(1,n+1))
    expected_dist = n * np.exp(-theta) / (1-np.exp(-theta)) - np.sum(rnge * np.exp(-rnge*theta) / (1 - np.exp(-rnge*theta)))

    return expected_dist

#************ Variance ************#

def variance_dist_mm(n, theta=None, phi=None):
    """ This function returns the variance of Kendall's-tau distance under the MMs.
        Parameters
        ----------
        n: int
            Length of the permutations
        theta: float
            Dispersion parameter, optional (if phi is given)
        phi  : float
            Dispersion parameter, optional (if theta is given)
        Returns
        -------
        float
            The variance of Kendall's-tau distance under the MMs.
    """
    theta, phi = mm.check_theta_phi(theta, phi)
    rnge = np.array(range(1,n+1))
    variance = (phi*n)/(1-phi)**2 - np.sum((pow(phi,rnge) * rnge**2)/(1-pow(phi,rnge))**2)

    return variance

#************ Learning ************#

def median(rankings): # Borda
    """ This function computes the central permutation (consensus ranking) given
    several permutations.
        Parameters
        ----------
        rankings: ndarray
            Matrix of several permutations
        Returns
        -------
        ndarray
            The central permutation of permutations given.
    """
    consensus =  np.argsort( # give the inverse of result --> sigma_0
                            np.argsort( # give the indexes to sort the sum vector --> sigma_0^-1
                                        rankings.sum(axis=0) # sum the indexes of all permutations
                                        )
                            )
    return consensus

def fit_mm(rankings, s0=None):
    """This function computes the consensus ranking and the MLE for the
    dispersion parameter phi for MM models.
        Parameters
        ----------
        rankings: ndarray
            The matrix of permutations
        s0: ndarray, optional
            The consensus ranking (default value is None)
        Returns
        -------
        tuple
            The ndarray corresponding to s0 the consensus permutation and the
            MLE for the dispersion parameter phi.
    """
    m, n = rankings.shape
    if s0 is None: s0 = np.argsort(np.argsort(rankings.sum(axis=0))) #borda
    dist_avg = np.mean(np.array([distance(s0, perm) for perm in rankings]))
    try:
        theta = sp_opt.newton(mle_theta_mm_f, 0.01, fprime=mle_theta_mm_fdev, args=(n, dist_avg), tol=1.48e-08, maxiter=500, fprime2=None)
    except:
        if dist_avg == 0.0:
            return s0, np.exp(-5)#=phi
        print("Error in function: fit_mm. dist_avg=",dist_avg, dist_avg == 0.0)
        print(rankings)
        print(s0)
        raise
    return s0, np.exp(-theta)#=phi

#************ Top-k rankings ************#
#****************************************#

#*************** Distance ***************#

def p_distance(beta_1, beta_2, k, p=0):
    """This function returns the distance between top-k rankings using
    the p-parametrized Kendall's-tau distance.
    Parameters
    ----------
    beta_1: ndarray
        A top-k permutation
    beta_2: ndarray
        A top-k permutation
    k: int
        Length of partial permutations (only top items)
    p: float
        The parameter in [0, 1]
    Returns
    -------
    float
        The p-parametrized Kendall's-tau distance.
    """

    alpha_1 = beta_to_alpha(beta_1, k=k)
    alpha_2 = beta_to_alpha(beta_2, k=k)
    d = 0
    p_counter = 0
    alpha_1Ualpha_2 = list(set(int(x) for x in np.union1d(alpha_1, alpha_2) if np.isnan(x) == False))
    for i_index, i in enumerate(alpha_1Ualpha_2):
        i_1_nan = np.isnan(beta_1[i])
        i_2_nan = np.isnan(beta_2[i])
        for j in alpha_1Ualpha_2[i_index + 1:] :
            j_1_nan = np.isnan(beta_1[j])
            j_2_nan = np.isnan(beta_2[j])
            if not i_1_nan and  not j_1_nan and not i_2_nan and not j_2_nan:
                if ( beta_1[i] > beta_1[j] and beta_2[i] > beta_2[j] ) or \
                ( beta_1[i] < beta_1[j] and beta_2[i] < beta_2[j] ):
                    continue
                elif ( beta_1[i] > beta_1[j] and beta_2[i] < beta_2[j] ) or \
                ( beta_1[i] < beta_1[j] and beta_2[i] > beta_2[j] ):
                    d += 1
            elif ( not i_1_nan and not j_1_nan and ( (not i_2_nan and j_2_nan) or (i_2_nan and not j_2_nan) ) ) or \
            ( not i_2_nan and not j_2_nan and ( (not i_1_nan and j_1_nan) or (i_1_nan and not j_1_nan) ) ):
                if i_1_nan:
                    d += int(beta_2[j] > beta_2[i])
                elif j_1_nan:
                    d += int(beta_2[i] > beta_2[j])
                elif i_2_nan:
                    d += int(beta_1[j] > beta_1[i])
                elif j_2_nan:
                    d += int(beta_1[i] > beta_1[j])
            elif ( not i_1_nan and j_1_nan and i_2_nan and not j_2_nan ) or \
            ( i_1_nan and not j_1_nan and not i_2_nan and j_2_nan ):
                d += 1
            elif ( not i_1_nan and not j_1_nan and i_2_nan and j_2_nan ) or \
            ( i_1_nan and j_1_nan and not i_2_nan and not j_2_nan ):
                p_counter += 1
    return d + p_counter*p
#********** Expected distance **********#

def expected_dist_top_k(n, k, theta=None, phi=None):
    """Compute the expected distance for top-k rankings, following
    a MM under the Kendall's-tau distance.
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        k: int
            Length of partial permutations (only top items)
        theta: float, optional (if phi is given)
            Real dispersion parameter
        phi  : float, optional (if theta is given)
            Real dispersion parameter
        Returns
        -------
        float
            The expected disance under the MMs.
    """
    theta, phi = mm.check_theta_phi(theta, phi)
    rnge = np.array(range(n-k+1,n+1))
    expected_dist = k * phi / (1-phi) - np.sum(rnge * pow(phi,rnge) / (1 - pow(phi, rnge)))
    return expected_dist

#************ Variance *************#

def variance_dist_top_k(n, k, theta=None, phi=None):
    """Compute the variance of the distance for top-k rankings, following
        a MM under the Kendall's-tau distance.
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        k: int
            Length of partial permutations (only top items)
        theta: float, optional (if phi is given)
            Real dispersion parameter
        phi  : float, optional (if theta is given)
            Real dispersion parameter
        Returns
        -------
        float
            The variance under the MMs.
    """
    theta, phi = mm.check_theta_phi(theta, phi)
    rnge = np.array(range(n-k+1,n+1))
    variance = (phi*k)/(1-phi)**2 - np.sum((pow(phi,rnge) * rnge**2)/(1-pow(phi,rnge))**2)
    return variance

#***** Expected/Variance vector *****#

def expected_v(n, theta=None, phi=None, k=None):#txapu integrar
    """This function computes the expected decomposition vector.
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float, optional (if phi is given)
            Real dispersion parameter
        phi  : float, optional (if theta is given)
            Real dispersion parameter
        k: int, optional
            Length of partial permutations (only top items)
        Returns
        -------
        ndarray
            The expected decomposition vector.
    """
    theta, phi = mm.check_theta_phi(theta, phi)
    if k is None: k = n-1
    if type(theta)!=list: theta = np.full(k, theta)
    rnge = np.array(range(k))
    expected_v = np.exp(-theta[rnge]) / (1-np.exp(-theta[rnge])) - (n-rnge) * np.exp(-(n-rnge)*theta[rnge]) / (1 - np.exp(-(n-rnge)*theta[rnge]))
    return expected_v

def variance_v(n, theta=None, phi=None, k=None):
    """This function computes the variance of the decomposition vector under GMM.
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float, optional (if phi is given)
            Real dispersion parameter
        phi  : float, optional (if theta is given)
            Real dispersion parameter
        k: int, optional
            Length of partial permutations (only top items)
        Returns
        -------
        ndarray
            The variance of the decomposition vector.
    """
    theta, phi = mm.check_theta_phi(theta, phi)
    if k is None:
        k = n-1
    if type(phi)!=list:
        phi = np.full(k, phi)
    rnge = np.array(range(k))
    var_v = phi[rnge]/(1-phi[rnge])**2 - (n-rnge)**2 * phi[rnge]**(n-rnge) / (1-phi[rnge]**(n-rnge))**2
    return var_v


#******** More functions *********#

def prob(sigma, sigma0, theta=None, phi=None):
    """Probability mass function of a MM with central ranking sigma0 and
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
    n = len(sigma)
    theta, phi = mm.check_theta_phi(theta, phi)
    sigma0_inv = pu.inverse(sigma0)
    rnge = np.array(range(n-1))

    psi = (1 - np.exp(( - n + rnge )*(theta)))/(1 - np.exp( -theta))
    psi = np.prod(psi)

    dist = distance( pu.compose(sigma, sigma0_inv) )

    return np.exp( - theta *  dist ) / psi

def borda_partial(rankings, w, k):
    """This function approximate the consensus ranking of a top-k rankings using Borda algorithm.
        Each nan-ranked item is assumed to have ranking $k$
        Parameters
        ----------
        rankings: ndarray
            The matrix of permutations
        w: float
            weight of each ranking
        k: int
            Length of partial permutations (only top items)
        Returns
        -------
        ndarray
            Consensus ranking.
    """
    a, b = rankings, w
    a, b = np.nan_to_num(rankings,nan=k), w
    aux = a * b
    borda = np.argsort(np.argsort(np.nanmean(aux, axis=0))).astype(float)
    mask = np.isnan(rankings).all(axis=0)
    borda[mask]=np.nan
    return borda

def psi_mm(n, theta=None, phi=None):
    """This function computes the normalization constant psi.
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter (optional if phi is given)
        phi: float
            Real dispersion parameter (optional if theta is given)
        Returns
        -------
        float
            The normalization constant psi.
    """
    rnge = np.array(range(2,n+1))
    if theta is not None:
        return np.prod((1-np.exp(-theta*rnge))/(1-np.exp(-theta)))
    if phi is not None:
        return np.prod((1-np.power(phi,rnge))/(1-phi))
    theta, phi = mm.check_theta_phi(theta, phi)



def fit_gmm(rankings, s0=None):
    """This function computes the consensus permutation and the MLE for the
    dispersion parameters theta_j for GMM models
        Parameters
        ----------
        rankings: ndarray
            The matrix of permutations
        s0: ndarray, optional
            The consensus permutation (default value is None)
        Returns
        -------
        tuple
            The ndarray corresponding to s0 the consensus permutation and the
            MLE for the dispersion parameters theta
    """
    m, n = rankings.shape
    if s0 is None:
        s0 = np.argsort(np.argsort(rankings.sum(axis=0))) #borda
    V_avg = np.mean(np.array([ranking_to_v(sigma)[:-1] for sigma in rankings]), axis = 0)
    try:
        theta = []
        for j in range(1, n):
            theta_j = sp_opt.newton(mle_theta_j_gmm_f, 0.01, fprime=mle_theta_j_gmm_fdev, args=(n, j, V_avg[j-1]), tol=1.48e-08, maxiter=500, fprime2=None)
            theta.append(theta_j)
    except:
        print("Error in function fit_gmm")
        raise
    return s0, theta


def mle_theta_mm_f(theta, n, dist_avg):
    """Compute the derivative of the likelihood.
    parameter
        Parameters
        ----------
        theta: float
            The dispersion parameter
        n: int
            Length of the permutations
        dist_avg: float
            Average distance of the sample (between the consensus and the
            permutations of the consensus)
        Returns
        -------
        float
            Value of the function for given parameters.
    """
    aux = 0
    rnge = np.array(range(1,n))
    aux = np.sum((n-rnge+1)*np.exp(-theta*(n-rnge+1))/(1-np.exp(-theta*(n-rnge+1))))
    aux2 = (n-1) / (np.exp( theta ) - 1) - dist_avg

    return aux2 - aux

def mle_theta_mm_fdev(theta, n, dist_avg):
    """This function computes the derivative of the function mle_theta_mm_f
    given the dispersion parameter and the average distance.
        Parameters
        ----------
        theta: float
            The dispersion parameter
        n: int
            Length of the permutations
        dist_avg: float
            Average distance of the sample (between the consensus and the
            permutations of the consensus)
        Returns
        -------
        float
            The value of the derivative of function mle_theta_mm_f for given
            parameters.
    """
    aux = 0
    rnge = np.array(range(1, n))
    aux = np.sum((n-rnge+1)*(n-rnge+1)*np.exp(-theta*(n-rnge+1))/pow((1 - np.exp(-theta * (n-rnge+1))), 2))
    aux2 = (- n + 1) * np.exp( theta ) / pow ((np.exp( theta ) - 1), 2)

    return aux2 + aux

def mle_theta_j_gmm_f(theta_j, n, j, v_j_avg):
    """Compute the derivative of the likelihood parameter theta_j in the GMM.
        Parameters
        ----------
        theta: float
            The jth dispersion parameter theta_j
        n: int
            Length of the permutations
        j: int
            The position of the theta_j in vector theta of dispersion parameters
        v_j_avg: float
            jth element of the average decomposition vector over the sample
        Returns
        -------
        float
            Value of the function for given parameters.
    """
    f_1 = np.exp( -theta_j ) / ( 1 - np.exp( -theta_j ) )
    f_2 = - ( n - j + 1 ) * np.exp( - theta_j * ( n - j + 1 ) ) / ( 1 - np.exp( - theta_j * ( n - j + 1 ) ) )
    return f_1 + f_2 - v_j_avg

def mle_theta_j_gmm_fdev(theta_j, n, j, v_j_avg):
    """This function computes the derivative of the function mle_theta_j_gmm_f
    given the jth element of the dispersion parameter and the jth element of the
    average decomposition vector.
        Parameters
        ----------
        theta: float
            The jth dispersion parameter theta_j
        n: int
            Length of the permutations
        j: int
            The position of the theta_j in vector theta of dispersion parameters
        v_j_avg: float
            jth element of the average decomposition vector over the sample
        Returns
        -------
        float
            The value of the derivative of function mle_theta_j_gmm_f for given
            parameters.
    """
    fdev_1 = - np.exp( - theta_j ) / pow( ( 1 - np.exp( -theta_j ) ), 2 )
    fdev_2 = pow( n - j + 1, 2 ) * np.exp( - theta_j * ( n - j + 1 ) ) / pow( 1 - np.exp( - theta_j * ( n - j + 1 ) ), 2 )
    return fdev_1 + fdev_2

def likelihood_mm(perms, s0, theta):
    """This function computes the log-likelihood for MM model given a matrix of
    permutation, the consensus permutation, and the dispersion parameter.
        Parameters
        ----------
        perms: ndarray
            A matrix of permutations
        s0: ndarray
            The consensus permutation
        theta: float
            The dispersion parameter
        Returns
        -------
        float
            Value of log-likelihood for given parameters.
    """
    m,n = perms.shape
    rnge = np.array(range(2,n+1))
    psi = 1.0 / np.prod((1-np.exp(-theta*rnge))/(1-np.exp(-theta)))
    probs = np.array([np.log(np.exp(-distance(s0, perm)*theta)/psi) for perm in perms])

    return probs.sum()




def alpha_to_beta(alpha,k): #aux for the p_distance
    inv = np.full(len(alpha), np.nan)
    for i,j in enumerate(alpha[:k]):
        inv[int(j)] = i
    return inv
def beta_to_alpha(beta,k): #aux for the p_distance
    inv = np.full(len(beta), np.nan)
    for i,j in enumerate(beta):
        if not np.isnan(j):
            inv[int(j)] = i
    return inv



def find_phi_n(n, bins):
    """ Divide the expected distances into bins and return both the expected
    distances and their corresponding values of dispersion parameter phi.
    Parameters
    ----------
    n: int
        Length of permutations
    bins: int
        Number of bins
    Returns
    -------
    tuple
        An array of expected distances and their corresponding dispersion parameter phi
    """
    ed, phi_ed = [], []
    ed_uniform = (n*(n-1)/2)/2
    for dmin in np.linspace(0,ed_uniform-1,bins):
        ed.append(dmin)
        phi_ed.append(find_phi(n, dmin, dmin+1))
    return ed, phi_ed

def find_phi(n, dmin, dmax):
    """Find the dispersion parameter phi that gives an expected distance between
    dmin and dmax where the length of rankings is n.
    Parameters
    ----------
    n: int
        Length of permutations
    dmin: int
        The minimum of expected distance
    dmax: int
        The maximum of expected distance
    Returns
    -------
    float
        The value of phi.
    """
    imin, imax = np.float64(0),np.float64(1)
    iterat = 0
    while iterat < 500:
        med = imin + (imax-imin)/2
        d = expected_dist_mm(n, mm.phi_to_theta(med))
        if d < dmax and d > dmin: return med
        elif d < dmin : imin = med
        elif d > dmax : imax = med
        iterat  += 1




# end