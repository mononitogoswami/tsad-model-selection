import numpy as np

def check_theta_phi(theta, phi):
    """This function automatically converts theta to phi or phi to theta as
    list or float depending on the values and value types given as input.
        Parameters
        ----------
        theta: float or list
            Dispersion parameter theta to convert to phi (can be None)
        phi: float or list
            Dispersion parameter phi to convert to theta (can be None)
        Returns
        -------
        tuple
            tuple containing both theta and phi (of list or float type depending on the input type)
    """
    if not ((phi is None) ^ (theta is None)):
        print("Set valid values for phi or theta")
    if phi is None and type(theta) != list:
        phi = theta_to_phi(theta)
    if theta is None and type(phi) != list:
        theta = phi_to_theta(phi)
    if phi is None and type(theta) == list:
        phi = [theta_to_phi(t) for t in theta]
    if theta is None and type(phi) == list:
        theta = [phi_to_theta(p) for p in phi]
    return np.array(theta), np.array(phi)


def theta_to_phi(theta):
    """ This functions converts theta dispersion parameter into phi
        Parameters
        ----------
        theta: float
            Real dispersion parameter
        Returns
        -------
        float
            phi real dispersion parameter
    """
    return np.exp(-theta)


def phi_to_theta(phi):
    """This functions converts phi dispersion parameter into theta
        Parameters
        ----------
        phi: float
            Real dispersion parameter
        Returns
        -------
        float
            theta real dispersion parameter
    """
    return -np.log(phi)
