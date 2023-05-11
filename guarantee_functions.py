from math import sqrt, log, ceil
from scipy import special as sc


def compute_zeta_spibb(Nwedge, delta, gamma, vmax, nb_states, nb_actions, rho_b, rho_opt):
    return (4.0 * vmax) / (1.0 - gamma) * sqrt(
        2.0 / Nwedge * log((2.0 * nb_states * nb_actions * 2 ** nb_states) / delta)) - rho_opt + rho_b


def compute_Nwedge_2s(zeta, delta, gamma, vmax, nb_states, nb_actions, rho_b, rho_opt):
    zeta = zeta + rho_opt - rho_b
    return ceil(
        (32.0 * vmax ** 2) / (zeta ** 2 * (1.0 - gamma) ** 2) * log((8.0 * nb_states ** 2 * nb_actions ** 2) / delta))


def compute_Nwedge_beta(zeta, delta, gamma, vmax, nb_states, nb_actions, n_init, rho_b, rho_opt):
    n = n_init
    delta_t = delta / ((nb_states ** 2) * (nb_actions ** 2))
    bnd = (4 * vmax) / (1.0 - gamma) * sqrt(1.0 - sc.betaincinv(n / 2.0 + 1.0, 0.5, delta_t)) - rho_opt + rho_b
    lower_bound = 0

    # find a safe upper bound by scaling n exponentially
    while zeta < bnd:
        lower_bound = n
        n *= 2
        bnd = (4 * vmax) / (1.0 - gamma) * (1 - 2 * sc.betaincinv(n / 2 + 1, n / 2 + 1, delta_t/2)) - rho_opt + rho_b
    upper_bound = n

    # do a binary search on the interval [lower_bound, upper_bound]
    while lower_bound < upper_bound - 1:
        n = int((lower_bound + upper_bound) / 2)
        bnd = (4 * vmax) / (1.0 - gamma) * (1 - 2 * sc.betaincinv(n / 2 + 1, n / 2 + 1, delta_t/2)) - rho_opt + rho_b
        if zeta < bnd:
            lower_bound = n
        else:
            upper_bound = n

    # return the upper bound, since the exact Nwedge is a fraction between lower and upper bound, so we round up
    return upper_bound
