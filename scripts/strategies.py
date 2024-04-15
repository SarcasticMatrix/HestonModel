from scripts.HestonModel import HestonModel
import numpy as np

def naive_strategy(
        percentage_in_bank_account: float,
        length: float
    ) -> np.array:
    """
    Implement a naive time-constant strategy. We put percentage_in_bank_account % at each time of the portfolio in the bank account.
    Args:
    - percentage_in_bank_account (float): % of portfolio to put into the bank account
    - length (int): number of time step, should be set at len(S)

    Returns:
    - np.array: Number of stocks over time to hold.
    """

    allocation = np.array([1 - percentage_in_bank_account] * length) 
    return allocation

def time_varying_strategy(
        premium_volatility_risk: float, 
        p: float, 
        heston: HestonModel, 
        V: np.array
    ) -> np.array:
    """
    Function to determine the allocation of the portfolio based on premium volatility risk and p.

    Args:
    - premium_volatility_risk (float): Premium volatility risk parameter.
    - p (float): Parameter.

    Returns:
    - np.array: Number of stocks over time to hold.
    """

    alpha = heston.r + np.sqrt(V)
    returns = heston.r + premium_volatility_risk * np.sqrt(V)

    allocation = (alpha - returns) / ((1-p) * V) 
    return allocation


def optimal_allocate_strategy(
        heston:HestonModel, 
        p:float, 
        time:np.array
    ) -> np.array:
    """
    Implement the optimal allocation of the portfolio based on premium volatility risk and p.

    Args:
    - premium_volatility_risk (float): Premium volatility risk parameter.
    - p (float): Parameter.

    Returns:
    - np.array: Number of stocks over time to hold over time.
    """

    k0 = p * heston.premium_volatility_risk**2 / (1-p)
    k1 = heston.kappa - p * heston.premium_volatility_risk * heston.sigma * heston.rho / (1-p)
    k2 = heston.sigma**2 + p * heston.sigma**2 * heston.rho**2 /(1-p)
    k3 = np.sqrt(k1**2 - k0*k2)
    
    b = lambda t: k0 * (np.exp(k3 * (heston.T-t)) - 1) / (np.exp(k3 * (heston.T-t)) * (k1 + k3) - k1 + k3)
    pi = lambda t: heston.premium_volatility_risk / (1-p) + (heston.sigma * heston.rho) / (1-p) * b(t) 

    allocation = pi(time) # np.array([pi(t) for t in time])
    return allocation