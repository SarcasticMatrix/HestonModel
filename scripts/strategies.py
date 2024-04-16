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

    allocation = pi(time) 
    return allocation


from scripts.HestonModel import HestonModel
from scripts.Portfolio import Portfolio
from tqdm import tqdm

import numpy as np
from tqdm import tqdm

def run_strategies(seeds, portfolio0):
    """
    Run len(seeds) simulations and compute the mean and standard deviation PnL of each strategy 
    """

    ### Initialisation of the model

    S0 = 100
    V0 = 0.06
    r = 0.05
    kappa = 1
    theta = 0.06
    drift_emm = 0.01 
    sigma = 0.3
    rho = -0.5
    T = 1
    K = 100

    premium_volatility_risk = 0.05

    N = len(seeds)

    PnL1_list = []
    PnL2_list = []
    PnL3_list = []
    PnL_opt_list = []

    for seed in tqdm(seeds):

        heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K, premium_volatility_risk, seed)

        S, V, _ = heston.simulate(scheme='milstein', n=1000, N=1)
        S = S.flatten()
        V = V.flatten()

        time = np.linspace(start=0, stop=T, num=len(S))
        dt = time[1] - time[0]

        portfolio = Portfolio(r=r, dt=dt)

        ### Naive constant allocation strategy

        allocation1 = naive_strategy(0.5, len(S))
        bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=portfolio0, allocation_strategy=allocation1)
        portfolio_value1 = bank_account + stocks_account
        PnL1_list.append(portfolio_value1[-1] - portfolio0)

        allocation2 = naive_strategy(0.7, len(S))
        bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=portfolio0, allocation_strategy=allocation2)
        portfolio_value2 = bank_account + stocks_account
        PnL2_list.append(portfolio_value2[-1] - portfolio0)

        allocation3 = naive_strategy(1, len(S))
        bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=portfolio0, allocation_strategy=allocation3)
        portfolio_value3 = bank_account + stocks_account
        PnL3_list.append(portfolio_value3[-1] - portfolio0)

        ### Optimal allocation strategy

        p = 0.02
        optimal_allocation = optimal_allocate_strategy(heston=heston, p=p, time=time)
        bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=portfolio0, allocation_strategy=optimal_allocation)
        portfolio_value_optimal = bank_account + stocks_account
        PnL_opt_list.append(portfolio_value_optimal[-1] - portfolio0)

    PnL1_arr = np.array(PnL1_list)
    PnL2_arr = np.array(PnL2_list)
    PnL3_arr = np.array(PnL3_list)
    PnL_opt_arr = np.array(PnL_opt_list)

    PnL1_mean = np.mean(PnL1_arr)
    PnL2_mean = np.mean(PnL2_arr)
    PnL3_mean = np.mean(PnL3_arr)
    PnL_opt_mean = np.mean(PnL_opt_arr)

    PnL1_std = np.std(PnL1_arr, ddof=1)
    PnL2_std = np.std(PnL2_arr, ddof=1)
    PnL3_std = np.std(PnL3_arr, ddof=1)
    PnL_opt_std = np.std(PnL_opt_arr, ddof=1)
        
    print("Mean and Standard Deviation of PnL for Strategy 1:", (PnL1_mean, PnL1_std))
    print("Mean and Standard Deviation of PnL for Strategy 2:", (PnL2_mean, PnL2_std))
    print("Mean and Standard Deviation of PnL for Strategy 3:", (PnL3_mean, PnL3_std))
    print("Mean and Standard Deviation of PnL for Optimal Strategy:", (PnL_opt_mean, PnL_opt_std))



        
