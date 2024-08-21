import numpy as np
from numpy import random
from random import gauss
from scipy.stats import norm
random.seed(42)
from scipy.integrate import quad 

import time
import matplotlib.pyplot as plt
from collections import namedtuple

import warnings
warnings.filterwarnings("ignore")

class HestonModel:
    """
    Class to represent a Heston Model: can simulate trajectories and price call options with this underlying.
    """
    
    def __init__(self, S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K, premium_volatility_risk = 0.0, seed=42):
        """
        Initialize the Heston Model with specified parameters.

        Parameters:
        - S0 (float): spot price
        - V0 (float): initial variance
        - r (float): interest rate
        - kappa (float): mean reversion speed
        - theta (float): long term variance
        - drift_emm (float): lambda from P to martingale measure Q (Equivalent Martingale Measure)
        - sigma (float): vol of variance 
        - rho (float): correlation
        - T (float): maturity
        - K (float): strike
        - premium_volatility_risk (float): premium for volatility risk by default is 0.0
        - seed (int): random seed, by default set at 42
        """

        # Simulation parameters
        self.S0 = S0                # spot price
        self.V0 = V0                # initial variance

        # Model parameters
        self.kappa = kappa          # mean reversion speed
        self.theta = theta          # long term variance
        self.sigma = sigma          # vol of variance
        self.rho = rho              # correlation
        self.drift_emm = drift_emm  # lambda from P to martingale measure Q (Equivalent Martingale Measure)
        self.premium_volatility_risk = premium_volatility_risk

        # Option parameters
        self.T = T                  # maturity
        self.K = K                  # strike
        self.r = r                  # interest rate
        
        self.seed = seed            # random seed

    def simulate(self, 
                scheme : str = "euler", 
                n: int = 100, 
                N:int = 1000,
        ) -> tuple:
        # generateHestonPathEulerDisc and generateHestonPathMilsteinDisc
        """
        Simulates and returns several simulated paths following the Heston model.

        Parameters:
        - scheme (str): the discretization scheme used
        - n (int): number of points in a path
        - N (int): number of simulated paths

        Returns:
        - S (np.array): stock paths
        - V (np.array): variance paths
        - null_variance (int): number of times the simulated variance has been null
        """
        random.seed(self.seed)

        dt = self.T / n
        S = np.zeros((N, n + 1))
        V = np.zeros((N, n + 1))
        S[:, 0] = self.S0
        V[:, 0] = self.V0

        null_variance = 0

        for i in range(1, n+1):

            # Apply reflection scheme 
            if np.any(V[:, i-1] < 0):
                V[:, i-1] = np.abs(V[:, i-1])

            if np.any(V[:, i-1] == 0):
                null_variance += np.sum(V[i-1, :] == 0) 

            # Brownian motion
            N1 = np.random.normal(loc=0, scale=1, size=N)
            N2 = np.random.normal(loc=0, scale=1, size=N)
            ZV = N1 * np.sqrt(dt)
            ZS = (self.rho * N1 + np.sqrt(1-self.rho**2) * N2) * np.sqrt(dt) 

            # Update the processes 
            #S[:, i] = S[:, i-1] + self.r * S[:, i-1] * dt + np.sqrt(V[:, i-1]) * S[:, i-1] * ZS
            S[:, i] = S[:, i-1] + (self.r + self.premium_volatility_risk * np.sqrt(V[:, i-1]))* S[:, i-1] * dt + np.sqrt(V[:, i-1]) * S[:, i-1] * ZS 

            V[:, i] = V[:, i-1] + (self.kappa * (self.theta - V[:, i-1]) - self.drift_emm * V[:, i-1]) * dt + self.sigma * np.sqrt(V[:, i-1]) * ZV 
            if scheme == "milstein":
                S[:, i] += 1/2 * V[:, i-1] * S[:, i-1] * (ZS**2 - dt) 
                #S[:, i] += 1/4 * S[:, i-1]**2 * (ZS**2 - dt) 
                V[:, i] += 1/4 * self.sigma**2 * (ZV**2 - dt)
            elif scheme == 'euler':
                pass
            else: 
                print("Choose a scheme between: 'euler' or 'milstein'")

        if N == 1:
            S = S.flatten()
            V = V.flatten()

        return S, V, null_variance

    def monte_carlo_price(self,
                               scheme: str = "euler",
                               n: int = 100,
                               N: int = 1000
                            ) -> float:
        # priceHestonCallViaEulerMC and priceHestonCallViaMilsteinMC
        """
        Simulates sample paths and estimates the call price with a simple Monte Carlo Method.

        Parameters:
        - scheme (str): the discretization scheme used
        - n (int): number of points in a path
        - N (int): number of simulated paths

        Returns:
        - result (namedtuple): with the following attributes:
            - price (float): estimation by Monte Carlo of the call price
            - standard_deviation (float): standard deviation of the option payoff
            - infimum (float): infimum of the confidence interval
            - supremum (float): supremum of the confidence interval
        """
        random.seed(self.seed)

        S, _, null_variance = self.simulate(scheme, n, N)
        print(f"Variance has been null {null_variance} times over the {n*N} iterations ({round(null_variance/(n*N)*100,2)}%) ")

        ST = S[:, -1]
        payoff = np.maximum(ST - self.K, 0)
        discounted_payoff = np.exp(-self.r * self.T) * payoff

        price = np.mean(discounted_payoff)
        standard_deviation = np.std(discounted_payoff, ddof=1)/np.sqrt(N)
        infimum = price - 1.96 * np.sqrt(standard_deviation / N)
        supremum = price + 1.96 * np.sqrt(standard_deviation / N)

        Result = namedtuple('Results','price std infinum supremum')
        return Result(price, standard_deviation, infimum, supremum) # price, standard_deviation, infimum, supremum

    def characteristic(
            self,
            j: int
        ) -> float:
        """
        Create the characteristic function Psi_j(x, v, t; u), for a given (x, v, t).

        Parameters:
        - j (int): index of the characteristic function

        Returns:
        - callable: characteristic function
        """

        if j == 1 : 
            uj = 1/2 
            bj = self.kappa + self.drift_emm - self.rho * self.sigma
        elif j == 2:
            uj = - 1/2
            bj = self.kappa + self.drift_emm
        else: 
            print('Argument j (int) must be 1 or 2')
            return 0
        a = self.kappa * self.theta 

        dj = lambda u : np.sqrt((self.rho * self.sigma * u * 1j - bj)**2 - self.sigma**2 * (2 * uj * u * 1j - u**2))
        gj = lambda u : (self.rho * self.sigma * u *1j - bj - dj(u))/(self.rho * self.sigma * u *1j - bj + dj(u))

        Cj = lambda tau, u : self.r * u * tau * 1j + a/self.sigma**2 * ((bj - self.rho * self.sigma * u * 1j + dj(u)) * tau - 2 * np.log((1-gj(u)*np.exp(dj(u)*tau))/(1-gj(u))))  
        Dj = lambda tau, u : (bj - self.rho * self.sigma * u * 1j + dj(u))/self.sigma**2 * (1-np.exp(dj(u) * tau))/(1-gj(u) * np.exp(dj(u)*tau))

        return lambda x, v, t, u : np.exp(Cj(self.T-t,u) + Dj(self.T-t,u)*v + u * x * 1j)
    
    def fourier_transform_price(
            self,
            t = 0
    ):
        """
        Computes the price of a European call option on the underlying asset S following a Heston model using the Heston formula.

        Parameters:
        - t (float): time

        Returns:
        - price (float): option price
        - error (float): error in the option price computation
        """

        x = np.log(self.S0)
        v = self.V0

        psi1 = self.characteristic(j=1)
        integrand1 = lambda u : np.real((np.exp(-u * np.log(self.K) * 1j) * psi1(x, v, t, u))/(u*1j)) 
        Q1 = 1/2 + 1/np.pi * quad(func = integrand1, a = 0, b = 1000)[0]
        error1 = 1/np.pi * quad(func = integrand1, a = 0, b = 1000)[1]

        psi2 = self.characteristic(j=2)
        integrand2 = lambda u : np.real((np.exp(-u * np.log(self.K) * 1j) * psi2(x, v, t, u))/(u*1j)) 
        Q2 = 1/2 + 1/np.pi * quad(func = integrand2, a = 0, b = 1000)[0]
        error2 = 1/np.pi * quad(func = integrand2, a = 0, b = 1000)[1]

        price = self.S0 * Q1 - self.K * np.exp(-self.r * (self.T - t)) * Q2
        error = self.S0 * error1 + self.K * np.exp(-self.r * (self.T - t)) * error2
        return price, error

    def carr_madan_price(self):
        """
        Computes the price of a European call option on the underlying asset S following a Heston model using Carr-Madan Fourier pricing.

        Returns:
        - price (float): option price
        - error (float): error in the option price computation
        """
        
        x = np.log(self.S0)
        v = self.V0
        t = self.T - 1
        alpha = 0.3

        price_hat = lambda u: np.exp(- self.r * self.T) / (alpha**2 + alpha - u**2 + u * (2 * alpha + 1) * 1j) \
            * self.characteristic(j=2)(x, v, t, u - (alpha + 1) * 1j)
        
        integrand = lambda u: np.exp(- u * np.log(self.K) * 1j) * price_hat(u)
        
        price = np.exp(- alpha * np.log(self.K)) / np.pi * quad(func = integrand, a = 0, b = 50)[0]
        error = np.exp(- alpha * np.log(self.K)) / np.pi * quad(func = integrand, a = 0, b = 50)[1]

        return price, error

    def plot_simulation(self, scheme : str = 'euler', n: int = 1000):
        """
        Plots the simulation of a Heston model trajectory.

        Parameters:
        - scheme (str): the discretization scheme used (euler or milstein)
        - n (int): number of points in a path
        """
        random.seed(self.seed)
        
        S, V, _ = self.simulate(n=n, scheme=scheme, N=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(np.linspace(0, 1, n+1), S, label='Risky asset', color='blue', linewidth=1)
        ax1.axhline(y=self.K, label=r'$K$', linestyle='--', color='black')
        ax1.set_ylabel('Value [$]', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8)
        ax1.minorticks_on()
        ax1.grid(which="minor", visible=False)

        ax2.plot(np.linspace(0,1,n+1), np.sqrt(V), label='Volatility', color='orange', linewidth=1)
        ax2.axhline(y=np.sqrt(self.theta), label=r'$\sqrt{\theta}$', linestyle='--', color='black')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Instantaneous volatility [%]', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8)
        ax2.minorticks_on()
        ax2.grid(which="minor", visible=False)       

        fig.suptitle(f'Heston Model Simulation with {scheme} scheme', fontsize=16)
        plt.tight_layout()
        plt.show()
    
        return S, V 




import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize
def calibrate(
    heston: HestonModel,
    option_type: str, 
    symbol: str = 'MSFT', 
    expiration_dates: list = None
):
    """
    Calibrer le modèle de Heston sur plusieurs dates d'expiration et strike associé.
    
    - expiration_dates : list of str
        Liste des dates d'expiration au format 'YYYY-MM-DD'.
    """

    if expiration_dates is None:
        expiration_dates = [
            '2024-05-24', '2024-05-31', 
            '2024-06-07', '2024-06-14', '2024-06-21', '2024-06-28', 
            # '2024-07-19', '2024-08-16', '2024-09-20', '2024-10-18', 
            # '2024-11-15', '2024-12-20', '2025-01-17', '2025-03-21', 
            # '2025-06-20', '2025-09-19', '2025-12-19', '2026-01-16', 
            # '2026-06-18', '2026-12-18'
        ]

    stock = yf.Ticker(symbol)

    start_date = datetime.now()
    history = stock.history(period="1d")
    spot = history['Close'].iloc[-1]
    heston.S0 = spot

    volumes = []
    strikes = []
    prices = []
    maturities = []
        
    for exp_date in expiration_dates:
        option_chain = stock.option_chain(exp_date)
        options_data = getattr(option_chain, "calls" if option_type.startswith("call") else "puts")

        volumes.append(options_data['volume'].values)
        strikes.append(options_data['strike'].values)
        prices.append(options_data['lastPrice'].values)

        expiration_date = datetime.strptime(exp_date, '%Y-%m-%d')
        time_remaining = expiration_date - start_date
        maturity = time_remaining.days / 365.25
        liquidity = len(options_data['volume'].values)
        maturities.append([maturity] * liquidity)

    volumes = np.hstack(volumes)
    strikes = np.hstack(strikes)
    prices = np.hstack(prices)
    maturities = np.hstack(maturities)
    nbr_data = len(prices)    

    is_nan_volumes = np.isnan(volumes)
    is_nan_prices = np.isnan(prices)
    mask = is_nan_volumes | is_nan_prices
    mask = ~ mask

    x0 = [heston.kappa, heston.theta, heston.sigma, heston.rho, heston.drift_emm, heston.V0]
    def objective_function(x):
        heston.theta = x[1]
        heston.sigma = x[2]
        heston.rho = x[3]
        heston.drift_emm = x[4]
        heston.V0 = x[5]

        model_prices = []
        for k in range(nbr_data):
            heston.K = strikes[k]
            heston.T = maturities[k]
            model_price, _ = heston.fourier_transform_price()
            model_prices.append(model_price)

        model_prices = np.array(model_prices)
        result = np.sum(
            volumes[mask] * np.abs(prices[mask] - model_prices[mask])
        )

        return result

    print('Callibration is running...')
    res = minimize(fun=objective_function, x0=x0)    

    return res












































