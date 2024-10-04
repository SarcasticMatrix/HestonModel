import numpy as np
from numpy import random
from scipy.stats import norm
from scipy.integrate import quad 

import matplotlib.pyplot as plt
from collections import namedtuple

import warnings
warnings.filterwarnings("ignore")

class Heston:
    """
    Class to represent a Heston Model: can simulate trajectories and price call options with this underlying.

    The Heston model is a popular stochastic volatility model used to describe the evolution of asset prices,
    particularly in the context of options pricing. This class allows for the simulation of price paths
    as well as the pricing of European call options using both Monte Carlo simulations and Fourier transform techniques.

    Attributes:
    - spot (float): Spot price of the underlying asset.
    - vol_initial (float): Initial variance of the asset's returns.
    - r (float): Risk-free interest rate.
    - kappa (float): Speed of mean reversion of the variance.
    - theta (float): Long-term average variance.
    - drift_emm (float): Drift term for the equivalent martingale measure.
    - sigma (float): Volatility of the variance process.
    - rho (float): Correlation between the Brownian motions driving the asset and variance.
    - T (float): Time to maturity for the options being priced.
    - K (float): Strike price of the options.
    - premium_volatility_risk (float): Premium for volatility risk (default is 0.0).
    - seed (int): Seed for random number generation (default is 42).

    Methods:
    - simulate(self, scheme: str = "euler", n: int = 100, N: int = 1000):
        Simulates and returns several simulated paths following the Heston model.
    - monte_carlo_price(self, scheme: str = "euler", n: int = 100, N: int = 1000):
        Simulates sample paths and estimates the call price with a simple Monte Carlo Method.
    - fourier_transform_price(self, t=0):
        Computes the price of a European call option on the underlying asset S following a Heston model using the Heston formula.
    - carr_madan_price(self):
        Computes the price of a European call option on the underlying asset S following a Heston model using Carr-Madan Fourier pricing.
    - plot_simulation(self, scheme: str = 'euler', n: int = 1000):
        Plots the simulation of a Heston model trajectory.
    - price_surface(self):
        Plot the Price of call option as a function of strike and time to maturity.

    Examples:
        # Parameters for the Heston model
        S0 = 100.0            # Initial spot price
        V0 = 0.06             # Initial volatility
        r = 0.05              # Risk-free interest rate
        kappa = 1.0           # Mean reversion rate
        theta = 0.06          # Long-term volatility
        drift_emm = 0.01      # Drift term
        sigma = 0.3           # Volatility of volatility
        rho = -0.5            # Correlation between asset and volatility
        T = 1.0               # Time to maturity in years
        K = 100.0             # Strike price

        # Create a Heston instance
        heston = Heston(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)

        print("\nPricing...")

        ### Price via Monte Carlo
        n = 100               # Number of paths
        N = 1000              # Number of time steps

        # Euler scheme pricing
        result = heston.monte_carlo_price(scheme="euler", n=n, N=N)
        price_euler = round(result.price, 2)
        std_euler = round(result.std, 2)
        infinum_euler = round(result.infinum, 2)
        supremum_euler = round(result.supremum, 2)
        print(f"Monte Carlo Euler scheme: price ${price_euler}, std {std_euler}, and Confidence interval [{infinum_euler},{supremum_euler}]\n")

        # Milstein scheme pricing
        result = heston.monte_carlo_price(scheme="milstein", n=n, N=N)
        price_milstein = round(result.price, 2)
        std_milstein = round(result.std, 2)
        infinum_milstein = round(result.infinum, 2)
        supremum_milstein = round(result.supremum, 2)
        print(f"Monte Carlo Milstein scheme: price ${price_milstein}, std {std_milstein}, and Confidence interval [{infinum_milstein},{supremum_milstein}]\n")

        ### Price via Fourier Transform
        price_FT, error_FT = heston.fourier_transform_price()
        infinum = round(price_FT - error_FT, 2)
        supremum = round(price_FT + error_FT, 2)
        price_FT = round(price_FT, 2)
        error_FT = round(error_FT, 8)
        print(f"Fourier Transform: price ${price_FT}, error ${error_FT}, and Confidence interval [{infinum},{supremum}]\n")

        ### Price via Carr-Madan formula 
        price_CM, error_CM = heston.carr_madan_price()
        infinum = round(price_CM - error_CM, 2)
        supremum = round(price_CM + error_CM, 2)
        price_CM = round(price_CM, 2)
        error_CM = round(error_CM, 14)
        print(f"Carr-Madan: price ${price_CM}, error ${error_CM}, and Confidence interval [{infinum},{supremum}]\n")

        print("\nPricing...finished\n")

        ### Path simulations
        scheme = 'milstein'
        heston.plot_simulation(scheme)
    """
    
    def __init__(self, spot, vol_initial, r, kappa, theta, drift_emm, sigma, rho, T, K, premium_volatility_risk = 0.0, seed=42):
        """
        Initialize the Heston Model with specified parameters.

        Parameters:
        - spot (float): spot price
        - vol_initial (float): initial variance
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
        self.spot = spot                # spot price
        self.vol_initial = vol_initial                # initial variance

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
        """
        Simulates multiple paths according to the Heston model.

        Parameters:
        - scheme (str): The discretization method to be used (e.g., "euler" or "milstein").
        - n (int): The number of discrete points in each path.
        - N (int): The total number of simulated paths.

        Returns:
        - S (np.array): Array of simulated stock price paths.
        - V (np.array): Array of simulated variance paths.
        - null_variance (int): Count of instances where simulated variance equals zero.
        """
        random.seed(self.seed)

        dt = self.T / n
        S = np.zeros((N, n + 1))
        V = np.zeros((N, n + 1))
        S[:, 0] = self.spot
        V[:, 0] = self.vol_initial

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
        """
        Simulates paths to estimate the price of a European call option using the Monte Carlo method.

        This method calculates the option price by averaging the discounted payoffs from simulated asset price paths.

        Parameters:
        - scheme (str): The discretization method used ("euler" or "milstein").
        - n (int): Number of discrete points in a path.
        - N (int): Number of paths to simulate.

        Returns:
        - result (namedtuple): Contains the following:
            - price (float): Estimated price of the call option.
            - standard_deviation (float): Standard deviation of the option payoffs.
            - infimum (float): Lower bound of the confidence interval.
            - supremum (float): Upper bound of the confidence interval.
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
        Creates the characteristic function Psi_j(x, v, t; u) for a given (x, v, t).

        This function returns the characteristic function based on the index provided.

        Parameters:
        - j (int): Index of the characteristic function (must be 1 or 2).

        Returns:
        - callable: The characteristic function.
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
        Calculates the price of a European call option using the Heston formula.

        This method computes the option price by evaluating the integral of the characteristic function.

        Parameters:
        - t (float): The current time for pricing (default is 0).

        Returns:
        - price (float): The calculated option price.
        - error (float): The error associated with the option price calculation.
        """

        x = np.log(self.spot)
        v = self.vol_initial

        psi1 = self.characteristic(j=1)
        integrand1 = lambda u : np.real((np.exp(-u * np.log(self.K) * 1j) * psi1(x, v, t, u))/(u*1j)) 
        Q1 = 1/2 + 1/np.pi * quad(func = integrand1, a = 0, b = 1000)[0]
        error1 = 1/np.pi * quad(func = integrand1, a = 0, b = 1000)[1]

        psi2 = self.characteristic(j=2)
        integrand2 = lambda u : np.real((np.exp(-u * np.log(self.K) * 1j) * psi2(x, v, t, u))/(u*1j)) 
        Q2 = 1/2 + 1/np.pi * quad(func = integrand2, a = 0, b = 1000)[0]
        error2 = 1/np.pi * quad(func = integrand2, a = 0, b = 1000)[1]

        price = self.spot * Q1 - self.K * np.exp(-self.r * (self.T - t)) * Q2
        error = self.spot * error1 + self.K * np.exp(-self.r * (self.T - t)) * error2
        return price, error

    def carr_madan_price(self):
        """
        Computes the price of a European call option using the Carr-Madan Fourier pricing method.

        This method employs the Carr-Madan approach, leveraging the characteristic function to calculate
        the option price.

        Returns:
        - price (float): The calculated option price.
        - error (float): The error associated with the option price calculation.
        """
        
        x = np.log(self.spot)
        v = self.vol_initial
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
        Visualizes the trajectory of the Heston model through simulation.

        This method generates plots for the simulated asset price and variance over time.

        Parameters:
        - scheme (str): The discretization method applied ("euler" or "milstein").
        - n (int): The number of points to include in each simulation path.
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
    
    def price_surface(self):
        """
        Visualizes the price of the call option in relation to strike price and time to maturity.

        This method creates a 3D surface plot illustrating how the call option price varies 
        based on different strike prices and times to maturity.
        """
        Ks = np.arange(start=20, stop=200, step=0.5)
        Ts = np.arange(start=0.1, stop=1.1, step=0.1)

        prices_surface = np.zeros((len(Ts), len(Ks)))

        for i, T in enumerate(Ts):
            for j, K in enumerate(Ks):
                heston = Heston(
                    self.spot, self.vol_initial, self.r, self.kappa, self.theta, self.drift_emm, self.sigma, self.rho, 
                    T=T, K=K)
                price, _ = heston.carr_madan_price()
                prices_surface[i, j] = price

        K_mesh, T_mesh = np.meshgrid(Ks, Ts)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_mesh, T_mesh, prices_surface, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.set_title('Call price as a function of strike and time to maturity')
        ax.set_xlabel(r'Strike ($K$)')
        ax.set_ylabel(r'Time to maturity ($T$)')
        ax.set_zlabel('Price')
        ax.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8)
        plt.show()

from datetime import datetime
from scipy.optimize import minimize
from hestonModel.option.data import get_options_data

def calibrate(
    flag_option: str, 
    heston: Heston,
    symbol: str = 'MSFT', 
):
    """
    Calibrates the Heston model using options data for various expiration dates and associated strikes.

    Parameters:
    - flag_option (str): Specifies the type of option (e.g., call or put).
    - heston (Heston): An instance of the Heston model to calibrate.
    - symbol (str): The stock symbol for which to gather options data; defaults to 'MSFT' for Microsoft Corporation.
    
    Returns:
    - res: The result of the optimization process, containing the optimized parameters for the Heston model.
    """
    # to do : implement for put options

    start_date = datetime.now()

    options_data, spot = get_options_data(symbol=symbol, flag_option=flag_option)
    heston.spot = spot

    # TEST
    mask = options_data['Volume'] > 0.1 * len(options_data)
    options_data = options_data.loc[mask]

    volumes = options_data["Volume"].values
    strikes = options_data["Strike"].values
    prices = options_data["Call Price"].values
    maturities = options_data["Time to Maturity"].values

    x0 = [
        heston.kappa, 
        heston.theta, 
        heston.sigma, 
        heston.rho, 
        heston.drift_emm, 
        heston.vol_initial
    ]
    
    def objective_function(x):
        heston.kappa = x[0]
        heston.theta = x[1]
        heston.sigma = x[2]
        heston.rho = x[3]
        heston.drift_emm = x[4]
        heston.vol_initial = x[5]

        model_prices = []
        for i in range(len(options_data)):
            heston.K = strikes[i]
            heston.T = maturities[i]
            model_price, _ = heston.fourier_transform_price()
            model_prices.append(model_price)

        model_prices = np.array(model_prices)
        weights = volumes / np.sum(volumes)
        
        result = np.sum(
            weights * (prices - model_prices)**2
        ) 

        return result

    print('Callibration is running...')
    res = minimize(fun=objective_function, x0=x0, method='Nelder-Mead')    

    return res

if __name__ == '__main__':

    # Parameters for the Heston model
    S0 = 100.0            # Initial spot price
    V0 = 0.06             # Initial volatility
    r = 0.05              # Risk-free interest rate
    kappa = 1.0           # Mean reversion rate
    theta = 0.06          # Long-term volatility
    drift_emm = 0.01      # Drift term
    sigma = 0.3           # Volatility of volatility
    rho = -0.5            # Correlation between asset and volatility
    T = 1.0               # Time to maturity in years
    K = 100.0             # Strike price

    # Create a Heston instance
    heston = Heston(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)

    print("\nPricing...")

    ### Price via Monte Carlo
    n = 100               # Number of paths
    N = 1000              # Number of time steps

    # Euler scheme pricing
    result = heston.monte_carlo_price(scheme="euler", n=n, N=N)
    price_euler = round(result.price, 2)
    std_euler = round(result.std, 2)
    infinum_euler = round(result.infinum, 2)
    supremum_euler = round(result.supremum, 2)
    print(f"Monte Carlo Euler scheme: price ${price_euler}, std {std_euler}, and Confidence interval [{infinum_euler},{supremum_euler}]\n")

    # Milstein scheme pricing
    result = heston.monte_carlo_price(scheme="milstein", n=n, N=N)
    price_milstein = round(result.price, 2)
    std_milstein = round(result.std, 2)
    infinum_milstein = round(result.infinum, 2)
    supremum_milstein = round(result.supremum, 2)
    print(f"Monte Carlo Milstein scheme: price ${price_milstein}, std {std_milstein}, and Confidence interval [{infinum_milstein},{supremum_milstein}]\n")

    ### Price via Fourier Transform
    price_FT, error_FT = heston.fourier_transform_price()
    infinum = round(price_FT - error_FT, 2)
    supremum = round(price_FT + error_FT, 2)
    price_FT = round(price_FT, 2)
    error_FT = round(error_FT, 8)
    print(f"Fourier Transform: price ${price_FT}, error ${error_FT}, and Confidence interval [{infinum},{supremum}]\n")

    ### Price via Carr-Madan formula 
    price_CM, error_CM = heston.carr_madan_price()
    infinum = round(price_CM - error_CM, 2)
    supremum = round(price_CM + error_CM, 2)
    price_CM = round(price_CM, 2)
    error_CM = round(error_CM, 14)
    print(f"Carr-Madan: price ${price_CM}, error ${error_CM}, and Confidence interval [{infinum},{supremum}]\n")

    print("\nPricing...finished\n")

    ### Path simulations
    scheme = 'milstein'
    heston.plot_simulation(scheme)












































