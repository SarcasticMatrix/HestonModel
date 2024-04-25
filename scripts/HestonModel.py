import numpy as np
from numpy import random
from random import gauss
from scipy.stats import norm
random.seed(42)
from scipy.integrate import quad 

import time
import matplotlib.pyplot as plt
from collections import namedtuple

import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined
from particles.collectors import Moments

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
        #alpha = 1/2 * (self.kappa**2 / (self.sigma * (2 * self.kappa - self.sigma)) + self.kappa**2 / (self.sigma * (- 2 * self.kappa - self.sigma)))
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
        
        S, V, _ = self.simulate(n=n, scheme=scheme)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(np.linspace(0,1,n+1), S[0, :], label='Risky asset', color='blue', linewidth=1)
        ax1.set_ylabel('Value [$]', fontsize=12)
        ax1.legend()

        #ax2.plot(np.linspace(0,1,n+1), np.sqrt(V[0, :]), label='Volatility', color='orange', linestyle='dotted', linewidth=1)
        ax2.plot(np.linspace(0,1,n+1), V[0, :], label='Variance', color='orange', linewidth=1)
        ax2.set_xlabel('Time [h]', fontsize=12)
        ax2.set_ylabel('Variance', fontsize=12)
        ax2.legend()

        fig.suptitle(f'Heston Model Simulation with {scheme} scheme', fontsize=16)
        plt.tight_layout()
        plt.show()

    def SIR_estimation(
            self,
            log_returns: np.ndarray,
            useHeston: bool = True,
            N: int = 1000,
        ):
        """
            useHeston = use parameters of the heston et don't estimate with MCMC
        """
            
        class StochVol(ssm.StateSpaceModel):

            def PX0(self):  # Distribution of X_0
                return dists.Normal(loc=self.theta, scale=self.sigma) # scale = self.sigma / np.sqrt(1. - (self.kappa + self.drift)**2)
            
            def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
                return dists.Normal(loc = self.kappa * (self.theta - xp) - self.drift * xp, scale=self.sigma)
            
            def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
                return dists.Normal(loc = self.log_return, scale = np.exp(x))
        
        if useHeston:
            my_model = StochVol(theta=self.theta, sigma=self.sigma, kappa=self.kappa, drift=self.drift_emm, log_return=self.r) 
            fk_model = ssm.Bootstrap(ssm=my_model, data=log_returns)  
            pf = particles.SMC(
                fk=fk_model, 
                N=N, 
                qmc=False, 
                resampling='systematic', 
                store_history=False, 
                verbose=False, 
                collect=[Moments()]
            )
            pf.run()
        estimation = np.array([m['mean'] for m in pf.summaries.moments])
        return estimation
            

if __name__ == "__main__":

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

    heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)

    print("\nPricing...")

    ### Price via Monte Carlo

    n = 100
    N = 10**3

    start_time = time.time()
    result = heston.monte_carlo_price(scheme="euler", n=n, N=N)
    time_delta = round(time.time() - start_time,4)
    price_euler = round(result.price, 2)
    std_euler = round(result.std, 2)
    infinum_euler = round(result.infinum, 2)
    supremum_euler = round(result.supremum, 2)
    print(f"Monte Carlo Euler scheme in {time_delta}s : price ${price_euler}, std {std_euler}, and Confidence interval [{infinum_euler},{supremum_euler}]")

    start_time = time.time()
    result = heston.monte_carlo_price(scheme="milstein", n=n, N=N)
    time_delta = round(time.time() - start_time,4)
    price_milstein = round(result.price, 2)
    std_milstein = round(result.std, 2)
    infinum_milstein = round(result.infinum, 2)
    supremum_milstein = round(result.supremum, 2)
    print(f"Monte Carlo Milstein scheme in {time_delta}s : price ${price_milstein}, std {std_milstein}, and Confidence interval [{infinum_milstein},{supremum_milstein}]")

    ### Price via Fourier Transform

    start_time = time.time()
    price_FT, error_FT = heston.fourier_transform_price()
    time_delta = round(time.time() - start_time,4)
    infinum = round(price_FT-error_FT, 2)
    supremum = round(price_FT+error_FT, 2)
    price_FT = round(price_FT, 2)
    error_FT = round(error_FT, 8)
    print(f"Fourier Transform in {time_delta}s : price ${price_FT}, error ${error_FT} , and Confidence interval [{infinum},{supremum}]")

    ### Price via Carr-Madan formula 

    start_time = time.time()
    price_CM, error_CM = heston.carr_madan_price()
    time_delta = round(time.time() - start_time,4)
    infinum = round(price_CM-error_CM, 2)
    supremum = round(price_CM+error_CM, 2)
    price_CM = round(price_CM, 2)
    #error_CM = round(error_CM, 8)
    print(f"Carr-Madan in {time_delta}s : price ${price_CM}, error ${error_CM} , and Confidence interval [{infinum},{supremum}]")

    print("Pricing...finished\n")

    ### Path simulations

    scheme = 'milstein'
    heston.plot_simulation(scheme)

    ### Characteristic function

    psi1 = heston.characteristic(j=1)
    psi2 = heston.characteristic(j=2)

    u = np.arange(start=-20, stop=20,step=0.01)

    x = np.log(S0)
    v = V0
    t = T - 1 


    # 2D plot
    # Create subplots for real and imaginary parts
    plt.figure()

    # Plot real part of psi1 and psi2
    plt.subplot(1, 2, 1)
    plt.title(r'$\mathfrak{Re}(\psi_1)$ and $\mathfrak{Re}(\psi_2)$')
    plt.plot(u, np.abs(psi1(x, v, t, u)), label=r'$|\psi_1|$', color='orange', linestyle='--')
    plt.plot(u, psi1(x, v, t, u).real, label=r'$\psi_1$', color='orange')
    plt.plot(u, psi2(x, v, t, u).real, label=r'$\psi_2$', color='blue')
    plt.grid(visible=True)
    plt.xlabel(r'$u$')
    plt.ylabel('Real part')
    plt.legend()

    # Plot imaginary part of psi1 and psi2
    plt.subplot(1, 2, 2)
    plt.title(r'$\mathfrak{Im}(\psi_1)$ and $\mathfrak{Im}(\psi_2)$')
    plt.plot(u, np.abs(psi1(x, v, t, u)), label=r'$|\psi_1|$', color='orange', linestyle='--')
    plt.plot(u, psi1(x, v, t, u).imag, label=r'$\psi_1$', color='orange')
    plt.plot(u, psi2(x, v, t, u).imag, label=r'$\psi_2$', color='blue')
    plt.grid(visible=True)
    plt.xlabel(r'$u$')
    plt.ylabel('Imaginary part')
    plt.legend()

    #plt.tight_layout()
    plt.show() 


    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(u, psi1(x, v, t, u).real, psi1(x, v, t, u).imag, label=r'$\psi_1$', color='orange')
    ax.plot(u, psi2(x, v, t, u).real, psi2(x, v, t, u).imag, label=r'$\psi_2$', color='blue')
    ax.set_xticks([-5*i for i in range(6)] + [5*i for i in range(6)])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.set_xlabel('u')
    ax.set_ylabel('Real part')
    ax.set_zlabel('Imaginary part')
    plt.legend()
    plt.show()

    ## Integration over R_+
    psi1 = heston.characteristic(j=1)
    integrand1 = lambda u : np.real((np.exp(-u * np.log(heston.K) * 1j) * psi1(x, v, t, u))/(u*1j)) 
    psi2 = heston.characteristic(j=2)
    integrand2 = lambda u : np.real((np.exp(-u * np.log(heston.K) * 1j) * psi2(x, v, t, u))/(u*1j)) 

    u = np.arange(start=0, stop=40,step=0.01)

    plt.figure()
    plt.plot(u, integrand1(u) * u**2, label="Integrand 1")
    plt.plot(u, integrand2(u) * u**2, label="Integrand 2")
    plt.xlabel(r'u')
    plt.ylabel(r'Integrand $\times u^2$')
    plt.legend()
    plt.grid(visible=True)
    plt.title(r'Existence of $Q_1$ and $Q_2$')
    plt.show()