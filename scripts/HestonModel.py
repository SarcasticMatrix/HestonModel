import numpy as np
import matplotlib.pyplot as plt 
from collections import namedtuple
from numpy import random
random.seed(42)

class HestonModel:
    def __init__(self, S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K):

        # Simulation parameters
        self.S0 = S0                # spot price
        self.V0 = V0                # initial variance

        # Model parameters
        self.kappa = kappa          # mean reversion speed
        self.theta = theta          # long term variance
        self.sigma = sigma          # vol of vol (vol of variance)
        self.rho = rho              # correlation
        self.drift_emm = drift_emm  # lambda from P to martingale measure Q (Equivalent Martingale Measure)

        # Option parameters
        self.T = T                  # maturity
        self.K = K                  # strike
        self.r = r                  # interest rate
        

    def simulate(self, 
                scheme : str = "euler", 
                n: int = 100, 
                N:int = 1000
        ) -> tuple:
        # generateHestonPathEulerDisc and generateHestonPathMilsteinDisc
        """
        Simulates and returns several simulated paths following the Heston model
        Input: 
            - scheme (str): the discretization scheme used
            - n (int): number of points in a path
            - N (int): number of simulated paths
        Ouput:
            - S (np.array): stock paths
            - V (np.array): variance paths
            - null_variance (int): number of time the simulated variance has been null 
        """

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
            S[:, i] = S[:, i-1] + self.r * S[:, i-1] * dt + np.sqrt(V[:, i-1]) * S[:, i-1] * ZS
            V[:, i] = V[:, i-1] + (self.kappa * (self.theta - V[:, i-1]) - self.drift_emm * V[:, i-1]) * dt + self.sigma * np.sqrt(V[:, i-1]) * ZV 
            if scheme == "milstein":
                S[:, i] += 1/2 * V[:, i-1] * S[:, i-1] * (ZS**2 - dt) 
                #S[:, i] += 1/4 * S[:, i-1]**2 * (ZS**2 - dt) 
                V[:, i] += 1/4 * self.sigma**2 * (ZV**2 - dt)
            elif scheme == 'euler':
                pass
            else: 
                print("Choose a scheme between: 'euler' or 'milstein'")

        return S, V, null_variance

    def monte_carlo_price(self,
                               scheme: str = "euler",
                               n: int = 100,
                               N: int = 1000
                            ) -> float:
        # priceHestonCallViaEulerMC and priceHestonCallViaMilsteinMC
        """
        Simulates sample paths, then estimation the call price with a simple Monte Carlo Method
        Input: 
            - n (int): number of points in a path
            - N (int): number of simulated paths
        Ouput:
            - result (namedtuple): with the following attribute
                - price (float): estimation by Monte Carlo of the call price
                - standard_deviation (float): standard deviation of the option payoff 
                - infimum (float): infimum of the confidence interval
                - supremum (float): supremum of the confidence interval
        """
                
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

    def plot_simulation(self, scheme : str = 'euler', n: int = 1000):
        S, V, _ = self.simulate(n=n, scheme=scheme)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(np.linspace(0,1,n+1), S[0, :], label='Risky asset', color='blue', linewidth=1)
        ax1.set_ylabel('Value [$]', fontsize=12)

        #ax2.plot(np.linspace(0,1,n+1), np.sqrt(V[0, :]), label='Volatility', color='orange', linestyle='dotted', linewidth=1)
        ax2.plot(np.linspace(0,1,n+1), V[0, :], label='Variance', color='orange', linewidth=1)
        ax2.set_xlabel('Time [h]', fontsize=12)
        ax2.set_ylabel('Variance', fontsize=12)

        fig.suptitle(f'Heston Model Simulation with {scheme} scheme', fontsize=16)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
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

    n = 100
    N = 1000

    result = heston.monte_carlo_price(scheme="euler", n=n, N=N)
    price_euler = round(result.price, 2)
    std_euler = round(result.std, 2)
    infinum_euler = round(result.infinum, 2)
    supremum_euler = round(result.supremum, 2)
    print(f"Euler scheme : price ${price_euler}, std {std_euler}, and Confidence interval [{infinum_euler},{supremum_euler}]")

    result = heston.monte_carlo_price(scheme="milstein", n=n, N=N)
    price_milstein = round(result.price, 2)
    std_milstein = round(result.std, 2)
    infinum_milstein = round(result.infinum, 2)
    supremum_milstein = round(result.supremum, 2)
    print(f"Milstein scheme : price ${price_milstein}, std {std_milstein}, and Confidence interval [{infinum_milstein},{supremum_milstein}]")

    # scheme = 'milstein'
    # heston.plot_simulation(scheme)