import numpy as np
import matplotlib.pyplot as plt 

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

            if np.any(V[i-1, :] == 0):
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
                V[:, i] += 1/4 * self.sigma**2 * (ZV**2 - dt)
            elif scheme == 'euler':
                pass
            else: 
                print("Choose a scheme between: 'euler' or 'milstein'")

        return S, V, null_variance

    def monte_carlo_call_price(self,
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
            - call_price (float): estimation by Monte Carlo of the call price
            - standard_deviation (float): standard deviation of the option payoff 
        """
                
        S, _, null_variance = self.simulate(scheme, n, N)
        print(f"Variance has been null {null_variance} times over the {n*N} iterations ({round(null_variance/(n*N)*100,2)}%) ")

        ST = S[:, -1]
        payoff = np.maximum(ST - self.K, 0)
        discounted_payoff = np.exp(-self.r * self.T) * payoff

        call_price = np.mean(discounted_payoff)
        standard_deviation = np.std(discounted_payoff, ddof=1)/np.sqrt(N)

        return call_price, standard_deviation

    def plot_simulation(self, n: int = 1000):
        S, V, _ = self.simulate(n=n)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(np.linspace(0,1,n+1), S[0, :], label='Stock', color='blue', linewidth=1)
        ax1.set_xlabel('Time [h]', fontsize=12)
        ax1.set_ylabel('Stock Price [$]', fontsize=12)

        ax2.plot(np.linspace(0,1,n+1), V[0, :], label='Variance', color='orange', linewidth=1)
        ax2.set_xlabel('Time [h]', fontsize=12)
        ax2.set_ylabel('Variance', fontsize=12)

        fig.suptitle('Heston Model Simulation', fontsize=16)
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
    call_price_euler, call_std_euler = heston.monte_carlo_call_price(scheme="euler", n=100, N=1000)
    call_price_euler = round(call_price_euler, 2)
    call_std_euler = round(call_std_euler, 2)
    print(f"Call price and std with Euler scheme: ${call_price_euler} and {call_std_euler}")

    call_price_milstein, call_std_milstein = heston.monte_carlo_call_price(scheme="milstein", n=100, N=1000)
    call_price_milstein = round(call_price_milstein, 2)
    call_std_milstein = round(call_std_milstein, 2)
    print(f"Call price and std with Milstein scheme: ${call_price_milstein} and {call_std_milstein}")


    heston.plot_simulation()