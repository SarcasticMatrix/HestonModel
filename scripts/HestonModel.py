import numpy as np

class HestonModel:
    def __init__(self, S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K, n):

        # Simulation parameters
        self.S0 = S0                # spot price
        self.V0 = V0                # initial variance
        self.n = n                  # number of points in path

        # Model parameters
        self.kappa = kappa          # mean reversion speed
        self.theta = theta          # long term variance
        self.sigma = sigma          # vol of vol (vol of variance)
        self.rho = rho              # correlation
        self.drift_emm = drift_emm  # lambda from P to martingale measure Q

        # Option parameters
        self.T = T                  # maturity
        self.K = K                  # strike
        self.r = r                  # interest rate
        

    def simulate_stock_and_variance(self, scheme="euler", N = 1000):
        dt = self.T / self.n
        S = np.zeros((N, self.n+1))
        V = np.zeros((N, self.n+1))
        S[:, 0] = self.S0
        V[:, 0] = self.V0

        for i in range(1, self.n+1):
            Z1 = np.random.normal(0, 1, N)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, N)

            if scheme == "euler":
                S[:, i] = S[:, i-1] + self.r * S[:, i-1] * dt + np.sqrt(V[:, i-1] * dt) * S[:, i-1] * Z1
                V[:, i] = V[:, i-1] + (self.kappa * (self.theta - V[:, i-1]) - self.drift_emm) * dt + self.sigma * np.sqrt(V[:, i-1] * dt) * Z2
            elif scheme == "milstein":
                S[:, i] = S[:, i-1] + self.r * S[:, i-1] * dt + np.sqrt(V[:, i-1] * dt) * S[:, i-1] * Z1
                V[:, i] = V[:, i-1] + (self.kappa * (self.theta - V[:, i-1]) - self.drift_emm) * dt + self.sigma * np.sqrt(V[:, i-1] * dt) * Z2 + \
                          0.5 * self.sigma**2 * dt * (Z2**2 - 1)
            else: 
                print('Please choose between: euler and milstein schema')

        return S, V

    def monte_carlo_call_price(self, scheme="euler"):
        S, V = self.simulate_stock_and_variance(scheme)
        ST = S[:, -1]
        payoff = np.maximum(ST - self.K, 0)
        discounted_payoff = np.exp(-self.r * self.T) * payoff
        call_price = np.mean(discounted_payoff)
        return call_price


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
    N = 1000
    n = 100

    heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K, n)
    call_price_euler = heston.monte_carlo_call_price(scheme="euler")
    call_price_milstein = heston.monte_carlo_call_price(scheme="milstein")

    print("Prix d'un call avec méthode d'Euler:", call_price_euler)
    print("Prix d'un call avec méthode de Milstein:", call_price_milstein)
