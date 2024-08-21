import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class BlackScholes:
    """
    Class representing a Black-Scholes model for stock price simulation.
    Parameters:
    - initial (float): initial value of the process
    - r (float): interest rate (drift coefficient) of the process
    - volatility (float): volatility coefficient of the process
    - T (float): time horizon of the simulation
    - seed (int): seed for random number generation
    Methods:
    - simulate(scheme='euler', n=100, N=1000): 
      Simulates and returns several simulated paths following the Black-Scholes model.
    - plot_simulation(scheme='euler', n=1000): 
      Plots the simulation of a Black-Scholes trajectory.
    """

    def __init__(
            self, 
            initial: float, 
            r: float, 
            mu: float,
            volatility: float, 
            T: float = 1, 
            seed: int = 42):
        """
        Initializes a Black-Scholes model object.
        Parameters:
        - initial (float): initial value of the process
        - r (float): interest rate (drift coefficient) of the process
        - mu (float): drift
        - volatility (float): volatility coefficient of the process
        - T (float): time horizon of the simulation
        - seed (int): seed for random number generation
        """

        # Black-Scholes parameters
        self.initial = initial
        self.r = r
        self.mu = mu
        self.volatility = volatility
        self.T = T
        self.seed = seed

    def simulate(
            self, 
            scheme: str = "euler", 
            n: int = 100, 
            N: int = 1000, 
        ) -> np.array:
        """
        Simulates and returns several simulated paths following the Black-Scholes model.
        Parameters:
        - scheme (str): the discretization scheme used ('euler' or 'milstein')
        - n (int): number of time steps in a path
        - N (int): number of simulated paths
        Returns:
        - S (np.array): simulated stock paths
        """
        np.random.seed(self.seed)

        dt = self.T / n
        S = np.zeros((N, n + 1))
        S[:, 0] = self.initial

        for i in range(1, n + 1):

            # Brownian motion
            N1 = np.random.normal(loc=0, scale=1, size=N)
            Z = N1 * np.sqrt(dt)

            # Update the processes 
            S[:, i] = S[:, i - 1] + self.mu * S[:, i - 1] * dt + self.volatility * S[:, i - 1] * Z

            if scheme == "milstein":
                S[:, i] += 1 / 2 * self.volatility * S[:, i - 1] * (Z ** 2 - dt)
            elif scheme != 'euler':
                print("Choose a scheme between: 'euler' or 'milstein'")

        return S

    def call_price(
            self,
            strike: float,
            initial: float = None, 
            r: float = None, 
            volatility: float = None, 
            T: float = None, 
        ):

        """
        Calculates the price of a European call option using the Black-Scholes formula.
        Parameters:
        - initial (float): initial value of the underlying asset
        - r (float): risk-free interest rate
        - volatility (float): volatility of the underlying asset
        - T (float): time to expiration (in years)
        - strike (float): strike price of the option
        Returns:
        - call_price (float): price of the European call option
        """
        if initial is None:
            initial = self.initial
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility
        if T is None:
            T = self.T

        d1 = (np.log(initial / strike) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        call_price = initial * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def put_price(
            self,
            strike: float,
            initial: float = None, 
            r: float = None, 
            volatility: float = None, 
            T: float = None, 
        ):        
        """
        Calculates the price of a European put option using the Black-Scholes formula and call-put parity.
        Parameters:
        - initial (float): initial value of the underlying asset. If None, defaults to model parameter.
        - r (float): risk-free interest rate. If None, defaults to model parameter.
        - volatility (float): volatility of the underlying asset. If None, defaults to model parameter.
        - T (float): time to expiration (in years). If None, defaults to model parameter.
        - strike (float): strike price of the option. If None, defaults to model parameter.
        Returns:
        - put_price (float): price of the European put option
        """
        if initial is None:
            initial = self.initial
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility
        if T is None:
            T = self.T

        call_price = self.call_price(initial, r, volatility, T, strike)
        put_price = call_price - initial + strike * np.exp(-r * T)
        return put_price

    def delta(
            self,
            strike: float,
            option: str,
            initial: float = None, 
            r: float = None, 
            volatility: float = None, 
            T: float = None, 
        ):
        """
        Calculates the delta of a European option using the Black-Scholes formula.
        Parameters:
        - strike (float): strike price of the option.
        - option (str): type of option, either 'call' or 'put'.
        - initial (float): initial value of the underlying asset. If None, defaults to model parameter.
        - r (float): risk-free interest rate. If None, defaults to model parameter.
        - volatility (float): volatility of the underlying asset. If None, defaults to model parameter.
        - T (float): time to expiration (in years). If None, defaults to model parameter.
        Returns:
        - delta (float): delta of the European option
        """
        if initial is None:
            initial = self.initial
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility
        if T is None:
            T = self.T

        if strike is None:
            raise ValueError("Please provide a strike price.")

        d1 = (np.log(initial / strike) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))

        if option == 'call':
            delta = norm.cdf(d1)
        elif option == 'put':
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return delta

    def gamma(
            self,
            strike: float,
            option: str,
            initial: float = None, 
            r: float = None, 
            volatility: float = None, 
            T: float = None, 
        ):
        """
        Calculates the gamma of a European option using the Black-Scholes formula.
        Parameters:
        - strike (float): strike price of the option.
        - option (str): type of option, either 'call' or 'put'.
        - initial (float): initial value of the underlying asset. If None, defaults to model parameter.
        - r (float): risk-free interest rate. If None, defaults to model parameter.
        - volatility (float): volatility of the underlying asset. If None, defaults to model parameter.
        - T (float): time to expiration (in years). If None, defaults to model parameter.
        Returns:
        - gamma (float): gamma of the European option
        """
        if initial is None:
            initial = self.initial
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility
        if T is None:
            T = self.T

        if strike is None:
            raise ValueError("Please provide a strike price.")

        d1 = (np.log(initial / strike) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
        gamma = norm.pdf(d1) / (initial * volatility * np.sqrt(T))

        return gamma

    def delta_hedging(self, option:str, strike:float, hedging_volatility:float, n:float=1000, N:float=100):

        time = np.linspace(start=0, stop=self.T, num=n+1)
        dt = self.T / n

        S = self.simulate(scheme='milstein', n=n, N=N)
        portfolio = np.zeros_like(S)

        if option == 'call':
            portfolio[:,0] = self.call_price(strike=strike, initial=S[:,0], volatility=hedging_volatility)
        else:
            portfolio[:,0] = self.put_price(strike=strike, initial=S[:,0], volatility=hedging_volatility)

        stocks = self.delta(
                initial=S[:,0], 
                T=self.T,
                volatility=hedging_volatility,
                strike=strike, 
                option=option, 
            )
        bank = portfolio[:, 0] - stocks * S[:, 0]

        for t in range(1, n):

            portfolio[:, t] = stocks * S[:, t] + bank * np.exp(dt * self.r) 
            stocks = self.delta(
                initial = S[:, t], 
                T = self.T - time[t],
                volatility = hedging_volatility,
                strike = strike, 
                option = option, 
            )
            bank = portfolio[:, t] - stocks * S[:, t]

        portfolio[:, -1] = stocks * S[:,-1] + bank * np.exp(dt * self.r) 

        return portfolio, S


    def plot_simulation(
            self, 
            scheme: str = 'euler', 
            n: int = 1000,
        ) -> np.array:
        """
        Plots the simulation of a Black-Scholes trajectory.
        Parameters:
        - scheme (str): the discretization scheme used ('euler' or 'milstein')
        - n (int): number of time steps in a path
        """        
        S = self.simulate(n=n, scheme=scheme, N=1)

        plt.figure(figsize=(10, 6))

        # Plot for the stock path
        plt.plot(np.linspace(0, self.T, n + 1), S[0], label='Risky asset', color='blue', linewidth=1)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value [$]', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.minorticks_on()
        plt.grid(which="minor", visible=False)
        plt.title(f'Black-Scholes Model Simulation with {scheme} scheme', fontsize=16)
        plt.show()

        return S


if __name__ == '__main__':

    initial = 100
    r = 0.05
    volatility = 0.06
    T = 1

    blackscholes = BlackScholes(initial=initial, r=r, volatility=volatility, T=T)
    # blackscholes.plot_simulation()

    initials = np.arange(start=25, stop=175, step=1)

    deltas_call = blackscholes.delta(option='call', strike=100, initial=initials, T=1)
    deltas_put = blackscholes.delta(option='put', strike=100, initial=initials, T=1)

    plt.figure()
    plt.plot(initials, deltas_call, label='call')
    plt.plot(initials, deltas_put, label='put')
    plt.xlabel('Spot value', fontsize=12)
    plt.ylabel('Delta', fontsize=12)
    plt.legend(loc='upper left')
    plt.xlim((0,200))
    plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
    plt.minorticks_on()
    plt.grid(which="minor", visible=False)
    plt.title("Delta for European Call and Put Options")
    plt.show()


    strike = 100
    initials = np.linspace(80, 120, 50)  
    T_values = np.linspace(0.1, 1, 50)
    option = 'call'

    deltas = np.zeros((len(initials), len(T_values)))
    for i, initial in enumerate(initials):
        for j, T in enumerate(T_values):
            deltas[i, j] = blackscholes.gamma(option=option, strike=strike, initial=initial, T=T)

    spot_grid, T_grid = np.meshgrid(initials, T_values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(spot_grid, T_grid, deltas.T, cmap='viridis')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Delta')
    plt.title(f'Delta Surface for European {option} Option')
    plt.show()