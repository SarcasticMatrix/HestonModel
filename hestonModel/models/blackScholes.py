import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Literal

class BlackScholes:
    """
    Class representing a Black-Scholes model.

    Parameters:
    - spot (float): The current price of the underlying asset.
    - r (float): The risk-free interest rate, used as the drift coefficient in the model.
    - mu (float): The drift rate of the underlying asset.
    - volatility (float): The volatility of the underlying asset.
    - seed (int): A seed for the random number generator for reproducibility.

    Methods:
    - simulate(scheme='euler', n=100, N=1000):
      Simulates and returns several paths for the stock price according to the Black-Scholes model.
    - plot_simulation(scheme='euler', n=1000):
      Plots a single simulated path of the Black-Scholes model.
    - call_price(strike, spot=None, r=None, volatility=None, T=1):
      Computes the price of a European call option using the Black-Scholes formula.
    - put_price(strike, spot=None, r=None, volatility=None, T=1):
      Computes the price of a European put option using the Black-Scholes formula and put-call parity.
    - delta(strike, flag_option='call', spot=None, r=None, volatility=None, T=1):
      Computes the delta of a European option based on the Black-Scholes model.
    - delta_surface(flag_option='call'):
      Plots the delta of the option as a function of the strike price and time to maturity.
    - gamma(strike, T=None, spot=None, r=None, volatility=None):
      Computes the gamma of a European option based on the Black-Scholes model.
    - gamma_surface():
      Plots the gamma as a function of the strike price and time to maturity.
    - delta_hedging(flag_option='call', strike, hedging_volatility, n=1000, N=100):
      Implements a delta hedging strategy for a European option.

    Example:
        # Parameters for the Black-Scholes model
        spot = 100.0          # Current spot price of the underlying asset
        r = 0.05              # Risk-free interest rate
        mu = 2 * r            # Expected return (often set to 2*r for analysis)
        volatility = 0.06     # Volatility of the underlying asset
        T = 1.0               # Time to maturity in years

        # Create a BlackScholes instance
        blackscholes = BlackScholes(spot=spot, r=r, volatility=volatility, mu=mu, T=T)

        # Plot simulation of the Black-Scholes model
        blackscholes.plot_simulation()

        # Generate spot prices for delta calculations
        spots = np.arange(start=25, stop=175, step=1)

        # Calculate delta for call and put options
        deltas_call = blackscholes.delta(flag_option='call', strike=100, spot=spots, T=1)
        deltas_put = blackscholes.delta(flag_option='put', strike=100, spot=spots, T=1)

        # Plotting delta values
        plt.figure()
        plt.plot(spots, deltas_call, label='Call Option Delta')
        plt.plot(spots, deltas_put, label='Put Option Delta')
        plt.xlabel('Spot Price', fontsize=12)
        plt.ylabel('Delta', fontsize=12)
        plt.legend(loc='upper left')
        plt.xlim((0, 200))
        plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.minorticks_on()
        plt.grid(which="minor", visible=False)
        plt.title("Delta for European Call and Put Options")
        plt.show()

        # Generate delta surfaces for call and put options
        blackscholes.delta_surface('call')
        blackscholes.delta_surface('put')

        # Generate gamma surface
        blackscholes.gamma_surface()

        # Calculate volatility surface for call option
        volatility_surface(flag_option='call')

        # Implied volatility calculation
        spot = 100.0          # Current spot price of the underlying asset
        r = 0.05              # Risk-free interest rate
        mu = 1.0              # Expected return of the underlying asset
        volatility = 0.5      # Volatility

        # Calculate implied volatility for a call option
        res = impliedVolatility(flag_option='call')
        print("Implied Volatility (Call):", res)

        # Calculate implied volatility with updated risk-free rate
        res = impliedVolatility(flag_option='call', r=res[1])
        print("Updated Implied Volatility (Call):", res)
    """

    def __init__(
            self, 
            spot: float, 
            r: float, 
            mu: float,
            volatility: float, 
            seed: int = 42):
        """
        Initializes a Black-Scholes model object.
        Parameters:
            - spot (float): spot value of the process
            - r (float): interest rate (drift coefficient) of the process
            - mu (float): drift
            - volatility (float): volatility coefficient of the process
            - seed (int): seed for random number generation
        """

        # Black-Scholes parameters
        self.spot = spot
        self.r = r
        self.mu = mu
        self.volatility = volatility
        self.seed = seed

    def simulate(
            self, 
            T: float = 1,
            scheme: str = "euler", 
            n: int = 100, 
            N: int = 1000, 
        ) -> np.array:
        """
        Simulates and returns several simulated paths following the Black-Scholes model.
        Parameters:
            - T (float): Time to maturity in years
            - scheme (str): the discretization scheme used ('euler' or 'milstein')
            - n (int): number of time steps in a path
            - N (int): number of simulated paths
        Returns:
            - S (np.array): simulated stock paths
        """
        np.random.seed(self.seed)

        dt = T / n
        S = np.zeros((N, n + 1))
        S[:, 0] = self.spot

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
        plt.plot(np.linspace(0, T, n + 1), S[0], label='Risky asset', color='blue', linewidth=1)
        plt.xlabel('Time to expiration', fontsize=12)
        plt.ylabel('Value [$]', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.minorticks_on()
        plt.grid(which="minor", visible=False)
        plt.title(f'Black-Scholes Model Simulation with {scheme} scheme', fontsize=16)
        plt.show()

        return S

    def call_price(
            self,
            strike: float,
            spot: float = None, 
            r: float = None, 
            volatility: float = None, 
            T: float = 1, 
        ):

        """
        Calculates the price of a European call option using the Black-Scholes formula.
        Parameters:
            - spot (float): spot value of the underlying asset
            - r (float): risk-free interest rate
            - volatility (float): volatility of the underlying asset
            - T (float): time to expiration (in years)
            - strike (float): strike price of the option
        Returns:
            - call_price (float): price of the European call option
        """
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility


        d1 = (np.log(spot / strike) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        call_price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def put_price(
            self,
            strike: float,
            spot: float = None, 
            r: float = None, 
            volatility: float = None, 
            T: float = 1, 
        ):        
        """
        Calculates the price of a European put option using the Black-Scholes formula and call-put parity.
        Parameters:
            - spot (float): spot value of the underlying asset. If None, defaults to model parameter.
            - r (float): risk-free interest rate. If None, defaults to model parameter.
            - volatility (float): volatility of the underlying asset. If None, defaults to model parameter.
            - T (float): time to expiration (in years). Defaults to 1.
            - strike (float): strike price of the option. If None, defaults to model parameter.
        Returns:
            - put_price (float): price of the European put option
        """
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility

        call_price = self.call_price(spot, r, volatility, T, strike)
        put_price = call_price - spot + strike * np.exp(-r * T)
        return put_price

    def delta(
            self,
            strike: float,
            flag_option: Literal['call', 'put'],
            spot: float = None, 
            r: float = None, 
            volatility: float = None, 
            T: float = 1, 
        ):
        """
        Calculates the delta of a European option using the Black-Scholes formula.
        Parameters:
            - strike (float): strike price of the option.
            - flag_option (str): type of option, either 'call' or 'put'.
        Optional parameters:
            - T (float): time to expiration (in years). Defaults to 1.
            - spot (float): spot value of the underlying asset. If None, defaults to model parameter.
            - r (float): risk-free interest rate. If None, defaults to model parameter.
            - volatility (float): volatility of the underlying asset. If None, defaults to model parameter.
        Returns:
            - delta (float): delta of the European option
        """
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility
        if strike is None:
            raise ValueError("Please provide a strike price.")

        d1 = (np.log(spot / strike) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))

        if flag_option == 'call':
            delta = norm.cdf(d1)
        elif flag_option == 'put':
            delta = norm.cdf(d1) - 1 
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return delta

    def delta_surface(self, flag_option:Literal['call', 'put']):    
        """"
        Plot the delta of the option as a function of strike and time to maturity
            - flag_option (str): 'call' of 'put', type of option
        """

        Ks = np.arange(start=20, stop=200, step=0.5)
        Ts = np.linspace(start=0.01, stop=1, num=500)
        deltas = np.zeros((len(Ks), len(Ts)))

        for i, K in enumerate(Ks):
            for j, T in enumerate(Ts):
                deltas[i, j] = blackscholes.delta(strike=K, T=T, flag_option=flag_option)
        K_grid, T_grid = np.meshgrid(Ks, Ts)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, deltas.T, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Delta')
        plt.title('Delta Surface for European options')
        plt.show()

    def gamma(
            self,
            strike: float,
            T: float = None, 
            spot: float = None, 
            r: float = None, 
            volatility: float = None, 
        ):
        """
        Calculates the gamma of a European option using the Black-Scholes formula.
        Parameters:
            - strike (float): strike price of the option.
            - spot (float): spot value of the underlying asset. If None, defaults to model parameter.
            - r (float): risk-free interest rate. If None, defaults to model parameter.
            - volatility (float): volatility of the underlying asset. If None, defaults to model parameter.
            - T (float): time to expiration (in years). If None, defaults to model parameter.
        Returns:
            - gamma (float): gamma of the European option
        """
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility
        if T is None:
            T = T

        if strike is None:
            raise ValueError("Please provide a strike price.")

        d1 = (np.log(spot / strike) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(T))

        return gamma
 
    def gamma_surface(self):    
        """"
        Plot the gamma as a function of strike and time to maturity.
        """

        Ks = np.arange(start=20, stop=200, step=0.5)
        Ts = np.linspace(start=0.01, stop=1, num=500)
        gammas = np.zeros((len(Ks), len(Ts)))

        for i, K in enumerate(Ks):
            for j, T in enumerate(Ts):
                gammas[i, j] = blackscholes.gamma(strike=K, T=T)
        K_grid, T_grid = np.meshgrid(Ks, Ts)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, gammas.T, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Gamma')
        plt.title('Gamma Surface for European options')
        plt.show()

    def delta_hedging(
            self, 
            flag_option: Literal['call', 'put'], 
            strike: float, 
            hedging_volatility: float, 
            n: float = 1000, 
            N: float = 100
        ):
        """
        Implement a delta hedging strategy using both a risky asset (underlying asset) 
        and a non-risky asset for a European option.    
        Parameters:
            - flag_option (str): 
                Type of option. Should be 'call' for a call option or 'put' for a put option.
            - strike (float): 
                The strike price of the option.
            - hedging_volatility (float): 
                The volatility used for hedging purposes.
            - n (float, optional): 
                The number of simulation steps or trading intervals over the life 
                of the option. Defaults to 1000. This parameter controls how often 
                the portfolio is rebalanced to maintain a delta-neutral position.
            - N (float, optional): 
                The number of simulations.
        Returns:
            - portfolio (np.array): allocation,
            - S (np.array): 
        """

        time = np.linspace(start=0, stop=T, num=n+1)
        dt = T / n

        S = self.simulate(scheme='milstein', n=n, N=N)
        portfolio = np.zeros_like(S)

        if flag_option == 'call':
            portfolio[:,0] = self.call_price(strike=strike, spot=S[:,0], volatility=hedging_volatility)
        else:
            portfolio[:,0] = self.put_price(strike=strike, spot=S[:,0], volatility=hedging_volatility)

        stocks = self.delta(
                spot=S[:,0], 
                T=T,
                volatility=hedging_volatility,
                strike=strike, 
                flag_option=flag_option, 
            )
        bank = portfolio[:, 0] - stocks * S[:, 0]

        for t in range(1, n):

            portfolio[:, t] = stocks * S[:, t] + bank * np.exp(dt * self.r) 
            stocks = self.delta(
                spot = S[:, t], 
                T = T - time[t],
                volatility = hedging_volatility,
                strike = strike, 
                flag_option = flag_option, 
            )
            bank = portfolio[:, t] - stocks * S[:, t]

        portfolio[:, -1] = stocks * S[:,-1] + bank * np.exp(dt * self.r) 

        return portfolio, S


from scipy.optimize import minimize
from hestonModel.option.data import get_options_data
from typing import Literal

def impliedVolatility(
    flag_option: Literal['call', 'put'], 
    symbol: str = 'MSFT',
    r: float = None,
    exponent: int = 1
):
    """
    Estimate the implied volatility of options using the Black-Scholes model.

    Parameters:
    - flag_option (Literal['call', 'put']): Type of option (call or put).
    - symbol (str): The underlying asset's symbol; defaults to 'MSFT' (Microsoft Corporation).
    - r (float): The risk-free interest rate; can be estimated if None.
    - exponent (int): The exponent used in the objective function. 
                      If 2, it uses squared error; if 1, it uses absolute error.

    Returns:
    - vol: The estimated implied volatility (and possibly the risk-free rate if estimated).
    """
    # to do : implement for put options - pas super stable pour le moment 

    options_data, spot = get_options_data(symbol=symbol, flag_option=flag_option)

    blackScholes = BlackScholes(
        spot=spot, r=0.03, mu=0.03, volatility=0.03
    )

    mask = options_data['Volume'] > 0.1 * len(options_data)
    options_data = options_data.loc[mask]

    volumes = options_data["Volume"].values
    strikes = options_data["Strike"].values
    prices = options_data[f"{flag_option.capitalize()} Price"].values
    maturities = options_data["Time to Maturity"].values

    if r is None:
        x0 = [
            blackScholes.volatility,
            blackScholes.r
        ]
    else:  
        # r is not estimated
        x0 = [
            blackScholes.volatility,
        ]
        blackScholes.r = r

    def objective_function(x):
        blackScholes.volatility = x[0]
        if r is None:
            # r is also estimated
            blackScholes.r = x[1]

        model_prices = []
        for i in range(len(options_data)):
            blackScholes.K = strikes[i]
            blackScholes.T = maturities[i]

            if flag_option == 'call':
                model_price = blackScholes.call_price(strike=strikes[i], T=maturities[i])
            else:
                model_price = blackScholes.put_price(strike=strikes[i], T=maturities[i])
            model_prices.append(model_price)

        model_prices = np.array(model_prices)
        weights = volumes / np.sum(volumes)
        
        result = np.sum(
            weights * np.abs(prices - model_prices)**exponent
        ) 

        return result

    print('Callibration is running...')
    res = minimize(fun=objective_function, x0=x0, method='Nelder-Mead')  

    if res.success:
        return res.x


import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from scipy.optimize import minimize

def volatility_surface(
    flag_option: Literal['call', 'put'], 
    symbol: str = 'MSFT',
    r: float = None,
    exponent: int = 1
):
    """
    Calculate the implied volatility surface from option prices.

    Parameters:
    - flag_option (Literal['call', 'put']): Type of option ('call' or 'put').
    - symbol (str): The symbol of the underlying asset.
    - r (float): Risk-free interest rate; can be None for estimation.
    - exponent (int): Exponent used for optimization.

    Returns:
    - vol_surface: Matrix of implied volatilities.
    """
    # Not working very well

    # Estimation of the interest rate
    res = impliedVolatility(flag_option=flag_option, symbol=symbol, exponent=exponent)
    r = res[1]

    market_data, spot = get_options_data(symbol=symbol, flag_option=flag_option)
    market_data = market_data.sort_values(['Strike', 'Time to Maturity'], ascending=[True, True])
    
    strikes = np.unique(market_data['Strike'])
    maturities = np.unique(market_data['Time to Maturity'])

    vol_surface = np.full((len(strikes), len(maturities)), np.nan)

    def implied_volatility(market_price, strike, maturity):
        """
        Calcule la volatilité implicite pour une option donnée via une optimisation numérique.
        """
        def objective_function(vol):
            bs = BlackScholes(spot=spot, r=r, mu=0, volatility=vol)
            if flag_option == 'call':
                model_price = bs.call_price(strike=strike, T=maturity)
            else:
                model_price = bs.put_price(strike=strike, T=maturity)
            return np.abs(model_price - market_price) ** exponent

        result = minimize(objective_function, x0=0.2)
        return result.x[0] if result.success else np.nan

    for _, row in market_data.iterrows():
        market_price = row[f'{flag_option.capitalize()} Price']
        strike = row['Strike']
        maturity = row['Time to Maturity']

        # Find indexes corresponding to (strike, maturity)
        strike_index = np.where(strikes == strike)[0][0]
        maturity_index = np.where(maturities == maturity)[0][0]

        # Computed implied vol
        iv = implied_volatility(market_price, strike, maturity)
        vol_surface[strike_index, maturity_index] = iv

    Strike_grid, Maturity_grid = np.meshgrid(strikes, maturities)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Strike_grid, Maturity_grid, vol_surface.T, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    plt.title('Volatility Surface')
    plt.show()

    return vol_surface









if __name__ == '__main__':

    # Parameters for the Black-Scholes model
    spot = 100.0          # Current spot price of the underlying asset
    r = 0.05              # Risk-free interest rate
    mu = 2 * r            # Expected return (often set to 2*r for analysis)
    volatility = 0.06     # Volatility of the underlying asset
    T = 1.0               # Time to maturity in years

    # Create a BlackScholes instance
    blackscholes = BlackScholes(spot=spot, r=r, volatility=volatility, mu=mu, T=T)

    # Plot simulation of the Black-Scholes model
    blackscholes.plot_simulation()

    # Generate spot prices for delta calculations
    spots = np.arange(start=25, stop=175, step=1)

    # Calculate delta for call and put options
    deltas_call = blackscholes.delta(flag_option='call', strike=100, spot=spots, T=1)
    deltas_put = blackscholes.delta(flag_option='put', strike=100, spot=spots, T=1)

    # Plotting delta values
    plt.figure()
    plt.plot(spots, deltas_call, label='Call Option Delta')
    plt.plot(spots, deltas_put, label='Put Option Delta')
    plt.xlabel('Spot Price', fontsize=12)
    plt.ylabel('Delta', fontsize=12)
    plt.legend(loc='upper left')
    plt.xlim((0, 200))
    plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
    plt.minorticks_on()
    plt.grid(which="minor", visible=False)
    plt.title("Delta for European Call and Put Options")
    plt.show()

    # Generate delta surfaces for call and put options
    blackscholes.delta_surface('call')
    blackscholes.delta_surface('put')

    # Generate gamma surface
    blackscholes.gamma_surface()

    # Calculate volatility surface for call option
    volatility_surface(flag_option='call')

    # Implied volatility calculation
    spot = 100.0          # Current spot price of the underlying asset
    r = 0.05              # Risk-free interest rate
    mu = 1.0              # Expected return of the underlying asset
    volatility = 0.5      # Volatility

    # Calculate implied volatility for a call option
    res = impliedVolatility(flag_option='call')
    print("Implied Volatility (Call):", res)

    # Calculate implied volatility with updated risk-free rate
    res = impliedVolatility(flag_option='call', r=res[1])
    print("Updated Implied Volatility (Call):", res)