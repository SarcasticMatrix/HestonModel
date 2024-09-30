import numpy as np
from scripts.models.HestonModel import HestonModel
from scripts.models.BlackScholes import BlackScholes
from typing import Literal

class Option:
    """
    Represents a financial option and provides methods to calculate its price using different models.

    Attributes:
        flag (Literal['call', 'put']): The type of the option ('call' or 'put').
        strike (float): The strike price of the option.
        time_to_maturity (float): The time to maturity of the option (in years).
        interest (float): The risk-free interest rate (annualized).
        spot (float): The current spot price of the underlying asset.
        current_vol (float): The current volatility of the underlying asset.

    Example:
        put = Option(
            flag='put',
            strike=K,
            time_to_maturity=T,
            interest=r,
            spot=S0,
            current_vol=V0
        )
        heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)
        price = put.price(heston)
    """

    def __init__(
            self, 
            flag: Literal['call', 'put'], 
            strike: float, 
            time_to_maturity: float,
            interest: float,
            spot: float,
            current_vol: float
        ) -> None:
        """
        Initializes an Option object.

        Parameters:
            flag (Literal['call', 'put']): The type of the option ('call' or 'put').
            strike (float): The strike price of the option.
            time_to_maturity (float): The time to maturity of the option (in years).
            interest (float): The risk-free interest rate (annualized).
            spot (float): The current spot price of the underlying asset.
            current_vol (float): The current volatility of the underlying asset.
        """
        self.flag = flag
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.interest = interest
        self.spot = spot
        self.current_vol = current_vol

    def price_call(self, flag_model: Literal['heston', 'blackScholes'], params: list) -> float:
        """
        Calculates the price of a call option using the specified model.

        Parameters:
            flag_model (Literal['heston', 'blackScholes']): The model to use for pricing ('heston' or 'blackScholes').
            params (list): The parameters required by the chosen model. 
                                a) For 'heston', this should include [kappa, theta, drift_emm, sigma, rho]. 
                                b) For 'blackScholes', this should be [mu].

        Returns:
            float: The price of the call option.
        """
        if flag_model == 'heston':
            heston = HestonModel(
                S0=self.spot,
                V0=self.current_vol,
                r=self.interest,
                T=self.time_to_maturity,
                K=self.strike,
                kappa=params[0],
                theta=params[1],
                drift_emm=params[2],
                sigma=params[3],
                rho=params[4],
           )
            price, _ = heston.carr_madan_price()

        elif flag_model == 'blackScholes':
            blackScholes = BlackScholes(
                initial=self.spot,
                r=self.interest, 
                volatility=self.current_vol, 
                T=self.time_to_maturity,
                mu=params[0],
            )
            price = blackScholes.call_price(
                strike=self.strike,
            )
        return price

    def price_put(self, flag_model: Literal['heston', 'blackScholes'], params: list) -> float:
        """
        Calculates the price of a put option using the specified model.

        Parameters:
            flag_model (Literal['heston', 'blackScholes']): The model to use for pricing ('heston' or 'blackScholes').
            params (list): The parameters required by the chosen model. For 'heston', this should include
                           [kappa, theta, drift_emm, sigma, rho]. For 'blackScholes', this should be [mu].

        Returns:
            float: The price of the put option.
        """
        call_price = self.price_call(flag_model, params)
        price = call_price - self.spot + self.strike * np.exp(- self.interest * self.time_to_maturity)
        return price

    def price(self, flag_model: Literal['heston', 'blackScholes'], params: list) -> float:
        """
        Calculates the price of the option based on its type ('call' or 'put') using the specified model.

        Parameters:
            flag_model (Literal['heston', 'blackScholes']): The model to use for pricing ('heston' or 'blackScholes').
            params (list): The parameters required by the chosen model. For 'heston', this should include
                           [kappa, theta, drift_emm, sigma, rho]. For 'blackScholes', this should be [mu].

        Returns:
            float: The price of the option.
        """
        if self.flag == 'put':
            price = self.price_put(flag_model, params)
        
        elif self.flag == 'call':
            price = self.price_call(flag_model, params)

        return price
    

def test():
    # Define parameters for the models
    S0 = 100.0  # Initial spot price
    V0 = 0.2    # Initial volatility
    r = 0.05    # Risk-free interest rate
    T = 1.0     # Time to maturity in years
    K = 100.0   # Strike price

    # Heston model parameters
    kappa = 1.5       # Mean reversion rate
    theta = 0.04      # Long-term volatility
    drift_emm = 0.0   # Drift term (not used in this context)
    sigma = 0.2       # Volatility of volatility
    rho = -0.5        # Correlation between asset and volatility

    # Black-Scholes model parameters
    mu = 0.05         # Expected return of the underlying asset

    # Create Option instances
    call_option = Option(
        flag='call',
        strike=K,
        time_to_maturity=T,
        interest=r,
        spot=S0,
        current_vol=V0
    )

    put_option = Option(
        flag='put',
        strike=K,
        time_to_maturity=T,
        interest=r,
        spot=S0,
        current_vol=V0
    )

    # Price options using Heston model
    heston_params = [kappa, theta, drift_emm, sigma, rho]
    call_price_heston = call_option.price('heston', heston_params)
    put_price_heston = put_option.price('heston', heston_params)

    print(f"Heston Model - Call Option Price: {call_price_heston:.2f}")
    print(f"Heston Model - Put Option Price: {put_price_heston:.2f}")

    # Price options using Black-Scholes model
    black_scholes_params = [mu]
    call_price_bs = call_option.price('blackScholes', black_scholes_params)
    put_price_bs = put_option.price('blackScholes', black_scholes_params)

    print(f"Black-Scholes Model - Call Option Price: {call_price_bs:.2f}")
    print(f"Black-Scholes Model - Put Option Price: {put_price_bs:.2f}")

if __name__ == "__main__":
    test()
