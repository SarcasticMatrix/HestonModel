# HestonModel Python Package
For the documentation see https://sarcasticmatrix.github.io/hestonModel/

This Python package implements the Heston model for option pricing and portfolio management using Monte Carlo simulations, the Carr-Madan method, and Fourier transforms. The package also includes functionality for optimal portfolio allocation using control techniques.

We assume that the stock is ruled by the Heston model under real world and martingale measures. Thus, skipping the reasoning from real-world measures $\mathbb P = (\mathbb P_1,\mathbb P_2)$ to martingale measures $\mathbb Q(\lambda) = [\mathbb Q_1(\lambda),\mathbb Q_2(\lambda)]$, parametrised by the drift parameter $\lambda$ under the equivalent martingale measure. Using the Cholesky decomposition and dsanov’s Theorem,

$$dS_t = rS_t dt + \sqrt{v_t}S_t (\rho dW^{Q_1}_t + \sqrt{1-\rho^2} dW^{Q_2}_t),$$

$$dv_t = \kappa (\theta - v_t)dt + \sigma\sqrt{v_t}dW_t^{Q_1}.$$

With $W^{Q_1}$ and $W^{Q_2}$ orthogonal Wiener processes under martingale measures. For the proofs, see `report.pdf`.

## Files
1. `HestonModel.py`: Contains the core class `HestonModel`, which simulates paths and prices options based on the Heston stochastic volatility model.
2. `Portfolio.py`: Manages portfolio allocation strategies.
3. `strategies.py`: Implements different control and optimization strategies for portfolio management.

See 
1. `pricing.py` for an example of pricing,
2. `asset_allocation.py` for an example of asset allocation,
---

## HestonModel.py

### `class HestonModel`
The `HestonModel` class simulates stock price paths and variance paths under the Heston stochastic volatility model and allows for the pricing of European call options.

- **Parameters:**
  - `S0 (float)`: Spot price of the asset ($S_0$).
  - `V0 (float)`: Initial variance ($v_0$).
  - `r (float)`: Risk-free interest rate ($r$).
  - `kappa (float)`: Rate at which variance reverts to its long-term mean ($\kappa$).
  - `theta (float)`: Long-term variance $\theta$.
  - `drift_emm (float)`: Drift parameter under the equivalent martingale measure ($\lambda$).
  - `sigma (float)`: Volatility of variance ($\sigma$).
  - `rho (float)`: Correlation between the Brownian motions of asset price and variance ($\rho$).
  - `T (float)`: Time to maturity.
  - `K (float)`: Strike price of the option.
  - `premium_volatility_risk (float, optional)`: Premium for volatility risk, default is `0.0`.
  - `seed (int, optional)`: Random seed for simulations, default is `42`.

#### `monte_carlo_price`
Simulates sample paths and estimates the call price with a simple Monte Carlo Method.
- **Parameters:**
        - scheme (str): the discretization scheme used
        - n (int): number of points in a path
        - N (int): number of simulated paths

- **Returns:**
        - result (namedtuple): with the following attributes:
            - price (float): estimation by Monte Carlo of the call price
            - standard_deviation (float): standard deviation of the option payoff
            - infimum (float): infimum of the confidence interval
            - supremum (float): supremum of the confidence interval

#### `fourier_transform_price`
Computes the price of a European call option on the underlying asset S following a Heston model using the Heston formula.
- **Parameters:**
        - t (float): time

- **Returns:**
        - price (float): option price
        - error (float): error in the option price computation

#### `carr_madan_price`
Computes the price of a European call option on the underlying asset S following a Heston model using Carr-Madan Fourier pricing.
- **Parameters:**
        - t (float): time, set at 0 by default

- **Returns:**
        - price (float): option price
        - error (float): error in the option price computation

        

---

Let me proceed with reviewing the other files to extend this documentation.

### Portfolio.py

The `Portfolio` class is used to model portfolio strategies and perform backtests.

#### `class Portfolio`
- **Parameters:**
  - `r (float)`: Interest rate.
  - `dt (float)`: Time step between each reallocation.

#### `grow_bank_account(self, bank_account)`
Calculates the growth of the bank account based on the interest rate.
- **Parameters:**
  - `bank_account (float)`: The current balance in the bank account.
- **Returns:**
  - `float`: The updated balance in the bank account after applying interest growth.

#### `grow_stocks_account(self, stocks_account, S_now, S_previous)`
Updates the stock portfolio by accounting for changes in the stock price.
- **Parameters:**
  - `stocks_account (float)`: The current value of the stocks account.
  - `S_now (float)`: The current stock price.
  - `S_previous (float)`: The previous stock price.
- **Returns:**
  - `float`: The updated stocks account balance.

#### `back_test(self, S, portfolio0, allocation_strategy)`
Backtests a portfolio strategy based on a sequence of stock prices and an allocation strategy over time.
- **Parameters:**
  - `S (np.array)`: The path of the stock prices over time.
  - `portfolio0 (float)`: The initial value of the portfolio.
  - `allocation_strategy (np.array)`: Allocation strategy over time; the same size as `S`.
- **Returns:**
  - `bank_account (np.array)`: Bank account value over time.
  - `stocks_account (np.array)`: Stocks account value over time.
