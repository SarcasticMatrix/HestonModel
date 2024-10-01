# HestonModel Python Package
Documentations :
* [Code documentation](https://sarcasticmatrix.github.io/hestonModel/),
* [Math documentation](https://github.com/SarcasticMatrix/hestonModel/blob/main/report.pdf).

This Python package implements the Heston model for option pricing and portfolio management using Monte Carlo simulations, the Carr-Madan method, and Fourier transforms. The package also includes functionality for optimal portfolio allocation using control techniques.

We assume that the stock is ruled by the Heston model under real world and martingale measures. Thus, skipping the reasoning from real-world measures $\mathbb P = (\mathbb P_1,\mathbb P_2)$ to martingale measures $\mathbb Q(\lambda) = [\mathbb Q_1(\lambda),\mathbb Q_2(\lambda)]$, parametrised by the drift parameter $\lambda$ under the equivalent martingale measure. Using the Cholesky decomposition and dsanov’s Theorem,

$$dS_t = rS_t dt + \sqrt{v_t}S_t (\rho dW^{Q_1}_t + \sqrt{1-\rho^2} dW^{Q_2}_t),$$

$$dv_t = \kappa (\theta - v_t)dt + \sigma\sqrt{v_t}dW_t^{Q_1}.$$

With $W^{Q_1}$ and $W^{Q_2}$ orthogonal Wiener processes under martingale measures. For the proofs, see `report.pdf`.

## Files
1. `HestonModel.py`: Contains the core class `HestonModel`, which simulates paths and prices options based on the Heston stochastic volatility model.
2. `Portfolio.py`: Manages portfolio allocation strategies.
3. `strategies.py`: Implements different control and optimization strategies for portfolio management.
---

## Pricing

**Initialisation of the model**
```
import matplotlib.pyplot as plt 
import time
import numpy as np
import tqdm

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

heston = Heston(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)
```

**Price via Monte Carlo with Euler-Maruyama discretization**
```
n = 100 # number of time points
N = 10**3 # number of path simulations

start_time = time.time()
result = heston.monte_carlo_price(scheme="euler", n=n, N=N)
time_delta = round(time.time() - start_time,4)
price_euler = round(result.price, 2)
std_euler = round(result.std, 2)
infinum_euler = round(result.infinum, 2)
supremum_euler = round(result.supremum, 2)
print(f"Monte Carlo Euler scheme in {time_delta}s : price ${price_euler}, std {std_euler}, and Confidence interval [{infinum_euler},{supremum_euler}]\n")
```
**Price via Monte Carlo with Milstein discretization**
```
start_time = time.time()
result = heston.monte_carlo_price(scheme="milstein", n=n, N=N)
time_delta = round(time.time() - start_time,4)
price_milstein = round(result.price, 2)
std_milstein = round(result.std, 2)
infinum_milstein = round(result.infinum, 2)
supremum_milstein = round(result.supremum, 2)
print(f"Monte Carlo Milstein scheme in {time_delta}s : price ${price_milstein}, std {std_milstein}, and Confidence interval [{infinum_milstein},{supremum_milstein}]\n")
```
**Price via Fourier Transform**
```
start_time = time.time()
price_FT, error_FT = heston.fourier_transform_price()
time_delta = round(time.time() - start_time,4)
infinum = round(price_FT-error_FT, 2)
supremum = round(price_FT+error_FT, 2)
price_FT = round(price_FT, 2)
error_FT = round(error_FT, 8)
print(f"Fourier Transform in {time_delta}s : price ${price_FT}, error ${error_FT} , and Confidence interval [{infinum},{supremum}]\n")
```
**Price via Carr-Madan formula**
```
start_time = time.time()
price_CM, error_CM = heston.carr_madan_price()
time_delta = round(time.time() - start_time,4)
infinum = round(price_CM-error_CM, 2)
supremum = round(price_CM+error_CM, 2)
price_CM = round(price_CM, 2)
error_CM = round(error_CM, 14)
print(f"Carr-Madan in {time_delta}s : price ${price_CM}, error ${error_CM} , and Confidence interval [{infinum},{supremum}]\n")
```
**Path simulations**
```
scheme = 'milstein'
heston.plot_simulation(scheme)
```
**Price surface : relations between price, strike and time to maturity**
```
Ks = np.arange(start=20, stop=200, step=1)
Ts = np.arange(start=0.1, stop=1.1, step=0.1)

prices_surface = np.zeros((len(Ts), len(Ks)))

for i, T in tqdm(enumerate(Ts)):
    for j, K in tqdm(enumerate(Ks)):
        heston = Heston(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K=K)
        price, _ = heston.carr_madan_price()
        prices_surface[i, j] = price

K_mesh, T_mesh = np.meshgrid(Ks, Ts)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_mesh, T_mesh, prices_surface, cmap='viridis')
ax.set_title('Surface de prix en fonction du strike et du temps de maturité')
ax.set_xlabel(r'Strike ($K$)')
ax.set_ylabel(r'Time to maturity ($T$)')
ax.set_zlabel('Price')
plt.show()
```