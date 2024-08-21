from scripts.models.HestonModel import HestonModel
from scripts.portfolio.Portfolio import Portfolio
from scripts.portfolio.strategies import time_varying_strategy, naive_strategy, optimal_allocate_strategy, run_strategies

import matplotlib.pyplot as plt 
import numpy as np


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

premium_volatility_risk = 0.05
seed = 380

heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K, premium_volatility_risk, seed)

S, V, _ = heston.simulate(scheme='milstein', n=1000, N=1)
S = S.flatten()
V = V.flatten()

time = np.linspace(start=0, stop=T, num=len(S))
dt = time[1] - time[0]

portfolio = Portfolio(r=r, dt=dt)

### Naive constant allocation strategy

allocation1 = naive_strategy(0.5, len(S))
bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=S0, allocation_strategy=allocation1)
portfolio_value1 = bank_account + stocks_account

allocation2 = naive_strategy(0.7, len(S))
bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=S0, allocation_strategy=allocation2)
portfolio_value2 = bank_account + stocks_account

allocation3 = naive_strategy(1, len(S))
bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=S0, allocation_strategy=allocation3)
portfolio_value3 = bank_account + stocks_account

### Time varying allocation strategy

percentage_in_bank_account = 1
p = 0.05
allocation4 = time_varying_strategy(premium_volatility_risk=premium_volatility_risk, p=p, heston=heston, V=V)
bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=S0, allocation_strategy=allocation4)
portfolio_value4 = bank_account + stocks_account


### Optimal allocation strategy

p = 0.02
optimal_allocation = optimal_allocate_strategy(heston=heston, p=p, time=time)
bank_account, stocks_account = portfolio.back_test(S=S, portfolio0=S0, allocation_strategy=optimal_allocation)
portfolio_value_optimal = bank_account + stocks_account

### Run strategies

seeds = np.arange(1,1000,1)
run_strategies(seeds, portfolio0=S0)


### Plot strategies

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  

ax1.plot(time, portfolio_value1, label='NS 50% bank account', color='blue', linewidth=1)
ax1.plot(time, portfolio_value2, label='NS 70% bank account', color='green', linewidth=1)
ax1.plot(time, portfolio_value3, label='NS 100% bank account', color='red', linewidth=1)
#ax1.plot(time, portfolio_value4, label=r'TVS $\bar\lambda=0.5$', color='orange', linewidth=1)

ax1.plot(time, portfolio_value_optimal, label=r'Optimal strategy', color='purple', linewidth=1)

ax1.plot(time, S, label='Stock', color='black', linewidth=1)
ax1.set_ylabel('Value [currency unit]')
ax1.set_title('Portfolio Value over Time')
ax1.legend()
ax1.grid(visible=True)

# P&L
PnL1 = np.diff(portfolio_value1)
PnL2 = np.diff(portfolio_value2)
PnL3 = np.diff(portfolio_value3)
#PnL4 = np.diff(portfolio_value4)

ax2.plot(time[1:], PnL1, label=r'$\pi_1$', color='blue', linewidth=0.7)
ax2.plot(time[1:], PnL2, label=r'$\pi_2$', color='green', linewidth=0.7)
ax2.plot(time[1:], PnL3, label=r'$\pi_3$', color='red', linewidth=0.7)
#ax2.plot(time, PnL4, label=r'$\pi_4$', color='orange', linewidth=0.7)

PnL_opt = np.diff(portfolio_value_optimal)
ax2.plot(time[1:], PnL_opt, label=r'$\pi^\star$', color='purple', linewidth=0.7)


ax2.set_xlabel('Time')
ax2.set_ylabel(r'$PnL(t)$')
ax2.set_title(r'$PnL$')
ax2.legend()
ax2.grid(visible=True)

plt.suptitle('Naive Strategy (NS) and Time Varying Strategy (TVS)')
plt.tight_layout()  
plt.show()


## Surface plot
p = np.arange(start=0, stop=1, step=0.01)
p = p[1:-1]
time = np.arange(start=0, stop=1, step=0.01)
p, tau = np.meshgrid(p, heston.T - time)

Z = optimal_allocate_strategy(heston=heston, p=p, time=tau)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(tau, p, Z)

ax.set_xlabel('p')
ax.set_ylabel(r'$\tau$')
ax.set_zlabel(r'$\%$ of portfolio in stock')

plt.show()
