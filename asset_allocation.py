from scripts.HestonModel import HestonModel
from scripts.PortfolioStrategy import PortfolioStrategy

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

heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K, premium_volatility_risk)

S, V, _ = heston.simulate(scheme='milstein', n=1000, N=1)
S = S.flatten()
V = V.flatten()

time = np.linspace(start=0, stop=1, num=len(S))
dt = time[1] - time[0]

### Naive constant allocation strategy

strategy = PortfolioStrategy(r=r, dt=dt)
allocate_portfolio = lambda s: (0.5,0.5)
bank_account, stocks_account = strategy.constant_back_test(S=S, portfolio0=S0, allocate_portfolio=allocate_portfolio)
portfolio_value1 = bank_account + stocks_account

strategy = PortfolioStrategy(r=r, dt=dt)
allocate_portfolio = lambda s: (0.7,0.3)
bank_account, stocks_account = strategy.constant_back_test(S=S, portfolio0=S0, allocate_portfolio=allocate_portfolio)
portfolio_value2 = bank_account + stocks_account


### Constant allocation strategy knowing return and volatility

strategy = PortfolioStrategy(r=r, dt=dt)
bank_account, stocks_account = strategy.optimal_back_test(S=S, portfolio0=S0, premium_volatility_risk=0.05, V=V, p=0.05)
portfolio_value3 = bank_account + stocks_account


strategy = PortfolioStrategy(r=r, dt=dt)
bank_account, stocks_account = strategy.optimal_back_test(S=S, portfolio0=S0, premium_volatility_risk=0.5, V=V, p=0.05)
portfolio_value4 = bank_account + stocks_account


### Plot strategies

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  

#ax1.plot(time, portfolio_value1, label='NS 50% bank account', color='blue', linewidth=1)
#ax1.plot(time, portfolio_value2, label='NS 70% bank account', color='green', linewidth=1)
ax1.plot(time, portfolio_value3, label=r'Time varying strategy $\bar\lambda=0.05$', color='red', linewidth=1)
ax1.plot(time, portfolio_value4, label=r'Time varying strategy $\bar\lambda=0.5$', color='orange', linewidth=1)
ax1.plot(time, S, label='Stock', color='black', linewidth=1)
ax1.set_ylabel('Value [currency unit]')
ax1.set_title('Portfolio Value over Time')
ax1.legend()
ax1.grid(visible=True)

# P&L
PnL1 = np.diff(portfolio_value1)
PnL2 = np.diff(portfolio_value2)
PnL3 = np.diff(portfolio_value3)
PnL4 = np.diff(portfolio_value4)
#ax2.plot(time[1:], PnL1, label=r'$\pi_1$', color='blue', linewidth=0.7)
#ax2.plot(time[1:], PnL2, label=r'$\pi_2$', color='green', linewidth=0.7)
ax2.plot(time[1:], PnL3, label=r'$\pi_3$', color='red', linewidth=0.7)
ax2.plot(time[1:], PnL4, label=r'$\pi_4$', color='orange', linewidth=0.7)
ax2.set_xlabel('Time')
ax2.set_ylabel('Profit & Loss [currency unit]')
ax2.set_title('Profit & Loss over Time')
ax2.legend()
ax2.grid(visible=True)

plt.tight_layout()  
plt.show()
