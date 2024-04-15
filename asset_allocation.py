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
bank_account, stocks_account = strategy.back_test(S=S, portfolio0=S0, allocate_portfolio=allocate_portfolio)
portfolio_value1 = bank_account + stocks_account


### Constant allocation strategy knowing return and volatility

alpha = lambda v: r + np.sqrt(v)
returns = lambda v: r + premium_volatility_risk * np.sqrt(v)
pi = lambda v, p: (alpha(v)-returns(v))/((1-p) * v) 

portfolio_value2 = pi(V, 0.5) * S 


### Plot strategies

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  

ax1.plot(time, portfolio_value1, label='Naive constant strategy', color='blue', linewidth=1)
ax1.set_ylabel('Value [currency unit]')
ax1.set_title('Portfolio Value over Time')
ax1.legend()
ax1.grid(visible=True)

# P&L
#PnL = [portfolio_value1[i] - portfolio_value1[i-1] for i in range(1, len(portfolio_value1))]
PnL = portfolio_value1 - S0
ax2.plot(time, PnL, label='Profit & Loss', color='green', linewidth=1)
ax2.set_xlabel('Time')
ax2.set_ylabel('Profit & Loss [currency unit]')
ax2.set_title('Profit & Loss over Time')
ax2.legend()
ax2.grid(visible=True)

plt.tight_layout()  
plt.show()
