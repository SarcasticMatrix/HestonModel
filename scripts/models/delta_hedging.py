import matplotlib.pyplot as plt
import numpy as np
from scripts.BlackScholes import BlackScholes

initial = 100
r = 0.05
mu = 0.1
volatility = 0.2
hedging_volatility = 0.3
T = 1

n = 1000
N = 1000

strike = 110

plt.figure()
blackscholes = BlackScholes(initial=initial, r=r, mu=mu, volatility=volatility, T=T)
portfolio, S = blackscholes.delta_hedging(option='call', strike=strike, N=N, hedging_volatility=hedging_volatility)
ST = S[:,-1]
plt.scatter(ST, portfolio[:, -1], label='Portfolio', color='blue', linewidth=1)

ST = np.arange(start=np.min(ST), stop=np.max(ST), step=0.1)
payoff = np.maximum(ST - strike, 0)
plt.plot(ST, payoff, label='Option payoff', color='black')

plt.xlabel(r'Final Price $S_T$', fontsize=12)
plt.ylabel('Value [$]', fontsize=12)
plt.legend(loc='upper left')
plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
plt.minorticks_on()
plt.grid(which="minor", visible=False)
plt.title(f'Delta Hedging with Black-Scholes', fontsize=16)
# plt.show()




N = 1000
blackscholes = BlackScholes(initial=initial, r=r, mu=mu, volatility=volatility, T=T)
portfolio, S = blackscholes.delta_hedging(option='call', strike=strike, N=N, hedging_volatility=hedging_volatility)
times = np.linspace(0, T, n+1)
portfolio = portfolio.T
plt.figure()
plt.plot(times, portfolio, label='P&L', color='blue', linewidth=1)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value [$]', fontsize=12)
plt.legend(loc='upper left')
plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
plt.minorticks_on()
plt.grid(which="minor", visible=False)
plt.title(f'Delta Hedging with Black-Scholes', fontsize=16)
plt.show()