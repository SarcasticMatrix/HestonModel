from scripts.HestonModel import HestonModel
from scripts.GeometricBrownianMotion import GeometricBrownianMotion as GBM

import matplotlib.pyplot as plt 
import numpy as np

### Initialisation of the model

n = 1000

S0 = 100
drift = 0.05
sigma = 0.3
T = 1

jump_nbr = 0
jump_size = -0.3
jump_std = 0.05

gbm = GBM(
    initial=S0,
    drift=drift, 
    volatility=sigma, 
    jump_nbr=jump_nbr, 
    jump_std=jump_std, 
    jump_size=jump_size, 
    T=T
) 
S, jumps = gbm.simulate(scheme='milstein', n=n, N=1)

from scripts.residual_analysis import residuals_analysis
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(S, order=(1, 0, 1)).fit()
time = np.linspace(0, 1, n + 1)

residuals_analysis(model.resid)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(time, S, label='actual', linewidth=0.7)
plt.plot(time[:-1], model.fittedvalues[1:], label='fitted', linewidth=0.7)
plt.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8)
plt.xlabel('Time')
plt.minorticks_on()
plt.grid(which="minor", visible=False)
plt.legend()
plt.show()



# heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)
# S,V = heston.plot_simulation(scheme='milstein')

