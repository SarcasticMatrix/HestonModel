from scripts.HestonModel import HestonModel

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

################################################################################
### Implied Volatility
Ks = np.arange(start=20, stop=200, step=1)
IVs = []
for K in tqdm(Ks):
    heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)
    price, error = heston.carr_madan_price()
    IV = np.sqrt(2 * np.pi / heston.T) * price / S0
    IVs.append(IV)

plt.figure()
plt.title(r"Implied Volatility")
plt.plot(Ks, IVs)
# plt.legend()
plt.xlabel(r'Strike ($K$)')
plt.ylabel("Implied Volatility")
plt.grid()
plt.show()

################################################################################
### Theta 
theta1 = 0.01
Ks = np.arange(start=20, stop=200, step=1)
prices_sigma1 = []
for K in tqdm(Ks):
    heston = HestonModel(S0, V0, r, kappa, theta1, drift_emm, sigma, rho, T, K=K)
    price, error = heston.carr_madan_price()
    prices_sigma1.append(price)

theta2 = 0.09
prices_sigma2 = []
for K in tqdm(Ks):
    heston = HestonModel(S0, V0, r, kappa, theta2, drift_emm, sigma, rho, T, K=K)
    price, error = heston.carr_madan_price()
    prices_sigma2.append(price)

theta3 = 1
prices_sigma3 = []
for K in tqdm(Ks):
    heston = HestonModel(S0, V0, r, kappa, theta3, drift_emm, sigma, rho, T, K=K)
    price, error = heston.carr_madan_price()
    prices_sigma3.append(price)


plt.figure()
plt.title(r"Call price as a function of price $K$")

plt.plot(Ks, prices_sigma1, label=rf'$\theta={theta1}$', linewidth=0.7, color='red')
plt.plot(Ks, prices_sigma2, label=rf'$\theta={theta2}$', linewidth=0.7, color='blue')
plt.plot(Ks, prices_sigma3, label=rf'$\theta={theta3}$', linewidth=0.7, color='green')

plt.legend()
plt.xlabel(r'Strike ($K$)')
plt.ylabel("Price [€]")
plt.grid()
plt.show()

###############################################################################
### Theta
Ks = np.arange(start=20, stop=200, step=1)
Ts = np.arange(start=0.1, stop=1.1, step=0.1)

prices_surface = np.zeros((len(Ts), len(Ks)))

for i, T in tqdm(enumerate(Ts)):
    for j, K in tqdm(enumerate(Ks)):
        heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K=K)
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

#from scripts.Option import Option
#option = Option(
#    flag='put',
#    strike=K,
#    time_to_maturity=T,
#    interest=r,
#    spot=S0,
#    current_vol=V0
#)
#heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)
#price, _ = option.price(heston)
#print(price)