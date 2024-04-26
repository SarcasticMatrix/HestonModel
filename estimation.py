from scripts.HestonModel import HestonModel

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

model = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)

import numpy as np

n = 10000
time = np.linspace(start=0, stop=model.T, num=n+1)
S, V, _ = model.simulate(scheme='milstein', n=n, N=1)
volatility = np.sqrt(V)

log_returns = 100 * np.diff(np.log(S))

import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined
from particles.collectors import Moments

class StochVol(ssm.StateSpaceModel):
    def PX0(self):  # Distribution of X_0
        return dists.Normal(loc=self.mu, scale=self.sigma / np.sqrt(1. - self.rho**2))
    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)
    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=0., scale=np.exp(x))
    
# prior_dict = {'mu':dists.Normal(),
#               'sigma': dists.Gamma(a=1., b=1.),
#               'rho':dists.Beta(9., 1.)}
# my_prior = dists.StructDist(prior_dict)

# from particles import mcmc  # where the MCMC algorithms (PMMH, Particle Gibbs, etc) live
# pmmh = mcmc.PMMH(ssm_cls=StochVol, prior=my_prior, data=log_returns, Nx=50, niter = 1000)
# pmmh.run()  # Warning: takes a few seconds

# import matplotlib.pyplot as plt
# import seaborn as sb
# burnin = 100  # discard the 100 first iterations
# for i, param in enumerate(prior_dict.keys()):
#     plt.subplot(2, 2, i+1)
#     sb.histplot(pmmh.chain.theta[param][burnin:])
#     plt.title(param)
# plt.show()

# mu = -0.55
# sigma = 0.05
# rho = 0.9
# my_model = StochVol(mu=-1., rho=.9, sigma=.1)  # actual model
# fk_model = ssm.Bootstrap(ssm=my_model, data=log_returns)  # we use the Bootstrap filter
# pf = particles.SMC(fk=fk_model, N=1000, qmc=False, resampling='systematic', store_history=False, verbose=False, collect=[Moments()])
# pf.run()  # actual computation

# estimation = [m['mean'] for m in pf.summaries.moments]

estimation = model.SIR_estimation(log_returns=log_returns)

import matplotlib.pyplot as plt 

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(time, S, label=r'$r_t$', color='blue', linewidth=1)
ax1.legend()
ax1.set_ylabel('Value [$]', fontsize=12)

ax2.plot(time, V, label=r'$v_t$', color='orange', linewidth=1)
ax2.plot(time[1:], estimation, label='Estimation', color='violet', linewidth=1)
ax2.legend()
ax2.set_xlabel('Time [h]', fontsize=12)
ax2.set_ylabel('Variance', fontsize=12)

# fig.suptitle(f'Heston Model Simulation', fontsize=16)
plt.tight_layout()
plt.show()

MSE = np.mean((estimation - volatility[1:])**2)
MAE = np.mean(np.sqrt((estimation - volatility[1:])**2))
print(f"MSE: {round(MSE,3)} - MAE: {round(MAE,3)}")