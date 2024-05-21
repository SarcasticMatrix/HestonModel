from scripts.HestonModel import HestonModel, calibrate

import matplotlib.pyplot as plt 
import time
import numpy as np

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

heston = HestonModel(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)
calibration = calibrate(option_type='calls', heston=heston)
print(calibration)