import numpy as np
import matplotlib.pyplot as plt

class GeometricBrownianMotion:
    """
    Class representing a Geometric Brownian Motion.

    Parameters:
    - initial (float): initial value of the process
    - drift (float): drift coefficient of the process
    - volatility (float): volatility coefficient of the process
    - jump_size (float): mean size of jumps
    - jump_std (float): variance of jump sizes
    - jump_nbr (float): intensity of jumps (average number of jumps per unit time)
    - T (float): time horizon of the simulation
    - seed (int): seed for random number generation

    Methods:
    - simulate(scheme='euler', n=100, N=1000): 
      Simulates and returns several simulated paths following the Geometric Brownian Motion model.
    - plot_simulation(scheme='euler', n=1000): 
      Plots the simulation of a Geometric Brownian Motion trajectory.
    """

    def __init__(
            self, 
            initial: float, 
            drift: float, 
            volatility: float, 
            jump_size: float,
            jump_std: float,
            jump_nbr: float = 1,
            T: float = 1, 
            seed: int = 42):
        """
        Initializes a Geometric Brownian Motion object.

        Parameters:
        - initial (float): initial value of the process
        - drift (float): drift coefficient of the process
        - volatility (float): volatility coefficient of the process
        - jump_size (float): mean size of jumps
        - jump_std (float): variance of jump sizes
        - jump_nbr (float): intensity of jumps (average number of jumps per unit time)
        - T (float): time horizon of the simulation
        - seed (int): seed for random number generation
        """

        # GBM parameters
        self.initial = initial
        self.drift = drift
        self.volatility = volatility

        # Jump parameters
        self.jump_nbr = jump_nbr
        self.jump_size = jump_size
        self.jump_std = jump_std

        # Other parameters
        self.T = T
        self.seed = seed

    def simulate(
            self, 
            scheme: str = "euler", 
            n: int = 100, 
            N: int = 1000, 
        ) -> np.array:
        """
        Simulates and returns several simulated paths following the Geometric Brownian Motion model.

        Parameters:
        - scheme (str): the discretization scheme used ('euler' or 'milstein')
        - n (int): number of time steps in a path
        - N (int): number of simulated paths

        Returns:
        - S (np.array): simulated stock paths
        """
        np.random.seed(self.seed)

        dt = self.T / n
        S = np.zeros((N, n + 1))
        S[:, 0] = self.initial

        # Jumps
        poissons = np.empty_like(S)
        for j in range(N):
            poissons[j,:] = np.random.poisson(lam=self.jump_nbr * dt, size=n+1)
        jump_sizes = np.random.normal(
                loc=self.jump_size, scale=self.jump_std, size=(N, n+1)
            )
        jumps = poissons * jump_sizes

        for i in range(1, n + 1):

            # Brownian motion
            N1 = np.random.normal(loc=0, scale=1, size=N)
            Z = N1 * np.sqrt(dt)

            # Update the processes 
            S[:, i] = S[:, i - 1] + self.drift * S[:, i - 1] * dt + self.volatility * S[:, i - 1] * Z

            if scheme == "milstein":
                S[:, i] += 1 / 2 * self.volatility * S[:, i - 1] * (Z ** 2 - dt)
            elif scheme != 'euler':
                print("Choose a scheme between: 'euler' or 'milstein'")
            S[:, i] = S[:, i] * (1 + jumps[:, i])

        if N == 1:
            S = S.flatten()
            jumps = jumps.flatten()
        return S, jumps

    def plot_simulation(
            self, 
            scheme: str = 'euler', 
            n: int = 1000,
        ) -> np.array:
        """
        Plots the simulation of a Geometric Brownian Motion trajectory.

        Parameters:
        - scheme (str): the discretization scheme used ('euler' or 'milstein')
        - n (int): number of time steps in a path
        """        
        S, jumps = self.simulate(n=n, scheme=scheme, N=1)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot for the stock path
        axs[0].plot(np.linspace(0, 1, n + 1), S, label='Risky asset', color='blue', linewidth=1)
        axs[0].set_ylabel('Value [$]', fontsize=12)
        axs[0].legend(loc='upper left')
        axs[0].grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        axs[0].minorticks_on()
        axs[0].grid(which="minor", visible=False)

        # Plot for the jumps
        # axs[1].stem(np.linspace(0, 1, n + 1), jumps, linefmt='r-', markerfmt='ro', basefmt='r-', label='Jumps')
        axs[1].step(np.linspace(0, 1, n + 1), jumps, label='Jumps')
        axs[1].set_xlabel('Time', fontsize=12)
        axs[1].set_ylabel('Jump Size', fontsize=12)
        axs[1].legend(loc='upper left')
        axs[1].grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        axs[1].minorticks_on()
        axs[1].grid(which="minor", visible=False)

        plt.suptitle(f'Geometric Brownian Motion Model Simulation with {scheme} scheme', fontsize=16)
        plt.show()

        return S, jumps
