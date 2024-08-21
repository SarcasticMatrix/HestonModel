import numpy as np

class Portfolio:
    """
    Abstract base class to model portfolio strategies.

    Attributes:
        r (float): Interest rate.
        dt (float): time step
    """
    def __init__(self, r, dt):
        """
        Initializes the Portfolio with the initial value of the portfolio and interest rate.

        Args:
            r (float): Interest rate.
            dt (float): time step
        """
        self.r = r    # Interest rate
        self.dt = dt
    
    def grow_bank_account(self, bank_account):
        """
        Method to grow the bank with interest.

        Args:
            bank_account (float): Current bank balance.

        Returns:
            float: The updated bank balance.
        """
        return bank_account * np.exp(self.r * self.dt)
    
    def grow_stocks_account(self, stocks_account, S_now, S_previous):
        """
        Method to grow the stocks account with interest.

        Args:
            stocks_account (float): Current stocks account balance.
            S_now (float): Current stock price.
            S_previous (float): Previous stock price.

        Returns:
            float: The updated stocks account balance.
        """
        number_of_stocks = stocks_account / S_previous
        return number_of_stocks * S_now
    
    def back_test(self, S, portfolio0, allocation_strategy):
        """
        Method to back test the optimal portfolio strategy.

        Args:
        - S (np.array): path of the stock
        - portfolio0 (float): value at time 0 of the portfolio
        - allocation_strategy (np.array): allocation strategy, same size as S

        Returns:
            bank_account (array_like): Money in the bank account over time.
            stocks_account (array_like): Money in stocks over time.
        """

        stock_allocation = allocation_strategy[0]
        bank_allocation = 1 - stock_allocation

        bank_account = np.empty_like(S)
        stocks_account = np.empty_like(S)

        stocks_account[0] = portfolio0 * stock_allocation
        bank_account[0] = portfolio0 * bank_allocation

        for t in range(1, len(S)):

            # Update the portfolio
            bank_account[t] = self.grow_bank_account(bank_account[t-1])
            stocks_account[t] = self.grow_stocks_account(stocks_account=stocks_account[t-1], S_now=S[t], S_previous=S[t-1])

            # Update the allocation
            stock_allocation = allocation_strategy[t]
            bank_allocation = 1 - stock_allocation
            total_value = bank_account[t] + stocks_account[t]
            bank_account[t] = total_value * bank_allocation
            stocks_account[t] = total_value * stock_allocation
        
        return bank_account, stocks_account

    # #### Strategies and backtesting 

    # ## Constant strategies

    # def constant_back_test(self, S, portfolio0, allocate_portfolio):
    #     """
    #     Method to back test the portfolio strategy for a constant strategy.

    #     Args:
    #         S (array_like): Array of stock prices over time.
    #         portfolio0 (float): Initial value of portfolio.
    #         allocate_portfolio (function): Allocation strategy function.

    #     Returns:
    #         bank_account (array_like): Money in the bank account over time.
    #         stocks_account (array_like): Money in stocks over time.
    #     """

    #     bank_account = np.empty_like(S)
    #     stocks_account = np.empty_like(S)

    #     bank_allocation, stock_allocation = allocate_portfolio(portfolio0)
    #     bank_account[0] = portfolio0 * bank_allocation
    #     stocks_account[0] = portfolio0 * stock_allocation
        
    #     for t in range(1, len(S)):

    #         # Update the portfolio
    #         bank_account[t] = self.grow_bank_account(bank_account[t-1])
    #         stocks_account[t] = self.grow_stocks_account(stocks_account=stocks_account[t-1], S_now=S[t], S_previous=S[t-1])

    #         # Update the allocation
    #         bank_allocation, stock_allocation = allocate_portfolio(S[t])
    #         total_value = bank_account[t] + stocks_account[t]
    #         bank_account[t] = total_value * bank_allocation
    #         stocks_account[t] = total_value * stock_allocation
        
    #     return bank_account, stocks_account

    # ## Time varying strategies

    # def time_varying_allocate_portfolio(self, premium_volatility_risk, p):
    #     """
    #     Function to determine the allocation of the portfolio based on premium volatility risk and p.

    #     Args:
    #         premium_volatility_risk (float): Premium volatility risk parameter.
    #         p (float): Parameter.

    #     Returns:
    #         function: Allocation function.
    #     """
    #     alpha = lambda v: self.r + np.sqrt(v)
    #     returns = lambda v: self.r + premium_volatility_risk * np.sqrt(v)
    #     pi = lambda v: (alpha(v)-returns(v))/((1-p) * v) 
    #     return pi

    # def time_varying_back_test(self, S, portfolio0, premium_volatility_risk, V, p):
    #     """
    #     Method to back test the portfolio strategy for a time-varying strategy.

    #     Args:
    #         S (array_like): Array of stock prices over time.
    #         portfolio0 (float): Initial value of portfolio.
    #         premium_volatility_risk (float): Premium volatility risk parameter.
    #         V (array_like): Array of values.
    #         p (float): Parameter.

    #     Returns:
    #         bank_account (array_like): Money in the bank account over time.
    #         stocks_account (array_like): Money in stocks over time.
    #     """

    #     allocate_portfolio = self.time_varying_allocate_portfolio(premium_volatility_risk, p)

    #     stock_allocation = allocate_portfolio(V[0])
    #     bank_allocation = 1 - stock_allocation

    #     bank_account = np.empty_like(S)
    #     stocks_account = np.empty_like(S)

    #     stocks_account[0] = portfolio0 * stock_allocation
    #     bank_account[0] = portfolio0 * bank_allocation

    #     for t in range(1, len(S)):

    #         # Update the portfolio
    #         bank_account[t] = self.grow_bank_account(bank_account[t-1])
    #         stocks_account[t] = self.grow_stocks_account(stocks_account=stocks_account[t-1], S_now=S[t], S_previous=S[t-1])

    #         # Update the allocation
    #         stock_allocation = allocate_portfolio(V[t])
    #         bank_allocation = 1 - stock_allocation
    #         total_value = bank_account[t] + stocks_account[t]
    #         bank_account[t] = total_value * bank_allocation
    #         stocks_account[t] = total_value * stock_allocation
        
    #     return bank_account, stocks_account

    # ## Optimal strategies

    # def optimal_allocate_portfolio(
    #         self, 
    #         p:float, 
    #         heston:HestonModel
    #     ):
    #     """
    #     Function to determine the optimal allocation of the portfolio based on premium volatility risk and p.

    #     Args:
    #         premium_volatility_risk (float): Premium volatility risk parameter.
    #         p (float): Parameter.

    #     Returns:
    #         function: Allocation function.
    #     """

    #     k0 = p * heston.premium_volatility_risk**2 / (1-p)
    #     k1 = heston.kappa - p * heston.premium_volatility_risk * heston.sigma * heston.rho / (1-p)
    #     k2 = heston.sigma**2 + p * heston.sigma**2 * heston.rho**2 /(1-p)
    #     k3 = np.sqrt(k1**2 - k0*k2)
    
    #     b = lambda t: k0 * (np.exp(k3 * (heston.T-t)) - 1) / (np.exp(k3 * (heston.T-t)) * (k1 + k3) - k1 + k3)
    #     pi = lambda t: heston.premium_volatility_risk / (1-p) + (heston.sigma * heston.rho) / (1-p) * b(t) 
    #     return pi
    
    # def optimal_back_test(self, S, portfolio0, p, heston):
    #     """
    #     Method to back test the optimal portfolio strategy.

    #     Args:


    #     Returns:
    #         bank_account (array_like): Money in the bank account over time.
    #         stocks_account (array_like): Money in stocks over time.
    #     """

    #     allocate_portfolio = self.optimal_allocate_portfolio(p, heston)

    #     stock_allocation = allocate_portfolio(0)
    #     bank_allocation = 1 - stock_allocation

    #     bank_account = np.empty_like(S)
    #     stocks_account = np.empty_like(S)

    #     stocks_account[0] = portfolio0 * stock_allocation
    #     bank_account[0] = portfolio0 * bank_allocation

    #     for t in range(1, len(S)):

    #         # Update the portfolio
    #         bank_account[t] = self.grow_bank_account(bank_account[t-1])
    #         stocks_account[t] = self.grow_stocks_account(stocks_account=stocks_account[t-1], S_now=S[t], S_previous=S[t-1])

    #         # Update the allocation
    #         stock_allocation = allocate_portfolio(t)
    #         bank_allocation = 1 - stock_allocation
    #         total_value = bank_account[t] + stocks_account[t]
    #         bank_account[t] = total_value * bank_allocation
    #         stocks_account[t] = total_value * stock_allocation
        
    #     return bank_account, stocks_account