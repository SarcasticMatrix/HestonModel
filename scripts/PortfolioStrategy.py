import numpy as np

class PortfolioStrategy:
    """
    Abstract base class to model portfolio strategies.

    Attributes:
        r (float): Interest rate.
        dt (float): time step
    """
    def __init__(self, r, dt):
        """
        Initializes the PortfolioStrategy with the initial value of the portfolio and interest rate.

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
            bank_account (float): Current bank bank.

        Returns:
            float: The updated bank balance.
        """
        return bank_account * np.exp(self.r * self.dt)
    
    def grow_stocks_account(self, stocks_account, S_now, S_previous):
        """
        Method to grow the cash with interest.

        Args:
            stocks_account (float): Current stock cash.

        Returns:
            float: The updated cash balance.
        """
        number_of_stocks = stocks_account / S_previous
        return number_of_stocks * S_now

    def back_test(self, S, portfolio0, allocate_portfolio):
        """
        Method to back test the portfolio strategy.

        Args:
            S (array_like): Array of stock prices over time.
            portfolio0 (float): initial value of portfolio.
            allocate_portfolio (function): allocation strategy

        Returns:
            bank_account (array_like), money in the bank account over time
            stocks_account (array_like), money in stocks over time
        """

        bank_account = np.empty_like(S)
        bank_account[0] = portfolio0/2

        stocks_account = np.empty_like(S)
        stocks_account[0] = portfolio0/2
        
        for t in range(1, len(S)):

            # Update the portfolio
            bank_account[t] = self.grow_bank_account(bank_account[t-1])
            stocks_account[t] = self.grow_stocks_account(stocks_account=stocks_account[t-1], S_now=S[t], S_previous=S[t-1])

            # Update the allocation
            bank_allocation, stock_allocation = allocate_portfolio(S[t])
            total_value = bank_account[t] + stocks_account[t]
            bank_account[t] = total_value * bank_allocation
            stocks_account[t] = total_value * stock_allocation
        
        return bank_account, stocks_account

