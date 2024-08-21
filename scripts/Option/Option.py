import numpy as np
from scripts.HestonModel import HestonModel
from typing import Literal

class Option:

    def __init__(
            self, 
            flag: Literal['call', 'put'], 
            strike: float, 
            time_to_maturity: float,
            interest: float,
            spot: float,
            current_vol: float
        ) -> None:

        self.flag = flag
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.interest = interest
        self.spot = spot
        self.current_vol = current_vol

    def price_call(self, heston: HestonModel):

        price, _ = heston.carr_madan_price()
        return price

    def price_put(self, heston: HestonModel):

        call_price, _ = heston.carr_madan_price()
        price = call_price - self.spot + self.strike * np.exp(- self.interest * self.time_to_maturity)
        return price

    def price(self, heston: HestonModel):

        heston.K = self.strike, 
        heston.T = self.time_to_maturity,
        heston.r = self.interest,
        heston.S0 = self.spot,
        heston.V0 = self.current_vol

        if self.flag == 'put':
            price = self.price_put(heston)
        
        elif self.flag == 'call':
            price = self.price_call(heston)

        return price