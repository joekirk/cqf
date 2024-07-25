import numpy as np
import pandas as pd

from enum import Enum

from src.monte_carlo import GBMMonteCarloSimulation, NumericalScheme, VarianceReduction
from src.black_scholes import CallPut, bs_price, bs_delta, bs_gamma


STOCK = 'STOCK'
OPTION = 'OPTION'


class VolatilityMeasure(Enum):
    ACTUAL = 'actual'
    IMPLIED = 'implied'


class ReplicatingPortfolio:

    def __init__(self, T, sigma_a, sigma_i, r, K, call_put, S, n_sims, N, dt) -> None:

        self.N = N
        self.n_sims = n_sims
        self.dt = dt
        self.S = S
        self.r = r

        self.tau = np.array(
            [
                np.array([T - (self.dt * i) for i in range(self.N)]) 
                for _ in range(n_sims)
            ]
        )

        self.delta_i = bs_delta(self.S, K, r, sigma_i, self.tau, call_put)
        self.delta_a = bs_delta(self.S, K, r, sigma_a, self.tau, call_put)

        self.option_price_i = bs_price(self.S, K, r, sigma_i, self.tau, call_put)
        self.option_price_a = bs_price(self.S, K, r, sigma_a, self.tau, call_put)

        self.bs_gamma_i = bs_gamma(self.S, K, r, sigma_i, self.tau)
        self.bs_gamma_a = bs_gamma(self.S, K, r, sigma_a, self.tau)

    @classmethod
    def from_monte_carlo(cls, T, k, S0, sigma_i, sigma_a, r, K, call_put, n_sims):
        mc = GBMMonteCarloSimulation(T=T, k=k, S0=S0, sigma=sigma_a, r=r, n_sims=n_sims)
        S = mc.run_sim(
            variance_reduction=VarianceReduction.SobolBrownianBridge,
            numerical_scheme=NumericalScheme.Milstein
        )
        replictator = cls(
            T=T, 
            sigma_a=sigma_a, 
            sigma_i=sigma_i, 
            r=r, 
            K=K,
            call_put=call_put,
            S=S, 
            n_sims=n_sims,
            N=mc.N,
            dt=mc.dt
        )
        return replictator

    def portfolio(self, volatility_measure: VolatilityMeasure):
        """
        Constructs a portfolio of stock and option positions. The portfolio contains
        contains a single long option position and a short stock position.  The keys
        OPTION / STOCK could be replaced with a key representing a stock / option pair.
        e.g SPX / SPX_2022_01_01_100_CALL.

        The volatility measure determines the quantity of the stock held since we hedge 
        the delta of the option which is calculated using actual or implied

        Parameters:
        - volatility_measure: A VolatilityMeasure enum value indicating the type of volatility measure to use.
           VolatilityMeasure.ACTUAL
           VolatilityMeasure.IMPLIED

        Returns:
        - A dictionary representing the constructed portfolio. The dictionary has two keys:
            - STOCK: The quantity of stock in the portfolio, which is determined by the delta value.
            - OPTION: The quantity of  the option positions in the portfolio. 
            
        """
        if volatility_measure == VolatilityMeasure.ACTUAL:
            delta = self.delta_a
        else:
            delta = self.delta_i

        return {
            STOCK: -delta,
            OPTION: np.ones((self.n_sims, self.N))    
        }

    def calculate_pnl(self, portfolio):
        v_option = self.option_price_i * portfolio[OPTION]
        v_stock = self.S * portfolio[STOCK]
        v_portfolio = v_option + v_stock

        cashflows = np.insert(np.diff(portfolio[STOCK]), 0, 0, axis=1) * self.S * -1

        cash_account = np.zeros((self.n_sims, self.N))
        cash_account[:, 0] = -1 * v_portfolio[:, 0]

        for i in range(1, self.N):
            cash_account[:, i] = cash_account[:, i-1] + (cash_account[:, i-1] * (np.exp(self.r * self.dt) - 1)) + cashflows[:, i]

        pnl = v_portfolio + cash_account

        return {
            'S': self.S,
            'Option': v_option,
            'Stock': v_stock,
            'Portfolio': v_portfolio,
            'Cash': cash_account,
            'CashFlows': cashflows,
            'PnL': pnl
        }
    

def pnl_statistics(idx, pnl):
    """
    Return the portfolio statistics for a given simulation index.
    """
    return pd.DataFrame({k: pnl[k][idx] for k in pnl.keys()})
       

if __name__ == "__main__":
    # Example usage

    n_sims = 2**9           # Number of simulations
    S0 = 100                # Initial stock price
    r = 0.02                # Risk free rate
    sigma_a = 0.35          # Actual Volatility
    sigma_i = 0.2           # Implied Volatility
    T = 1                 # Time to maturity
    K = 100                 # Strike price
    call_put=CallPut.CALL   # Option type
    k = 8                   # Used to calulate the number of steps in the simulation

    # Run a number of different path projections for the evolution of the underlying
    replicating_portfolio = ReplicatingPortfolio.from_monte_carlo(
        T=T, 
        k=k, 
        S0=S0, 
        sigma_a=sigma_a, 
        sigma_i=sigma_i, 
        r=r, 
        K=K,
        n_sims=n_sims,
        call_put=call_put
    )

    # Calculate the PNL if the the options is hedged using implied volatility
    implied_portfolio = replicating_portfolio.portfolio(VolatilityMeasure.IMPLIED)
    implied_pnl = replicating_portfolio.calculate_pnl(implied_portfolio)
    
    # Inspect the PNL for the first simulation
    df = pnl_statistics(idx=0, pnl=implied_pnl)

    # Calculate the PNL if the the options is hedged using actual volatility
    actual_portfolio = replicating_portfolio.portfolio(VolatilityMeasure.ACTUAL)
    actual_pnl = replicating_portfolio.calculate_pnl(actual_portfolio)
    
    # Inspect the PNL for the first simulation
    df = pnl_statistics(idx=0, pnl=actual_pnl)
