import enum
import numpy as np

from functools import partial
from scipy.stats import qmc, norm

from src.black_scholes import payoff, CallPut, bs_price


def exact_gbm(S, dt, sigma, r, dW):
    """
    Calculates the next value pf a GBM SDE using the analytical solution
    
    Parameters:
        S (float): Stock price at t.
        dt (float): Time step size.
        sigma (float): Volatility of the stock.
        r (float): Risk-free interest rate.
        dW (float): Random variable representing the Wiener process.
    
    Returns:
        float: The stock price at the next time step.
    """
    return S * np.exp((r - sigma ** 2 / 2) * dt + (sigma * np.sqrt(dt) * dW))


def euler_maruyama_gbm(S, dt, sigma, r, dW):
    """
    Calculates the next value of a GBM SDE using the Euler-Maryuama approximation
    
    Parameters:
        S (float): Stock price at t.
        dt (float): Time step size.
        sigma (float): Volatility of the stock.
        r (float): Risk-free interest rate.
        dW (float): Random variable representing the Wiener process.
    
    Returns:
        float: The stock price at the next time step.
    """
    return S + (r * S * dt) + (sigma * S * np.sqrt(dt) * dW)


def milstein_gbm(S, dt, sigma, r, dW):
    """
    Calculates the next value of a GBM SDE using the Milstein approximation
    
    Parameters:
        S (float): Stock price at t.
        dt (float): Time step size.
        sigma (float): Volatility of the stock.
        r (float): Risk-free interest rate.
        dW (float): Random variable representing the Wiener process.
    
    Returns:
        float: The stock price at the next time step.
    """
    return S + (r * S * dt) + (sigma * S * np.sqrt(dt) * dW) + (0.5 * sigma**2 * S * ((dW * np.sqrt(dt))**2 - dt))


class VarianceReduction(enum.Enum):
    Gaussian = 'gaussian'
    Antithetic = 'antithetic'
    Sobol = 'sobol'
    BrownianBridge = 'brownian_bridge'
    SobolBrownianBridge = 'sobol_brownian_bridge'


class NumericalScheme(enum.Enum):
    Exact = 'exact'
    Milstein = 'milstein'
    Euler = 'euler'


SCHEME = {
    NumericalScheme.Exact: exact_gbm,
    NumericalScheme.Euler: euler_maruyama_gbm,
    NumericalScheme.Milstein: milstein_gbm
}


class BrownianBridge:

    def __init__(self, N, n_sims) -> None:
        """
        Calculate the indices determining the order of the point construction,
        the weighting coefficients and the conditional variances
        """
        self.n_sims = n_sims
        self.N = N  

        self.bridge_index = np.full(N, np.nan)
        self.left_index = np.full(N, np.nan)
        self.right_index = np.full(N, np.nan)
        self.left_weight = np.full(N, np.nan)
        self.right_weight = np.full(N, np.nan)
        self.stdev = np.full(N, np.nan)

        map = np.zeros(N)
        map[N-1] = 1

        self.bridge_index[0] = N-1
        self.stdev[0] = np.sqrt(N)

        self.left_weight[0] = 0
        self.right_weight[0] = 0

        j = 0
        for i in range(1, N):
            while map[j]:
                # find the first unpopulated entry
                j += 1

            k = j
            while not map[k]:
                # find the next populated entry from j
                k += 1
            
            l = j + ((k -1 - j) // 2)

            map[l] = i

            self.bridge_index[i] = l
            self.left_index[i] = j 
            self.right_index[i] = k
            self.left_weight[i] = (k - l) / (k + 1 - j)
            self.right_weight[i] = (l+1 -j) / (k + 1 - j)
            self.stdev[i] = np.sqrt(((l+1-j) * (k-l)) / (k+1-j))
            j = k+1
            if j >= N:
                j=0
    
    def build_paths(self, dZ):
        paths = np.full([self.n_sims, self.N], np.nan)
        paths[:, self.N -1] = self.stdev[0] * dZ[:, 0]

        for i in range(1, self.N):
            j = int(self.left_index[i])
            k = int(self.right_index[i])
            l = int(self.bridge_index[i])

            if j:
                paths[:, l] = self.left_weight[i] * paths[:, j-1] + self.right_weight[i] * paths[:, k] + self.stdev[i] * dZ[:, i]
            else:
                paths[:, l] = self.right_weight[i] * paths[:, k] + self.stdev[i] * dZ[:, i]

        return paths

    def dW(self, dZ):
        paths = self.build_paths(dZ)
        return np.diff(paths)


class GBMMonteCarloSimulation:

    def __init__(self, T=1, k=8, S0=100, sigma=0.2, r=0.05, n_sims=1000):
        """
        Initialize the Monte Carlo Simulation object.

        Parameters:
        - T (float): Time horizon in years (default: 1)
        - k (int): Number of time steps is 2**k (default: 8)
        - S0 (float): Initial stock price (default: 100)
        - sigma (float): Volatility of the stock (default: 0.2)
        - r (float): Risk-free interest rate (default: 0.05)
        - n_sims (int): Number of simulations (default: 1000)
        """
        self.T = T
        self.N = 2 ** k   # power of 2 makes brownian bridge easier to implement
        self.dt = T / (self.N - 1)  # time step size TODO - CHECK THIS!!!
        self.n_sims = n_sims
        self.k = k
        self.S0 = S0
        self.sigma = sigma
        self.r = r

    def simulate_paths(self):
        return partial(simulate_path, self.S0, self.n_sims, self.N, self.sigma, self.r, self.dt)

    def gaussian_variate(self):
        samples = np.random.uniform(0, 1, size=[self.n_sims, self.N])
        dW = norm.ppf(samples, loc=0, scale=1)
        return dW

    def anithetic_variate(self):
        if self.n_sims % 2 != 0:
            raise("Number of simulations must be even for antithetic variates")
        
        dW = np.random.normal(0, 1, [self.n_sims // 2, self.N])
        dW = np.concatenate((dW, -dW))
        return dW

    def sobol(self):
        sobol_engine = qmc.Sobol(d=self.N, scramble=True)
        sobol_samples = sobol_engine.random(self.n_sims)
        dW = norm.ppf(sobol_samples)
        return dW

    def brownian_bridge(self):
        dZ = np.random.normal(0, 1, [self.n_sims, self.N])
        bb = BrownianBridge(self.N, self.n_sims)
        dW = bb.dW(dZ) 
        return dW

    def sobol_brownian_bridge(self):
        sobol_engine = qmc.Sobol(d=self.N, scramble=True)
        sobol_samples = sobol_engine.random(self.n_sims)
        dZ = norm.ppf(sobol_samples)
        bb = BrownianBridge(self.N, self.n_sims)
        dW = bb.dW(dZ)
        return dW

    def run_sim(self, variance_reduction=None, numerical_scheme=NumericalScheme.Euler):

        dW = {
            VarianceReduction.Antithetic: self.anithetic_variate,
            VarianceReduction.BrownianBridge: self.brownian_bridge,
            VarianceReduction.Sobol: self.sobol,
            VarianceReduction.SobolBrownianBridge: self.sobol_brownian_bridge
        }.get(variance_reduction, self.gaussian_variate)()  # default to gaussian

        paths = self.simulate_paths()(dW, numerical_scheme)
        return paths
    
    def option_price(self, K, call_put, paths):
        payoffs = payoff(paths[:, -1], K, call_put)
        average_payoff = payoffs.mean()
        prc =  np.exp(-self.r * self.T) * average_payoff
        return prc

    def mc_pricer(self, K, call_put, variance_reduction=None, numerical_scheme=NumericalScheme.Euler):
        """
        Monte Carlo pricer for European options
        """
        paths = self.run_sim(variance_reduction, numerical_scheme)
        return self.option_price(K, call_put, paths)
    

def simulate_path(S0, n_sims, N, sigma, r, dt, dW, numerical_scheme):
    """
    Simulates the path of an asset price using a specified numerical scheme.

    Parameters:
    S0 (float): Initial asset price.
    n_sims (int): Number of simulations.
    N (int): Number of time steps.
    sigma (float): Volatility of the asset price.
    r (float): Risk-free interest rate.
    dt (float): Time step size.
    dW (ndarray): Array of random Wiener increments.
    numerical_scheme (str): Name of the numerical scheme to use (e.g. 'euler', 'milstein').

    Returns:
    ndarray: Array of simulated asset price paths.
    """

    fn = SCHEME.get(numerical_scheme, None)

    if fn is None:
        raise ValueError(f"Unsupported numerical scheme {numerical_scheme} supplied")
    
    S = np.zeros([n_sims, N])

    S[:, 0] = S0
    for n in range(1, N):
        S[:, n] = fn(S[:, n-1], dt, sigma, r, dW[:, n-1])
    return S


if __name__ == "__main__":
    # Example usage
    sim = GBMMonteCarloSimulation(
        S0 = 100,
        r = 0.05,
        sigma = 0.2,
        T = 1,
        k = 8,
        n_sims=1000
    )

    # Generate the paths
    paths = sim.run_sim(
        variance_reduction=VarianceReduction.SobolBrownianBridge, 
        numerical_scheme=NumericalScheme.Euler
    )

    # Generate paths and price the option
    mc =sim.mc_pricer(
        K=100, 
        call_put=CallPut.CALL, 
        variance_reduction=VarianceReduction.SobolBrownianBridge, 
        numerical_scheme=NumericalScheme.Exact
    )

    bs = bs_price(100, 100, 0.05, 0.2, 1, CallPut.CALL)
    print("Pricing European Option: K=100, T-1, r=0.05, sigma=0.2")
    print(f"BS price: {bs}")
    print(f"MC Euler Sobol Brownian Bridge: {mc}")
