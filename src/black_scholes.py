import enum
import numpy as np

from scipy.stats import norm

class CallPut(enum.Enum):
    CALL = 'C'
    PUT = 'P'


def d1(S, K, r, sigma, T):
    """
    Calculates the d1 value for the Black-Scholes formula.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
    T (float): The time to expiration of the option.

    Returns:
    float: The d1 value.
    """
    a = np.log(S/K)
    b = T * (r + sigma**2 * 0.5)
    c = sigma * np.sqrt(T)
    return (a + b) / c


def d2(S, K, r, sigma, T):
    """
    Calculates the d2 value for the Black-Scholes formula.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
    T (float): The time to expiration of the option.

    Returns:
    float: The d2 value.
    """
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)


def bs_call_price(S, K, r, sigma, T):
    """
    Calculates the price of a call option using the Black-Scholes formula.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
    T (float): The time to expiration of the option.

    Returns:
    float: The price of the call option.
    """
    _d1 = d1(S, K, r, sigma, T)
    _d2 = d2(S, K, r, sigma, T)

    _call = S * norm.cdf(_d1) - K * np.exp(np.multiply(-1, r) * T) * norm.cdf(_d2)

    return _call


def bs_put_price(S, K, r, sigma, T):
    """
    Calculates the price of a put option using the Black-Scholes formula.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
    T (float): The time to expiration of the option.

    Returns:
    float: The price of the put option.
    """
    _d1 = d1(S, K, r, sigma, T)
    _d2 = d2(S, K, r, sigma, T)

    _put = K * np.exp(np.multiply(-1, r) * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)

    return _put


def bs_call_delta(S, K, r, sigma, T):
    """
    Calculates the delta of an option using the Black-Scholes formula.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
    T (float): The time to expiration of the option.
    call_put (str): The type of option, either 'CALL' or 'PUT'.

    Returns:
    float: The delta of the option.
    """
    
    _d1 = d1(S, K, r, sigma, T)
    n_d1 = norm.cdf(_d1)
    _delta = n_d1
    return _delta


def bs_put_delta(S, K, r, sigma, T):
    """
    Calculates the delta of an option using the Black-Scholes formula.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
    T (float): The time to expiration of the option.
    call_put (str): The type of option, either 'CALL' or 'PUT'.

    Returns:
    float: The delta of the option.
    """
    
    _d1 = d1(S, K, r, sigma, T)
    n_d1 = norm.cdf(_d1)
    _delta = n_d1 - 1
    return _delta


def bs_price(S, K, r, sigma, T, call_put):
    """
    Calculates the price of a European option using the Black-Scholes formula.

    Parameters:
    S (float): Current price of the underlying asset.
    K (float): Strike price of the option.
    r (float): Risk-free interest rate.
    sigma (float): Volatility of the underlying asset.
    T (float): Time to expiration of the option.
    call_put (CallPut): Type of option (Call or Put).

    Returns:
    float: The price of the option.

    """
    if call_put == CallPut.CALL:
        return bs_call_price(S, K, r, sigma, T)
    else:
        return bs_put_price(S, K, r, sigma, T)


def bs_delta(S, K, r, sigma, T, call_put):
    """
    Calculates the delta of an option using the Black-Scholes model.

    Parameters:
    S (float): Current price of the underlying asset.
    K (float): Strike price of the option.
    r (float): Risk-free interest rate.
    sigma (float): Volatility of the underlying asset.
    T (float): Time to expiration of the option.
    call_put (CallPut): Type of option (Call or Put).

    Returns:
    float: The delta of the option.
    """
    if call_put == CallPut.CALL:
        return bs_call_delta(S, K, r, sigma, T)
    else:
        return bs_put_delta(S, K, r, sigma, T)
    

def bs_gamma(S, K, r, sigma, T):
    """
    Calculates the gamma of an option using the Black-Scholes formula.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
    T (float): The time to expiration of the option.

    Returns:
    float: The gamma of the option.
    """
    _d1 = d1(S, K, r, sigma, T)
    _gamma = norm.pdf(_d1) / (S * sigma * np.sqrt(T))
    return _gamma


def bs_vega(S, K, r, sigma, T):
    """
    Calculates the vega of an option using the Black-Scholes formula.
    
    Parameters:
    S (float): Current price of the underlying asset.
    K (float): Strike price of the option.
    r (float): Risk-free interest rate.
    sigma (float): Volatility of the underlying asset.
    T (float): Time to expiration of the option.
    
    Returns:
    float: The vega of the option.
    """
    _d1 = d1(S, K, r, sigma, T)
    _vega = S * np.sqrt(T) * norm.pdf(_d1) 
    return _vega


def bs_rho(S, K, r, sigma, T, call_put):
    _d2 = d2(S, K, r, sigma, T)
    _rho = K * T * np.exp(-r * T) * norm.cdf(_d2)

    if call_put == CallPut.PUT:
        _rho *= -1
        
    return _rho


def solve_r(S, K, T, sigma, call_put, v_market, r_init, tol=1e-6, max_iter=100):
    """
    Solves for the implied interest rate (r) using the Newton-Raphson method.

    Parameters:
    - S (float): Current price of the underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration of the option.
    - sigma (float): Volatility of the underlying asset.
    - call_put (str): Type of option, either 'call' or 'put'.
    - v_market (float): Market price of the option.
    - r_init (float): Initial guess for the interest rate.
    - tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
    - max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
    - float: The implied interest rate (r) that solves the Black-Scholes equation.

    Raises:
    - RuntimeError: If the Newton-Raphson method does not converge within the maximum number of iterations.
    """
    
    r = r_init
    for _ in range(max_iter):
        v_bs = bs_price(S, K, r, sigma, T, call_put)
        f = v_market - v_bs
        f_prime = -bs_rho(S, K, r, sigma, T, call_put)
        r_next = r - f / f_prime
        if all(abs(r_next - r) < tol):
            return r_next
        r = r_next

    raise RuntimeError("Newton-Raphson did not converge")


def payoff(S, K, call_put):
    """
    Calculates the payoff of an option

    Parameters:
    S (float): The price of the underlying asset.
    K (float): The strike price of the option.
    call_put (str): The type of option, either 'CALL' or 'PUT'.

    Returns:
    float: The payoff of the option.
    """
    if call_put == CallPut.CALL:
        return np.maximum(S - K, 0)
    else:
        return np.maximum(K - S, 0)
