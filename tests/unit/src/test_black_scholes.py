import pytest
import numpy as np

from src.black_scholes import d1, d2, bs_call_price, bs_put_price, bs_call_delta, bs_put_delta, bs_price, bs_delta, bs_gamma, bs_vega, CallPut


@pytest.mark.parametrize("S, K, r, sigma, T, expected_d1", [
    (100, 100, 0.05, 0.2, 1, 0.35),
])
def test_d1(S, K, r, sigma, T, expected_d1):
    _d1 = d1(S, K, r, sigma, T)
    assert np.isclose(_d1, expected_d1)


@pytest.mark.parametrize("S, K, r, sigma, T, expected_d2", [
    (100, 100, 0.05, 0.2, 1, 0.15),
])
def test_d2(S, K, r, sigma, T, expected_d2):
    _d2 = d2(S, K, r, sigma, T) 
    assert np.isclose(_d2, expected_d2)


@pytest.mark.parametrize("S, K, r, sigma, T, expected_call_price", [
    (100, 100, 0.05, 0.2, 1, 10.450583572185565),
])
def test_call_price(S, K, r, sigma, T, expected_call_price):
    prc = bs_call_price(S, K, r, sigma, T)
    assert np.isclose(prc, expected_call_price)


@pytest.mark.parametrize("S, K, r, sigma, T, expected_put_price", [
    (100, 100, 0.05, 0.2, 1, 5.573526022256971),
])
def test_put_price(S, K, r, sigma, T, expected_put_price):
    prc = bs_put_price(S, K, r, sigma, T)
    assert np.isclose(prc, expected_put_price)


@pytest.mark.parametrize("S, K, r, sigma, T, call_put, expected_price", [
    (100, 100, 0.05, 0.2, 1, CallPut.CALL, 10.450583572185565),
    (100, 100, 0.05, 0.2, 1, CallPut.PUT, 5.573526022256971),
])
def test_price(S, K, r, sigma, T, call_put, expected_price):
    prc = bs_price(S, K, r, sigma, T, call_put)
    assert np.isclose(prc, expected_price)


@pytest.mark.parametrize("S, K, r, sigma, T, expected_call_delta", [
    (100, 100, 0.05, 0.2, 1, 0.6368306511756191),
])
def test_call_delta(S, K, r, sigma, T, expected_call_delta):
    _delta = bs_call_delta(S, K, r, sigma, T) 
    assert np.isclose(_delta, expected_call_delta)


@pytest.mark.parametrize("S, K, r, sigma, T, expected_put_delta", [
    (100, 100, 0.05, 0.2, 1, -0.3631693488243809),
])
def test_put_delta(S, K, r, sigma, T, expected_put_delta):
    _delta = bs_put_delta(S, K, r, sigma, T)
    assert np.isclose(_delta, expected_put_delta)


@pytest.mark.parametrize("S, K, r, sigma, T, call_put, expected_delta", [
    (100, 100, 0.05, 0.2, 1, CallPut.CALL, 0.6368306511756191),
    (100, 100, 0.05, 0.2, 1, CallPut.PUT, -0.3631693488243809),
])
def test_delta(S, K, r, sigma, T, call_put, expected_delta):
    _delta = bs_delta(S, K, r, sigma, T, call_put)
    assert np.isclose(_delta, expected_delta)


@pytest.mark.parametrize("S, K, r, sigma, T, expected_gamma", [
    (100, 100, 0.05, 0.2, 1, 0.018762017345846895),
])
def test_gamma(S, K, r, sigma, T, expected_gamma):
    _gamma = bs_gamma(S, K, r, sigma, T)
    assert np.isclose(_gamma, expected_gamma)


@pytest.mark.parametrize("S, K, r, sigma, T, expected_vega", [
    (100, 100, 0.05, 0.2, 1, 37.52403469169379),
])
def test_vega(S, K, r, sigma, T, expected_vega):
    _vega = bs_vega(S, K, r, sigma, T)
    assert np.isclose(_vega, expected_vega)

