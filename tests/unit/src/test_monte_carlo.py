import numpy as np
import pytest

from src.monte_carlo import GBMMonteCarloSimulation, VarianceReduction, NumericalScheme, BrownianBridge


# @pytest.mark.parametrize('variance_reduction, expected_path', [
#     (VarianceReduction.Antithetic, [100, 95.13243586, 91.29696478, 82.64229899]),
#     (VarianceReduction.BrownianBridge, [100, 105.98681962, 106.81628475, 123.86061682]),
#     (VarianceReduction.Sobol, [100, 94.39324039,  90.52358801,  98.93324721]),
#     (VarianceReduction.SobolBrownianBridge, [100, 106.68325711,  94.04083003,  82.98163739]),
#     (None, [100, 107.08805744,  68.96526939,  66.25697902]),
# ]) 
# def test_simulate_path(variance_reduction, expected_path):
#     np.random.default_rng(1)
#     np.random.seed(1)
#     mc =  GBMMonteCarloSimulation(S0=100, r=0.05, sigma=0.2, T=1, k=2, n_sims=1)
#     path = mc.run_sim(variance_reduction, NumericalScheme.Euler)
#     assert np.allclose(path, expected_path)

def test_brownian_bridge():


    """
    


    +-----------------+-----------------+-----------------+-----------------+
    0                 1                 2                 3                 4                  
    """

    # N is the number of steps
    bb = BrownianBridge(N=5, n_sims=1)
    dZ = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

    bridge = bb.build_paths(dZ)

    import pdb; pdb.set_trace()

    assert np.allclose(bridge, [0.1, 0.2, 0.3, 0.4, 0.5])
    import pdb; pdb.set_trace()

    pass