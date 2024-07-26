# CQF Assignment - Optimal Hedging with Advanced Delta Modelling

The implementation for all models are found in the src. 

 - black_scholes.py 
     Contains Black-Scholes utility functions (e.g call option price, black scholes delta etc.)

 - data.py
     Loads OptionsDX EOD SPX option data for 2023 

 - minimum_variance_delta.py
     Implements MVD regression

 - monte_carlo.py
     Implements Monte Carlo simulation for pricing options including a number of variance reduction techniques including Brownian Bridge

 - replication.py
     Implements a portfolio replication model using actual or implied volatility and accompanying PNL analysis

Most files (where relevant) contain an example of how to run the model

Analysis for the accompanying report is conducted in the notebooks folder.  For the most part we use the notebooks to generate plots.

### How to run the code

The code has been developed against python 3.8

```
python3 -m venv .venv
source .venv/bin/activate
python setup.py develop
```

To understand a model you can run and set breakpoints accordingly

```
python src/monte_carlo.py
python src/data.py
python src/black_scholes.py
python src/replication.py
python src/minimum_variance_delta.py
```

To understand the experiment and the results the following notebooks are runnable

 - notebooks/assignment/actual_implied_hedging.ipynb
 - notebooks/assignment/monte_carlo.ipynb
 - notebooks/assignment/mvd.ipynb

Unit tests have been implemented for the black scholes module

```
py.test tests/unit/src/test_black_scholes.py
```
