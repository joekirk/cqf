# CQF Assignment - Optimal Hedging with Advanced Delta Modelling

The implementation for all models are found in the src. 

 - black_scholes.py 
     Contains Black-Scholes utility functions  (e.g call option price)

 - data.py
     Loads OptionsDX EOD SPX option data for 2023 

 - minimum_variance_delta.py
     Implements a MVD regression

 - monte_carlo.py
     Implements monte carlo simulation for pricing options including a number of variance reduction techniques including brownian bridge

 - replication.py
     Implements a portfolio replication model using actual or implied volatility and accompanying PNL analysis

Most files (where relevant) contain an example of how to run the model

Analysis for the accompanying report is conducted in the notebooks folder.  For the most part we use the notebooks to generate plots.

### How to run the code

The code has been developed against python 3.8

```
python -m venv myenv
source myenv/bin/activate
python setup.py develop
```

To understand a model you can run it using. Otherwise recommend running through the notebooks.

```
python src/monte_carlo.py
```
