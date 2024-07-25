import numpy as np
import pandas as pd

from datetime import datetime as dt, timedelta

from sklearn.linear_model import LinearRegression

from src.data import get_spx_data, IV, TAU, UNDERLYING, STRIKE, MinMax, CallPut, EXPIRE_DATE, CALL_PUT, DELTA, VEGA, MID, QUOTE_DATE
from src.black_scholes import  bs_call_price, bs_call_delta, bs_vega, CallPut

from src.monte_carlo import GBMMonteCarloSimulation, VarianceReduction, NumericalScheme

FACTOR = 'FACTOR'
UNDERLYING_1 = 'UNDERLYING_1'
OPTION_PRICE = 'OPTION_PRICE'
OPTION_PRICE_1 = 'OPTION_PRICE_1'
F = 'F'
S = 'S'
IV_1 = 'IV_1'
MVD_DELTA = 'MVD_DELTA'
EIV = 'EIV'
MVDBS = 'MVD-BS'
Y_HAT = 'Y_HAT'
Y='Y'


def _make_X(data):

    X = pd.DataFrame({
        'x1': data[FACTOR].values,
        'x2': (data[DELTA] * data[FACTOR]).values,
        'x3': (np.square(data[DELTA]) * data[FACTOR]).values
    }, index=data[QUOTE_DATE].values)

    return X


def _make_y(data):

    dependent = _y(
        data[OPTION_PRICE], 
        data[OPTION_PRICE_1], 
        data[UNDERLYING], 
        data[UNDERLYING_1], 
        data[DELTA]
    )

    y = pd.Series(
        data=dependent.values,
        index=data[QUOTE_DATE].values) 

    return y
    

def _y(f0, f1, s0, s1, delta_bs):
    return (f1 - f0) - (delta_bs * (s1 - s0))


def _y_hat(a, b, c, delta_bs, vega, tau, s0, s1):
    return _factor(vega, tau, s0, s1) * (a + (b * delta_bs) + (c * np.square(delta_bs)))
    
    
def _factor(vega, tau, s0, s1):
    return vega * (s1 - s0) / (s0 * np.sqrt(tau))


def delta_mvd(a, b, c, S, delta_bs, vega, tau):
    return delta_bs + (vega / (S * np.sqrt(tau)) * (a + (b * delta_bs) + (c * np.square(delta_bs))))


def expected_iv(a, b, c, delta_s, delta, tau):
    _eiv = ((a + (b * delta) + (c * np.square(delta))) / np.sqrt(tau)) * delta_s
    return _eiv


def error_mv(a, b, c, delta_bs, vega, tau, s0, s1, f0, f1):
    _eiv = _y(f0, f1, s0, s1, delta_bs) - _y_hat(a, b, c, delta_bs, vega, tau, s0, s1)
    return _eiv


def error_bs(f0, f1, s0, s1, delta_bs):
    _ebs = _y(f0, f1, s0, s1, delta_bs)


class MinimumVarianceDelta:

    def __init__(
            self, 
            min_delta, 
            max_delta, 
            min_dte, 
            max_dte, 
            call_put, 
            start_date, 
            end_date,
        ):

        spx_df = get_spx_data(
            call_put,
            start_date,
            end_date,
            days_to_expiry=MinMax(min_dte, max_dte),
            delta=MinMax(min_delta, max_delta),
        )

        self._data = self._prepare_data(spx_df)
        # self._data = self.get_synthetic_data()
        self.X = _make_X(self._data)
        self.y = _make_y(self._data)

    
    def get_synthetic_data(self):

        # start with just a call option for now with fixed expiry bucket
        # delta bucket between 0.35 and 0.45
        # this is just a test to try to understand the data
        
        sim = GBMMonteCarloSimulation(
            S0 = 4000,
            r = 0.05,
            sigma = 0.2,
            T = 1,
            k = 8,
            n_sims=1
        )

        # Generate the paths
        path = sim.run_sim(
            variance_reduction=VarianceReduction.SobolBrownianBridge, 
            numerical_scheme=NumericalScheme.Euler
        )[0]

        # lets start really simple and assume the spot just walks town


        tau = 1 / 12
        r = 0.05
        K = [x for x in range(3800, 4000, 5)]
        options = []
        underlying_price = 4000
        for day, s in enumerate(path):
            underlying_price *= 0.999
            for strike in K:
                iv = np.random.uniform(0, 0.25)
                price = bs_call_price(s, strike, r, iv, tau)
                delta = bs_call_delta(s, strike, r, iv, tau)
                vega = bs_vega(s, strike, r, iv, tau)
                options.append(
                    (
                        day,
                        tau,
                        underlying_price,
                        delta,
                        vega,
                        price,
                        strike,
                        'C',
                        iv
                    )
                )

        df = pd.DataFrame(
            options,
            columns=[QUOTE_DATE, TAU, UNDERLYING, DELTA, VEGA, OPTION_PRICE, STRIKE, CALL_PUT, IV]
        )
        df[OPTION_PRICE_1] = df.groupby([STRIKE, CALL_PUT])[OPTION_PRICE].shift(-1)
        df[UNDERLYING_1] = df.groupby([STRIKE, CALL_PUT])[UNDERLYING].shift(-1)
        df[IV_1] = df.groupby([STRIKE, CALL_PUT])[IV].shift(-1)
        df = df.dropna()

        df[F] = df[OPTION_PRICE_1] - df[OPTION_PRICE]
        df[S] = df[UNDERLYING_1] - df[UNDERLYING]

        # need to filter by delta
        df[FACTOR] = _factor(
            df[VEGA], 
            df[TAU], 
            df[UNDERLYING], 
            df[UNDERLYING_1]
        )

        df = df[(df.DELTA < 0.45) & (df.DELTA > 0.35)].copy()
        return df


    @staticmethod
    def _prepare_data(spx_df):
        spx_df = spx_df.reset_index()
        
        original = spx_df.copy()

        spx_df = spx_df.rename(columns={MID: OPTION_PRICE})
        spx_df[OPTION_PRICE_1] = spx_df.groupby([EXPIRE_DATE, STRIKE, CALL_PUT])[OPTION_PRICE].shift(-1)
        spx_df[UNDERLYING_1] = spx_df.groupby([EXPIRE_DATE, STRIKE, CALL_PUT])[UNDERLYING].shift(-1)
        spx_df[IV_1] = spx_df.groupby([EXPIRE_DATE, STRIKE, CALL_PUT])[IV].shift(-1)
        
        import pdb; pdb.set_trace()
        
        spx_df = spx_df.dropna()

        data = spx_df[[
            EXPIRE_DATE, 
            STRIKE, 
            CALL_PUT, 
            DELTA,
            VEGA,
            UNDERLYING, 
            UNDERLYING_1, 
            TAU, 
            OPTION_PRICE, 
            OPTION_PRICE_1, 
            IV, 
            IV_1, 
            QUOTE_DATE
        ]].copy()

        # multiply out VEGA by factor 100 
        data[VEGA] = data[VEGA] * 100

        data[F] = data[OPTION_PRICE_1] - data[OPTION_PRICE]
        data[S] = data[UNDERLYING_1] - data[UNDERLYING]

        data[FACTOR] = _factor(
            data[VEGA], 
            data[TAU], 
            data[UNDERLYING], 
            data[UNDERLYING_1]
        )

        return data

    def results_df(self, coefficients, columns=None):

        key = [EXPIRE_DATE, STRIKE, CALL_PUT]

        df = pd.merge(self._data, coefficients, left_on=QUOTE_DATE, right_index=True, how='left')

        df[MVD_DELTA] = delta_mvd(
            a=df['a'],
            b=df['b'],
            c=df['c'],
            S=df[UNDERLYING],
            delta_bs=df[DELTA],
            vega=df[VEGA],
            tau=df[TAU]
        )
        df[Y] = _y(
            df[OPTION_PRICE], 
            df[OPTION_PRICE_1], 
            df[UNDERLYING], 
            df[UNDERLYING_1], 
            df[DELTA]
        )
        df[Y_HAT] = _y_hat(
            a=df['a'],
            b=df['b'],
            c=df['c'],
            s0=df[UNDERLYING],
            s1=df[UNDERLYING_1],
            delta_bs=df[DELTA],
            vega=df[VEGA],
            tau=df[TAU]
        )

        df[EIV] = expected_iv(
            a=df['a'],
            b=df['b'],
            c=df['c'],
            delta_s=(df[UNDERLYING_1] - df[UNDERLYING]) / df[UNDERLYING],
            delta=df[DELTA],
            tau=df[TAU]
        )
        
        df[MVDBS] = df[MVD_DELTA] - df[DELTA]

        if columns is not None:
            df = df[key + columns]

        return df


    def fit(self, rolling_window_size=None):
        X = self.X
        y = self.y

        coefficients = []

        train_dt = X.index.unique()

        if rolling_window_size is None:
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            coefficients.append(
                (
                    train_dt[0],
                    model.coef_[0],
                    model.coef_[1],
                    model.coef_[2]
                )
            )
        else:
            for _dt in train_dt:

                if _dt + timedelta(days=rolling_window_size) > train_dt[-1]:
                    # terminate early and forward fill
                    break

                start = _dt
                end = _dt + timedelta(days=rolling_window_size)
                model = LinearRegression()
                model.fit(X[start:end], y[start:end])
                coefficients.append(
                    (
                        _dt,
                        model.coef_[0],
                        model.coef_[1],
                        model.coef_[2]
                    )
                )

        df = pd.DataFrame(
            coefficients, 
            columns=['tstamp', 'a', 'b', 'c']
        ).set_index('tstamp')

        df = df.reindex(train_dt, method='ffill')
        return df


if __name__ == "__main__":
    start_date = dt(2023, 1, 1)
    end_date = dt(2023, 3, 31)

    mvd = MinimumVarianceDelta(
        min_delta=0.25, 
        max_delta=0.35, 
        min_dte=15, 
        max_dte=45, 
        call_put=CallPut.CALL, 
        start_date=start_date, 
        end_date=end_date
    )
    coefficients = mvd.fit(rolling_window_size=None)
    results = mvd.results_df(coefficients)
    print(results)