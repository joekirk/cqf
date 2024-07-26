import numpy as np
import pandas as pd

from datetime import datetime as dt, timedelta
from sklearn.linear_model import LinearRegression

from src.black_scholes import CallPut
from src.data import get_spx_data, IV, TAU, UNDERLYING, STRIKE, MinMax, EXPIRE_DATE, CALL_PUT, DELTA, VEGA, MID, QUOTE_DATE


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
ERROR_BS = 'ERROR_BS'
ERROR_MV = 'ERROR_MV'


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


def delta_mvd(a, b, c, s0, delta_bs, vega, tau):
    return delta_bs + (vega / (s0 * np.sqrt(tau)) * (a + (b * delta_bs) + (c * np.square(delta_bs)))) 


def expected_iv(a, b, c, delta_s, delta, tau):
    _eiv = ((a + (b * delta) + (c * np.square(delta))) / np.sqrt(tau)) * delta_s
    return _eiv


def error_mv(a, b, c, delta_bs, vega, tau, s0, s1, f0, f1):
    _eiv = _y(f0, f1, s0, s1, delta_bs) - _y_hat(a, b, c, delta_bs, vega, tau, s0, s1)
    return _eiv


def error_bs(f0, f1, s0, s1, delta_bs):
    _ebs = _y(f0, f1, s0, s1, delta_bs)
    return _ebs


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
        self.X = _make_X(self._data)
        self.y = _make_y(self._data)

    @staticmethod
    def _prepare_data(spx_df):
        spx_df = spx_df.reset_index()
        
        spx_df = spx_df.rename(columns={MID: OPTION_PRICE})
        spx_df[OPTION_PRICE_1] = spx_df.groupby([EXPIRE_DATE, STRIKE, CALL_PUT])[OPTION_PRICE].shift(-1)
        spx_df[UNDERLYING_1] = spx_df.groupby([EXPIRE_DATE, STRIKE, CALL_PUT])[UNDERLYING].shift(-1)
        spx_df[IV_1] = spx_df.groupby([EXPIRE_DATE, STRIKE, CALL_PUT])[IV].shift(-1)
        
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
            s0=df[UNDERLYING],
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

        df[ERROR_BS] =  error_bs(
            f0=df[OPTION_PRICE],
            f1=df[OPTION_PRICE_1],
            s0=df[UNDERLYING],
            s1=df[UNDERLYING_1],
            delta_bs=df[DELTA]
        )

        df[ERROR_MV] = error_mv(
            a=df['a'],
            b=df['b'],
            c=df['c'],
            delta_bs=df[DELTA],
            vega=df[VEGA],
            tau=df[TAU],
            s0=df[UNDERLYING],
            s1=df[UNDERLYING_1],
            f0=df[OPTION_PRICE],
            f1=df[OPTION_PRICE_1]
        )

        if columns is not None:
            df = df[key + columns]

        return df


    def fit(self, rolling_window_size=None, fit_intercept=True):
        X = self.X
        y = self.y

        coefficients = []

        train_dt = X.index.unique()

        if rolling_window_size is None:
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X, y)
            coefficients.append(
                (
                    train_dt[0],
                    model.coef_[0],
                    model.coef_[1],
                    model.coef_[2],
                    model.intercept_
                )
            )
        else:
            for _dt in train_dt:

                if _dt + timedelta(days=rolling_window_size) > train_dt[-1]:
                    # terminate early and forward fill
                    break

                start = _dt
                end = _dt + timedelta(days=rolling_window_size)
                model = LinearRegression(fit_intercept=fit_intercept)
                model.fit(X[start:end], y[start:end])
                coefficients.append(
                    (
                        _dt,
                        model.coef_[0],
                        model.coef_[1],
                        model.coef_[2],
                        model.intercept_
                    )
                )

        df = pd.DataFrame(
            coefficients, 
            columns=['tstamp', 'a', 'b', 'c', 'eps']
        ).set_index('tstamp')

        df = df.reindex(train_dt, method='ffill')
        return df


if __name__ == "__main__":
    # Short period of example data
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

    # Fit the model
    coefficients = mvd.fit(rolling_window_size=None)

    # Generate results dataframe for analysis
    results = mvd.results_df(coefficients)

    print(results)