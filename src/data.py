import pandas as pd
import pkg_resources
import functools

from typing import Optional

from datetime import datetime as dt

from collections import namedtuple

from src.black_scholes import CallPut

from logging import getLogger

LOG = getLogger(__name__)

MinMax = namedtuple('MinMax', ['min', 'max'])

CALL_PUT = 'CALL_PUT'
QUOTE_UNIXTIME = 'QUOTE_UNIXTIME'
QUOTE_READTIME = 'QUOTE_READTIME'
QUOTE_DATE= 'QUOTE_DATE'        
QUOTE_TIME_HOURS = 'QUOTE_TIME_HOURS'
UNDERLYING_LAST = 'UNDERLYING_LAST'
EXPIRE_DATE = 'EXPIRE_DATE'
EXPIRE_UNIX = 'EXPIRE_UNIX'
DTE = 'DTE'
DELTA = 'DELTA'
GAMMA = 'GAMMA'
VEGA = 'VEGA'
THETA = 'THETA'
RHO = 'RHO'
TAU = 'TAU'
IV = 'IV'
VOLUME = 'VOLUME'
LAST = 'LAST'
SIZE = 'SIZE'
BID = 'BID'
ASK = 'ASK'
MID = 'MID'
STRIKE = 'STRIKE'
VOLUME = 'VOLUME'
STRIKE_DISTANCE = 'STRIKE_DISTANCE'
STRIKE_DISTANCE_PCT = 'STRIKE_DISTANCE_PCT'
UNDERLYING = 'UNDERLYING'
DAYS_IN_YEAR = 365


OPTION_COLUMNS = [
    DTE,
    IV,
    DELTA,
    GAMMA,
    VEGA,
    THETA,
    RHO,
    BID,
    ASK,
]


COLUMNS = [
    QUOTE_DATE,
    EXPIRE_DATE,
    CALL_PUT,
    STRIKE,
    UNDERLYING,
 ] + OPTION_COLUMNS


MEMO = {}


def cache(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in MEMO:
            MEMO[key] = func(*args, **kwargs)
        return MEMO[key]

    return wrapper
    
    
def clear_cache():
    MEMO.clear()


@cache
def _spx_raw(
    call_put: CallPut,   # loading call and put separately due to low memory on PC
    start_date: dt = dt(2023, 1, 1), 
    end_date: dt = dt(2023, 12, 31), 
):
    if start_date < dt(2023, 1, 1) or end_date > dt(2023, 12, 31):
        raise ValueError("We only support data for 2023")

    LOG.info(f"Loading SPX data from {start_date} to {end_date}")

    start_month = start_date.month
    end_month = end_date.month

    files = [f"spx_eod_2023{month:02d}.txt" for month in range(start_month, end_month+1)]

    spx_data = []
    for file in files:
        df = pd.read_csv(pkg_resources.resource_filename(__name__, f'../data/SPX/{file}'), na_values=['', ' '])
        df.columns = df.columns.str.strip().str.replace(r'[\[\]()]', '', regex=True)
        df = _format_spx_data(df, call_put)
        spx_data.append(df)

    spx_df = pd.concat(spx_data)
    
    return spx_df


def _format_spx_data(spx_df, call_put):
    
    if call_put == CallPut.CALL:
        spx_df = spx_df.drop([c for c in spx_df.columns if c.startswith(f"P_")], axis=1)
    else:
        spx_df = spx_df.drop([c for c in spx_df.columns if c.startswith(f"C_")], axis=1)
    

    spx_df = spx_df.rename(columns={UNDERLYING_LAST: UNDERLYING})
    spx_df = spx_df.rename(columns={f"{call_put.value}_{c}": c for c in OPTION_COLUMNS})
    spx_df[CALL_PUT] = call_put.value

    spx_df[QUOTE_READTIME] = pd.to_datetime(spx_df[QUOTE_READTIME]).dt.date
    spx_df[QUOTE_DATE] = pd.to_datetime(spx_df[QUOTE_DATE]).dt.date
    spx_df[EXPIRE_DATE] = pd.to_datetime(spx_df[EXPIRE_DATE]).dt.date

    spx_df = spx_df[COLUMNS]

    spx_df = spx_df.set_index([EXPIRE_DATE , CALL_PUT , STRIKE])

    spx_df[IV] = spx_df[IV].astype('float')

    return spx_df


@cache
def _enrich_spx_data(call_put, start_date, end_date):
    """
    Load SPX data from start_date to end_date and process it.

    Parameters:
    - start_date (str): The start date of the data to load.
    - end_date (str): The end date of the data to load.

    Returns:
    - df (pandas.DataFrame): Processed SPX data.
    """
    spx_df = _spx_raw(call_put, start_date, end_date)
    spx_df[MID] = (spx_df[BID] + spx_df[ASK]) / 2
    spx_df[TAU] = spx_df[DTE] / DAYS_IN_YEAR
    return spx_df


def get_spx_data(
    call_put,
    start_date,
    end_date,
    iv: Optional[MinMax] = None,
    days_to_expiry: Optional[MinMax] = None,
    delta: Optional[MinMax] = None,
):
    df = _enrich_spx_data(call_put, start_date, end_date)
    
    # Apply filters
    if iv is not None:
        df = df[(df[IV] >= iv.min) & (df[IV] <= iv.max)]
    
    if days_to_expiry is not None:
        df = df[(df[DTE] >= days_to_expiry.min) & (df[DTE] <= days_to_expiry.max)]
    
    if delta is not None:
        df = df[(df[DELTA] >= delta.min) & (df[DELTA] <= delta.max)]

    return df


if __name__ == "__main__":
    """
    Example usage. 

    Put options with 15 to 45 days to expiry and a delta between -0.55 and -0.45

                                 QUOTE_DATE  UNDERLYING   DTE       IV    DELTA    GAMMA     VEGA    THETA      RHO   BID   ASK    MID       TAU
    EXPIRE_DATE CALL_PUT STRIKE                                                                                                                 
    2023-01-19  P        3845.0  2023-01-04     3853.39  15.0  0.21166 -0.46072  0.00239  3.17216 -2.02396 -0.74934  60.7  61.4  61.05  0.041096
                         3850.0  2023-01-04     3853.39  15.0  0.21216 -0.47251  0.00240  3.17996 -2.02719 -0.76630  63.2  63.8  63.50  0.041096
                         3855.0  2023-01-04     3853.39  15.0  0.21091 -0.48413  0.00239  3.18451 -2.01965 -0.78297  65.5  66.0  65.75  0.041096
                         3860.0  2023-01-04     3853.39  15.0  0.21012 -0.49643  0.00236  3.18622 -2.00730 -0.79984  67.6  68.3  67.95  0.041096
                         3865.0  2023-01-04     3853.39  15.0  0.21112 -0.50863  0.00242  3.18573 -2.00566 -0.81662  70.3  71.0  70.65  0.041096
    
    """
    start_date = dt(2023, 1, 1)
    end_date = dt(2023, 1, 31)

    spx_df = get_spx_data(
        CallPut.PUT,
        start_date,
        end_date,
        days_to_expiry=MinMax(15, 45),
        delta=MinMax(-0.55, -0.45),
    )