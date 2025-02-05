"""
This module provides functions for analysis of fixed income assets.
Functions:
- get_coupon_dates: gets dates of coupon payments of bond
- price_bond: prices a bond (dirty price) using TTM approximation
- closed_form_duration: gets duration using a closed form formula
- ddur_hedge_ratio: gets hedge ratio between two assets based on dollar duration
- build_factors: builds level, slope, and curvature yield-curve factors from time series yields (1, 2, 5, 7, 10, 20, 30)
- mvts_regression: time series regression that reports alpha, beta, R^2
- convert_to_ttm: converts dates to ttm
"""

from datetime import datetime
from pandas.core.series import Series
import pandas as pd
import numpy as np
from typing import Any, Tuple
import statsmodels.api as sm

def get_coupon_dates(issue_date: str, maturity_date: str, freq: int) -> Series:
    """
    Gets dates of coupon payments of bond

    Args:
        issue_date (str): string for issue date. "2025-01-26"  # Example date string
        maturity_date (str): string for date of maturity. "2025-01-26"  # Example date string
        freq (int): how many times a year the bond pays coupons.

    Returns:
        Series: series of dates
    """
    issue_date = datetime.strptime(issue_date, "%Y-%m-%d").date()
    maturity_date = datetime.strptime(maturity_date, "%Y-%m-%d").date()

    freq = pd.DateOffset(months=int(12/freq))

    dates = pd.date_range(start = issue_date, end = maturity_date, freq = freq)
    dates = pd.to_datetime(dates)
    dates = pd.DataFrame(data=dates[dates > pd.to_datetime(issue_date)])
    
    return dates[0]


def price_bond(ytm: float, quote_date: str, maturity_date: str, coupon_rate: float, freq: int, fv: float = 100, exact: bool = False, return_dirty: bool = True) -> float:
    """
    Prices a bond (with or without coupon) given a YTM, quote date, maturity date, and coupon payment frequency, returning either dirty or clean price

    Args:
        ytm (float): yield to maturity in percentage form.
        quote_date (str): string for quote date. "2025-01-26"  # Example date string
        maturity_date (str): string for date of maturity. "2025-01-26"  # Example date string
        coupon_rate (float): percentage rate for coupon
        freq (int): how many times a year the bond pays coupons.
        fv (float): face value of bond (default 100)
        return_dirty (bool): dirty or clean price (default dirty, true)

    Returns:
        float: price of bond
    """
    quote_date = datetime.strptime(quote_date, "%Y-%m-%d").date()
    maturity_date = datetime.strptime(maturity_date, "%Y-%m-%d").date()
    coupon_rate = (coupon_rate/freq)/100
    ytm = (ytm/freq)/100
    ttm = abs((maturity_date - quote_date).days)/365.25

    # Here, we're checking time till maturity
    # If we are exactly on a issue date or after payment of coupon, no need to approximate
    if exact:
        tau_left = 0
        step = 1 / freq
        ttm = round(ttm / step) * step
        tau = round(freq * ttm)
    else:
        decimal_portion = ttm % 1
        # If the remainder is less than a day, round it out
        if (1 - decimal_portion) * 365.25 < 1 or decimal_portion * 365.25 < 1:
            ttm = round(ttm)
            
        # Getting remaining value to make a dirty price
        tau_left = freq * (ttm % (1/freq))
        tau = round(freq * ttm - tau_left)
    
    
    pv = 0
    for i in range(1,tau):
        pv += (coupon_rate) / (1+ytm)**i

    pv += (1+coupon_rate)/(1+ytm)**tau
    pv *= fv
    if tau_left > 0:
        pv += (coupon_rate * fv) 
        pv /= (1 + ytm) ** tau_left
    
    if not return_dirty:
        accrued_interest = coupon_rate * fv * tau_left  # Accrued Interest Calculation
        pv = pv - accrued_interest

    return pv

def closed_form_duration(freq: int, ytm: float, coupon_rate: float, time_till_mat: float) -> float:
    """
    Provides the duration of a fixed rate bond taking in frequency of coupon payment, ytm, coupon rate, and time till maturity

    Args:
        freq (int): how many times a year the bond pays coupons.
        ytm (float): yield to maturity in percentage form.
        coupon_rate (float): percentage rate for coupon
        time_till_mat (float): time till maturity of the bond in years

    Returns:
        float: duration of the bond (Macaulay Duration)
    """
    y_t = (ytm/100) / freq
    c_t = (coupon_rate/100) / freq
    t_t = freq * time_till_mat

    duration = 1 / freq * ((1 + y_t) / y_t - (1 + y_t + t_t * (c_t - y_t)) / (c_t * ((1 + y_t) ** t_t - 1) + y_t))
    return duration

def ddur_hedge_ratio(n_i: float, ddur_i: float, ddur_j: float) -> float:
    """
    Calculates the quantity of asset j required to hedge duration of asset i.

    Args:
        n_i (float): number of contracts for asset i
        ddur_i (float): dollar duration of asset i
        ddur_j (float): dollar duration of asset j

    Returns:
        float: number of contracts needed in asset j to perfectly hedge dollar duration
    """
    return - n_i * (ddur_i/ddur_j)

def build_factors(yields: pd.DataFrame, keep_columns: list[str] = ['Level Factor', 'Slope Factor', 'Curvature Factor']) -> pd.DataFrame:
    """
    Provides common yield curve factors in time series (value for factors for each time t) where:
    - level is an average of the yields
    - slope is the 30 year yield minus the 1 year yield
    - curvature is (- 1 year yield + 2 * 10 year yield - 30 year yield)
    
    Args:
        yields (pd.DataFrame): pandas dataframe of yields in time series (1, 2, 5, 7, 10, 20, 30)
        keep_columns (list[str]): indicates which factors to keep

    Returns:
        pd.DataFrame: time series of factors chosen
    """
    y = yields.copy()
    y['Level Factor'] = 1/7 * y.sum(axis = 1)
    y['Slope Factor'] = y.iloc[:, 6] - y.iloc[:, 0]
    y['Curvature Factor'] = - y.iloc[:, 0] + 2 * y.iloc[:, 4] - y.iloc[:, 6]
    return y[keep_columns]

def mvts_regression(y_df: pd.DataFrame, x_df: pd.DataFrame, report: list[str] = ['Coef', 'R_squared'], const: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs multivariate regression
    
    Args:
        y_df (pd.DataFrame): pandas dataframe of single column of y values
        x_df (pd.DataFrame): pandas dataframe of single or multiple columns of x values
        report (list[str]): list of things to report (coef, R^2)
        const (bool): to include constant or not

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: results of regression, a coef df and metrics df
    """
    # Make sure dates/indices are merged
    y_df, x_df = y_df.align(x_df, axis=0)

    # Add const if necessary
    y = y_df
    if const:
        X = sm.add_constant(x_df)
    else:
        X = x_df
    
    # Model
    model = sm.OLS(y, X).fit()

    # Add coefficients
    if 'Coef' in report:
        coefficients = model.params
        coef_df = pd.DataFrame(coefficients, columns=['Coefficient'])
    else:
        coef_df = pd.DataFrame(columns=['Coefficient'])
    
    # Add metrics
    metrics_df = pd.DataFrame(columns=['Values'])

    if 'R_squared' in report:
        metrics_df.loc['R-squared'] = model.rsquared

    return coef_df, metrics_df

def convert_to_ttm(maturity_dates: pd.Series, present_date: str) -> pd.Series:
    """
    Converts maturity dates into time to maturity (in years) given a present date.

    Args:
        maturity_dates (pd.Series): maturity dates 
        present_date (str): current date as a string (format: YYYY-MM-DD) or a datetime

    Returns:
        pd.Series: Time to maturity in years.
    """
    maturity_dates = pd.to_datetime(maturity_dates)
    present_date = pd.to_datetime(present_date)

    ttm = (maturity_dates - present_date).dt.days / 365.25
    return ttm


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
These are functions that are still in the work/not tested
- price_bond_date_ver (not tested): prices a bond (dirty price) using dates of cashflows
"""

def price_bond_date_ver(ytm: float, issue_date: str, quote_date: str, maturity_date: str, coupon_rate: float, freq: int, fv: float = 100) -> float:
    """
    Prices a bond (with or without coupon) given a YTM, quote date, maturity date, and coupon payment frequency

    Args:
        ytm (float): yield to maturity in percentage form.
        issue_date (str): string for issue date. "2025-01-26"  # Example date string
        quote_date (str): string for quote date. "2025-01-26"  # Example date string
        maturity_date (str): string for date of maturity. "2025-01-26"  # Example date string
        coupon_rate (float): percentage rate for coupon
        freq (int): how many times a year the bond pays coupons.
        fv (float): face value of bond (default 100)

    Returns:
        float: price of bond
    """
    if issue_date == quote_date:
        date_series = get_coupon_dates(quote_date, maturity_date, freq)
    else:
        date_series = get_coupon_dates(issue_date, maturity_date, freq)
        date_series = date_series[date_series >= pd.to_datetime(quote_date)]

    # Getting tuples of time in years till each cash flow and the value of the cashflow
    # Approximating 365.25 days per year
    quote_date = pd.Timestamp(datetime.strptime(quote_date, "%Y-%m-%d").date())
    cashflows = [[abs((date - quote_date).days)/365.25, (coupon_rate/freq)/100 * fv] for date in date_series]
    cashflows[-1][1] = cashflows[-1][1] + fv

    price = 0

    for cf in cashflows:
        pv = cf[1] / ((1 + (ytm/freq)/100) ** (2*cf[0]))
        price += pv

    return price
