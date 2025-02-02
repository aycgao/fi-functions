"""
This module provides functions for specifically for dealing with spot/forward discount rates, spot/forward discount factors, and changing compounding frequencies
Functions:
- convert_discount_rate_to_discount_factor
- convert_discount_factor_to_discount_rate
- spot_forward_factors_conversion
- rate_factor_converter
- generate_discount_perfect_system
"""

from datetime import datetime
import pandas as pd
import numpy as np

def convert_discount_rate_to_discount_factor(discount_rate: float, ttm: float, compounding_freq):
    ''' 
    Given an annualized discount rate, a time to maturity, and a compounding frequency, convert the discount rate into a discount factor.

    Params:
        discount_rate (float): the annualized discount rate (as a decimal).
        ttm (float): time to maturity in years.
        compounding_freq (int or str): the number of compounding periods in a year. "continuous" if continuous compounding. 
    '''
    if compounding_freq=='continuous':
        return np.exp(-discount_rate * ttm)
    else:
        return (1 + discount_rate/compounding_freq)^(-compounding_freq * ttm)

def convert_discount_factor_to_discount_rate(discount_factor: float, ttm: float, compounding_freq):
    ''' Given a discount factor, a time to maturity in years, and a compounding frequency, convert the discount factor to an annualized discount rate.

        Params:
            discount_factor (float): the number f such that cash flow * f = present value.
            ttm (float): time to maturity in years.
            compounding_freq (int or str): the number of compounding periods in a year. "continuous" if continuous compounding. 
        
    '''
    if compounding_freq=='continuous':
        return np.log(1/discount_factor)/ttm
    else:
        return ((1/discount_factor)**(1/compounding_freq/ttm) - 1) * compounding_freq

def spot_forward_factors_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts spot discount factors to forward discount factors

    Parameters:
        df (pd.DataFrame): a dataframe with discount factors and time increments associated with the factors

    Returns:
        pd.DataFrame: A DataFrame with the same time increments given, but with forward discount factors added
    """
    df['Forward Discount Factors'] = df['Discount Factors'] / df['Discount Factors'].shift(1)
    df.iloc[0, df.columns.get_loc('Forward Discount Factors')] = df.iloc[0, df.columns.get_loc('Discount Factors')]
    return df

def rate_factor_converter(input: str, input_df: pd.DataFrame, compounding_freq: int = 2, forward_time_length: int = 0.5) -> pd.DataFrame:
    ''' 
    Given any of discount rates, discount factors, forward rates, forward factors, can produce information on the other three in a dataframe

    Params:
        input (str): string of which input metrics is given (DR: discount rate, DF: discount factor, FR: forward rate, FF: forward factor)
        input_df (pd.DataFrame): a dataframe with any of the four metrics listed, and the times at which this is occurring (names of columns must be 'Metric', 'Time')
        compounding_freq (int): the number of compounding periods in a year. "continuous" if continuous compounding. 
        forward_time_lenght (int): lenght of time for forward rate (T2 - T1)
    
    Returns:
        pd.DataFarme: A dataframe with all four metrics returned
    '''
    output = input_df.copy()

    if input == 'DR':
        output = output.rename(columns = {'Metric': 'Discount Rates'})
        output['Discount Factors'] = convert_discount_rate_to_discount_factor(output['Discount Rates'], output['Time'], compounding_freq)
        w = spot_forward_factors_conversion(output)
        output['Forward Discount Factors'] = w['Forward Discount Factors']
        output['Forward Discount Rates'] = convert_discount_factor_to_discount_rate(output['Forward Discount Factors'], forward_time_length, compounding_freq)
        return output
    
    if input == 'DF':
        output = output.rename(columns = {'Metric': 'Discount Factors'})
        output['Discount Rates'] = convert_discount_factor_to_discount_rate(output['Discount Factors'], output['Time'], compounding_freq)
        w = spot_forward_factors_conversion(output)
        output['Forward Discount Factors'] = w['Forward Discount Factors']
        output['Forward Discount Rates'] = convert_discount_factor_to_discount_rate(output['Forward Discount Factors'], forward_time_length, compounding_freq)
        return output

    if input == 'FR':
        output = output.rename(columns = {'Metric': 'Forward Discount Rates'})
        output['Forward Discount Factors'] = convert_discount_rate_to_discount_factor(output['Forward Discount Rates'], forward_time_length, compounding_freq)
        output['Discount Factors'] = output['Forward Discount Factors'].cumprod()
        output['Discount Rates'] = convert_discount_factor_to_discount_rate(output['Discount Factors'], output['Time'], compounding_freq)
        return output

    if input == 'FF':
        output = output.rename(columns = {'Metric': 'Forward Discount Factors'})
        output['Discount Factors'] = output['Forward Discount Factors'].cumprod()
        output['Forward Discount Rates'] = convert_discount_factor_to_discount_rate(output['Forward Discount Factors'], forward_time_length, compounding_freq)
        output['Discount Rates'] = convert_discount_factor_to_discount_rate(output['Discount Factors'], output['Time'], compounding_freq)
        return output

def generate_discount_perfect_system(cash_flow_matrix: pd.DataFrame, prices: pd.DataFrame, freq: int = 2) -> pd.DataFrame:
    """
    Generates discount factors/rates with TTM given cash flow matrix, prices (only for perfect system)

    Parameters:
        cash_flow_matrix (pd.DataFrame): A DataFrame where rows are indexed by bond IDs,
                                         columns are cash flow dates, and values are cash flows.
        prices (pd.DataFrame): dataframe of the prices of the bonds
        freq (int): how often we want to get the discount factor (2 would be semiannual)
        

    Returns:
        pd.DataFrame: A DataFrame where the discount factors, rates , ttm are listed
    """
    matrix = cash_flow_matrix.values
    inverse_matrix = np.linalg.inv(matrix)
    z = inverse_matrix @ prices
    z.rename(columns={z.columns[0]: 'Spot Discount Factor'}, inplace = True)
    z.index = cash_flow_matrix.columns
    z['T - t'] = np.arange(1/freq, 1/freq * (len(z) + 1), 1/freq)
    z['Spot Discount Rates'] = convert_discount_factor_to_discount_rate(z['Spot Discount Factor'], z['T - t'], 2)

    return z