from datetime import datetime
from pandas.core.series import Series
import pandas as pd
import numpy as np
from typing import Dict, Union
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ====================================
# Section: Rate Factor Conversions
# ====================================

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
        return (1 + discount_rate/compounding_freq)**(-compounding_freq * ttm)

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

# ====================================
# Section: Cash Flow Matrix
# ====================================

def get_bond_cash_flows(
    quote_df: pd.DataFrame,
    bond_id: Union[int, str],
    freq: int = 2,
    fv: float = 100,
    cols: Dict[str, str] = {
        'treasury number': 'KYTREASNO',
        'quote date': 'quote date',
        'issue date': 'issue date',
        'maturity date': 'maturity date',
        'cpn rate': 'cpn rate'
    }
) -> pd.DataFrame:
    """
    Generate a series of future cash flows for a single bond given its ID and a dataframe of all the quotes.

    Parameters:
        quote_df (pd.DataFrame): DataFrame containing many bonds quote information.
        bond_id (int | str): ID of the bond for which to generate cash flows.
        freq (int): frequency of coupon payment per year (default 2: semiannual)
        fv (float): face value of bond (default 100)
        cols (Dict[str, str]): names of columns necessary (just in case they are different)

    Returns:
        pd.DataFrame: A DataFrame containing the future cash flow dates and corresponding cash flows.
    """
    # Locate the bond by ID
    bond = quote_df[quote_df[cols['treasury number']] == bond_id].iloc[0]
    
    # Extract bond details
    quote_date = pd.to_datetime(bond[cols['quote date']])
    issue_date = pd.to_datetime(bond[cols['issue date']])
    maturity_date = pd.to_datetime(bond[cols['maturity date']])
    coupon_rate = bond[cols['cpn rate']]
    
    # Initialize an empty list to store cash flow information
    cash_flows = []
    
    # Generate cash flow dates (every 6 months after the issue date)
    current_date = issue_date
    while current_date < maturity_date:
        cash_flows.append((current_date, coupon_rate / freq))
        current_date += pd.DateOffset(months=int(12/freq))


    cash_flows.append((maturity_date, fv + coupon_rate / freq))
    
    # Convert to a DataFrame
    cash_flows_df = pd.DataFrame(cash_flows, columns=['date', 'cash flow'])
    
    return cash_flows_df[cash_flows_df['date'] > quote_date]

def calculate_ytm(cash_flow_df: pd.DataFrame, current_date: str, market_price: float) -> float:
    """
    Calculate the yield to maturity (YTM) of a bond given its cash flows and current price.

    Parameters:
        cash_flow_df (pd.DataFrame): DataFrame with columns 'date' (datetime) and 'cash flow' (float) for a single bond.
        current_date (str): The current date as a string. Eg. "2025-01-26".
        market_price (float): The current market price of the bond.

    Returns:
        float: Yield to maturity (YTM) as an annualized rate (semi-annual compounding).
    """
    
    # Ensure date inputs are in datetime format
    current_date = pd.to_datetime(current_date)
    cash_flow_df['date'] = pd.to_datetime(cash_flow_df['date'])

    # Filter out dates from before the current date
    cash_flow_df = cash_flow_df[cash_flow_df['date'] >= current_date]
    
    # Calculate the time to cash flows in years (semi-annual periods)
    cash_flow_df['TTCF'] = (cash_flow_df['date'] - current_date).dt.days / 365.25 

    # Define the present value function
    def pv_function(ytm):
        pv = sum(
            row['cash flow'] / (1 + ytm / 2) ** (row['TTCF'] * 2)
            for _, row in cash_flow_df.iterrows()
        )
        return (pv - market_price) ** 2  # Return the squared error between PV and actual price

    # Initial guess for the yield 
    initial_guess = 0.04

    # Minimize the present value function to find the YTM
    result = minimize(pv_function, initial_guess, bounds=[(0, None)])

    # Return the annualized YTM
    return result.x[0] if result.success else None

def make_cashflow_matrix(quote_df: pd.DataFrame, bond_ids: list[str] | Series) -> pd.DataFrame:
    """
    Combine cash flow streams of multiple bonds into a matrix.

    Parameters:
        quote_df (pd.DataFrame): DataFrame containing bond information.
        bond_ids (list[str] | Series): List/series of bond IDs to include in the matrix.

    Returns:
        pd.DataFrame: A DataFrame where rows are indexed by bond IDs, 
                      columns are unique cash flow dates, and values are cash flows.
    """
    # Initialize a dictionary to store cash flow data for each bond
    all_cash_flows = {}
    
    for bond_id in bond_ids:
        # Generate cash flows for the bond
        cash_flows_df = get_bond_cash_flows(quote_df, bond_id)
        
        # Store the cash flow data with the bond ID as the key
        all_cash_flows[bond_id] = cash_flows_df.set_index('date')['cash flow']
    
    # Combine all cash flow series into a single DataFrame
    combined_cash_flows = pd.DataFrame(all_cash_flows).fillna(0).T
    
    # Sort the columns (dates) for clarity
    combined_cash_flows = combined_cash_flows.rename_axis('bond id', axis='index')
    
    return combined_cash_flows

def consolidate_by_month(cash_flow_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    This function should be sufficient if he gives us idealized data like in HW 1.
    Consolidate cash flows within the same month from an existing cash flow matrix. Only works for idealized data (maturity in perfect intervals).

    Parameters:
        cash_flow_matrix (pd.DataFrame): A DataFrame where rows are indexed by bond IDs,
                                         columns are cash flow dates, and values are cash flows.

    Returns:
        pd.DataFrame: A DataFrame where columns represent months, and cash flows within
                      the same month are aggregated.
    """
    # Convert column names to datetime for grouping
    cash_flow_matrix.columns = pd.to_datetime(cash_flow_matrix.columns)
    
    # Group columns by month (YYYY-MM format)
    grouped_columns = cash_flow_matrix.groupby(cash_flow_matrix.columns.to_period('M'), axis=1).sum()
    
    # Rename columns back to strings (YYYY-MM format)
    grouped_columns.columns = grouped_columns.columns.astype(str)

    # Make sure last cash flow is correct (assuming FV 100 for each treasury)
    fv = 100
    for i in range(len(grouped_columns)): 
        if i == 0:
            continue
        nonzero_indices = np.flatnonzero(grouped_columns.iloc[i].to_numpy())
        coupon = grouped_columns.iloc[i][0]
        correct_value = fv + coupon
        if len(nonzero_indices) > 0:  # If there is at least one nonzero value
            last_nonzero_idx = nonzero_indices[-1]  # Get the last nonzero index
            grouped_columns.iat[i, last_nonzero_idx] = correct_value

    return grouped_columns

def remove_singular_securities(cash_flow_matrix):
    '''
    this function must be used if he gives us real data, with linear dependence
    Given a cash flow matrix, remove the securities which pay cash flows on dates where no other security pays cash flows. Then, check for 
        dates which now have no cash flows and remove these.
        Intuition: these steps will prevent linear dependence in the cash flow matrix.
    '''
    
    # Step 1: Find columns where all elements are < 100
    columns_to_exclude = cash_flow_matrix.columns[(cash_flow_matrix < 100).all(axis=0)]
    
    # Step 2: Identify rows with non-zero values in these columns
    rows_to_exclude = cash_flow_matrix[columns_to_exclude].any(axis=1)
    
    # Step 3: Filter out rows with non-zero values in those columns (preserve index)
    filtered_matrix = cash_flow_matrix[~rows_to_exclude]
    
    # Step 4: Remove columns containing only zeros
    filtered_matrix = filtered_matrix.loc[:, (filtered_matrix != 0).any(axis=0)]

    return filtered_matrix

# ====================================
# Section: Curve Fitting
# ====================================

def generate_discount_perfect_system(cash_flow_matrix: pd.DataFrame, prices: pd.DataFrame, freq: Union[int, str] = 2) -> pd.DataFrame:
    """
    Generates discount factors/rates with TTM given cash flow matrix, prices (only for perfect system)

    Parameters:
        cash_flow_matrix (pd.DataFrame): A DataFrame where rows are indexed by bond IDs,
                                         columns are cash flow dates, and values are cash flows.
        prices (pd.DataFrame): dataframe of the prices of the bonds
        freq (Union[int, str]): how often we want to get the discount factor (2 would be semiannual, continuous for continuously compounded)
        

    Returns:
        pd.DataFrame: A DataFrame where the discount factors, rates , ttm are listed
    """
    matrix = cash_flow_matrix.values
    inverse_matrix = np.linalg.inv(matrix)
    z = inverse_matrix @ prices
    z.rename(columns={z.columns[0]: 'Spot Discount Factor'}, inplace = True)
    z.index = cash_flow_matrix.columns
    z['T - t'] = np.arange(1/freq, 1/freq * (len(z) + 1), 1/freq)
    z['Spot Discount Rates'] = convert_discount_factor_to_discount_rate(z['Spot Discount Factor'], z['T - t'], freq)

    return z

# ====================================
# Subsection: Nelson Siegel
# ====================================

def nelson_siegel(theta, T):
    """
    Applies the Nelson-Siegel model to predict the spot rate for a given maturity. 

    Parameters:
    theta (iterable): a list of parameters [theta_0, theta_1, theta_2, lambda].
    T (float): time to maturity.

    """
    theta_0, theta_1, theta_2, lambd = theta

    # Apply the Nelson-Siegel formula
    term1 = theta_0
    term2 = (theta_1 + theta_2) * (1 - np.exp(-T / lambd)) / (T / lambd)
    term3 = -theta_2 * np.exp(-T / lambd)
    
    r = term1 + term2 + term3
    return r

def nelson_siegel_extended(theta, T):
    """
    Applies the extended Nelson-Siegel model to predict the spot rate for a given maturity.

    Parameters:
    theta (iterable): a list of parameters [theta_0, theta_1, theta_2, lambda_1, theta_3, lambda_2].
    T (float): time to maturity.

    Returns:
    float: The predicted spot rate.
    """
    theta_0, theta_1, theta_2, lambda_1, theta_3, lambda_2 = theta

    # Apply the extended Nelson-Siegel formula
    term1 = theta_0
    term2 = (theta_1 + theta_2) * (1 - np.exp(-T / lambda_1)) / (T / lambda_1)
    term3 = -theta_2 * np.exp(-T / lambda_1)
    term4 = theta_3 * ((1 - np.exp(-T / lambda_2)) / (T / lambda_2) - np.exp(-T / lambda_2))

    r = term1 + term2 + term3 + term4
    return r

def years_between(quote_date, target_date):
    """
    Compute the number of years between a quote date and a target date.
    """
    # Convert both dates to pandas datetime
    quote_date = pd.to_datetime(quote_date, format='%Y-%m')
    target_date = pd.to_datetime(target_date, format='%Y-%m')
    
    # Compute the time difference in years
    years_between = (target_date - quote_date).days / 365.0
    return years_between

def calculate_discount_factors(quote_date, cash_flow_dates, theta, extended):
    """
    Calculate discount factors for each cash flow date using the Nelson-Siegel model.

    Parameters:
    quote_date : str
        The quote date in the format 'YYYY-MM-DD'.
    cash_flow_dates : list of str
        List of cash flow dates in the format 'YYYY-MM-DD'.
    theta : list or array-like
        A vector of parameters

    Returns:
    numpy.ndarray
        A vector of discount factors, one for each cash flow date.
    """

    # Calculate discount factors
    discount_factors = []
    for cash_flow_date in cash_flow_dates:
        ttm = years_between(quote_date, cash_flow_date)  # Time to maturity
        if ttm <= 0:
            discount_factors.append(1.0)  # If ttm is zero or negative, no discounting
            continue
        if extended:
            r = nelson_siegel_extended(theta, ttm)
        else:
            r = nelson_siegel(theta, ttm)  # Get discount rate
        discount_factor = np.exp(-r * ttm)  # Convert rate to discount factor
        discount_factors.append(discount_factor)
    
    return np.array(discount_factors)

def compute_loss(cash_flow_matrix, theta, quote_date, market_prices, extended):
    ''' 
    Calculate the loss for a given set of NS parameters theta.
    
    Inputs: 
    cash_flow_matrix: processed matrix of cash flows
    theta: vector of NS parameters
    market_prices: the market prices of all securities in the cash flow matrix. 
    extended: bool
        If we want to use nelson siegel extended or not, default is false
    '''
    # Get a series of cash flow dates.
    cash_flow_dates = cash_flow_matrix.columns
    
    # Compute the discount factors by applying NS. 
    discount_factors = calculate_discount_factors(quote_date, cash_flow_dates, theta, extended)

    # Predict the bond prices based on the discount factors.
    price_predictions = cash_flow_matrix @ discount_factors

    # Compute the mean squared error. This is what we want to minimize.
    mse = np.mean((market_prices - price_predictions) ** 2)
    return mse

def find_optimal_theta(cash_flow_matrix, quote_date, market_prices, initial_theta, extended = False):
    """
    Finds the optimal theta that minimizes the loss function.
    
    Parameters:
    cash_flow_matrix : pd.DataFrame
        The cash flow matrix where columns represent cash flow dates and rows represent securities.
    quote_date : str
        The quote date in the format 'YYYY-MM-DD'.
    market_prices : pd.Series
        The market prices of the securities.
    initial_theta : list or array-like
        The initial guess for the parameters theta.
    extended: bool
        If we want to use nelson siegel extended or not, default is false

    Returns:
    result : OptimizeResult
        The result of the optimization containing the optimal theta.
    """
    # Define the objective function (loss function)
    def objective(theta):
        return compute_loss(cash_flow_matrix, theta, quote_date, market_prices, extended)
    
    # Perform the minimization using 'BFGS' (or another method if preferred)
    result = minimize(objective, initial_theta, method='BFGS')
    
    return result

# ====================================
# Section: PCA
# ====================================


def perform_pca(yield_data: pd.DataFrame, n_components: 2):
    """
    PCA Function from GPT
    Perform PCA on the yield curve data.

    Parameters:
    yield_data (pd.DataFrame): DataFrame containing yield curve data (excluding date column).
    n_components (int): Number of principal components to retain.

    Returns:
    tuple: Explained variance ratio and loadings of the first two PCA factors.
    """

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(yield_data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(standardized_data)

    # Extract explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Extract PCA loadings
    loadings = pca.components_

    return explained_variance, loadings

# ====================================
# Section: PCA
# ====================================

def asset_duration(cash_flow_df, current_date_str):
    """
    GPT WRITTEN FUNCTION
    Computes the (cash flow weighted average) duration of each asset.
    
    Parameters:
      cash_flow_df : pd.DataFrame
          A DataFrame where each row corresponds to an asset and 
          each column corresponds to a cash flow date (as a string or datetime),
          with the cell values being the cash flow amounts.
      current_date_str : str
          A string representing the current date (e.g., '2025-02-04').
    
    Returns:
      pd.Series
          A Series with the asset index and the corresponding duration (in years).
          If an asset has no future cash flows (or the sum is 0) its duration will be NaN.
    """
    # Convert the current date and the cash flow dates to datetime objects.
    current_date = pd.to_datetime(current_date_str)
    cf_dates = pd.to_datetime(cash_flow_df.columns)
    
    # Only consider future cash flows (dates >= current_date)
    future_mask = cf_dates >= current_date
    if not future_mask.any():
        # If no cash flows are in the future, return NaN for all assets.
        return pd.Series(np.nan, index=cash_flow_df.index)
    
    # Filter the dataframe to only include future cash flows.
    cf_future = cash_flow_df.loc[:, future_mask]
    
    # Compute the time (in years) from current_date to each cash flow date.
    # Here we use days/365.25 to approximate the year fraction.
    time_diffs = (cf_dates[future_mask] - current_date).days / 365.25
    
    # Compute the weighted average time:
    #   duration = sum(cash_flow * time) / sum(cash_flow)
    weighted_times = cf_future.multiply(time_diffs, axis=1)
    numerator = weighted_times.sum(axis=1)
    denominator = cf_future.sum(axis=1)
    
    # Avoid division by zero: if an asset's denominator is 0, its duration becomes NaN.
    duration = numerator / denominator
    return duration
