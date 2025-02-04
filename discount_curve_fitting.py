


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