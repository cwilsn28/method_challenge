import numpy as np
import pandas as pd

from category_encoders import BinaryEncoder, CountEncoder
from os import PathLike
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from typing import List, Optional, Union

from shared import*


def load_raw_data(filepath: Union[str, PathLike]) -> pd.DataFrame:
    filepath = Path(filepath)
    if filepath.suffix != '.csv':
        raise ValueError('Customer data must be in .csv format.')

    return pd.read_csv(filepath)


def merge_col_vals(df: pd.DataFrame) -> pd.DataFrame:
    # Combine col values where appropriate.
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(TO_MERGE['PreferredPaymentMode'])
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace(TO_MERGE['PreferredLoginDevice'])
    return df


def encode_col_count(df: pd.DataFrame) -> pd.DataFrame:
    encoder = CountEncoder(cols=TO_ENCODE_COUNT)
    # Fit and transform the DataFrame
    return encoder.fit_transform(df)


def encode_col_ohe(df: pd.DataFrame) -> pd.DataFrame:
    ohe = OneHotEncoder(sparse_output=False)

    # Apply one-hot encoding to the categorical columns
    encoded = ohe.fit_transform(df[TO_ENCODE_OHE])
    # Create a DataFrame with the one-hot encoded columns
    # We use get_feature_names_out() to get the column names for the encoded data
    new_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(TO_ENCODE_OHE))
    # Concatenate the one-hot encoded dataframe with the original dataframe
    encoded_df = pd.concat([df, new_df], axis=1)

    # Drop the original categorical columns
    return encoded_df.drop(TO_ENCODE_OHE, axis=1)


def drop_encoded_cols(df: pd.DataFrame, col:Optional[str]=None, prefix: Optional[str]=None) -> pd.DataFrame:
    if col is not None:
        return df.drop([col], axis=1)

    if prefix is not None:
        to_drop = [v for v in list(df.columns) if prefix in v]
        return df.drop(to_drop, axis=1)

    return df


def impute_na(df: pd.DataFrame, grouped:Optional[bool]=False) -> pd.DataFrame:
    # Account for missing data in the following cols:
    # Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder

    df['WarehouseToHome'] = df['WarehouseToHome'].fillna(df['WarehouseToHome'].mean())

    if grouped:
        # Use grouped mean.
        df['Tenure'] = df['Tenure'].fillna(df.groupby('Gender')['Tenure'].transform('mean'))
        df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df.groupby('Gender')['HourSpendOnApp'].transform('mean'))
        df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(df.groupby('Gender')['OrderAmountHikeFromlastYear'].transform('mean'))
        df['CouponUsed'] = df['CouponUsed'].fillna(df.groupby('Gender')['CouponUsed'].transform('mean'))
        df['OrderCount'] = df['OrderCount'].fillna(df.groupby('Gender')['OrderCount'].transform('mean'))
        df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df.groupby('Gender')['DaySinceLastOrder'].transform('mean'))
        return df

    # Impute missing data with col mean.
    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].mean())
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mean())
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].mean())
    df['CouponUsed'] = df['CouponUsed'].fillna(df['CouponUsed'].mean())
    df['OrderCount'] = df['OrderCount'].fillna(df['OrderCount'].mean())
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].mean())
    return df