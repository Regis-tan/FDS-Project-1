import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df1 = pd.read_csv('sp500_companies-1.csv')
df2 = pd.read_csv('sp500_companies-2.csv')

df1.rename(columns={'Symbol': 'Ticker', 'Company': 'Company_Name'}, inplace=True)
df2.rename(columns={'Symbol': 'Ticker', 'Company': 'Company_Name'}, inplace=True)

merged_df = pd.merge(df1, df2, on='Ticker', how='outer', suffixes=('_x', '_y'))

for col in merged_df.columns:
    if col.endswith('_x'):
        base_col = col[:-2]
        if base_col + '_y' in merged_df.columns:
            merged_df[base_col] = merged_df[col].combine_first(merged_df[base_col + '_y'])
        else:
            merged_df[base_col] = merged_df[col]

        merged_df.drop([col, base_col + '_y'], axis=1, inplace=True)

print(merged_df.info())

merged_df.set_index('Ticker', inplace=True)

for col in merged_df.select_dtypes(include=[object]).columns:
    merged_df[col] = merged_df[col].map(lambda x: x.lower() if isinstance(x, str) else x)

if 'Marketcap' in merged_df.columns:
    merged_df['Log_Market_Cap'] = merged_df['Marketcap'].apply(lambda x: np.log(x) if pd.notnull(x) and x > 0 else None)
else:
    print("Column 'Marketcap' is not present in the DataFrame.")

if 'Currentprice' in merged_df.columns:
    large_cap_companies = merged_df.query('Marketcap > 100_000_000_000')
else:
    large_cap_companies = pd.DataFrame()
    print("Column 'Currentprice' (or equivalent) is not present in the DataFrame.")

if 'Sector' in merged_df.columns and 'Marketcap' in merged_df.columns:
    sector_market_cap = merged_df.groupby('Sector').agg({'Marketcap': 'mean'})
else:
    sector_market_cap = pd.DataFrame()
    print("Columns 'Sector' and/or 'Marketcap' are not present in the DataFrame.")

if 'Currentprice' in merged_df.columns:
    merged_df.dropna(subset=['Currentprice'], inplace=True)
else:
    print("Column 'Currentprice' is not present in the DataFrame.")

merged_df.ffill(inplace=True)
merged_df.fillna(value=0, inplace=True)

if 'Marketcap' in merged_df.columns:
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    merged_df['Marketcap'] = imputer.fit_transform(merged_df[['Marketcap']])
else:
    print("Column 'Marketcap' is not present in the DataFrame.")

print(merged_df.head())
