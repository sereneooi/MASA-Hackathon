from asyncio.windows_events import NULL
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dython import nominal

# Missing value treatment
def deal_missing_value(df):
    # check on missing value
    print(pd.isna(df).sum())

    # 45107 rows out of 63326 rows (gender) are null value, remove the whole column
    df = df.drop(['Gender'], axis = 1)

    return df

# plot boxplot
def plot_boxplot(df, col, desc):
    sns.boxplot(x = col, data = df).set(title = 'Boxplot of ' + col + " " + desc)
    plt.show()

# detecting the outliers using boxplot
def deal_outliers(df, col):
    Q1, Q3 = np.percentile(df[col] , [25, 75])
    IQR = Q3 - Q1
    ul = Q3 + 1.5 * IQR
    ll = Q1 - 1.5 * IQR

    outliers = df[(df[col] > ul) | (df[col] < ll)]
    print(len(outliers))
    
    df[col] = np.where(df[col] > ul, ul, np.where(df[col] < ll, ll, df[col]))
    
    return df

def replace_negative_value(df, col):
    mean_v = df[col].mean()

    df.loc[df[col] <= 0, col] = mean_v

    return df
        
def obtain_unique_value(df, col):
    return dict(enumerate(df[col].unique()))

def heatmap(df):
    nominal.associations(df, figsize=(20, 10), mark_columns = True)

# Read csv data file
df = pd.read_csv('dataset.csv') 

df = deal_missing_value(df) 
df = replace_negative_value(df, 'Duration')
df = replace_negative_value(df, 'Age')

# loop all the numeric features
for feature in df.select_dtypes(exclude = 'object'):
    plot_boxplot(df, feature)
    df = deal_outliers(df, feature)
    plot_boxplot(df, feature)

transfomed_df = df.copy()

# convert the categorical data into numeric column
for feature in df.select_dtypes(include = 'object'):
    unique_value = obtain_unique_value(df, feature)
    unique_value = {v: k for k, v in unique_value.items()}
    transfomed_df[feature] = df[feature].replace(unique_value)
    
# feature selection
heatmap(df)

# decide to delete both columns, since high correlated to each other
del transfomed_df['Agency']
del transfomed_df['Product Name']

heatmap(transfomed_df)

transfomed_df.to_csv('dataset2.csv', index = False)