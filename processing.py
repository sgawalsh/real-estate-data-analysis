import pandas as pd
from sklearn.preprocessing import StandardScaler

def processData(data: pd.DataFrame):
    data.drop(columns=["fullAddress", "postcode", "country", "outcode", "saleEstimate_valueChange.saleDate", "history_date"], inplace=True)
    dropMissing(data)
    fillMissing(data)
    encodeCategorical(data)
    normalizeNumerical(data)
    print(data.head())



def fillMissing(data: pd.DataFrame):
    missing = data.isna().sum() / data.shape[0] > 0
    for colName in missing[missing].index:
        if data[colName].dtype == 'object':
            data[colName] = data[colName].fillna("Unknown")
        else:
            data[colName] = data[colName].fillna(data[colName].median())

def dropMissing(data: pd.DataFrame, threshold = 0.1):
    missing = data.isna().sum() / data.shape[0] > threshold
    toDrop = missing[missing].index
    data.drop(columns=toDrop, inplace=True)

def encodeCategorical(data: pd.DataFrame):
    categorical = data.dtypes == 'object'
    data = pd.get_dummies(data, columns = categorical[categorical].index)

def normalizeNumerical(data: pd.DataFrame):
    numerical = data.dtypes == 'float64'
    scaler = StandardScaler()
    data[numerical] = scaler.fit_transform(data[numerical])



myData = pd.read_csv("data/kaggle_london_house_price_data.csv")
# print(myData.head())
# print(myData.shape)
# print(myData.columns.values)
# print(myData.dtypes)
# print(myData.isna().sum()/ my_data.shape[0])
processData(myData)