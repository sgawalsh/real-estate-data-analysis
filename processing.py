import pandas as pd, numpy as np, seaborn

def processData(data: pd.DataFrame):
    data.insert(0, 'newId', range(len(data)))
    toMap = data[['newId', 'fullAddress', 'latitude', 'longitude']]
    data.drop(columns=["fullAddress", "postcode", "country", "outcode", "saleEstimate_valueChange.saleDate", "history_date"], inplace=True)
    toCategorical(data, ['bedrooms', 'livingRooms'])
    dropMissing(data)
    fillMissing(data)
    encodeCategorical(data)
    normalizeNumerical(data)
    visualizeCorrelations(data)
    print(data.head())
    print(data.info())
    print(toMap.head())

def toCategorical(data: pd.DataFrame, columns):
    for col in columns:
        data[col] = data[col].astype("object")

def fillMissing(data: pd.DataFrame):
    missing = data.isna().sum() / data.shape[0] > 0
    for colName in missing[missing].index:
        if data[colName].dtype == 'object':
            data[colName] = data[colName].fillna("Unknown")
        else:
            data[colName] = data[colName].fillna(data[colName].median())

def dropMissing(data: pd.DataFrame, threshold = 0.15):
    missing = data.isna().sum() / data.shape[0] > threshold
    toDrop = missing[missing].index
    print(f"Dropping {toDrop}")
    data.drop(columns=toDrop, inplace=True)

def encodeCategorical(data: pd.DataFrame):
    categorical = data.dtypes == 'object'
    data = pd.get_dummies(data, columns = categorical[categorical].index)

def normalizeNumerical(data: pd.DataFrame, std = True):
    numerical = data.dtypes == 'float64'
    toNormalize = data[numerical[numerical].index]
    if std:
        data[numerical[numerical].index] = (toNormalize-toNormalize.mean())/toNormalize.std()
    else:
        data[numerical[numerical].index] = (toNormalize-toNormalize.min())/(toNormalize.max()-toNormalize.min())

def visualizeCorrelations(data: pd.DataFrame):
    seaborn.heatmap(data.select_dtypes(include='float64').corr()).get_figure().savefig('heatmap.png', dpi=400)


myData = pd.read_csv("data/kaggle_london_house_price_data.csv")
# print(myData.head())
# print(myData.shape)
print(myData.columns.values)
# print(myData.dtypes)
# print(myData.isna().sum()/ my_data.shape[0])
processData(myData)