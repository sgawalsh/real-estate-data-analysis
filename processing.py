import pandas as pd, seaborn

def processData(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dropMissing(data, ['saleEstimate_currentPrice'])
    toMap = data[['fullAddress', 'latitude', 'longitude', 'bedrooms', 'bathrooms', 'floorAreaSqM', 'saleEstimate_currentPrice']].copy() # save labelling information
    data.drop(columns=["fullAddress", "postcode", "country", "outcode", "saleEstimate_valueChange.saleDate", "history_date"], inplace=True) # drop non-info columns
    toCategorical(data, ['bedrooms', 'livingRooms'])
    dropMissingThreshold(data)# drop columns with more than n% missing values
    fillMissing(data)
    normalizeNumerical(data)
    # visualizeCorrelations(data)
    data = encodeCategorical(data)
    # print(data.head())
    # print(data.info())
    return data, toMap

def toCategorical(data: pd.DataFrame, columns):
    for col in columns:
        data[col] = data[col].astype("object")

def fillMissing(data: pd.DataFrame):
    missing = data.isna().sum() > 0
    for colName in missing[missing].index:
        if data[colName].dtype == 'object':
            data[colName] = data[colName].fillna("Unknown")
        else:
            data[colName] = data[colName].fillna(data[colName].median())

def dropMissing(data: pd.DataFrame, columns):
    data.dropna(subset=columns, inplace=True)

def dropMissingThreshold(data: pd.DataFrame, threshold = 0.15):
    missing = data.isna().sum() / data.shape[0] > threshold
    toDrop = missing[missing].index
    print(f"Dropping {toDrop}")
    data.drop(columns=toDrop, inplace=True)

def encodeCategorical(data: pd.DataFrame) -> pd.DataFrame:
    categorical = data.dtypes == 'object'
    return pd.get_dummies(data, columns = categorical[categorical].index, drop_first=True)

def normalizeNumerical(data: pd.DataFrame, std = True):
    numericalCols = data.select_dtypes(include=['float64', 'int64']).columns
    if std: # Standard score normalization (Z-score)
        data[numericalCols] = (data[numericalCols] - data[numericalCols].mean()) / data[numericalCols].std()
    else: # Min-max normalization
        data[numericalCols] = (data[numericalCols] - data[numericalCols].min()) / (data[numericalCols].max() - data[numericalCols].min())

def printUniqueCategoricals(data):
    for col in data.select_dtypes(include='object').columns:
        unique_vals = data[col].unique()
        print(f"Column: {col}")
        print(unique_vals)
        print("-" * 40)

def visualizeCorrelations(data: pd.DataFrame, fileName: str = 'correlation_heatmap'):
    seaborn.heatmap(data.select_dtypes(include='float64').corr()).get_figure().savefig(f'{fileName}.png', dpi=400, bbox_inches="tight")

def descalePreds(preds: pd.DataFrame, y: pd.Series, mean:float, std:float) -> pd.DataFrame:
    preds = preds * std + mean
    y = y * std + mean
    return preds, y