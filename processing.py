import pandas as pd, seaborn

def processData(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dropMissing(data, ['saleEstimate_currentPrice'])
    toMap = data[['fullAddress', 'latitude', 'longitude', 'bedrooms', 'bathrooms', 'floorAreaSqM', 'saleEstimate_currentPrice']].copy() # save labelling information
    data = data.drop(columns=["fullAddress", "postcode", "country", "outcode", "saleEstimate_valueChange.saleDate", "history_date"]) # drop non-info columns
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
    for colName in data.columns:
        if not data[colName].isna().any():
            continue

        if pd.api.types.is_numeric_dtype(data[colName]):
            data[colName] = data[colName].fillna(data[colName].median())
        else:
            data[colName] = data[colName].fillna("Unknown")

    return data

def dropMissing(data: pd.DataFrame, columns):
    data = data.dropna(subset=columns)

def dropMissingThreshold(data: pd.DataFrame, threshold = 0.15):
    missing = data.isna().sum() / data.shape[0] > threshold
    toDrop = missing[missing].index
    print(f"Dropping {toDrop}")
    data = data.drop(columns=toDrop)

def encodeCategorical(data: pd.DataFrame) -> pd.DataFrame:
    categorical = [
        col for col in data.columns
        if (
            pd.api.types.is_object_dtype(data[col].dtype)
            or pd.api.types.is_string_dtype(data[col].dtype)
            or isinstance(data[col].dtype, pd.CategoricalDtype)
        )
    ]

    return pd.get_dummies(
        data,
        columns=categorical,
        drop_first=True,
        dtype=int
    )

def normalizeNumerical(data: pd.DataFrame, std=True):
    numericalCols = data.select_dtypes(include="number").columns

    if std:  # Standard score normalization (Z-score)
        means = data[numericalCols].mean()
        stds = data[numericalCols].std().replace(0, 1)
        data[numericalCols] = (data[numericalCols] - means) / stds
    else:  # Min-max normalization
        mins = data[numericalCols].min()
        ranges = (data[numericalCols].max() - mins).replace(0, 1)
        data[numericalCols] = (data[numericalCols] - mins) / ranges

def printUniqueCategoricals(data):
    for col in data.select_dtypes(include='object').columns:
        unique_vals = data[col].unique()
        print(f"Column: {col}")
        print(unique_vals)
        print("-" * 40)

def visualizeCorrelations(data: pd.DataFrame, fileName: str = 'correlation_heatmap'):
    seaborn.heatmap(data.select_dtypes(include='float64').corr()).get_figure().savefig(f'heatmaps/{fileName}.png', dpi=400, bbox_inches="tight")

def descalePreds(preds: pd.DataFrame, y: pd.Series, rawTargets: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    preds = preds * rawTargets.std() + rawTargets.mean()
    return preds, rawTargets[y.index]