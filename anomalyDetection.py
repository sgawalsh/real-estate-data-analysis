from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, mean_squared_error, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from typing import Optional
from xgboost import XGBRegressor
import pandas as pd, numpy as np, folium, seaborn

#TODO additional underpriced predictions and comparison

def isolationForest(data: pd.DataFrame, info: pd.DataFrame, anomalyRate = 0.01):
    iso = IsolationForest(contamination=anomalyRate, random_state=0)
    iso.fit(data)
    info['anomaly_score'] = iso.decision_function(data)
    info['is_anomaly'] = iso.predict(data)

    return info[info['is_anomaly'] == -1], 'isolation_forest'

def getModelPredictions(model, xData: pd.DataFrame, yData: pd.DataFrame, featureDf: dict = None):
    model.fit(xData, yData)
    predictedPrice = model.predict(xData)

    if hasattr(model, 'feature_importances_'):
        featureDf[str(len(featureDf))] = model.feature_importances_

    # return (predictedPrice - yData) / abs(yData)
    return predictedPrice

def getRmseWeights(y_data: pd.Series, predsDf: pd.DataFrame) -> np.ndarray:
    rmseList = []
    for _, preds in predsDf.items():
        rmseList.append(root_mean_squared_error(y_data, preds))

    invErrors = 1 / np.array(rmseList)
    return invErrors / invErrors.sum(), rmseList

def addCols(y_data, predsDf, verbal: bool = True, n: int = 5):
    predWeights, rmseList = getRmseWeights(y_data, predsDf)
    print(predWeights)
    predsDf['weighted_agg'] = predsDf.dot(predWeights)
    predsDf['agg_price_diff'] = predsDf.mean(axis=1)

    for col in predsDf.iloc[:, -2:]:
        rmseList.append(root_mean_squared_error(y_data, predsDf[col]))

    if verbal:
        mseDict = {}
        for i, colName in enumerate(predsDf.columns):
            mseDict[colName] = round(rmseList[i] ** 2, n)
        print(mseDict)
    
    return rmseList

def showFeatures(featureDf: pd.DataFrame):
    featureDf['average'] = featureDf.iloc[:, 1:].mean(axis=1)
    featureDf = featureDf.sort_values(by='average', ascending=False)
    print(featureDf)

def compareModelPreds(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = data.sample(5000)
    y_data = data['saleEstimate_currentPrice']
    x_data = data[['latitude', 'longitude', 'floorAreaSqM', 'bedrooms_2.0', 'bedrooms_3.0', 'bedrooms_4.0', 'bedrooms_5.0', 'bedrooms_6.0', 'bedrooms_7.0', 'bedrooms_8.0', 'bedrooms_9.0','bedrooms_Unknown',
             'livingRooms_2.0', 'livingRooms_3.0', 'livingRooms_4.0', 'livingRooms_5.0', 'livingRooms_6.0', 'livingRooms_7.0', 'livingRooms_8.0', 'livingRooms_9.0', 'livingRooms_Unknown',
            'tenure_Freehold', 'tenure_Leasehold', 'tenure_Shared', 'tenure_Unknown',
            'propertyType_Converted Flat', 'propertyType_Detached Bungalow', 'propertyType_Detached House', 'propertyType_Detached Property', 'propertyType_End Terrace Bungalow', 'propertyType_End Terrace House', 'propertyType_End Terrace Property', 'propertyType_Flat/Maisonette', 'propertyType_Mid Terrace Bungalow', 'propertyType_Mid Terrace House', 'propertyType_Mid Terrace Property', 'propertyType_Purpose Built Flat', 'propertyType_Semi-Detached Bungalow', 'propertyType_Semi-Detached House', 'propertyType_Semi-Detached Property', 'propertyType_Terrace Property', 'propertyType_Terraced', 'propertyType_Terraced Bungalow', 'propertyType_Unknown']].copy()

    featureDf = {'features': x_data.columns}
    predsDf = pd.DataFrame({
        'random_forest': getModelPredictions(RandomForestRegressor(), x_data, y_data, featureDf),
        'xgboost': getModelPredictions(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=0), x_data, y_data, featureDf),
        'linear_regression': getModelPredictions(LinearRegression(), x_data, y_data),
        'knn': getModelPredictions(KNeighborsRegressor(), x_data, y_data)
    })
    showFeatures(pd.DataFrame(featureDf))

    addCols(y_data, predsDf, verbal=True, n=5)
    
    scaler = StandardScaler()
    scaledPredsDf = pd.DataFrame(scaler.fit_transform(predsDf), columns=predsDf.columns)
    seaborn.heatmap(predsDf.corr()).get_figure().savefig(f'underpriced_preds_correlation.png', dpi=400, bbox_inches="tight")
    print(predsDf.var())
    print(predsDf.describe())
    print(scaledPredsDf.var())
    print(scaledPredsDf.describe())
    print(y_data.var())
    print(y_data.describe())

    inputDf = predsDf

    stackedFeatureDf = {'features': inputDf.columns}
    stackedPredsDf = pd.DataFrame({
        'random_forest': getModelPredictions(RandomForestRegressor(), inputDf, y_data, stackedFeatureDf),
        'xgboost': getModelPredictions(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=0), inputDf, y_data, stackedFeatureDf),
        'linear_regression': getModelPredictions(LinearRegression(), inputDf, y_data),
        'knn': getModelPredictions(KNeighborsRegressor(), inputDf, y_data)
    })

    rmseList = addCols(y_data, stackedPredsDf, verbal=True, n=5)
    showFeatures(pd.DataFrame(stackedFeatureDf))
    print(stackedPredsDf.head())

    bestColumn = stackedPredsDf.columns[rmseList.index(min(rmseList))]

    stackedPredsDf.index = y_data.index
    return stackedPredsDf, y_data, bestColumn

    # pctDifference = stackedPredsDf.subtract(y_data, axis=0).div(y_data, axis=0) * 100
    # pctDifference['average'] = pctDifference.mean(axis=1)

    # ars = adjusted_rand_score(knnPreds, xgbPreds)
    # nmi = normalized_mutual_info_score(knnPreds, xgbPreds)

    # print("Forest/xgb: ", ars, nmi)
    # ars = adjusted_rand_score(linearRegressionPreds, xgbPreds)
    # nmi = normalized_mutual_info_score(linearRegressionPreds, xgbPreds)
    # print("linear/xgb: ", ars, nmi)

def mapUnderpriced(data: pd.DataFrame, y: pd.Series, info: pd.Series, sortColumn: str = 'xgboost', topN = 0.01) -> pd.DataFrame:
    percentDiff = data.subtract(y, axis=0).div(y, axis=0) * 100
    percentDiff = percentDiff.sort_values(by=sortColumn, ascending=False).head(round(len(percentDiff) * topN))
    mapAnomalies(info.loc[percentDiff.index], 'underpriced', percentDiff[sortColumn])

def mapAnomalies(labelledData: pd.DataFrame, fileName, additionalInfo: Optional[pd.Series] = None):

    mean_lat = labelledData['latitude'].mean()
    mean_lon = labelledData['longitude'].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

    for i, row in labelledData.iterrows():
        popupInfo = f"""
            <b>{row['fullAddress']}</b><br>
            Bedrooms: {row['bedrooms']}<br>
            Bathrooms: {row['bathrooms']}<br>
            Square Footage: {row['floorAreaSqM']}<br>
            Sales Price Estimate: {row['saleEstimate_currentPrice']}
            {f"<br>Underpriced %: {round(additionalInfo[i], 2)}" if additionalInfo is not None else ""}
        """
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popupInfo,
            icon=folium.Icon(color='red')
            ).add_to(m)
    m.save(f'folium_maps\\anomalies\\{fileName}.html')