from sklearn.ensemble import IsolationForest, RandomForestRegressor
import pandas as pd, folium

def isolationForest(data: pd.DataFrame, info: pd.DataFrame, anomalyRate = 0.01):
    iso = IsolationForest(contamination=anomalyRate, random_state=0)
    iso.fit(data)
    info['anomaly_score'] = iso.decision_function(data)
    info['is_anomaly'] = iso.predict(data)

    anomalies = info[info['is_anomaly'] == -1]

    print(anomalies.head())
    mapAnomalies(anomalies, "isolation_forest")

def randomForest(data: pd.DataFrame, info: pd.DataFrame, anomalyRate = 0.01):
    y = data['saleEstimate_currentPrice']
    X = data.drop(columns=['newId', 'rentEstimate_lowerPrice', 'rentEstimate_currentPrice', 'rentEstimate_upperPrice', 'saleEstimate_lowerPrice', 'saleEstimate_currentPrice', 'saleEstimate_upperPrice', 'saleEstimate_valueChange.numericChange', 'saleEstimate_valueChange.percentageChange', 'history_price'])

    model = RandomForestRegressor()
    model.fit(X, y)

    info['predicted_price'] = model.predict(X)
    info['price_diff'] = info['predicted_price'] - info['saleEstimate_currentPrice']

    info['underpriced_flag'] = info['price_diff'] < info['price_diff'].quantile(anomalyRate)

    anomalies = info[info['underpriced_flag']]

    mapAnomalies(anomalies, "random_forest")

def mapAnomalies(labelledData: pd.DataFrame, fileName):

    mean_lat = labelledData['latitude'].mean()
    mean_lon = labelledData['longitude'].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

    for _, row in labelledData.iterrows():
        popupInfo = f"""
            <b>{row['fullAddress']}</b><br>
            Bedrooms: {row['bedrooms']}<br>
            Bathrooms: {row['bathrooms']}<br>
            Square Footage: {row['floorAreaSqM']}<br>
            Sales Price Estimate: {row['saleEstimate_currentPrice']}<br>
        """
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popupInfo,
            icon=folium.Icon(color='red')
            ).add_to(m)
    m.save(f'folium_maps\\anomalies\\{fileName}.html')