import folium, pandas as pd
from typing import Optional

clusterColours = ['lightblue', 'gray', 'blue', 'darkred', 'lightgreen', 'purple', 'red', 'green', 'lightred', 'white', 'darkblue', 'darkpurple', 'cadetblue', 'orange', 'pink']

def mapClusters(labelledData: pd.DataFrame, fileName: str, clusterGroups: dict, meanLat: float, meanLon: float, nSamples: int = 1000):
    labelledData = labelledData.sample(n=nSamples)
    for _, row in labelledData.iterrows():
        popupInfo = f"""
            <b>{row['fullAddress']}</b><br>
            Cluster: {row['clusterLabels']}<br>
            Bedrooms: {row['bedrooms']}<br>
            Bathrooms: {row['bathrooms']}<br>
            Square Footage: {row['floorAreaSqM']}<br>
            Sales Price Estimate: {row['saleEstimate_currentPrice']}<br>
        """
        marker = folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popupInfo,
            icon=folium.Icon(color=clusterColours[row['clusterLabels'] % len(clusterColours)])
            )
        clusterGroups[row['clusterLabels']].add_child(marker)

    
    foliumMap = folium.Map(location=[meanLat, meanLon], zoom_start=12)
    # Add all FeatureGroups to the map
    for group in clusterGroups.values():
        group.add_to(foliumMap)

    # Add layer control to toggle visibility
    folium.LayerControl(collapsed=False).add_to(foliumMap)
    foliumMap.save(f'folium_maps\\pca_clustering\\{fileName}_map.html')

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