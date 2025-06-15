from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd, folium, matplotlib.pyplot as plt, math

clusterColours = [
    'red', 'blue', 'green', 'purple', 'orange',
    'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen'
]

def applyPCA(data: pd.DataFrame, numComponents):
    pca = PCA(n_components=numComponents)
    X_pca = pca.fit_transform(data)

    dfDict = {}
    for i in range(X_pca.shape[1]):
        dfDict[f"col{i+1}"] = X_pca[:, i]

    pcaDf = pd.DataFrame.from_dict(dfDict)

    # print(pca.explained_variance_ratio_)
    # print(X_pca)

    return pcaDf

def pcaCompare(data: pd.DataFrame, mapData: pd.DataFrame, stop: int = 1, kRange: range = range(1, 16, 2), step = 10):
    
    while True:
        inertias = []
        for numClusters in kRange:
            kmeans = KMeans(n_clusters=numClusters, random_state=42).fit(data)
            inertias.append(kmeans.inertia_)
            mapCluster(pd.concat([mapData, pd.Series(kmeans.labels_, name="clusterLabels")], axis=1), numClusters, data.shape[1])

        graphInertias(kRange, inertias, data.shape[1])

        if data.shape[1] - step <= stop:
            break

        data = applyPCA(data, data.shape[1] - step)

def graphInertias(clusterRange, inertias, numColumns):
    plt.figure()
    plt.plot(clusterRange, inertias, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig(f'inertias\\{numColumns}_cols.png')

def mapCluster(labelledData: pd.DataFrame, numClusters: int, numColumns: int, verbal: bool = True):
    meanLat = labelledData['latitude'].mean()
    meanLon = labelledData['longitude'].mean()

    if verbal:
        # print(labelledData.head())
        labels = sorted(labelledData['clusterLabels'].unique())
        print(f"{len(labels)} Clusters:")
        labelCounts = labelledData['clusterLabels'].value_counts()
        summaryDF = pd.DataFrame(columns=['Count', 'Price', 'Sq. Ft.', 'Bedrooms', 'Bathrooms', 'Lat', 'Long', 'Lat Bias', 'Long Bias', 'Spread'])

        clusterGroups = {}
        for i, label in enumerate(labels):
            clusterGroups[label] = folium.FeatureGroup(name=f"Cluster {label}")
            clusterDf = labelledData[labelledData['clusterLabels'] == label]
            # print(f"Cluster {label} - Count: {labelCounts[label]} Price: {clusterDf['saleEstimate_currentPrice'].mean():,.0f} - Sq. Ft. {clusterDf['floorAreaSqM'].mean():.0f} - Bedrooms: {clusterDf['bedrooms'].mean():.2f} - Bathrooms: {clusterDf['bathrooms'].mean():.2f}")
            clusterLatMean = clusterDf['latitude'].mean()
            clusterLonMean = clusterDf['longitude'].mean()

            clusterSpread = clusterDf.apply(
                lambda row: haversine(meanLat, meanLon, row['latitude'], row['longitude']),
                axis=1
            ).std() # std dev cluster spread in km

            summaryDF.loc[i] = [
                format(labelCounts[label], ","),
                format(round(clusterDf['saleEstimate_currentPrice'].mean()), ","),
                round(clusterDf['floorAreaSqM'].mean()),
                round(clusterDf['bedrooms'].mean(), 2),
                round(clusterDf['bathrooms'].mean(), 2),
                round(clusterLatMean, 2),
                round(clusterLonMean, 2),
                round((clusterLatMean - meanLat) * 111, 4), # latitude to km
                round((clusterLonMean - meanLon) * 111 * math.cos(math.radians((clusterLatMean + meanLat) / 2)), 4), # longitude to km
                round(clusterSpread, 2),
                ]
        print(summaryDF)
        clusterCoordsFigure(labelledData, summaryDF, numColumns, numClusters)
        clusterPriceVsLongBias(summaryDF, numColumns, numClusters)
    
    labelledData = labelledData.sample(n=1000)
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
    foliumMap.save(f'folium_maps\\pca_clustering\\{numColumns}_columns_{numClusters}_clusters_map.html')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def clusterCoordsFigure(labelledData: pd.DataFrame, clusterDF: pd.DataFrame, numColumns: int, numClusters: int):
    plt.figure()
    plt.scatter(labelledData['longitude'], labelledData['latitude'], s=1, alpha=0.3)
    plt.scatter(clusterDF['Long'], clusterDF['Lat'], color='red')
    plt.title("Cluster Biases (km)")
    plt.xlabel("Longitude Bias (km)")
    plt.ylabel("Latitude Bias (km)")
    plt.grid(True)
    plt.savefig(f'cluster_graphs\\cluster_center_coords\\{numColumns}_columns_{numClusters}_clusters.png')

def clusterPriceVsLongBias(clusterDF: pd.DataFrame, numColumns: int, numClusters: int):
    plt.figure()
    plt.scatter(clusterDF['Long Bias'], clusterDF['Price'].str.replace(',', '').astype(float), c='blue')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Longitudinal Bias (km from mean)")
    plt.ylabel("Average Price (£)")
    plt.title("Property Price vs. Longitude Bias")
    plt.grid(True)
    plt.savefig(f'cluster_graphs\\price_vs_long_bias\\{numColumns}_columns_{numClusters}_clusters.png')