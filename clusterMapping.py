from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd, folium, matplotlib.pyplot as plt

clusterColours = [
    'red', 'blue', 'green', 'purple', 'orange',
    'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen'
]

def applyPCA(data: pd.DataFrame, numComponents, idColName="newId"):
    ids = data[idColName]
    data.drop(columns=[idColName], inplace=True)
    pca = PCA(n_components=numComponents)
    X_pca = pca.fit_transform(data)

    dfDict = {}
    for i in range(X_pca.shape[1]):
        dfDict[f"col{i+1}"] = X_pca[:, i]

    pcaDf = pd.DataFrame.from_dict(dfDict)

    # print(pca.explained_variance_ratio_)
    # print(X_pca)

    return pd.concat([ids, pcaDf], axis=1)

def pcaCompare(data: pd.DataFrame, mapData: pd.DataFrame, stop: int = 1, kRange: range = range(1, 16, 2), step = 10):
    
    while data.shape[1] > stop:
        # print(data.head())
        inertias = []
        for numClusters in kRange:
            kmeans = KMeans(n_clusters=numClusters, random_state=42).fit(data.iloc[:, 1:])
            inertias.append(kmeans.inertia_)
            mapCluster(pd.concat([mapData, pd.Series(kmeans.labels_, name="clusterLabels")], axis=1), numClusters, data.shape[1])

        graphInertias(kRange, inertias, data.shape[1])

        data = applyPCA(data, data.shape[1] - step -1)

def graphInertias(clusterRange, inertias, numColumns):
    plt.figure() 
    plt.plot(clusterRange, inertias, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig(f'inertias\\{numColumns}_cols.png')

def mapCluster(labelledData: pd.DataFrame, numClusters: int, numColumns: int, verbal: bool = True):
    if verbal:
        # print(labelledData.head())
        labels = sorted(labelledData['clusterLabels'].unique())
        print(f"{len(labels)} Clusters:")
        labelCounts = labelledData['clusterLabels'].value_counts()
        summaryDF = pd.DataFrame(columns=['Count', 'Price', 'Sq. Ft.', 'Bedrooms', 'Bathrooms'])
        for i, label in enumerate(labels):
            clusterDf = labelledData[labelledData['clusterLabels'] == label]
            # print(f"Cluster {label} - Count: {labelCounts[label]} Price: {clusterDf['saleEstimate_currentPrice'].mean():,.0f} - Sq. Ft. {clusterDf['floorAreaSqM'].mean():.0f} - Bedrooms: {clusterDf['bedrooms'].mean():.2f} - Bathrooms: {clusterDf['bathrooms'].mean():.2f}")
            summaryDF.loc[i] = [format(labelCounts[label], ","), format(round(clusterDf['saleEstimate_currentPrice'].mean()), ","), round(clusterDf['floorAreaSqM'].mean()), round(clusterDf['bedrooms'].mean(), 2), round(clusterDf['bathrooms'].mean(), 2)]
        print(summaryDF)

    labelledData = labelledData.sample(n=1000)
    mean_lat = labelledData['latitude'].mean()
    mean_lon = labelledData['longitude'].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

    for _, row in labelledData.iterrows():
        popupInfo = f"""
            <b>{row['fullAddress']}</b><br>
            Cluster: {row['clusterLabels']}<br>
            Bedrooms: {row['bedrooms']}<br>
            Bathrooms: {row['bathrooms']}<br>
            Square Footage: {row['floorAreaSqM']}<br>
            Sales Price Estimate: {row['saleEstimate_currentPrice']}<br>
        """
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popupInfo,
            icon=folium.Icon(color=clusterColours[row['clusterLabels'] % len(clusterColours)])
            ).add_to(m)
    m.save(f'folium_maps\\pca_clustering\\{numColumns}_columns_{numClusters}_clusters_map.html')