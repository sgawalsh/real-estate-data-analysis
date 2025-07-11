from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from processing import visualizeCorrelations
import pandas as pd, numpy as np, matplotlib.pyplot as plt, folium, math, seaborn

clusterColours = ['lightblue', 'gray', 'blue', 'darkred', 'lightgreen', 'purple', 'red', 'green', 'lightred', 'white', 'darkblue', 'darkpurple', 'cadetblue', 'orange', 'pink']

def pcaCompare(data: pd.DataFrame, mapData: pd.DataFrame, stop: int = 1, step = 10, kRange: range = range(1, 16, 2), comparePcaLabels = True):
    
    labels = {}
    while True:
        inertias = []
        labels[data.shape[1]] = []
        for numClusters in kRange:
            kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(data)
            inertias.append(kmeans.inertia_)
            labels[data.shape[1]].append(kmeans.labels_)
            mapClusters(pd.concat([mapData, pd.Series(kmeans.labels_, name="clusterLabels")], axis=1), f"{data.shape[1]}_columns_{numClusters}_clusters")

        graphInertias(kRange, inertias, data.shape[1])

        if data.shape[1] - step <= stop:
            break

        data = applyPCA(data, data.shape[1] - step)
    
    if comparePcaLabels:
        labelDf = pd.DataFrame.from_dict(labels)
        labelDf.index = kRange
        keys = labelDf.columns
        max, min = keys[0], keys[len(keys) - 1]
        resultDf = pd.DataFrame(columns = [f'ari_{max}_{min}', f'nmi_{max}_{min}'], index=kRange)
        for clusterCount, row in labelDf.iterrows():
            resultDf.loc[clusterCount, f'ari_{max}_{min}'] = adjusted_rand_score(row[max], row[min])
            resultDf.loc[clusterCount, f'nmi_{max}_{min}'] = normalized_mutual_info_score(row[max], row[min])

        print(resultDf)
        graphPcaSimilarity(resultDf, f"{max}_{min}")

def applyPCA(data: pd.DataFrame, numComponents: int) -> pd.DataFrame:
    pca = PCA(n_components=numComponents)
    X_pca = pca.fit_transform(data)

    dfDict = {}
    for i in range(X_pca.shape[1]):
        dfDict[f"col{i+1}"] = X_pca[:, i]

    pcaDf = pd.DataFrame.from_dict(dfDict)

    # print(pca.explained_variance_ratio_)
    # print(X_pca)

    return pcaDf

def graphInertias(clusterRange, inertias, numColumns):
    plt.figure()
    plt.plot(clusterRange, inertias, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig(f'inertias\\{numColumns}_cols.png')
    plt.close()

def graphPcaSimilarity(pcaData: pd.DataFrame, title: str):

    # Create two subplots and unpack the output array immediately
    cols = pcaData.columns
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle(f"Cluster Information Retention Post PCA {title}")
    ax1.plot(pcaData.index, pcaData[cols[0]], 'bo-')
    ax1.set_ylabel('Adjusted Random Score')
    ax2.plot(pcaData.index, pcaData[cols[1]], 'go-')
    ax2.set_ylabel('Normalized Mutual Info')
    plt.tight_layout()
    plt.savefig(f'pca\\{title}.png')
    plt.close()
    
def mapClusters(labelledData: pd.DataFrame, fileName: str, verbal: bool = True):
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
        clusterCoordsFigure(labelledData, summaryDF, fileName)
        clusterPriceVsLongBias(summaryDF, fileName)
    
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
    foliumMap.save(f'folium_maps\\pca_clustering\\{fileName}_map.html')

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def clusterCoordsFigure(labelledData: pd.DataFrame, clusterDF: pd.DataFrame, fileName: str, folderPath: str = 'cluster_graphs\\cluster_center_coords'):
    plt.figure()
    for label in labelledData['clusterLabels'].unique():
        toPlot = labelledData.loc[labelledData['clusterLabels'] == label]
        plt.scatter(toPlot['longitude'], toPlot['latitude'], s=1, alpha=0.1)
    plt.scatter(clusterDF['Long'], clusterDF['Lat'], color='red')
    plt.title("Cluster Biases (km)")
    plt.xlabel("Longitude Bias (km)")
    plt.ylabel("Latitude Bias (km)")
    plt.grid(True)
    plt.savefig(f'{folderPath}\\{fileName}.png')
    plt.close()

def clusterPriceVsLongBias(clusterDF: pd.DataFrame, fileName: str):
    plt.figure()
    plt.scatter(clusterDF['Long Bias'], clusterDF['Price'].str.replace(',', '').astype(float), c='blue')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Longitudinal Bias (km from mean)")
    plt.ylabel("Average Price (£)")
    plt.title("Property Price vs. Longitude Bias")
    plt.grid(True)
    plt.savefig(f'cluster_graphs\\price_vs_long_bias\\{fileName}.png')
    plt.close()

def dbScan(data: pd.DataFrame, mapData: pd.DataFrame, targetDimension: int = 5, neighbourCount: int = 5, findElbow: bool = False, autoEps: bool = False, epsPadding: float = .5, nEpsSteps: int = 5, nSamples: int = 10000):
    mapData, data = combineAndSample(mapData, data, nSamples)
    ogData = data.copy()
    mapData.reset_index(drop=True, inplace=True)
    data = applyPCA(data, targetDimension)

    if findElbow:
        findElbowFn(data, neighbourCount)

    if autoEps:
        e = getEps(data, neighbourCount, False)
        epsVals = np.linspace(e * (1 - epsPadding), e * (1 + epsPadding), nEpsSteps)
    else:
        epsVals = [.5, .8, 1, 1.2, 1.5, 2]

    for e in epsVals:
        db = DBSCAN(eps=e, min_samples=neighbourCount).fit(data)
        labels = db.labels_
        # unique, counts = np.unique(labels, return_counts=True)
        # print(e)
        # print(np.asarray((unique, counts)).T)

        mapClusters(pd.concat([mapData, pd.Series(labels, name='clusterLabels')], axis=1), f"{targetDimension}d_PCA_{e}_eps_{neighbourCount}_samples_{nSamples}_pop")
    visualizeCorrelations(pd.concat([ogData, data], axis=1), f"{targetDimension}d_PCA_correlation")

def dbScan2D(data: pd.DataFrame, mapData: pd.DataFrame, nSamples: int = 10000, minSamples: int = 5):
    if nSamples:
        mapData, data = combineAndSample(mapData, data, nSamples)
    data = applyPCA(data, 2)

    # findElbowFn(data, minSamples)
    eps = getEps(data, minSamples, False)

    db = DBSCAN(eps=eps, min_samples = minSamples).fit(data)
    labels = db.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print(np.asarray((unique, counts)).T)

    scatterData = pd.concat([data, pd.Series(labels, name='clusterLabels')], axis=1)
    plt.figure()
    for label in unique:
        scatterLabel = scatterData.loc[scatterData['clusterLabels'] == label]
        plt.scatter(scatterLabel.iloc[:, 0], scatterLabel.iloc[:, 1], label=label)
    plt.title("2d PCA")
    plt.legend()
    plt.show()

    mapClusters(pd.concat([mapData.reset_index(drop=True), pd.Series(labels, name='clusterLabels')], axis=1), f"2d_PCA_{eps}_eps_{minSamples}_samples_{nSamples}_pop")

def getEps(data: pd.DataFrame, n: int, plot: bool = True) -> float:
    neigh = NearestNeighbors(n_neighbors=n)
    nbrs = neigh.fit(data)
    kDistances = np.sort(nbrs.kneighbors(data)[0][:, n - 1])

    nPoints = len(kDistances)

    # Create line from first to last point
    allIndices = np.arange(nPoints)
    firstPoint = np.array([0, kDistances[0]])
    lastPoint = np.array([nPoints - 1, kDistances[-1]])
    
    # Compute distances to the line
    lineVec = lastPoint - firstPoint
    lineVecNorm = lineVec / np.linalg.norm(lineVec)
    
    vecFromFirst = np.stack([allIndices, kDistances], axis=1) - firstPoint
    scalarProj = np.dot(vecFromFirst, lineVecNorm)
    proj = np.outer(scalarProj, lineVecNorm)
    vecToLine = vecFromFirst - proj
    distToLine = np.linalg.norm(vecToLine, axis=1)
    
    # Elbow point is where distance to line is maximum
    elbowIdx = np.argmax(distToLine)
    elbowEps = kDistances[elbowIdx]

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(kDistances, label=f"{n}-NN Distances")
        plt.axvline(x=elbowIdx, color='red', linestyle='--', label=f"Elbow @ index {elbowIdx}")
        plt.axhline(y=elbowEps, color='green', linestyle='--', label=f"eps ≈ {elbowEps:.2f}")
        plt.scatter(elbowIdx, elbowEps, color='black')
        plt.xlabel("Sorted data points")
        plt.ylabel(f"{n}-NN Distance")
        plt.title("Automatic Elbow Detection")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return elbowEps

def compareDbScanKmeansLabels(data: pd.DataFrame, showHeatmaps: bool = False, dropOutliers: bool = False, pcaDimension: int = 5, nSamples: int = 10000):
    data = data.sample(nSamples, random_state=0)
    if pcaDimension:
        data = applyPCA(data, pcaDimension)

    minSamplesList = [5, 7, 9]
    colNum = 5

    scanGrid = buildDbscanParamGrid(data, minSamplesList, nEpsSteps=colNum)

    summaryDfArs, summaryDfNmi = pd.DataFrame(columns=range(colNum), index=minSamplesList, dtype=np.float64), pd.DataFrame(columns=range(colNum), index=minSamplesList, dtype=np.float64)
    numLabelsDf, epsValuesDf = pd.DataFrame(columns=range(colNum), index=minSamplesList, dtype=np.int64), pd.DataFrame(columns=range(colNum), index=minSamplesList, dtype=np.int64)

    for minSamples, epsList in scanGrid.items():
        for i, eps in enumerate(epsList):
            dbLabels = DBSCAN(eps=eps, min_samples = minSamples).fit(data).labels_
            if dropOutliers:
                filteredData = pd.concat([data.reset_index(drop=True), pd.Series(dbLabels, name='dbLabels')], axis=1)
                filteredData = filteredData.loc[filteredData['dbLabels'] != -1].iloc[:, :-1]
                dbLabels = dbLabels[dbLabels != -1]
                kmLabels = KMeans(n_clusters=(len(np.unique(dbLabels))), random_state=0).fit(filteredData).labels_
            else:
                kmLabels = KMeans(n_clusters=(len(np.unique(dbLabels))), random_state=0).fit(data).labels_
            
            numLabelsDf.loc[minSamples, i] = len(np.unique(dbLabels))
            epsValuesDf.loc[minSamples, i] = eps
            summaryDfArs.loc[minSamples, i] = adjusted_rand_score(dbLabels, kmLabels)
            summaryDfNmi.loc[minSamples, i] = normalized_mutual_info_score(dbLabels, kmLabels)

    rowIdx, colIdx = np.unravel_index(summaryDfNmi.values.argmax(), summaryDfNmi.shape)

    hyperParams = (f"{nSamples} Samples / PCA{pcaDimension}" if pcaDimension else "") + (" / Dropped Outliers" if dropOutliers else "")
    print(hyperParams)
    print(f"NMI:\n{summaryDfNmi}\nMax of {summaryDfNmi.loc[minSamplesList[rowIdx], colIdx]:.2f} at {minSamplesList[rowIdx]}, {colIdx}\n")
    rowIdx, colIdx = np.unravel_index(summaryDfArs.values.argmax(), summaryDfArs.shape)
    print(f"ARS:\n{summaryDfArs}\nMax of {summaryDfArs.loc[minSamplesList[rowIdx], colIdx]:.2f} at {minSamplesList[rowIdx]}, {colIdx}\n")
    print(f"Num Labels:\n{numLabelsDf}\n")
    print(f"Epsilons:\n{epsValuesDf}\n")

    if showHeatmaps:
        buildHeatmap(summaryDfNmi, "Normalized Mutual Info Scores" + hyperParams)
        buildHeatmap(summaryDfArs, "Adjusted Random Scores" + hyperParams)

def buildDbscanParamGrid(data: pd.DataFrame, minSamplesList: list = [3, 5, 7, 9], epsPadding: float = 0.3, nEpsSteps: int = 5) -> dict:
    
    paramGrid = {}

    for minSamples in minSamplesList:
        base_eps = getEps(data, minSamples, plot=False)

        # Create a range of eps values ±epsPadding (e.g. ±20%)
        paramGrid[minSamples] = np.linspace(base_eps * (1 - epsPadding), base_eps * (1 + epsPadding), nEpsSteps)

    return paramGrid

def buildHeatmap(data: pd.DataFrame, title: str, xTitle: str = "Espilons", yTitle: str = "Min Samples"):
    plt.figure()
    ax = seaborn.heatmap(data)
    ax.set_title(title)   
    ax.set_xlabel(xTitle)
    ax.set_ylabel(yTitle)
    plt.show()
    plt.close()

def combineAndSample(d1: pd.DataFrame, d2: pd.DataFrame, sampleNum: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    numCols = d1.shape[1]
    combined = pd.concat([d1, d2], axis=1)
    combined = combined.sample(sampleNum)

    return combined.iloc[:, :numCols], combined.iloc[:, numCols:]

def findElbowFn(data: pd.DataFrame, neighbourCount: int):
    neigh = NearestNeighbors(n_neighbors=neighbourCount)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)
    k_distances = np.sort(distances[:, neighbourCount - 1])
    plt.figure()
    plt.plot(k_distances)
    plt.xlabel("Data Points sorted by distance")
    plt.ylabel(f"{neighbourCount}-NN Distance")
    plt.title("K-Distance Graph")
    plt.grid(True)
    plt.show()