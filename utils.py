import folium, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn
from scipy.stats import norm

clusterColours = ['lightblue', 'gray', 'blue', 'darkred', 'lightgreen', 'purple', 'red', 'green', 'lightred', 'white', 'darkblue', 'darkpurple', 'cadetblue', 'orange', 'pink']

def mapLabelledGroups(labelledData: pd.DataFrame, fileName: str, clusterGroups: dict, meanLat: float, meanLon: float, nSamples: int = 1000, folderName = 'pca_clustering', labelColumnName = 'clusterLabels'):
    colourMap = {}
    colourI = 0

    def getColourInt(label: str):
        try:
            return colourMap[label]
        except KeyError:
            nonlocal colourI
            colourMap[label] = colourI
            colourI += 1
            return colourMap[label]
    
    def getAddInfo(items):
        popupInfo = ""
        for l, v in items:
            popupInfo += f"<br>{l}: {v}"
        return popupInfo
    
    addInfo = False
    defaultEnd = labelledData.columns.get_loc('saleEstimate_currentPrice')
    end = len(labelledData.columns) - 1
    if end > defaultEnd:
        addInfo = True
        defaultEnd += 1

    labelledData = labelledData.sample(n=nSamples)
    for _, row in labelledData.iterrows():
        popupInfo = f"""
            <b>{row['fullAddress']}</b><br>
            Cluster: {row[labelColumnName]}<br>
            Bedrooms: {row['bedrooms']}<br>
            Bathrooms: {row['bathrooms']}<br>
            Square Footage: {row['floorAreaSqM']}<br>
            Sales Price Estimate: {row['saleEstimate_currentPrice']}{getAddInfo(row.iloc[defaultEnd:end].items()) if addInfo else ''}
        """

        marker = folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popupInfo,
            icon=folium.Icon(color=clusterColours[getColourInt(row[labelColumnName]) % len(clusterColours)])
            )
        clusterGroups[row[labelColumnName]].add_child(marker)

    
    foliumMap = folium.Map(location=[meanLat, meanLon], zoom_start=12)
    # Add all FeatureGroups to the map
    for group in clusterGroups.values():
        group.add_to(foliumMap)

    # Add layer control to toggle visibility
    folium.LayerControl(collapsed=False).add_to(foliumMap)
    foliumMap.save(f'folium_maps\\{folderName}\\{fileName}_map.html')

def buildHeatmap(data: pd.DataFrame, title: str, xTitle: str = "Espilons", yTitle: str = "Min Samples", saveFlag: bool = False, fileName: str = "heatmap"):
    plt.figure()
    ax = seaborn.heatmap(data)
    ax.set_title(title)   
    ax.set_xlabel(xTitle)
    ax.set_ylabel(yTitle)
    if saveFlag:
        plt.savefig(f'heatmaps\\{fileName}.png', dpi=400, bbox_inches="tight")
    else:
        plt.show(bbox_inches="tight")
    plt.close()

def plotGaussian(percent_diff):
    # Compute mean and std
    mu, std = norm.fit(percent_diff)

    # Plot histogram
    plt.figure()
    plt.hist(percent_diff, bins=30, density=True, alpha=0.6, color='skyblue')

    # Generate fitted Gaussian curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2)

    plt.title(f"Gaussian Fit: μ = {mu:.2f}, σ = {std:.2f}")
    plt.xlabel("Percent Difference")
    plt.ylabel("Density")
    plt.show()