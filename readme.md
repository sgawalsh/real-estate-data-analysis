# Real Estate Data

This repo applies various data processing methods to a dataset containing information on London real estate listings.

For ease of interpretation, I've added two main notebooks: [Identifying Underpriced Properties](https://nbviewer.org/github/sgawalsh/real-estate-data-analysis/blob/main/identifyingUnderpriced.ipynb) and [Real Estate Cluster Mapping](https://nbviewer.org/github/sgawalsh/real-estate-data-analysis/blob/main/realEstateClusterMapping.ipynb). The first notebook leverages several machine learning methods, models, and an original consensus-finding technique in order to identify and map underpriced properties from our dataset. The cluster mapping notebook details the data-preparation techniques, and the effects of PCA processes on k-means clustering. These clusters are then used as a means of analysing market trends in our dataset.

The repo also contains several folders with various graphs generated depicting trends within the data. The `cluster_graphs` folder contains graphs of clusters mapped using both DBscan and k-means clustering methods, displaying geographic clustering biases. The folder also contains a set of images depicting how the average price of a cluster changes as it moves latitudinally. The `heatmaps` folder contains images depicting alignment between various data used in the repo such as simlarity between k-means and dbscan clusters, a heatmap of model predictions, or a heatmap of normalized column values. Finally, the `inertias` folder contains a simple graph of how the inertia value changes as cluster count is increased for k-means clustering on this dataset.

# How to Run

This project contains four main functions implementing different data analysis techniques. All of these functions are available in the `main.py` file and allow for some flexibility the range of parameters used.

- `generateClusters` investigates the relationship between PCA and k-means clustering on this dataset, seeing how clusters are effected as the number of components and number of clusters is adjusted.
- `anomalyDetect` uses machine learning and consensus-gathering techniques in order to predict underpriced properties from our dataset
- `dbScan` also investigates the effect of PCA transformations, this time against the DBscan algorithm
- `compareDbScanKmeansLabels` investigates the similarity of the clusters generated with various DBscan values against k-means clusters