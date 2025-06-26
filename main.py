import processing, clusterMapping, anomalyDetection
import pandas as pd

def generateClusters(data: pd.DataFrame, info: pd.DataFrame):
    clusterMapping.pcaCompare(processed, info, kRange=range(1, 15, 3), stop = 0, step=15)

def dbScan(data: pd.DataFrame, mapData: pd.DataFrame):
    clusterMapping.dbScan(data, mapData)

def anomalyDetect(data: pd.DataFrame, info: pd.DataFrame):
    # anomalyDetection.isolationForest(processed, info)
    anomalyDetection.randomForest(data, info)

if __name__ == "__main__":
    data = pd.read_csv("data/kaggle_london_house_price_data.csv")
    processed, info = processing.processData(data)
    generateClusters(processed, info)
    # anomalyDetect(processed, info)
    # dbScan(processed, info)
    # clusterMapping.compareDbScanKmeansLabels(processed, info)