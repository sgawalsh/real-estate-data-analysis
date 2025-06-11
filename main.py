import processing, clusterMapping, anomalyDetection
import pandas as pd

def generateClusters(data: pd.DataFrame):
    processed, info = processing.processData(data)
    clusterMapping.pcaCompare(processed, info, stop = 10)

def anomalyDetect(data: pd.DataFrame):
    processed, info = processing.processData(data)
    # anomalyDetection.isolationForest(processed, info)
    anomalyDetection.randomForest(processed, info)

if __name__ == "__main__":
    data = pd.read_csv("data/kaggle_london_house_price_data.csv")

    # generateClusters(data)
    anomalyDetect(data)