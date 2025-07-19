import processing, clustering, anomalyDetection
import pandas as pd

def generateClusters(data: pd.DataFrame, info: pd.DataFrame):
    clustering.pcaCompare(data, info, kRange=range(1, 15, 3), stop = 0, step=15)

def dbScan(data: pd.DataFrame, mapData: pd.DataFrame):
    clustering.dbScan(data, mapData)

def anomalyDetect(data: pd.DataFrame, info: pd.DataFrame, rawTargets: pd.Series):
    # anomalyDetection.mapAnomalies(anomalyDetection.isolationForest(data, info))
    preds, y, bestColumn = anomalyDetection.compareModelPreds(data)
    preds, y = processing.descalePreds(preds, y, rawTargets.mean(), rawTargets.std())
    anomalyDetection.mapUnderpriced(preds[bestColumn], y, info)

if __name__ == "__main__":
    data = pd.read_csv("data/kaggle_london_house_price_data.csv")
    rawTargets = data['saleEstimate_currentPrice'].copy()
    processed, info = processing.processData(data)
    generateClusters(processed, info)
    # anomalyDetect(processed, info, rawTargets)
    # dbScan(processed, info)
    # clustering.compareDbScanKmeansLabels(processed)