import processing, clusterMapping
import pandas as pd

if __name__ == "__main__":
    myData = pd.read_csv("data/kaggle_london_house_price_data.csv")
    # print(myData.head())
    # print(myData.shape)
    print(myData.columns.values)
    # print(myData.dtypes)
    # print(myData.isna().sum()/ my_data.shape[0])
    data, map = processing.processData(myData)
    clusterMapping.pcaCompare(data, map, stop = 10)