import processing, clustering
import pandas as pd

if __name__ == "__main__":
    myData = pd.read_csv("data/kaggle_london_house_price_data.csv")
    # print(myData.head())
    # print(myData.shape)
    print(myData.columns.values)
    # print(myData.dtypes)
    # print(myData.isna().sum()/ my_data.shape[0])
    data, map = processing.processData(myData)

    clustered = clustering.applyPCA(data, 2)
    print(clustered)
