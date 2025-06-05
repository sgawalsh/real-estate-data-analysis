from sklearn.decomposition import PCA
import pandas as pd

def applyPCA(data: pd.DataFrame, numComponents, idColName="newId"):
    ids = data[idColName]
    data.drop(columns=[idColName], inplace=True)
    pca = PCA(n_components=numComponents)
    X_pca = pca.fit_transform(data)

    dfDict = {}
    for i in range(X_pca.shape[1]):
        dfDict[f"col{i+1}"] = X_pca[:, i]

    pcaDf = pd.DataFrame.from_dict(dfDict)

    print(pca.explained_variance_ratio_)
    print(X_pca)

    return pd.concat([ids, pcaDf], axis=1)