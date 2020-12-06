import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

def elbowgraph(knndf):
    distortions = []
    for k in range(1, 10):
        kmm = KMeans(n_clusters=k)
        kmm.fit(knndf)
        #plotted against inertia value which is the distortion value which represents
        #the sum of the squared distances between each observation and its dominating centroid
        inertiavalue = kmm.inertia_
        distortions = distortions + [inertiavalue]

    plt.plot(range(1,10),distortions,'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()

def scorelist(finaldf):
    lendf = len(finaldf.columns)-1
    score = []
    name = []

    for i in range(0,lendf):
        name = name +[finaldf.columns[i]]
        linreg = linear_model.LinearRegression()
        x = finaldf[finaldf.columns[i]].values
        y = finaldf[finaldf.columns[13]].values
        #had to reshape for linear Regression to work
        x = x.reshape(506, 1)
        y = y.reshape(506, 1)
        reg1 = linreg.fit(x, y)
        sc1 = reg1.score(x, y)
        score = score + [sc1]


    finald = {'Name': name, 'Score': score}
    outputdf = pd.DataFrame(finald)
    #outputs the table from highest to lowest absolute values of scores
    outputdf= outputdf.iloc[(-np.abs(outputdf['Score'].values)).argsort()]
    return(outputdf)



if __name__ == '__main__':
    bostondata = load_boston()
    bostondf = pd.DataFrame(bostondata.data, columns=bostondata.feature_names)
    bostondf['MEDV'] = bostondata.target
    print(scorelist(bostondf))

    wine = load_wine()
    winedf = pd.DataFrame(wine['data'])
    elbowgraph(winedf)

    iris = load_iris()
    irisdf = pd.DataFrame(iris['data'])
    elbowgraph(irisdf)
