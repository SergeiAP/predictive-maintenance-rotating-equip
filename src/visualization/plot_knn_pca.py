from tqdm.autonotebook import tqdm
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def plot_elbow(df: pd.DataFrame,
               seed: int,
               first_cluster: int = 2,
               clusters: int = 10,
               n_init: int = 3,
               max_iter: int = 5,
               metric: str = "euclidean",
               n_jobs: int = 6):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        seed (int): _description_
        first_cluster (int, optional): _description_. Defaults to 2.
        clusters (int, optional): _description_. Defaults to 10.
        n_init (int, optional): _description_. Defaults to 3.
        max_iter (int, optional): _description_. Defaults to 5.
        metric (str, optional): _description_. Defaults to "euclidean".
        n_jobs (int, optional): _description_. Defaults to 6.
    """
    # Сумма кв расстояний от объектов до центра кластера, взвешенная по весам (при их наличии)
    distortions = []
    silhouette = []
    K = range(first_cluster, first_cluster + clusters) # num of clusters
    for k in tqdm(K):
        kmeanModel = TimeSeriesKMeans(n_clusters=k,
                                      metric=metric,
                                      n_jobs=n_jobs,
                                      max_iter=max_iter,
                                      n_init=n_init,
                                      random_state=seed)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)
        # silhouette_score считает насколько чисты кластеры
        silhouette.append(silhouette_score(df, kmeanModel.labels_, metric=metric))
        
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(K, distortions, 'b-')
    ax2.plot(K, silhouette, 'r-')

    ax1.set_xlabel('# clusters')
    ax1.set_ylabel('Distortion', color='b')
    ax2.set_ylabel('Silhouette', color='r')


def plot_and_get_pca(df: pd.DataFrame,
                     seed: int,
                     is_plot: bool = False,
                     explained_tresh: float = 0.95) -> np.ndarray:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        seed (int): _description_
        is_plot (bool, optional): _description_. Defaults to False.
        explained_tresh (float, optional): _description_. Defaults to 0.95.

    Returns:
        np.ndarray: _description_
    """
    pca = PCA(random_state=seed)
    pca.fit(df)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    components_threshold = np.argwhere(explained_variance > explained_tresh).reshape(-1)[0]
    
    pca = PCA(n_components=components_threshold)
    pca_transformed = pca.fit_transform(df)
    print("Explained variance of 2 components", np.sum(pca.explained_variance_ratio_[:2]))
    if is_plot:
        plt.vlines(components_threshold, explained_variance.min(), explained_tresh, linestyle='dashed')
        plt.plot(explained_variance)
        plt.show()
        
    return pca_transformed
        
def plot_tsne(predicted: pd.Series, representation: np.ndarray):
    """tSNE plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    scatter = ax.scatter(representation[:, 0], representation[:, 1], c=predicted)
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)
    
    ax.set_title("Predicted clusters")
    plt.legend()
    plt.show()
