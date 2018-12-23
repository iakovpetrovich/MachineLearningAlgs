# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:29:27 2018

@author: jasap
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

data = pd.read_csv('C:\\Users\\jasap\\Documents\\MASTER\\AlgMasUc\\2. Linear regression\\Domaci\\Boston_Housing.txt',
                   sep='\t')


def euclidean(data, centroids):
    return cdist(data, centroids, metric='euclidean')


def city_block(data, centroids):
    return cdist(data, centroids, metric='cityblock')


def sqeuclidean(data, centroids):
    return cdist(data, centroids, metric='sqeuclidean')


def chebyshev(data, centroids):
    return cdist(data, centroids, metric='chebyshev')


def normalize_data(data):
    normalized_data = (data - data.mean()) / data.std()
    return normalized_data


def init_centroids(data, k):
    norm_data = normalize_data(data)
    centroids = norm_data.sample(1)
    for i in range(1, k):
        temp_data = norm_data.drop(centroids.index)
        next_centroid = norm_data.iloc[cdist(temp_data, centroids).sum(axis=1).argmax(), :]
        centroids = centroids.append(next_centroid)
    return centroids


def cluster(data, k=2, ponder=[], dist_fun=euclidean, max_iter=20):
    norm_data = normalize_data(data)
    n, m = norm_data.shape
    if ponder != [] and len(ponder) == m:
        norm_data *= ponder
    centroids = init_centroids(norm_data, k).reset_index(drop=True)
    assignment = np.zeros(n)
    old_quality = float('inf')
    quality = np.zeros(k)

    for it in range(max_iter):
        distance = dist_fun(norm_data, centroids)
        for index, row in norm_data.iterrows():
            # assignment[index] = cdist(pd.DataFrame(row).transpose(), centroids, metric='cityblock').argmin()
            # assignment[index] = dist_fun(row, centroids)
            assignment[index] = distance[index].argmin()
        for index in range(len(centroids)):
            subset = norm_data[assignment == index]
            centroids.loc[index] = subset.mean()
            quality[index] = subset.var().sum() * len(subset)

        print('Iteration:', it, ' Quality:', quality.sum())

        if abs(old_quality - quality.sum()) < 0.1:
            break

        old_quality = quality.sum()
    # print(' Quality:', quality.sum())
    return quality.sum(), assignment, centroids


def KMeans(data, k=2, ponder=[], dist_fun=euclidean, max_iter=20, max_clusterizations=5):
    n, m = data.shape
    best_quality = float('inf')
    best_assignment = np.array(n)
    best_centroids = None

    for it in range(max_clusterizations):
        #print('Cluster NO: ', it)
        q, a, c = cluster(data, k, ponder, dist_fun, max_iter)
        if q <= best_quality:
            best_quality = q
            best_assignment = a
            best_centroids = c

    return best_quality, best_assignment, best_centroids


def find_k(data, max_k=10, dist_function=sqeuclidean):
    norm_data = normalize_data(data)
    result = {}
    for i in range(2, max_k):
        quality, assignment, centroids = KMeans(data, i, dist_fun=dist_function)
        a = 0
        b = 0
        for j in range(i):
            a += cdist(norm_data[assignment == j], norm_data[assignment == j]).mean()
            index_of_closest_centroid = cdist(pd.DataFrame(centroids.iloc[j, :]).transpose(),
                                              centroids.drop(j)).argmin()
            b += cdist(norm_data[assignment == j], norm_data[assignment == index_of_closest_centroid]).mean()
        sil_score = (b - a) / max(a, b)
        result['k:'+str(i)] = sil_score
        print('k: ', i, ' sil. score:', sil_score)
    return result

quality, assignment, centroids = KMeans(data, k=3, dist_fun=city_block)
rez =find_k(data, 10, sqeuclidean)
