from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN

from collections import deque
import numpy as np

euclidean = lambda x,y: np.sqrt(sum((x - y)**2))

def DBSCAN_mine(X, dist_func=euclidean, eps=0.3, min_samples=10):
    n = len(X)

    all_dist = {}

    for i in range(n):
        for j in range(i, n):
            all_dist[(i, j)] = dist_func(X[i], X[j])

    def get_dist(a, b):
        if a > b:
            return all_dist[(b, a)]
        else:
            return all_dist[(a, b)]

    def get_ngbr(p):
        ngbr = set()
        for q in range(n):
            if p != q and get_dist(p, q) <= eps:
                ngbr.add(q)
        return ngbr

    ngbrhood = {}
    noise = set()
    for p in range(n):
        ngbrhood[p] = get_ngbr(p)
        # + 1 to include self
        if len(ngbrhood[p]) + 1 < min_samples:
            noise.add(p)

    c = 0
    labels_map = {}
    for p in range(n):
        if p in labels_map or p in noise:
            continue
        c += 1

        queue = deque()
        queue.append(p)

        while len(queue) > 0:
            v = queue.popleft()
            if v in labels_map:
                continue

            labels_map[v] = c

            if v not in noise:
                for r in ngbrhood[v]:
                    queue.append(r)

    labels = []
    for p in range(n):
        if p not in labels_map:
            labels.append(-1)
        else:
            labels.append(labels_map[p])
    return labels

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

labels = DBSCAN_mine(X, eps=0.3, min_samples=10)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('*'*20)
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))