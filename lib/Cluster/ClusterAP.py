# sklearn
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

def APWithSimilaryMatrix(similaryMatrix):
	p = np.mean(similaryMatrix) * 2
	af = AffinityPropagation(max_iter=2000, preference = p, affinity = 'precomputed')
	af.fit(similaryMatrix)
	return (af.cluster_centers_indices_, af.labels_)