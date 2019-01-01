from numpy import genfromtxt
import numpy as np
from scipy.spatial import distance_matrix
import heapq

k = 3
temp = genfromtxt('Q2Q3_input.csv', delimiter=',')
inp = temp[1:, 1:]

#create a distance matrix
mat = distance_matrix(inp, inp, p=2)
k_nearest = mat.argsort()[:, :k+1][:, k] #Indices of k-nearest neigbours
dist_k_o = mat[np.arange(695), k_nearest] #Slice to get distances

#Code to calculate nko
n_k_o = []
for i in range(695):
	n_i_o = []
	for j in range(695):
		if (mat[i, j] <= dist_k_o[i]) and (i != j):
			n_i_o.append(j)
	n_k_o.append(n_i_o)


def reachdist_k_o(o, o_d):
	return max(dist_k_o[o], mat[o, o_d])

#Code to calculate lrd
lrd = []
print(reachdist_k_o(0,1))
for i in range(695):
	lrd.append(len(n_k_o[i]) / sum([reachdist_k_o(j, i) for j in n_k_o[i]]))

#Code to calculate lof
lof = []
for i in range(695):
	lof.append(sum([lrd[j] / lrd[i] for j in n_k_o[i]]) / len(n_k_o[i]))

#Print outliers
outliers = heapq.nlargest(5, range(len(lof)), lof.__getitem__)
print(outliers)
print([lof[i] for i in outliers])
