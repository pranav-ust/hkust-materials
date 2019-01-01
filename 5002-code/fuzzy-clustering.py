from numpy import genfromtxt
import numpy as np

temp = genfromtxt('Q2Q3_input.csv', delimiter=',') #Generate the numpy array
inp = temp[1:, 1:]
mean = inp.mean(axis = 0)
std = inp.std(axis = 0)
#normed = (inp - inp.mean(axis = 0)) / inp.std(axis = 0)
normed = inp

def sq_dist(a, b):
	return np.sum((a - b) ** 2, axis = 1) #will return squared distance

c1 = np.array([1,1,1,1,1,1])
c2 = np.array([0,0,0,0,0,0])

for i in range(50):
	d_a = sq_dist(c1, normed) #Distance from cluster to points
	d_b = sq_dist(c2, normed)

	w1 = np.divide(d_b, d_a + d_b) #Find the fuzzy weights
	w2 = np.divide(d_a, d_a + d_b)

	sse = np.sum(np.multiply(d_a, w1) + np.multiply(d_b, w2)) #Calculate SSE

	w_1 = w1.reshape(695, 1)
	w_2 = w2.reshape(695, 1)

	w1_sq = np.multiply(w_1**2, np.ones((695, 6))) #Broadcasting
	w2_sq = np.multiply(w_2**2, np.ones((695, 6)))

	w1_m = np.multiply(w1_sq, normed) #M step starts here
	w2_m = np.multiply(w2_sq, normed) #This is the numerator

	new_c1 = np.sum(w1_m, axis = 0) / np.sum(w1**2) #Sum up the numerator and divide it by weights
	new_c2 = np.sum(w2_m, axis = 0) / np.sum(w2**2)
	dist = np.sum(np.abs(new_c1 - c1)) + np.sum(np.abs(new_c2 - c2))

	print(sse)
	print(new_c1)
	print(new_c2)

	if dist < 0.001: #break conditions
		break

	c1 = new_c1
	c2 = new_c2
