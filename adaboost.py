import numpy as np
y = np.array([-1, 1, 1, 1, -1, -1, 1, -1]) #This is the y vector
a = np.array([-1, -1, 1, 1, 1, 1, 1, -1]) #This is the h1 vector
b = np.array([-1, 1, 1, -1, 1, 1, -1, -1]) #This is h2 vector

length = len(y)
w1 = np.array([1 / length] * length) #Initialize w1 with 1/8
w2 = np.array([1 / length] * length) #Initialize w2 with 1/8
alpha1 = 1
alpha2 = 1
alpha_list = [] #List of alpha to be kept in ensembles
h_list = [] #List of classifiers to be kept in ensembles
for i in range(3):
	print("Iteration ", i+1)
	e1 = np.dot((y != a), w1) #This is error function. Calculate mismatches and multiply with weights
	print("Error 1 is ", e1)
	e2 = np.dot((y != b), w1) #This is error function. Calculate mismatches and multiply with weights
	print("Error 2 is ", e2)
	if e1 < e2:
		alpha1 = 0.5 * np.log((1 - e1)/ e1) #Calculate alpha
		print("Alpha 1 is ",alpha1)
		for j in range(len(y)):
			w1[j] = w1[j] * np.exp(-alpha1 * y[j] * a[j]) #Weight update equation
		print("W1 vector is ", w1)
		w1 = np.divide(w1, np.sum(w1)) #Normalize the weights
		print("W1 vector after normalization  ", w1)
		alpha_list.append(alpha1) #Add alpha to ensemble
		h_list.append(a) #And h to ensemble
	if e1 > e2:
		alpha2 = 0.5 * np.log((1 - e2)/ e2) #Calculate alpha
		print("Alpha 2 is ",alpha2)
		for j in range(len(y)):
			w1[j] = w1[j] * np.exp(-alpha2 * y[j] * b[j]) #Update weights
		print("W2 vector is ", w1)
		w1 = np.divide(w1, np.sum(w1)) #Normalize weights
		print("W2 vector after normalization  ", w1)
		alpha_list.append(alpha2) #Add alpha to ensemble
		h_list.append(b) #Add h to ensemble
	#You have a running list of alphas and h. To calculate the response, multiply them together.
	product = np.array([np.multiply(np.array(h_list[j]), alpha_list[j]) for j in range(len(h_list))])
	#Sum them for the iterations you specified and take the sign of it.
	print(np.sign(np.sum(product, axis=0)) == y)
