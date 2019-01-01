import re
import matplotlib.pyplot as plt
from pylab import savefig
import numpy as np

def learning_curve(filename):
	f = open(filename, 'r+')
	loss = []
	pre = []
	for line in f:
		figures = re.findall("\d+\.\d+", line.rstrip()) #Extract decimals from a string
		loss.append(float(figures[0])) #First number is the loss
		pre.append(float(figures[1]))
	print("Loss array is ", loss)
	print("Precision array is ", pre)
	y1_loss = loss[::2] #x1 is the training numbers at even positions
	y2_loss = loss[1::2] #x2 is the validation numbers at odd positions
	y1_pre = pre[::2] #x1 is the training numbers at even positions
	y2_pre = pre[1::2] #x2 is the validation numbers at odd positions
	x = np.arange(1, len(y1_loss) + 1, 1)
	#Plot loss learning curve
	loss = plt.figure(1)
	plt.plot(x, y1_loss, 'r', label="Training")
	print(x, y2_loss)
	plt.plot(x, y2_loss, 'g', label="Validation")
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.legend(loc='upper right')
	loss = plt.gcf()
	plt.draw()
	loss.savefig('Loss Learning Curve.pdf')
	plt.close()
	#Plot precision learning curve
	pre = plt.figure(2)
	plt.plot(x, y1_pre, 'r', label="Training")
	plt.plot(x, y2_pre, 'g', label="Validation")
	plt.xlabel('Iterations')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	pre = plt.gcf()
	plt.draw()
	plt.ylim([0.85, 1.0])
	pre.savefig('Accuracy Learning Curve.pdf')
	plt.close()

learning_curve("Iterations.txt")
