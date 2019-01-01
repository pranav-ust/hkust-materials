#Coded by Pranav
from pprint import pprint

global S
global E
global match
global mismatch
global MIN
S   = -4. #Cost of opening
E   = -1. #Cost of extending
match = 1.
mismatch = -1.
MIN = -float("inf")

#return match or mismatch score
def _match(s, t, i, j):
	if t[i-1] == s[j-1]:
		return match
	else:
		return mismatch

#initializers for matrices
def _init_x(i, j):
	if i > 0 and j == 0:
		return MIN
	else:
		if j > 0:
			return S + (E * j)
		else:
			return 0

def _init_y(i, j):
	if j > 0 and i == 0:
		return MIN
	else:
		if i > 0:
			return S + (E * i)
		else:
			return 0

def _init_m(i, j):
	if j == 0 and i == 0:
		return 0
	else:
		if j == 0 or i == 0:
			return MIN
		else:
			return 0

def _format_tuple(inlist, i, j):
	return 0

def distance_matrix(s, t):
	dim_i = len(t) + 1
	dim_j = len(s) + 1
	#abuse list comprehensions to create matrices
	X = [[_init_x(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]
	Y = [[_init_y(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]
	M = [[_init_m(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]

	for j in range(1, dim_j):
		for i in range(1, dim_i):
			X[i][j] = max((S + E + M[i][j-1]), (E + X[i][j-1]), (S + E + Y[i][j-1]))
			Y[i][j] = max((S + E + M[i-1][j]), (S + E + X[i-1][j]), (E + Y[i-1][j]))
			M[i][j] = max(_match(s, t, i, j) + M[i-1][j-1], X[i][j], Y[i][j])

	return [X, Y, M]

def backtrace(s, t, X, Y, M):
	sequ1 = ''
	sequ2 = ''
	i = len(t)
	j = len(s)
	while (i>0 or j>0):
		if (i>0 and j>0 and M[i][j] == M[i-1][j-1] + _match(s, t, i, j)):
			sequ1 += s[j-1]
			sequ2 += t[i-1]
			i -= 1; j -= 1
		elif (i>0 and M[i][j] == Y[i][j]):
			sequ1 += '_'
			sequ2 += t[i-1]
			i -= 1
		elif (j>0 and M[i][j] == X[i][j]):
			sequ1 += s[j-1]
			sequ2 += '_'
			j -= 1

	sequ1r = ' '.join([sequ1[j] for j in range(-1, -(len(sequ1)+1), -1)])
	sequ2r = ' '.join([sequ2[j] for j in range(-1, -(len(sequ2)+1), -1)])

	return [sequ1r, sequ2r]

[X, Y, M] = distance_matrix("AAT", "ACACT")
[str1, str2] = backtrace("AAT", "ACACT", X, Y, M)
print("-=Alignment=-")
print(str1)
print(str2)
print("\nThe Ix matrix is\n")
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in X]))
print("\nThe Iy matrix is\n")
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in Y]))
print("\nThe M matrix is\n")
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in M]))
print("\n")
