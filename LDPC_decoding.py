from numpy.random import random
import numpy
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import csv
import pickle
# decoder
def bitFlippingDecoding(y, B, maxIterations=100):
	M = y.copy()
	i = 0
	
	while i < maxIterations:
		E = []
		### getting checks ###
		for m in range(len(B)):
			E.append([])
			for n in range(len(B[m])):
				if B[m][n] == -1:
					continue
				### summing ###
				summation = 0
				for bit in B[m]:
					if bit != B[m][n]:
						summation += M[bit-1]
				E[m].append(summation % 2)
		'''
		creating bitFlips object to check whether 0s or 1s are
		the majority for each bit. Then filling up the object with
		the number occurences of 0s and 1s for each bit
		'''
		bitFlips = {}
		for b in range(1,len(M)+1):
			bitFlips[b] = {0:0, 1:0}

		for m in range(len(B)):
			for n in range(len(B[m])):
				nth_bit = B[m][n]
				if nth_bit == -1:
					continue
				bitFlips[nth_bit][E[m][n]] += 1

		### checking and flipping bits ###
		for bit in bitFlips.keys():
			one  = bitFlips[bit][1]
			zero = bitFlips[bit][0]
			if one > zero and M[bit-1] == 0:
				M[bit-1] = 1
			elif zero > one and M[bit-1] == 1:
				M[bit-1] = 0

		### parity-check ###
		L = []

		for row in E:
			L.append(sum(row) % 2)

		if sum(L) == 0:
			break
		else:
			i += 1
	return M
def sumProductDecoding(r, H, maxIterations=100):
	
	I = 0
	M = []
	B = hammToB(H)

	# initialization
	for i in range(len(B)):
		M.append([0]*len(r))
	for i in range(len(B)):
		for j in B[i]:
			M[i][j-1] = r[j-1]

	while I < maxIterations:

		# initialize E
		E = []
		for i in range(len(B)):
			E.append([0]*len(r))

		# get checks
		for i in range(len(E)):
			for j in range(len(E[i])):
				if M[i][j] == 0:
					continue
				product = 1.0
				for n in B[i]:
					if n-1 != j:
						product *= numpy.tanh(M[i][n-1] / 2)
				if product == 1:
					print("here!", M)
				E[i][j] = float(numpy.log((1 + product)/(1 - product)))

		
		L = r.copy()

		for i in range(len(E)):
			for j in range(len(E[i])):
				L[j] += E[i][j]


		# converting to binary format
		z = []
		for bit in L:
			if bit > 0:
				z.append(0)
			else:
				z.append(1)
		s = []

		for i in range(len(H)):
			total = numpy.dot(H[i], z)
			s.append(total % 2)
		if sum(s) == 0:
			return z
		else:
			# initializing M
			M = []
			for i in range(len(E)):
				M.append([0]*len(r))

			# setting up M from E and r
			for i in range(len(E)):
				for j in range(len(E[i])):
					val = 0
					if E[i][j] != 0:
						for row in range(len(E)):
							if row != i:
								val += E[row][j]
						M[i][j] = val + r[j]

						# limit bit to check values
						if M[i][j] < 0 and M[i][j] < -20:
							M[i][j] = -20
						elif M[i][j] > 0 and M[i][j] > 20:
							M[i][j] = 20

		I += 1

	return z
def BSC(y, error):
	error = error / 100
	flip_locs = (random(len(y)) <= error)
	for i in range(len(y)):
		if flip_locs[i]:
			y[i] = (y[i] + 1) % 2
	return y
def errorCounter(x, y):
	bitErrors = 0
	blockErrors = 0
	for i in range(len(y)):
		if x[i] != y[i]:
			bitErrors += 1
	if bitErrors > 0:
		blockErrors = 1

	return [bitErrors, blockErrors]
# returns [p of biterrors, p of blockerrors]	
def hardDecisionSimulation(B, error, lengthOfCode, maxBlockErros = 100):
	x = [0]*lengthOfCode
	bitErrors = 0
	blockErrors = 0
	i = 0

	while blockErrors < maxBlockErros:

		# pass through BSC
		y = BSC([0]*lengthOfCode, error)

		# decode for errors
		decodedY = bitFlippingDecoding(y, B)

		# count number of errors the decoder missed
		errors = errorCounter(x, decodedY)

		blockErrors += errors[1]
		bitErrors += errors[0]
		i += 1

	return [bitErrors/i, blockErrors/i]
def softDecisionSimulation(H, r, EbN0dB, maxBlockErros = 100, maxIterations = 200, seed = 0):

	EbN0 = pow(10, EbN0dB / 10)
	EsN0 = EbN0 * r
	EsN0dB = 10*numpy.log(EsN0)

	lengthOfCode = len(H[0])
	x = [0]*lengthOfCode
	bitErrors = 0
	blockErrors = 0
	I = 0

	numpy.random.seed(seed)

	while blockErrors < maxBlockErros and I < maxIterations:

		# pass through AWGN

		y = list(awgn([1]*lengthOfCode, EsN0))

		# decode for errors
		decodedY = sumProductDecoding(y, H)

		# count number of errors the decoder missed
		errors = errorCounter(x, decodedY)

		blockErrors += errors[1]
		bitErrors += errors[0]
		I += 1

	return [bitErrors/I, blockErrors/I]
def hammToB(hamm):
	B = []
	for r in hamm:
		temp = []
		for x in range(len(r)):
			if r[x] == 1:
				temp.append(x+1)
		B.append(temp)

	return B
def BToHamm(B):
	maximum = 0
	for check in B:
		maximum = max(max(check), maximum)
	
	size = len(B)
	hamm = []
	for i in range(size):
		hamm.append([0]*maximum)

	for i in range(len(B)):
		for j in range(len(B[i])):
			hamm[i][B[i][j] - 1] = 1

	return hamm
def getCorruptedCode(filename):
	with open(filename, newline='') as file:
		code = list(csv.reader(file))[0]
	
	for i in range(len(code)):
		code[i] = float(code[i])

	return code
def awgn(data, EsN0):
	variance = 1 / (2 * EsN0)
	dataLength = len(data)

	noise = numpy.random.normal(0, variance, dataLength)

	for i in range(len(data)):
		data[i] += noise[i]

	return data
def printMatrix(matrix):
	for row in matrix:
		print(row)


H = [[1,1,0,1,0,0],
	 [0,1,1,0,1,0],
	 [1,0,0,0,1,1],
	 [0,0,1,1,0,1]]

#print(hammToB(H))

#E = [[0]*len(H[0])]*len(H)
#print(E)
#with open('mat_3_6.txt', 'rb') as f:
#    H = pickle.load(f)

print(softDecisionSimulation(H, 0.5, 0.0005)[0])