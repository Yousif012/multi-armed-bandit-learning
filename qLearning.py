import numpy as np
import copy
import random
import math
import pickle
from LDPC_decoding import awgn, propogateMessage




def quant(x, thre, repre):
	"""Quantization operation. 
	"""
	thre = np.append(thre, np.inf)
	thre = np.insert(thre, 0, -np.inf)
	x_hat_q = np.zeros(np.shape(x))
	for i in range(len(thre)-1):
		if i == 0:
			x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]),
                               np.full(np.size(x_hat_q), repre[i]), x_hat_q)
		elif i == range(len(thre))[-1]-1:
			x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]), 
                               np.full(np.size(x_hat_q), repre[i]), x_hat_q)
		else:
			x_hat_q = np.where(np.logical_and(x > thre[i], x < thre[i+1]), 
                      		   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
	return x_hat_q

# Syndrome Realizations map to index
class RealizationMap:
	def __init__(self):
		self.index = 0
		self.hashMap = {}

	def generateRealizationMap(self, repre, n):
		
		self.hashMap = {}

		arr = ([0] * n)

		self.generateRealizations(n, arr, 0, repre)

		return self.hashMap

	def generateRealizations(self, n, arr, i, repre):

		if i == n:
			self.hashMap[str(arr)] = self.index
			self.index+=1
			return

		for j in repre:
			arr[i] = round(j, 3)
			self.generateRealizations(n, arr, i + 1, repre)

def getMaximumCluster(qTable):
	maxValue = qTable[0][0]
	index = 0

	for i in range(len(qTable)):
		for j in range(len(qTable[0])):
			prevMaxValue = maxValue
			maxValue = max(qTable[i][j], maxValue)
			if prevMaxValue != maxValue:
				index = i

	return index

def getClusterIndex(checkNodeIndex, clusters):
	i = 0
	while i < len(clusters):
		if checkNodeIndex in clusters[i]:
			return i
		i += 1

	return -1

def getSyndromeIndex(syndromeVector, cluster, realizationMap):
	sequence = []
	for index in cluster:
		sequence.append(syndromeVector[index])
	return realizationMap[str(sequence)]

def hammToB(hamm):
	B = []
	for r in hamm:
		temp = []
		for x in range(len(r)):
			if r[x] == 1:
				temp.append(x+1)
		B.append(temp)

	return B

def updateSyndrome(checkToBitEdges, checkNodeIndex, syndromeVector, L, repre, thre):
	neighboursOfCheckNode = checkToBitEdges[checkNodeIndex]

	for vk in neighboursOfCheckNode:
		for i in range(len(checkToBitEdges)):
			if vk in checkToBitEdges[i]:
				for vi in checkToBitEdges[i]:
					syndromeVector[i] += L[vi-1]
				syndromeVector = quant(syndromeVector, thre, repre)

def quantize(S, quantizationRange):

	for i in range(len(S)):
		val = int(S[i])
		if S[i] < 0 and abs(S[i]) > abs(val) + 0.5:
			val-=1
		elif S[i] >= 0 and S[i] > val + 0.5:
			val+=1
		if val not in quantizationRange:
			if val < quantizationRange[0]:
				S[i] = quantizationRange[0]
			else:
				S[i] = quantizationRange[-1]
		else:
			S[i] = val

def getResidual(checkNodeIndex, prevCheckToBit, checkToBit, edges):

	maxValue = abs(checkToBit[checkNodeIndex][edges[0]-1] - prevCheckToBit[edges[0]-1])

	for edge in edges:
		maxValue = max(abs(checkToBit[checkNodeIndex][edge-1] - prevCheckToBit[edge-1]), maxValue)

	return maxValue

def initBitToCheck(L, bitToCheck):
	for i in range(len(bitToCheck)):
		bitToCheck[i] = L[:]

def computeQ(qValue, alpha, beta, residual, maxQ):

	return ( (1 - alpha) * qValue + alpha * (residual + beta * maxQ) )
	

def qLearning(H, lmax, e, clusters, M, numberOfVectors, EsN0, seed, name):

	n = len(H)
	maxQ = 0

	repre = list(np.load('repre.npy'))
	for i in range(len(repre)):
		repre[i] = round(repre[i], 3)
	thre = list(np.load('thre.npy'))

	realizationMap = RealizationMap().generateRealizationMap(repre, len(clusters[0]))


	Q = [[0 for i in range(len(realizationMap))] for j in range(n)]

	# 1-indexed
	B = hammToB(H)
	E = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	M = [[0 for i in range(len(H[0]))] for j in range(len(H))]

	#np.random.seed(seed)

	indexOfVector = 0
	L = []

	for _ in range(numberOfVectors):
		L = generateNoiseVectors(len(H[0]), 1, EsN0)[0]
		l=0

		# S = HL
		S = matrix_vector_multiplicationV2(B, L)

		# Quantization ##
		S_hat = quant(S, thre, repre)

		# initialize bit to check messages
		for i in range(len(B)):
			M.append([0]*len(L))
		for i in range(len(B)):
			for j in B[i]:
				M[i][j-1] = L[j-1]   

		# Initializing the main variables
		action = 0
		u = 0
		state = 0
		nextState = 0
		residual = 0


		while l < lmax:
			# generating random number between 0 and 1
			eRandom = random.uniform(0, 1)

			# selecting whether to explore or exploit
			if(eRandom > e):
				##
				action = getMaximumCluster(Q)

			else:
				exploreRandom = random.uniform(0, len(S_hat))
				action = math.floor(exploreRandom)

			u = getClusterIndex(action, clusters)

			# get syndrome index
			state = getSyndromeIndex(S_hat, clusters[u], realizationMap)

			# get previous check to bit messages
			prevE_row = E[action][::] ##

			# belief propagation ##
			propogateMessage(E, M, action, B) ##

			# update syndrome
			updateSyndrome(B, action, S_hat, L, repre, thre)

			S_hat = quant(S_hat, thre, repre)

			nextState = getSyndromeIndex(S_hat, clusters[u], realizationMap)

			residual = getResidual(action, prevE_row, E, B[action]) ##
			
			Q_val = computeQ(Q[action][state], 0.1, 0.9, residual, maxQ)
			Q[action][state] = Q_val

			maxQ = max(Q[action][state], maxQ)

			l+=1

		indexOfVector += 1

		del S
		del S_hat

	printMatrix(Q)
	
	with open(name, "wb") as fp:
		pickle.dump(Q, fp)


def printMatrix(matrix):
	for row in matrix:
		print(row)

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

def generateNoiseVectors(bitsWidth, numberOfVectors, EsN0):
	L = []
	sigma = np.sqrt(1 / (2 * EsN0))
	for i in range(numberOfVectors):
		newArr = list(awgn([1]*bitsWidth, EsN0))
		for j in range(len(newArr)):
			newArr[j] = 2*newArr[j]*EsN0/(sigma*sigma)
		L.append(newArr)
	return L

def generateClusters(numberOfActions, numberOfClusters):
	clusters = []

	temp = []
	for i in range(numberOfActions):
		temp.append(i)
		if len(temp) == numberOfClusters:
			clusters.append(temp[:])
			temp = []
	
	return clusters

def matrix_vector_multiplication(matrix, vector):
    # Check if dimensions match for multiplication
    if len(matrix[0]) != len(vector):
        raise ValueError("Matrix columns must match vector length for multiplication.")
    
    # Initialize result vector with appropriate dimensions
    result = [0] * len(matrix)

    # Perform matrix-vector multiplication
    for i in range(len(matrix)):
        for j in range(len(vector)):
            result[i] += matrix[i][j] * vector[j]

    return result

def matrix_vector_multiplicationV2(B, vector):
    
    # Initialize result vector with appropriate dimensions
    result = [0] * len(B)

    # Perform matrix-vector multiplication
    for i in range(len(B)):
        for j in B[i]:
            result[i] += vector[j-1]

    return result


'''
B1 = [[10,30,40],[5,32,45],[16,18,39],[12,22,38],[15,19,47],[2,17,34],[9,24,42],[1,29,33],[4,27,36],[3,26,35],[11,31,43],[7,21,44],[8,20,48],[14,23,46],[6,28,37],[13,25,41],[14,32,43],[5,23,37],[2,31,36],[1,28,34],[7,25,47],[10,21,33],[15,30,35],[16,26,48],[3,22,46],[12,20,41],[8,18,38],[4,19,45],[6,24,40],[9,27,39],[13,17,42],[11,29,44],[8,24,34],[6,25,36],[9,19,43],[1,20,46],[14,27,42],[7,22,39],[13,18,35],[4,26,40],[16,29,38],[15,21,48],[11,23,45],[3,17,47],[5,28,44],[12,32,33],[2,30,41],[10,31,37],[10,18,36],[4,23,44],[9,29,40],[2,27,38],[8,30,42],[12,28,43],[11,20,37],[1,19,35],[15,31,39],[16,32,41],[5,26,33],[3,25,45],[13,21,34],[14,24,48],[7,17,46],[6,22,47],[7,27,40],[11,18,33],[2,32,35],[10,28,47],[5,24,41],[12,25,37],[3,19,39],[14,31,44],[16,30,34],[13,20,38],[9,22,36],[6,17,45],[4,21,42],[15,29,46],[8,26,43],[1,23,48],[1,25,42],[15,22,40],[8,21,41],[9,18,47],[6,27,43],[11,30,46],[7,31,35],[5,20,36],[14,17,38],[16,28,45],[4,32,37],[13,23,33],[12,26,44],[3,29,48],[2,24,39],[10,19,34],[8,20,36,56,80,81],[6,19,47,52,67,95],[10,25,44,60,71,94],[9,28,40,50,77,91],[2,18,45,59,69,88],[15,29,34,64,76,85],[12,21,38,63,65,87],[13,27,33,53,79,83],[7,30,35,51,75,84],[1,22,48,49,68,96],[11,32,43,55,66,86],[4,26,46,54,70,93],[16,31,39,61,74,92],[14,17,37,62,72,89],[5,23,42,57,78,82],[3,24,41,58,73,90],[6,31,44,63,76,89],[3,27,39,49,66,84],[5,28,35,56,71,96],[13,26,36,55,74,88],[12,22,42,61,77,83],[4,25,38,64,75,82],[14,18,43,50,80,92],[7,29,33,62,69,95],[16,21,34,60,70,81],[10,24,40,59,79,93],[9,30,37,52,65,85],[15,20,45,54,68,90],[8,32,41,51,78,94],[1,23,47,53,73,86],[11,19,48,57,72,87],[2,17,46,58,67,91],[8,22,46,59,66,92],[6,20,33,61,73,96],[10,23,39,56,67,87],[9,19,34,49,75,88],[15,18,48,55,70,91],[4,27,41,52,74,89],[3,30,38,57,71,95],[1,29,40,51,65,82],[16,26,47,58,69,83],[7,31,37,53,77,81],[11,17,35,54,79,85],[12,32,45,50,72,93],[2,28,43,60,76,90],[14,25,36,63,78,86],[5,21,44,64,68,84],[13,24,42,62,80,94]]
B2 = [[1,2,4], [2,3,5], [1,5,6], [3,4,6]]
B3 = [[1,2,3,4,5,6],[2,4,6,7,8,12],[5,8,9,10,11,12],[3,6,7,10,11,12],[1,2,3,5,9,11],[1,4,7,8,9,10]]

H = BToHamm(B1)

with open('mat_3_6.txt', 'rb') as f:
    H = pickle.load(f)

clusters = generateClusters(len(H), 2)
lmax = 100
e = 0.6
M = 2
numberOfVectors = 100000
EsN0 = 0.5



qLearning(H, lmax, e, clusters, M, numberOfVectors, EsN0, 0)
'''
'''

if(s_hatt_ell^(j)<=(quantizationRange[0]+quantizationRange[1])/2) s_ell^(j)=0; 
else if(s_hatt_ell^(j)>(quantizationRange[0]+quantizationRange[1])/2 && s_hatt_ell^(j)<=(quantizationRange[1]+quantizationRange[2])/2) s_ell^(j)=1;
else if(s_hatt_ell^(j)>(quantizationRange[1]+quantizationRange[2])/2 && s_hatt_ell^(j)<=(quantizationRange[2]+quantizationRange[3])/2) s_ell^(j)=2;
else if(s_hatt_ell^(j)>=(quantizationRange[2]+quantizationRange[3])/2) s_ell^(j)=3;

'''
