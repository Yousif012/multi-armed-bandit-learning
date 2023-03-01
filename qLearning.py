import numpy as np
import copy
import random
import math
import pickle

# Syndrome Realizations map to index
class RealizationMap:
	def __init__(self):
		self.index = 0
		self.hashMap = {}

	def generateRealizationMap(self, M, n):
		
		self.hashMap = {}

		power = int(math.log2(M))

		arr = [None] * n

		self.generateRealizations(n, arr, 0, power)

		return self.hashMap

	def generateRealizations(self, n, arr, i, power):

		if i == n:
			self.hashMap[str(arr)] = self.index
			self.index+=1
			return

		for j in range(pow(-2, power), pow(2, power-1) + 1):
			arr[i] = j
			self.generateRealizations(n, arr, i + 1, power)

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

def updateSyndrome(checkToBitEdges, checkNodeIndex, syndromeVector, L, M):
	neighboursOfCheckNode = checkToBitEdges[checkNodeIndex]

	for vk in neighboursOfCheckNode:
		for i in range(len(checkToBitEdges)):
			if vk in checkToBitEdges[i]:
				for vi in checkToBitEdges[i]:
					syndromeVector[i] += L[vi-1]
				syndromeVector = quantize(syndromeVector, M)

def quantize(S, M):
	power = int(math.log2(M))
	quantizationRange = [i for i in range(pow(-2, power), pow(2, power-1) + 1)]
	quantizedVector = S[:]
	for i in range(len(quantizedVector)):
		val = int(quantizedVector[i])
		if quantizedVector[i] > val + 0.5:
			val+=1
		if val not in quantizationRange:
			if val < quantizationRange[0]:
				quantizedVector[i] = quantizationRange[0]
			else:
				quantizedVector[i] = quantizationRange[-1]
		else:
			quantizedVector[i] = val

	quantizedVector = np.int_(quantizedVector)

	return quantizedVector

def getResidual(checkNodeIndex, prevCheckToBit, checkToBit, edges):

	maxValue = abs(checkToBit[checkNodeIndex][edges[0]] - prevCheckToBit[checkNodeIndex][edges[0]])

	for edge in edges:
		maxValue = max(abs(checkToBit[checkNodeIndex][edge-1] - prevCheckToBit[checkNodeIndex][edge-1]), maxValue)

	return maxValue

def initBitToCheck(L, bitToCheck):
	for i in range(len(bitToCheck)):
		bitToCheck[i] = L[:]

def computeQ(qValue, alpha, beta, residual, maxQ):

	return ( (1 - alpha) * qValue + alpha * (residual + beta * maxQ) )

def sumProductDecoding(checkToBit, bitToCheck, checkNodeIndex, tannerGraphEdges):

	edges = tannerGraphEdges[checkNodeIndex]

	for edge in edges:
		product = 1
		for otherEdge in edges:
			if otherEdge != edge:
				product *= np.tanh(bitToCheck[checkNodeIndex][otherEdge - 1] / 2)
		checkToBit[checkNodeIndex][edge - 1] = float(np.log((1 + product)/(1 - product)))
		for otherEdge in edges:
			if otherEdge != edge:
				bitToCheck[checkNodeIndex][otherEdge - 1] += checkToBit[checkNodeIndex][edge - 1]

				# limit values of bit to check
				if bitToCheck[checkNodeIndex][otherEdge - 1] < 0 and bitToCheck[checkNodeIndex][otherEdge - 1] < -20:
					bitToCheck[checkNodeIndex][otherEdge - 1] = -20
				if bitToCheck[checkNodeIndex][otherEdge - 1] > 0 and bitToCheck[checkNodeIndex][otherEdge - 1] > 20:
					bitToCheck[checkNodeIndex][otherEdge - 1] = 20

def qLearning(Ls, H, lmax, e, clusters, M, seed):

	n = len(H)

	power = int(math.log2(M))

	maxQ = 0

	realizationMap = RealizationMap().generateRealizationMap(M, len(clusters))

	Q = [[0 for i in range(len(realizationMap))] for j in range(n)]

	# 1-indexed
	edges = hammToB(H)
	checkToBit = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	bitToCheck = [[0 for i in range(len(H[0]))] for j in range(len(H))]

	np.random.seed(seed)

	for L in Ls:
		l=0

		# S = HL
		S = np.matmul(H, L)

		# Quantization ##
		S = quantize(S, M)

		# initialize bit to check messages
		initBitToCheck(L, bitToCheck)

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
				exploreRandom = random.uniform(0, len(S))
				action = math.floor(exploreRandom)

			u = getClusterIndex(action, clusters)

			# get syndrome index
			state = getSyndromeIndex(S, clusters[u], realizationMap)

			# get previous check to bit messages
			prevCheckToBit = copy.deepcopy(checkToBit)
			# belief propagation ##
			sumProductDecoding(checkToBit, bitToCheck, action, edges)

			# update syndrome ##
			updateSyndrome(edges, action, S, L, M)

			S = quantize(S, M)

			# do we actually need this variable?
			nextState = getSyndromeIndex(S, clusters[u], realizationMap)

			##
			residual = getResidual(action, prevCheckToBit, checkToBit, edges[action])

			Q[action][state] = computeQ(Q[action][state], 0.1, 0.9, residual, maxQ)

			# is this the right way to get the maximum Q?
			maxQ = max(Q[action][state], maxQ)

			l+=1

	with open("results.txt", "wb") as fp:
		pickle.dump(Q, fp)

def awgn(data, EsN0):
	variance = 1 / (2 * EsN0)
	dataLength = len(data)

	noise = np.random.normal(0, variance, dataLength)

	for i in range(len(data)):
		data[i] += noise[i]

	return data

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
	for i in range(numberOfVectors):
		newArr = [0 for i in range(bitsWidth)]
		newArr = awgn(newArr, EsN0)
		L.append(list(newArr))
	return L

def generateClusters(numberOfActions):
	clusters = []

	temp = []
	for i in range(numberOfActions):
		temp.append(i)
		if len(temp) == 2:
			clusters.append(temp[:])
			temp = []
	return clusters




#B = [[10,30,40],[5,32,45],[16,18,39],[12,22,38],[15,19,47],[2,17,34],[9,24,42],[1,29,33],[4,27,36],[3,26,35],[11,31,43],[7,21,44],[8,20,48],[14,23,46],[6,28,37],[13,25,41],[14,32,43],[5,23,37],[2,31,36],[1,28,34],[7,25,47],[10,21,33],[15,30,35],[16,26,48],[3,22,46],[12,20,41],[8,18,38],[4,19,45],[6,24,40],[9,27,39],[13,17,42],[11,29,44],[8,24,34],[6,25,36],[9,19,43],[1,20,46],[14,27,42],[7,22,39],[13,18,35],[4,26,40],[16,29,38],[15,21,48],[11,23,45],[3,17,47],[5,28,44],[12,32,33],[2,30,41],[10,31,37],[10,18,36],[4,23,44],[9,29,40],[2,27,38],[8,30,42],[12,28,43],[11,20,37],[1,19,35],[15,31,39],[16,32,41],[5,26,33],[3,25,45],[13,21,34],[14,24,48],[7,17,46],[6,22,47],[7,27,40],[11,18,33],[2,32,35],[10,28,47],[5,24,41],[12,25,37],[3,19,39],[14,31,44],[16,30,34],[13,20,38],[9,22,36],[6,17,45],[4,21,42],[15,29,46],[8,26,43],[1,23,48],[1,25,42],[15,22,40],[8,21,41],[9,18,47],[6,27,43],[11,30,46],[7,31,35],[5,20,36],[14,17,38],[16,28,45],[4,32,37],[13,23,33],[12,26,44],[3,29,48],[2,24,39],[10,19,34],[8,20,36,56,80,81],[6,19,47,52,67,95],[10,25,44,60,71,94],[9,28,40,50,77,91],[2,18,45,59,69,88],[15,29,34,64,76,85],[12,21,38,63,65,87],[13,27,33,53,79,83],[7,30,35,51,75,84],[1,22,48,49,68,96],[11,32,43,55,66,86],[4,26,46,54,70,93],[16,31,39,61,74,92],[14,17,37,62,72,89],[5,23,42,57,78,82],[3,24,41,58,73,90],[6,31,44,63,76,89],[3,27,39,49,66,84],[5,28,35,56,71,96],[13,26,36,55,74,88],[12,22,42,61,77,83],[4,25,38,64,75,82],[14,18,43,50,80,92],[7,29,33,62,69,95],[16,21,34,60,70,81],[10,24,40,59,79,93],[9,30,37,52,65,85],[15,20,45,54,68,90],[8,32,41,51,78,94],[1,23,47,53,73,86],[11,19,48,57,72,87],[2,17,46,58,67,91],[8,22,46,59,66,92],[6,20,33,61,73,96],[10,23,39,56,67,87],[9,19,34,49,75,88],[15,18,48,55,70,91],[4,27,41,52,74,89],[3,30,38,57,71,95],[1,29,40,51,65,82],[16,26,47,58,69,83],[7,31,37,53,77,81],[11,17,35,54,79,85],[12,32,45,50,72,93],[2,28,43,60,76,90],[14,25,36,63,78,86],[5,21,44,64,68,84],[13,24,42,62,80,94]]
#B = [[1,2,4], [2,3,5], [1,5,6], [3,4,6]]
#H = BToHamm(B)
#L = generateNoiseVectors(len(H[0]), 100, 1)


#qLearning(L, H, 100, 0.2, generateClusters(len(H)), 2, 0)



