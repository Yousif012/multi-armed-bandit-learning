import numpy as np
import copy
import random
import math

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

		for j in range(pow(-2, power), pow(2, power-1)):
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
	quantizationRange = [i for i in range(pow(-2, power), pow(2, power-1))]
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

# problem here
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

# need to fix this. Should look similar to my notes.
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


def qLearning(Ls, H, lmax, e, clusters, M):

	n = len(H)

	power = int(math.log2(M))

	maxQ = 0

	realizationMap = RealizationMap().generateRealizationMap(M, len(clusters))

	Q = [[0 for i in range(len(realizationMap))] for j in range(n)]

	# 1-indexed
	edges = hammToB(H)
	checkToBit = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	bitToCheck = [[0 for i in range(len(H[0]))] for j in range(len(H))]


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
		print(Q)

L = [[-0.5, 2.5, -4, 5, -3.5, 2.5]]
H = [[1,1,0,1,0,0],
	 [0,1,1,0,1,0],
	 [1,0,0,0,1,1],
	 [0,0,1,1,0,1]]



qLearning(L, H, 100, 0.2, [[0,1],[2,3]], 2)
