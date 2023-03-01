from qLearning import *

@profile
def beliefPropogation(L, H, Q, clusters, realizationMap, M, lmax):

	checkToBit = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	bitToCheck = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	initBitToCheck(L, bitToCheck)

	edges = hammToB(H)

	S = quantize(np.matmul(H, L), M)

	l = 0

	while l < lmax:
		for cluster in clusters:

			# get syndrome index
			subSyndrome = [S[i] for i in cluster]
			i = realizationMap[str(subSyndrome)]
			

			# find best action
			action = Q[0][i]
			actionIndex = 0
			for j in range(len(Q)):
				if Q[j][i] > action:
					action = Q[j][i]
					actionIndex = j

			# run belief propagation with best action
			sumProductDecoding(checkToBit, bitToCheck, actionIndex, edges)
		
		# update L vector
		for i in range(len(checkToBit)):
			for j in range(len(checkToBit[i])):
				L[j] += checkToBit[i][j]

		S = quantize(np.matmul(H, L), M)

		# if the vector is all zeros, then we have decoded successfully
		if(sum(S) == 0):
			break

		l+=1
	
	# return number of trials
	return l

B = [[1,2,4], [2,3,5], [1,5,6], [3,4,6]]
H = BToHamm(B)
with open('results.txt', 'rb') as f:
    Q = pickle.load(f)
clusters = [[0,1],[2,3]]
M = 2
realizationMap = RealizationMap().generateRealizationMap(M, len(clusters))



decoded = 0

numberOfTrials = 100

for _ in range(numberOfTrials):

	l = beliefPropogation(generateNoiseVectors(len(H[0]), 1, 1)[0], H, Q, clusters, realizationMap, M, numberOfTrials)
	if l < numberOfTrials:
		decoded+=1

print(decoded/100)

