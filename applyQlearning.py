from qLearning import *

def beliefPropogation(L, H, Q, clusters, realizationMap, M, lmax, repre, thre):

	checkToBit = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	bitToCheck = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	initBitToCheck(L, bitToCheck)

	edges = hammToB(H)

	S = list(np.matmul(H, L))
	S = quant(S, thre, repre)

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
    
		Lbinary = []

		for l in L:
			if l >= 0:
				Lbinary.append(0)
			else:
				Lbinary.append(1)

		Sreal = list(np.matmul(H, Lbinary))

		for i in range(len(Sreal)):
			Sreal[i] = Sreal[i] % 2

		# if the vector is all zeros, then we have decoded successfully
		if(sum(Sreal) == 0):
			break

		l+=1
	
	# return number of trials
	return l

#B = [[1,2,3,4,5,6],[2,4,6,7,8,12],[5,8,9,10,11,12],[3,6,7,10,11,12],[1,2,3,5,9,11],[1,4,7,8,9,10]]
#B = [[10,30,40],[5,32,45],[16,18,39],[12,22,38],[15,19,47],[2,17,34],[9,24,42],[1,29,33],[4,27,36],[3,26,35],[11,31,43],[7,21,44],[8,20,48],[14,23,46],[6,28,37],[13,25,41],[14,32,43],[5,23,37],[2,31,36],[1,28,34],[7,25,47],[10,21,33],[15,30,35],[16,26,48],[3,22,46],[12,20,41],[8,18,38],[4,19,45],[6,24,40],[9,27,39],[13,17,42],[11,29,44],[8,24,34],[6,25,36],[9,19,43],[1,20,46],[14,27,42],[7,22,39],[13,18,35],[4,26,40],[16,29,38],[15,21,48],[11,23,45],[3,17,47],[5,28,44],[12,32,33],[2,30,41],[10,31,37],[10,18,36],[4,23,44],[9,29,40],[2,27,38],[8,30,42],[12,28,43],[11,20,37],[1,19,35],[15,31,39],[16,32,41],[5,26,33],[3,25,45],[13,21,34],[14,24,48],[7,17,46],[6,22,47],[7,27,40],[11,18,33],[2,32,35],[10,28,47],[5,24,41],[12,25,37],[3,19,39],[14,31,44],[16,30,34],[13,20,38],[9,22,36],[6,17,45],[4,21,42],[15,29,46],[8,26,43],[1,23,48],[1,25,42],[15,22,40],[8,21,41],[9,18,47],[6,27,43],[11,30,46],[7,31,35],[5,20,36],[14,17,38],[16,28,45],[4,32,37],[13,23,33],[12,26,44],[3,29,48],[2,24,39],[10,19,34],[8,20,36,56,80,81],[6,19,47,52,67,95],[10,25,44,60,71,94],[9,28,40,50,77,91],[2,18,45,59,69,88],[15,29,34,64,76,85],[12,21,38,63,65,87],[13,27,33,53,79,83],[7,30,35,51,75,84],[1,22,48,49,68,96],[11,32,43,55,66,86],[4,26,46,54,70,93],[16,31,39,61,74,92],[14,17,37,62,72,89],[5,23,42,57,78,82],[3,24,41,58,73,90],[6,31,44,63,76,89],[3,27,39,49,66,84],[5,28,35,56,71,96],[13,26,36,55,74,88],[12,22,42,61,77,83],[4,25,38,64,75,82],[14,18,43,50,80,92],[7,29,33,62,69,95],[16,21,34,60,70,81],[10,24,40,59,79,93],[9,30,37,52,65,85],[15,20,45,54,68,90],[8,32,41,51,78,94],[1,23,47,53,73,86],[11,19,48,57,72,87],[2,17,46,58,67,91],[8,22,46,59,66,92],[6,20,33,61,73,96],[10,23,39,56,67,87],[9,19,34,49,75,88],[15,18,48,55,70,91],[4,27,41,52,74,89],[3,30,38,57,71,95],[1,29,40,51,65,82],[16,26,47,58,69,83],[7,31,37,53,77,81],[11,17,35,54,79,85],[12,32,45,50,72,93],[2,28,43,60,76,90],[14,25,36,63,78,86],[5,21,44,64,68,84],[13,24,42,62,80,94]]
B = [[1,2,4], [2,3,5], [1,5,6], [3,4,6]]
H = BToHamm(B)
with open('results4.txt', 'rb') as f:
    Q = pickle.load(f)
clusters = generateClusters(len(H), 2)
M = 2

repre = list(np.load('repre.npy'))
for i in range(len(repre)):
	repre[i] = round(repre[i], 3)
thre = list(np.load('thre.npy'))

realizationMap = RealizationMap().generateRealizationMap(repre, len(clusters[0]))



decoded = 0

numberOfTrials = 100

for _ in range(10):

	l = beliefPropogation(generateNoiseVectors(len(H[0]), 1, 1)[0], H, Q, clusters, realizationMap, M, numberOfTrials, repre, thre)
	if l < numberOfTrials:
		decoded+=1

print(decoded/numberOfTrials)


'''
Make sure the sequential decoding is working
Make sure to not use the soft syndrome to make decisions
Make sure to use a hard syndrome by converting the values in the LLR vector to binary values then multiplying the H matrix by the LLR vector
Pick more than one SNR or EbN0
Pick the range EbN0 from Salman's paper
'''

