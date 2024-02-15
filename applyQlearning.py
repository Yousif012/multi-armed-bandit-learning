from qLearning import initBitToCheck, hammToB, quant
from LDPC_decoding import getMessageToCheckEdges, propogateMessage
import numpy as np

def beliefPropogation(L, H, Q, clusters, realizationMap, M, lmax, repre, thre):

	E = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	M = [[0 for i in range(len(H[0]))] for j in range(len(H))]
	initBitToCheck(L, M)

	S = list(np.matmul(H, L))
	S = quant(S, thre, repre)
	
	A = getMessageToCheckEdges(H)
	B = hammToB(H)

	l = 0

	while l < lmax:
		for cluster in clusters:

			# get syndrome index
			subSyndrome = [S[i] for i in cluster]
			i = realizationMap[str(subSyndrome)]
			

			# find best action
			'''
			action = Q[0][i]
			actionIndex = 0
			for j in range(len(Q)):
				if Q[j][i] > action:
					action = Q[j][i]
					actionIndex = j
			'''
			actions = []
			for j in range(len(Q)):
				actions.append((j, Q[j][i]))
			actions.sort(key = lambda x : x[1], reverse=True)
			
			for action, _ in actions:
				# run belief propagation with best action
				propogateMessage(E, M, action, B)
		
		# update L vector
		L_cp = L.copy()
		
		for i in range(len(B)):
			for j in B[i]:
				L_cp[j-1] += E[i][j-1]
    
		z = []
		for bit in L_cp:
			if bit > 0:
				z.append(0)
			else:
				z.append(1)
		s = []
		
		for i in range(len(B)):
			total = 0
			for j in B[i]:
				total += (z[j-1])
			s.append(total % 2)

		# if the vector is all zeros, then we have decoded successfully
		if sum(s) == 0:
			return True
		else:
            # resetting up M
			for i in range(len(A)):
				val = 0
				for j in A[i]:
					val += E[j-1][i]
				for j in A[i]:
					M[j-1][i] = val + L[i] - E[j-1][i]
					
					if M[j-1][i] < -20:
						M[j-1][i] = -20
					elif M[j-1][i] > 20:
						M[j-1][i] = 20

		l+=1
	
	# return failed decoding
	return False

'''
Make sure the sequential decoding is working
Make sure to not use the soft syndrome to make decisions
Make sure to use a hard syndrome by converting the values in the LLR vector to binary values then multiplying the H matrix by the LLR vector
Pick more than one SNR or EbN0
Pick the range EbN0 from Salman's paper
'''

