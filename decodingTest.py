import numpy as np
import pickle
from qLearning import hammToB, generateNoiseVectors, initBitToCheck, BToHamm
import random


def sumProductDecoding(E, M, checkNodeIndex, tannerGraphEdges):

	edges = tannerGraphEdges[checkNodeIndex]
	

	for edge in edges:
		product = 1
		for otherEdge in edges:
			if otherEdge != edge:
				product *= np.tanh(M[checkNodeIndex][otherEdge - 1] / 2)
		E[checkNodeIndex][edge - 1] = round(float(np.log((1 + product)/(1 - product))), 3)


'''
with open("mat_3_6 2.txt","r") as outfile:
    data = outfile.readlines()

H = []
for d in data:
    bits = d.split()
    for i in range(len(bits)):
        bits[i] = int(bits[i])
    H.append(bits)


with open("mat_3_6.txt", "wb") as fp:
	pickle.dump(H, fp)

'''

EsN0 = 1

with open('mat_3_6.txt', 'rb') as f:
    H = pickle.load(f)
B2 = [[1,2,4], [2,3,5], [1,5,6], [3,4,6]]
H = BToHamm(B2)

decoded = 0
for n in range(100):
    L = generateNoiseVectors(len(H[0]), 1, EsN0)[0]

    edges = hammToB(H)
    M = [[0 for i in range(len(H[0]))] for j in range(len(H))]
    initBitToCheck(L, M)
    E = [[0 for i in range(len(H[0]))] for j in range(len(H))]


    for l in range(10):
                
        checkNode = int(len(H)*random.uniform(0, 1))
        sumProductDecoding(E, M, checkNode, edges)
        
        noiseVector = L.copy()
        for i in range(len(E)):
            for j in range(len(E[i])):
                noiseVector[j] += E[i][j]
        
        Lbinary = []
        for noise in noiseVector:
            if noise >= 0:
                Lbinary.append(0)
            else:
                Lbinary.append(1)
                
        Sreal = list(np.matmul(H, Lbinary))
        for i in range(len(Sreal)):
            Sreal[i] = Sreal[i] % 2
        
        sumOfBits = sum(Sreal)
        if(sumOfBits == 0):
            decoded += 1
            break
        else:
            M = [[0 for i in range(len(H[0]))] for j in range(len(H))]
            for i in range(len(E)):
                for j in range(len(E[0])):
                    val = 0
                    if E[i][j] != 0:
                        for row in range(len(E)):
                            if row != i:
                                val += E[row][j]
                        M[i][j] = val + L[j]
                        
                        if M[i][j] < -20:
                            M[i][j] = -20
                        elif M[i][j] > 20:
                            M[i][j] = 20
print(decoded/100)

    
#sumProductDecoding(checkToBit, bitToCheck, 4, edges)

#print(hammToB(H))

