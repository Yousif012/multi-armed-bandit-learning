import numpy as np


def encode(Rs, G):
    # Rs: vector of binary information vectors to be encoded
    # G: generator matrix

    # initializing output array
    output = [[0] * len(G[0])] * len(Rs)

    # this is similar to the A matrix you mentioned in Las Vegas
    A = getRowsThatAreOne(G)
    
    for R_index in range(len(output)):
        for r_index in range(len(output[R_index])):
            
            sum = 0
            
            # looping over A array
            for one_index in A[r_index]:
                sum += Rs[R_index][one_index - 1]
            
            output[R_index][r_index] = sum % 2
    
    return output


def getRowsThatAreOne(H):
    A = [[] for _ in range(len(H[0]))]
    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i][j] != 0:
                A[j].append(i+1)
    return A


G = [[1,1,0,1],
     [1,0,1,1],
     [1,0,0,0],
     [0,1,1,1],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]

Rs = [[1,0,1,1]]

print(encode(Rs, np.transpose(G)))