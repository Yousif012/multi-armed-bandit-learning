from numpy.random import random
import numpy as np
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal
import csv
import pickle
import random as rnd
import time
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
        for b in range(1, len(M)+1):
            bitFlips[b] = {0: 0, 1: 0}

        for m in range(len(B)):
            for n in range(len(B[m])):
                nth_bit = B[m][n]
                if nth_bit == -1:
                    continue
                bitFlips[nth_bit][E[m][n]] += 1

        ### checking and flipping bits ###
        for bit in bitFlips.keys():
            one = bitFlips[bit][1]
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


def floodingDecoding(r, H, A, B, maxIterations=100):
    
    I = 0
    M = []

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
        for i in range(len(B)):
            product = 1.0
            for j in B[i]:
                product *= np.tanh(M[i][j-1] / 2)
            for j in B[i]:
                newProduct = product / (np.tanh(M[i][j-1] / 2))
                E[i][j-1] = float(np.log((1 + newProduct)/(1 - newProduct)))

        L = r.copy()

        for i in range(len(B)):
            for j in B[i]:
                L[j-1] += E[i][j-1]

        # converting to binary format
        z = []
        for bit in L:
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

        if sum(s) == 0:
            break
        else:
            # resetting up M
            for i in range(len(A)):
                val = 0
                for j in A[i]:
                    val += E[j-1][i]
                for j in A[i]:

                    M[j-1][i] = val + r[i] - E[j-1][i]

                    if M[j-1][i] < -20:
                        M[j-1][i] = -20
                    elif M[j-1][i] > 20:
                        M[j-1][i] = 20

        I += 1

    return z


def sequentialDecoding(r, H, A, B, maxIterations=100):

    # initialize bit node to check node messages
    M = []

    # initialization
    for i in range(len(B)):
        M.append([0]*len(r))
    for i in range(len(B)):
        for j in B[i]:
            M[i][j-1] = r[j-1]   

    for _ in range(maxIterations):

        checkSequence = rnd.sample(range(len(H)), len(H))

        # initialize E
        E = []
        for i in range(len(B)):
            E.append([0]*len(r))

        for checkNode in checkSequence:
            propogateMessage(E, M, checkNode, B)

        L = r.copy()

        for i in range(len(B)):
            for j in B[i]:
                L[j-1] += E[i][j-1]

        # converting to binary format
        z = []
        for bit in L:
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

        if(sum(s) == 0):
            return z
        else:
            # resetting up M
            for i in range(len(A)):
                val = 0
                for j in A[i]:
                    val += E[j-1][i]
                for j in A[i]:

                    M[j-1][i] = val + r[i] - E[j-1][i]

                    if M[j-1][i] < -20:
                        M[j-1][i] = -20
                    elif M[j-1][i] > 20:
                        M[j-1][i] = 20

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


def hardDecisionSimulation(B, error, lengthOfCode, maxBlockErros=100):
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
# sequential decoding simulation


def softDecisionSequentialSimulation(H, r, EbN0dB, maxBlockErros=10, maxSamples=10000, seed=0):

    EbN0 = pow(10, EbN0dB / 10)
    EsN0 = EbN0 * r

    lengthOfCode = len(H[0])
    x = [0]*lengthOfCode
    bitErrors = 0
    blockErrors = 0
    sample_index = 0

    A = getMessageToCheckEdges(H)
    B = hammToB(H)

    np.random.seed(seed)

    while blockErrors < maxBlockErros and sample_index < maxSamples:

        # pass through AWGN
        y = list(awgn([1]*lengthOfCode, EsN0))
        r = [0]*len(y)

        for i in range(len(r)):
            r[i] = 4*y[i]*EsN0

        # decode for errors
        decodedY = sequentialDecoding(r, H, A, B)

        # count number of errors the decoder missed
        errors = errorCounter(x, decodedY)

        blockErrors += errors[1]
        bitErrors += errors[0]
        sample_index += 1

    return [bitErrors/(sample_index*lengthOfCode), blockErrors/sample_index]
# flooding decoding simulation
# H is the code matrix
# r is code rate
# returns [bitErrorRate, blockErrorRate]


def softDecisionFloodingSimulation(H, rate, EbN0dB, maxBlockErros=10, maxSamples=10000, seed=0):

    EbN0 = pow(10, EbN0dB / 10)
    EsN0 = EbN0 * rate

    lengthOfCode = len(H[0])
    x = [0]*lengthOfCode
    bitErrors = 0
    blockErrors = 0
    sample_index = 0

    B = hammToB(H)
    A = getMessageToCheckEdges(H)

    np.random.seed(seed)

    while blockErrors < maxBlockErros and sample_index < maxSamples:

        # pass through AWGN
        y = list(awgn([1]*lengthOfCode, EsN0))
        r = [0]*len(y)

        for i in range(len(r)):
            r[i] = 4*y[i]*EsN0

        # decode for errors
        decodedY = floodingDecoding(r, H, A, B)

        # count number of errors the decoder missed
        errors = errorCounter(x, decodedY)

        blockErrors += errors[1]
        bitErrors += errors[0]
        sample_index += 1

    return [bitErrors/(sample_index*lengthOfCode), blockErrors/sample_index]


def propogateMessage(E, M, checkNode, B):
    
    product = 1.0
    for j in B[checkNode]:
        product *= np.tanh(M[checkNode][j-1] / 2)
    for j in B[checkNode]:
        newProduct = product / (np.tanh(M[checkNode][j-1] / 2))
        E[checkNode][j-1] = float(np.log((1 + newProduct)/(1 - newProduct)))


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

    noise = list(np.random.normal(0, np.sqrt(variance), dataLength))

    for i in range(len(data)):
        data[i] += noise[i]

    return data


def printMatrix(matrix):
    for row in matrix:
        print(row)


def initBitToCheck(L, bitToCheck):
    for i in range(len(bitToCheck)):
        bitToCheck[i] = L[:]


def getMessageToCheckEdges(H):
    A = [[] for _ in range(len(H[0]))]
    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i][j] != 0:
                A[j].append(i+1)
    return A


## ignore this part

'''
with open('mat_3_6.txt', 'rb') as f:
    H = pickle.load(f)
np.random.seed(1)
lengthOfCode = len(H[0])
EsN0 = 0.8
y = list(awgn([1]*lengthOfCode, EsN0))
r = [0]*len(y)
  
for i in range(len(r)):
	r[i] = 4*y[i]*EsN0

res = floodingDecoding(r, H)

errors = errorCounter([0]*len(H[0]), res)

print(errors)



with open('mat_3_6.txt', 'rb') as f:
    H = pickle.load(f)

# Generate data
EbN0dB = 0

error = softDecisionFloodingSimulation(H, 0.5, EbN0dB, 1, 1)

'''