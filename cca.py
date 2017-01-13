from scipy.linalg import eigh
import numpy as np

def cca(X, Y, numCC=None):
    print X.shape, Y.shape
    assert X.shape[0] == Y.shape[0]
    print 'Calculating correlation matrices'
    cxx = np.dot(X.T, X)
    cxy = np.dot(X.T, Y)
    cyx = cxy.T
    cyy = np.dot(Y.T, Y)

    nX = X.shape[1]
    nY = Y.shape[1]
    numCC = X.shape[0] if numCC is None else numCC

    LH = np.zeros((nX + nY, nX + nY), dtype=np.float32)
    RH = np.zeros((nX + nY, nX + nY), dtype=np.float32)

    LH[0:nX, 0:nX] = cxx
    LH[0:nX, nX:nX + nY] = cxy
    LH[nX:nX + nY, 0:nX] = cyx
    LH[nX:nX + nY, nX:nX + nY] = cyy
    RH[0:nX, 0:nX] = cxx
    RH[nX:nX + nY, nX:nX + nY] = cyy

    print 'Solving general eigenvalue problem'
    v, W = eigh(LH, RH, overwrite_a=True, overwrite_b=True, eigvals=(nX + nY - numCC, nX + nY - 1))
    return W[0:nX, :numCC], W[nX:nX + nY, :numCC]
