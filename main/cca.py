from scipy.linalg import eigh
import numpy as np

def correlations(X, Y, useGPU):
    if useGPU:
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import skcuda.linalg as linalg
        linalg.init()

        X_gpu = gpuarray.to_gpu(X)
        XT_gpu = linalg.transpose(X_gpu)
        cxx = linalg.mdot(XT_gpu, X_gpu).get()

        XT_gpu = linalg.transpose(X_gpu)
        X_gpu.gpudata.free()
        del X_gpu
        Y_gpu = gpuarray.to_gpu(Y)
        cxy = linalg.mdot(XT_gpu, Y_gpu).get()

        cyx = cxy.T

        YT_gpu = linalg.transpose(Y_gpu)
        cyy = linalg.mdot(YT_gpu, Y_gpu).get()
    else:
        cxx = np.dot(X.T, X)
        cxy = np.dot(X.T, Y)
        cyx = cxy.T
        cyy = np.dot(Y.T, Y)

    return cxx, cxy, cyx, cyy

def fit(X, Y, numCC=None, useGPU=False):
    print X.shape, Y.shape
    assert X.shape[0] == Y.shape[0]

    print 'Calculating correlation matrices'
    cxx, cxy, cyx, cyy = correlations(X, Y, useGPU)

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
