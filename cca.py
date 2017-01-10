from scipy.linalg import eigh, eigvalsh
import numpy as np

def cca(X, Y, numCC=None):
    print X.shape, Y.shape
    assert X.shape[0] == Y.shape[0]
    assert np.isreal(X).all() and np.isreal(Y).all()
    cxx = np.dot(X.T, X)
    cxy = np.dot(X.T, Y)
    cyx = cxy.T
    cyy = np.dot(Y.T, Y)

    print 'cxx', cxx.shape
    assert (cxx == cxx.T).all()
    assert np.isreal(cxx).all()
    #vx = eigvalsh(cxx)
    #print vx.shape, vx
    #assert (vx >= -2e-3).all()
    print 'cyy', cyy.shape
    assert (cyy == cyy.T).all()
    assert np.isreal(cyy).all()
    #vy = eigvalsh(cyy)
    #print vy.shape, vy
    #assert (vy >= -2e-4).all()

    nX = X.shape[1]
    nY = Y.shape[1]
    numCC = X.shape[0] if numCC is None else numCC

    LH = np.zeros((nX + nY, nX + nY))
    RH = np.zeros((nX + nY, nX + nY))

    LH[0:nX, 0:nX] = cxx
    LH[0:nX, nX:nX + nY] = cxy
    LH[nX:nX + nY, 0:nX] = cyx
    LH[nX:nX + nY, nX:nX + nY] = cyy
    RH[0:nX, 0:nX] = cxx
    RH[nX:nX + nY, nX:nX + nY] = cyy
    print X.T[0:3,:]
    print RH[0:5,0:5], cxx[0:5,0:5]

    print RH.shape
    assert (RH == RH.T).all()
    vrh = eigvalsh(RH)
    print vrh
    #assert (vrh >= 0).all()

    v, W = eigh(LH, RH, overwrite_a=True, overwrite_b=True, eigvals=(nX + nY - numCC, nX + nY - 1))
    print v, v.shape
    print W, W.shape
    return W[0:nX, :numCC], W[nX:nX + nY, :numCC]
