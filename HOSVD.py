## hosvd_R generaliza la descomposición SVD de una matriz para un tensor de orden n>=2.
## Fuentes:
## 1.)   https://doi.org/10.1016/j.expthermflusci.2016.06.017
## 2.)   DOI: 10.1137/S0895479896305696
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VERSION 1.0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## S almacena el núcleo de la descomposición.
## U es un cell de matrices. Cada Matriz almacena los modos 1-D del tensor
## sv son los valores singulares de las matrices B


import numpy as np
import pandas as pd


from scipy.interpolate import Akima1DInterpolator #https://pypi.org/project/akima/

#import matplotlib.pyplot as plt

###############################################################
def hosvd_R(T):
    #print(T)
    if isinstance(T, pd.DataFrame):
        T.to_numpy()

    ndim = T.ndim
    vdim = np.array(range(ndim),dtype=int)
    sz = np.array(T.shape,dtype=int)

    S = T
    U = []
    SV = []
    #Vh = []
    shiftdimr = np.zeros(ndim, dtype=int)

    dimorder = np.array(range(ndim))

    for i in range(ndim): #Para la dimensión i
        shiftdiml = np.roll(dimorder, -i)
        shiftdiml[1:] = np.roll(shiftdiml[1:], 1)
        for j in range(ndim-1, -1, -1):
            shiftdimr[j] = np.where(shiftdiml == j)[0]
        #print(shiftdimr)
        #print(shiftdiml)
        ## Matricization para construir la matriz B^(i) de [ref. 1]
        B = np.moveaxis(T, vdim, shiftdiml)
        B = np.reshape(B, (B.shape[0], B.size//B.shape[0]), order='F')
        #print(B)
        ## SVD De la matriz B
        DmyU, DmySV, _ = np.linalg.svd(B, full_matrices=True, compute_uv=True, hermitian=False)
        U.append(DmyU)
        SV.append(DmySV)
        ## Matrizar S
        S =  np.moveaxis(S, vdim, shiftdiml)
        S = np.reshape(S, (S.shape[0], S.size//S.shape[0]), order='F')
        ##  Multiplicar por la conjugada transpuesta de U (despejar U*S = T)
        S = np.dot( DmyU.conj().T, S )
        ## volver a ensamblar en forma de tensor
        S = np.reshape(S, sz[shiftdiml], order='F')
        S = np.moveaxis(S, vdim, shiftdimr)
    return U, S, SV
###############################################################
###############################################################
def full_hosvd_R(S, U, DIMS = None):
    T = S
    ndim = T.ndim
    if DIMS == None:
        DIMS = range(ndim)

    vdim = np.array(range(ndim),dtype=int)
    sz = np.zeros(ndim, dtype=int)
    szTrc = np.zeros(ndim, dtype=int)
    siz = np.zeros(ndim, dtype=int)
    shiftdimr = np.zeros(ndim, dtype=int)

    for i in range(ndim-1,-1,-1):
        ## En caso de descomposiciones truncadas, el tamaño original
        sz[i] = U[i].shape[0]
        ##y el tamaño truncado size(Struncado) = szTrc
        szTrc[i] = U[i].shape[1]
        siz[i] = U[i].shape[1]
    dimorder = range(ndim)
    #siz = szTrc

    for i in DIMS:
        ## Controlar como se expande el tensor
        siz[i] = sz[i]
        ## Almacenar las permutaciones de las dimeniones
        shiftdiml = np.roll(dimorder, -i)
        shiftdiml[1:] = np.roll(shiftdiml[1:], 1)
        for j in range(ndim - 1, -1, -1):
            shiftdimr[j] = np.where(shiftdiml == j)[0]

        ## Rotar dimensiones y matrizar
        T = np.moveaxis(T, vdim, shiftdiml)
        T = np.reshape(T, (szTrc[i], T.size // szTrc[i]), order='F')
        ##  Multiplicar
        T = np.dot(U[i], T)
        ## volver a ensamblar en forma de tensor
        T = np.reshape(T, siz[shiftdiml], order='F')
        T = np.moveaxis(T, vdim, shiftdimr)
    return T
###############################################################
###############################################################
def trunc_hosvd_R(S, U, threashold, SV = None):
    ndims = S.ndim

    if SV != None and isinstance(threashold, float): #Threashold is integer
        VEC = np.zeros(ndims,dtype=int)
        for i in range(ndims):
            E = SV[i][0]
            ET = np.sum(SV[i])

            while  E < ET*threashold:
                VEC[i] = VEC[i] + 1
                E = E + SV[i][VEC[i]]
    else: #Threashold is an index vector
        VEC = threashold

    VEC = VEC + 1 #para indexar decentemtente 0:3 = 0, 1, 2

    if ndims == 2:
        S = S[0:VEC[0], 0:VEC[1]]
    elif ndims == 3:
        S = S[0:VEC[0], 0:VEC[1], 0:VEC[2]]

    for i in range(ndims):
        U[i] = U[i][:,0:VEC[i]]

    T = full_hosvd_R(S, U)
    return T, S, U, VEC
###############################################################
###############################################################
def Akima1D_hosvd_R(S, U, xq, threashold = None, SV = None):
    if threashold != None:
        _, S, U, _ = trunc_hosvd_R(S, U, threashold, SV)
    U_int = []
    for i in range(S.ndim):
        U_int.append( np.zeros( (xq[i].size, U[i].shape[1]), dtype=int ) )
        x = np.linspace(0, U[i].shape[0]-1, U[i].shape[0] ,dtype=int)
        for j in range(U[i].shape[1]):
            U_int[i][:,j] = Akima1DInterpolator(x, U[i][:,j])(xq[i])

    T = full_hosvd_R(S, U)
    return T, U_int, S