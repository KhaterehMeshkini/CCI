import os
import numpy as np
import scipy as sp
import h5py
from libs.RSdatamanager import filemanager as fm
from scipy.ndimage import binary_dilation as bindilation
from sklearn.metrics import mean_squared_error as MSE
from multiprocessing import Pool, Lock, Value
from functools import partial

#---------------------------------------------------------------------------------------------------#
def manager(tilename, **kwargs):
    #SETUP DEFAULT OPTIONS
    info = kwargs.get('info', True)
    blocksize = kwargs.get('blocksize', 200)
    mappath = kwargs.get('mappath', None)

    #PATHS
    loadpath = fm.check_folder(kwargs.get('savepath', None), 'Features')
    savepath = fm.check_folder(kwargs.get('savepath', None), 'NDI_TimeSeries')
    maindir = kwargs.get('maindir', None)
    temppath = fm.joinpath(maindir, 'numpy', tilename)

    #GET IMAGE INFO
    fn = [f for f in os.listdir(loadpath) if f.endswith('.tif')]
    if len(fn)>0:
        img = fm.readGeoTIFFD(fm.joinpath(loadpath,fn[0]), metadata=False)
        height, width, totfeatures = img.shape
    else:
        raise IOError('Unable to find input data!')

    #LOAD CLASSIFICATION MAP
    if mappath is not None:
        classmap = fm.readGeoTIFFD(mappath, metadata=False)
    else:
        classmap = np.empty(heigth, width)

    #ALLOC VARIABLES
    npixels = blocksize*blocksize
    rects = np.empty((npixels,368))
    mse = np.empty((npixels,4))

    #LOOP THROUGH FEATURES
    for feature in range(totfeatures):
        if info:
            print('Reconstructing feature %i/%i...' % ( (feature+1), totfeatures ), end='\r')

        folder = 'NDI' + str(feature+1)
        path = fm.check_folder(savepath, folder)

        #FOR EACH BLOCK POSITION
        for i in range(0, width, blocksize):
            for j in range(0, height, blocksize):

                matr, mask, days = loadts_block(i, j, feature, loadpath, temppath, **kwargs)
                
                counter = Value('i', 0)
                corecount = int( os.cpu_count()/2 - 1 ) #half to account for virtual cores

                p = Pool(corecount, initializer=counterinit, initargs=(counter,))
                results = p.map( partial(parallel_manager, matr=matr, mask=mask, days=days,
                                blocksize=blocksize), 
                                range(npixels) )
                p.close()
                p.join()

                for npx in range(npixels):
                    row, col = divmod(npx, blocksize)
                    row = row + i
                    col = col + j
                        
                    rects[npx,0] = width*row + col
                    rects[npx,1] = classmap[row,col]
                    rects[npx,2] = feature+1
                    mse[npx,0] = width*row + col
                    mse[npx,1] = classmap[row,col]
                    mse[npx,2] = feature+1

                    rects[npx,3:] = results[npx][0]
                    mse[npx,3] = results[npx][1]

                filename = fm.joinpath(path, 'ts.h5')
                if not os.path.isfile(filename):
                    with h5py.File(filename, 'w') as hf:
                        hf.create_dataset("ts",  data=rects, chunks=True, maxshape=(None, rects.shape[1]))
                else:
                    with h5py.File(filename, 'a') as hf:
                        hf["ts"].resize((hf["ts"].shape[0] + rects.shape[0]), axis = 0)
                        hf["ts"][-rects.shape[0]:] = rects

                filename = fm.joinpath(path, 'mse.h5')
                if not os.path.isfile(filename):
                    with h5py.File(filename, 'w') as hf:
                        hf.create_dataset("mse",  data=mse, chunks=True, maxshape=(None,mse.shape[1]))
                else:
                    with h5py.File(filename, 'a') as hf:
                        hf["mse"].resize((hf["mse"].shape[0] + mse.shape[0]), axis = 0)
                        hf["mse"][-mse.shape[0]:] = mse

#---------------------------------------------------------------------------------------------------#
def counterinit(c):
    global counter
    counter = c

def loadts_block(i, j, feature, loadpath, temppath, **kwargs):
    blocksize = kwargs.get('blocksize', 200)

    fn = [f for f in os.listdir(loadpath) if f.endswith('.tif')]
    totimg = len(fn)
    matr = np.empty((blocksize,blocksize,totimg))
    mask = np.empty((blocksize,blocksize,totimg), dtype=bool)
    days = []

    #RECONSTRUCT TS
    for idx, f in enumerate(fn):
        mtr = fm.readGeoTIFFD(fm.joinpath(loadpath,f), band=feature, metadata=False)
        matr[...,idx] = mtr[i:i+blocksize,j:j+blocksize]

        #MASK FOR VALID VALUES
        f = f.split('_') #split filename
        name = f[0] + '_' + f[1]
        maskpath = fm.joinpath(temppath, name, 'MASK.npy')
        msk = np.load(maskpath)
        msk = (msk==3) | (msk==4) | bindilation(msk==2, iterations=50) | (msk==1)
        msk = np.logical_not(msk)
        mask[...,idx] = msk[i:i+blocksize,j:j+blocksize]

        date = f[1].split('T') #split date and time
        days.append(date[0])

    start = fn[0].split('_')
    start = start[1].split('T')
    start = start[0]

    firstday = fm.string2ordinal(start) - 1
    days = [(fm.string2ordinal(d) - firstday) for d in days]
    days = np.array(days)

    return matr, mask, days

def parallel_manager(npx, matr, mask, days, **kwargs):
    blocksize = kwargs.get('blocksize', 200)

    #PIXEL COORDINATES IN BLOCK
    row, col = divmod(npx, blocksize)

    totimg = len(days)

    mask = mask[row,col,:]
    matr = matr[row,col,:]
    matr = matr[mask]
    days = days[mask]

    if len(days) == 0:
        rects =  np.full(365, np.nan)
        mse = np.nan
        return rects, mse
    else:
        #PADDING
        if days[0] != 1:
            days = np.concatenate( ([1], days) )
            matr = np.concatenate( ([0], matr) )
        if days[-1] != 365:
            days = np.concatenate( (days, [365]) )
            matr = np.concatenate( (matr, [0]) )

        #RECONSTRUCT TS
        rects, mse = fitting(days, matr, **kwargs)

        return rects, mse

#---------------------------------------------------------------------------------------------------#
#TS-Fitting FUNCTION(S) 
def fitting(days, feature_ts, **kwargs):
    #PREPARE FUNCTION PARAMETERS
    scale = kwargs.get('scale', 100)

    #GET INFO
    firstday = days[0]
    lastday = days[-1]
    totdays = lastday - firstday + 1 #get number of days

    #ALLOC VARIABLES
    linear_ts = np.empty(totdays)
    upenv_ts = np.empty(totdays)
    
    #CREATE VECTOR WITH ALL DAYS
    fDays = ( np.arange( firstday, lastday+1 ) ).reshape(-1,1)

    #FILTER INPUT DATA
    x, y = _filterxy(days, feature_ts)
    y = y*scale #scale output to [-100;100] range

    #LINEAR INTERPOLATION
    f = sp.interpolate.interp1d(x, y)
    linear_ts = f(fDays)
    linear_ts = np.ravel(linear_ts)

    #UPPER ENVELOPE CUBIC INTERPOLATION
    upenv_ts = upperenvelope_dropout(x, y, fDays)

    #COMPUTE MSE
    mse = MSE(linear_ts, upenv_ts)
    return upenv_ts, mse

def _filterxy(x,y): 
    mask = np.logical_not( np.isnan(y) )

    if ( np.isnan(y[0]).any() ):
        mask[0] = True
        y[0] = 0

    ynew = y[mask]
    xnew = x[mask]

    xunique = np.unique(xnew)
    yunique = np.empty(xunique.shape)

    for idx,var in enumerate(xunique):
        yunique[idx] = np.mean( ynew[xnew==var] )

    return xunique, yunique

#---------------------------------------------------------------------------------------------------#
#UPPER ENVELOPE
def upperenvelope_dropout(xold, yold, xnew, threshold=40):
    #STEP 1: filter local minima greater than the threshold
    xkeep = np.array(xold)
    ykeep = np.array(yold)
    stth = ykeep<threshold #"smaller than threshold"-mask

    m1 = np.concatenate( (ykeep[:-1]>ykeep[1:], [False]) )
    m2 = np.concatenate( ([False], ykeep[1:]>ykeep[:-1]) )
    localmaxima = m1 & m2

    mask = stth | localmaxima
    mask[0] = True
    mask[-1] = True
    xkeep = xold[mask]
    ykeep = yold[mask]

    #STEP 2: compute intemediate PCHIP interpolation
    f = sp.interpolate.pchip(xkeep,ykeep)
    y = f(xnew)
    y = np.ravel(y)

    #STEP 3: add-back original values that are above envelope values    
    ytemp = y[xold-1] #add offset:xold is in [1:k] range, y in [0:k-1]
    stor = (ytemp<yold) #"smaller than original"-mask
    mask = mask | stor
    xkeep = xold[mask]
    ykeep = yold[mask]

    #STEP 4: compute final PCHIP interpolation
    f = sp.interpolate.pchip(xkeep,ykeep)
    y = f(xnew)
    y = np.ravel(y)

    return y
