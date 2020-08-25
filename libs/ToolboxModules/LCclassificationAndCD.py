import os
import numpy as np
import h5py
import math
from libs.RSdatamanager import filemanager as fm


#---------------------------------------------------------------------------------------------------#
"""
Classes:
1 artificial
2 bareland
3 grassland
4 crops
5 broadleaves
6 conifers
7 snow
8 water
9 shrub
"""
#---------------------------------------------------------------------------------------------------#
def manager(tile, **kwargs):
    #SETUP VARIBLES  
    info = kwargs.get('info', True)
    years = kwargs.get('years', None)
    outpath = kwargs.get('outpath', None)
    loadpath = '' #TODO: where is the test data?
    savepath = fm.check_folder(outpath, tile, 'LCclassificationAndCD')

    blocksize = kwargs.get('blocksize', 200)
    n_classes = kwargs.get('n_classes', 9)
    DTW_max_samp = kwargs.get('DTW_max_samp', 15)   # max number of samples of DTW
    MAX_CD = kwargs.get('MAX_CD', 1)                # max number of detected changes

    col_nPIXEL = 0
    col_nCLASS = 1
    col_nBAND  = 2
    col_DATA = 3


    ###############################
    # GET INFO AND INITIALIZATION #
    ###############################
    for rootname, _, filenames in os.walk(loadpath):
        for f in filenames:
            if (f.endswith('.tif')):
                path = fm.joinpath(rootname, f) 
    img, geotransform, projection = fm.readGeoTIFFD(path, metadata=True)
    width, height, totfeature = img.shape

    for rootname, _, filenames in os.walk(loadpath):
        for f in filenames:
            if (f.endswith('ts.h5')):
                path = fm.joinpath(rootname)
    with h5py.File(fm.joinpath(path,f), 'r') as hf:
        NDI_ = np.array(hf["ts"])

    
    #Get classes intervals
    class_int = np.zeros(n_classes)
    class_int_mask = np.unique(NDI_[:,col_nCLASS]).astype(int).tolist()
    for n in class_int_mask:
        class_int[n-1] = n
    class_int = class_int.astype(int).tolist()

    #Get number of seeds
    n_seeds = len(np.unique(NDI_[:,col_nPIXEL]))
    
    #Get number of features
    n_features = totfeature

    #Get number of seeds per class and class seeds mask
    n_seeds_c = np.zeros(n_classes)
    for nc in class_int:
        n_seeds_c[nc-1] = np.size(NDI_[NDI_[:,col_nCLASS]==nc, :], axis=0)
    n_seeds_c = n_seeds_c.astype(int).tolist()

    seed_class_mask = NDI_[:,col_nCLASS]

    #Define blocksize
    nseeds_b = blocksize


    #Multi feature DTW maximum distance
    path = fm.check_folder(outpath, tile, 'LCTraining_DTW', 'Multifeature')

    DTW_max_d = 0
    for b1 in range(0, n_seeds, nseeds_b):
        for b2 in range(0, n_seeds, nseeds_b):
            with h5py.File(filename, 'r') as hf:
                max_d_block = np.nanmax(np.array(hf["DTW_matrix"][b1:b1+nseeds_b, b2:b2+nseeds_b]))
                if max_d_block > DTW_max_d:
                    DTW_max_d = max_d_block

    #Loading the models
    path = fm.check_folder(outpath, tile, 'LCTraining_DTW')
    models = np.load(fm.joinpath(path, "models.npy"))

    
    ############################
    # LC CLASSIFICATION AND CD #
    ############################
    #Time array definition
    t_seq_st = np.array([1, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73])
    t_seq_en = np.array([366, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73])
    t_seq_st = np.cumsum(t_seq_st)
    t_seq_en = np.cumsum(t_seq_en)
    
    #Similarity trends computation and classification
    Test_simi_traj = [None]*n_seeds
    LC_seq = [None]*n_seeds

    for ns in range(n_seeds):
        Traj1 = None
        for nb, band in enumerate(n_features):
            Seeds = load_seeds(tile, ns, nb, col_DATA, **kwargs)
            if Traj1 is None:
                Traj1 = np.zeros((len(n_features), len(Seeds[col_DATA:])))
                Traj1[nb,:] = Seeds[col_DATA:]
            else:
                Traj1[nb,:] = Seeds[col_DATA:]

        pixnr = Seeds[col_nPIXEL]
        Test_simi_traj[pixnr] = np.empty((n_classes, np.size(t_seq_st)))
        LC_seq[pixnr] = np.empty((2, np.size(t_seq_st)))

        for ts in range(np.size(t_seq_st)):
            Traj1_T = Traj1[:, t_seq_st[ts]:t_seq_en[ts]]
            Traj1_T = np.roll(Traj1_T, 73*ts, axis=1)

            for nc in range(n_classes):
                max_simi = 0

                for nm in range(len(models[nc])):
                    Traj2 = models[nc][nm]
                    simi = (DTW_max_d - DTW(Traj1_T, Traj2, DTW_max_samp=DTW_max_samp)) / DTW_max_d #TODO: distance_fast
                    max_simi = max(max_simi, simi)

                Test_simi_traj[pixnr][nc,ts] = max_simi

            LC_seq[pixnr][0,ts] = np.argmax(Test_simi_traj[ns][:,ts]) + 1      # +1 number of class vs index
    
    #Stability rule application
    CD_counter = np.empty(n_seeds)
    break_p = np.empty((n_seeds, MAX_CD))
    LC_seq_bp = np.empty((n_seeds, MAX_CD+1))

    for ns in range(n_seeds):
        counter = 0

        for ts in range(np.size(t_seq_st)):
            if ts == 0:
                LC_seq[ns][1,ts] = LC_seq[ns][0,ts]
            else:
                if (LC_seq[ns][0,ts] == LC_seq[ns][0,ts-1]) and (counter == 0):
                    LC_seq[ns][1,ts] = LC_seq[ns][1,ts-1]
                elif LC_seq[ns][0,ts] != LC_seq[ns][0,ts-1]:
                    LC_seq[ns][1,ts] = LC_seq[ns][1,ts-1]
                    counter = 1
                elif LC_seq[ns][0,ts] == LC_seq[ns][0,ts-1]:
                    counter = counter + 1
                    if counter<4:
                        LC_seq[ns][1,ts] = LC_seq[ns][1,ts-1]
                    else:
                        LC_seq[ns][1,ts-3] = LC_seq[ns][0,ts]
                        LC_seq[ns][1,ts-2] = LC_seq[ns][0,ts]
                        LC_seq[ns][1,ts-1] = LC_seq[ns][0,ts]
                        LC_seq[ns][1,ts] = LC_seq[ns][0,ts]
                        counter = 0

        CD_counter[ns] = 0

        for ts in range(1, np.size(t_seq_st)):
            if LC_seq[ns][1,ts] != LC_seq[ns][1,ts-1]:
                CD_counter[ns] = CD_counter[ns] + 1
                if CD_counter[ns] <= MAX_CD:
                    break_p[ns, CD_counter[ns]-1] = ts
                    LC_seq_bp[ns, CD_counter[ns]-1] = LC_seq[ns][1,ts-1]
                    LC_seq_bp[ns, CD_counter[ns]] = LC_seq[ns][1,ts]

        if CD_counter[ns] == 0:
            break_p[ns,0] = 0
            LC_seq_bp[ns,0] = LC_seq[ns][1,0]
            LC_seq_bp[ns,1] = LC_seq[ns][1,0]

    np.save(fm.joinpath(savepath, "LC_seq.npy"), LC_seq)
    np.save(fm.joinpath(savepath, "Test_simi_traj.npy"), Test_simi_traj)

    #Output maps
    nyears = len(years)
    outmaps = [None]*nyears
    for ny in range(nyears):
        outmaps[ny] = np.zeros((height, width, 2))

    for row in range(height):
        for col in range(width):
            ns = width*row + col

            if break_p[ns,0] == 0:
                pass
            else:
                z = break_p[ns,0]
                start_z = t_seq_st[z]
                end_z = t_seq_en[z]
                int_z = np.arange(start_z, end_z)
                int_z = np.ceil(int_z/365)

                maxperc = 0
                for ny in range(nyears):
                    perc = np.sum(int_z[int_z == (ny+1)]) / (365*(ny+1))
                    perc = perc*100

                    if perc > maxperc:
                        maxperc = perc
                        outmaps[ny][row,col,0] = LC_seq_bp[ns,1]
                        outmaps[ny][row,col,1] = perc

    for ny in range(nyears):
        outname = tile + '_' + 'CD' + (ny+1) +'.tif'
        sp = fm.joinpath(savepath, outname)
        fm.writeGeoTIFFD(sp, outmaps[ny], geotransform, projection)


#---------------------------------------------------------------------------------------------------#

def load_seeds(tile, ns, feature, col_DATA, **kwargs):
    #SETUP OPTIONS
    years = kwargs.get('years', None)
    loadpath = kwargs.get('loadpath', None)

    Seeds = None
    min_f = []
    max_f = []
    n2 = 'NDI' + str(feature+1)

    for y in years:
        n1 = tile + '_' + y
        filename = fm.joinpath(loadpath, n1, 'NDI_TimeSeries', n2, 'ts.h5')

        with h5py.File(filename, 'r') as hf:
            temp = np.array(hf["ts"])

        min_f.append(np.amin(temp[:,col_DATA:]))
        max_f.append(np.amax(temp[:,col_DATA:]))

        if Seeds is None:
            Seeds = temp[ns,:]
        else:
            Seeds = np.concatenate((Seeds, temp[ns,col_DATA:]), axis=1)
    
    min_f = min(min_f)
    max_f = max(max_f) - min_f

    Seeds = Seeds - min_f
    Seeds = Seeds / max_f
    
    return Seeds
 