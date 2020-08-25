import os, time
import numpy as np
import h5py
import scipy
import scipy.stats
import math
from libs.RSdatamanager import filemanager as fm 
from multiprocessing import Pool, Lock, Value
from functools import partial
from pandas.core.common import flatten
from skimage.measure import block_reduce

from pydtw import dtw1d
from pydtw import dtw2d

# import pyximport
# pyximport.install()
# from libs.ToolboxModules import dtw_c


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
    #SETUP DEFAULT OPTIONS 
    info = kwargs.get('info', True)
    years = kwargs.get('years', None)
    outpath = kwargs.get('outpath', None)
    savepath = fm.check_folder(outpath, tile, 'LCTraining_DTW')

    blocksize = kwargs.get('blocksize', 500)
    n_classes = kwargs.get('n_classes', 9)
    multiprocessing = kwargs.get('multiprocessing', True)
    weekly = kwargs.get('weekly', True)

    singlefeaturedtw = kwargs.get('singlefeaturedtw', False)
    featureselection = kwargs.get('featureselection', False)
    multifeatureDTW = kwargs.get('multifeatureDTW', False)
    similarity = kwargs.get('similarity', False)
    classprototypes = kwargs.get('classprototypes', False)

    DTW_max_samp = kwargs.get('DTW_max_samp', 15)   # max number of samples of DTW

    col_nPIXEL = 0
    col_nCLASS = 1
    col_nBAND  = 2
    col_DATA = 3


    ###############################
    # GET INFO AND INITIALIZATION #
    ###############################
    for rootname, _, filenames in os.walk(outpath):
        for f in filenames:
            if (f.endswith('.tif')):
                loadpath = fm.joinpath(rootname, f) 
    img = fm.readGeoTIFFD(loadpath, metadata=False)
    width, height, totfeature = img.shape

    for rootname, _, filenames in os.walk(outpath):
        for f in filenames:
            if (f.endswith('ts.h5')):
                loadpath = fm.joinpath(rootname)
    #NDI_ = fm.loadh5(loadpath, f)
    with h5py.File(fm.joinpath(loadpath,f), 'r') as hf:
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
    
    #Space of analysis parameters
    min_perc_samp_V = np.arange(1, 0.64, -0.03).tolist()                # minimum percentage of total used samples
    min_perc_samp_mod_V = np.ones(12, dtype=float)/np.arange(1,13)      # minimum percentage of used samples per model
    min_perc_samp_mod_V = min_perc_samp_mod_V.tolist()
    
    sepa_b_vs_b = np.zeros((12,12,n_features))


    ##########################################
    # SINGLE FEATURE DTW SIMILARITY MATRICES #
    ##########################################
    if singlefeaturedtw:

        for feature in range(n_features):
            if info:
                t_start = time.time()
                print('Computing DTW feature %i/%i...' % ( (feature+1), n_features ), end='\r') 

            path = fm.check_folder(savepath, "Singlefeature", 'DTW_matrix_B'+str(feature+1))

            for b1 in range(0, n_seeds, nseeds_b):
                Seeds_B_B1 = load_block(tile, b1, feature, col_DATA, **kwargs)
                for b2 in range(0, n_seeds, nseeds_b):
                    Seeds_B_B2 = load_block(tile, b2, feature, col_DATA, **kwargs)
                    singledtw(Seeds_B_B1, Seeds_B_B2, b1, b2, nseeds_b, n_seeds, path, **kwargs)
            
            if info:
                t_end = time.time()
                print('\nMODULE 3.2: calculating DTW for %ith feature..Took %i' % (feature+1 , (t_end-t_start)/60), 'min')


    #Single feature DTW maximum distance
    DTW_max_d_B = np.array(n_features)

    for feature in range(n_features):
        path = fm.check_folder(savepath, "Singlefeature", 'DTW_matrix_B'+str(feature+1))

        max_d = 0
        for b1 in range(0, n_seeds, nseeds_b):
            for b2 in range(0, n_seeds, nseeds_b):
                with h5py.File(filename, 'r') as hf:
                    max_d_block = np.nanmax(np.array(hf["DTW_matrix_B"][b1:b1+nseeds_b, b2:b2+nseeds_b]))
                    if max_d_block > max_d:
                        max_d = max_d_block
        
        DTW_max_d_B[feature] = max_d


    ######################################################
    # FEATURE SPACE ANALYSIS AND FEATURE SPACE REDUCTION #
    ######################################################
    if featureselection:

        for feature in range(n_features):
            if info:
                t_start = time.time()
                print('Feature %i/%i...' % ( (feature+1), n_features ), end='\r')

            sepa_c_vs_c = np.zeros((12,12))
            sepa_c_vs_c_N = np.zeros((12,12))
            
            for i, nc in enumerate(class_int_mask):
                c_r = np.delete(class_int_mask, i).tolist()
                for nc1 in c_r:
                    simi_c_W, simi_c_C = load_block_DTW(seed_class_mask, feature, DTW_max_d_B[feature], nc, nc1, savepath)
                    
                    for col_i, min_perc_samp in enumerate(min_perc_samp_V):
                        for row_i, min_perc_samp_mod in enumerate(min_perc_samp_mod_V):
                            sepa_mea = np.zeros(n_seeds_c[nc-1])
                            for nsc in range(n_seeds_c[nc-1]):
                                simi_c_C_s = simi_c_C[:,nsc]
                                simi_c_C_s = simi_c_C_s[~np.isnan(simi_c_C_s)]
                                simi_c_C_s = sorted(simi_c_C_s, reverse=True)
                                simi_c_C_s = simi_c_C_s[0:math.ceil(n_seeds_c[nc-1]*min_perc_samp_mod*min_perc_samp)]
                                simi_c_W_s = simi_c_W[:,nsc]
                                simi_c_W_s = sorted(simi_c_W_s, reverse=True)
                                simi_c_W_s = simi_c_W_s[0:math.ceil(n_seeds_c[nc-1]*min_perc_samp_mod*min_perc_samp)]
                                pd_C_mu, pd_C_sigma = scipy.stats.distributions.norm.fit(simi_c_C_s)
                                pd_W_mu, pd_W_sigma = scipy.stats.distributions.norm.fit(simi_c_W_s)
                                if pd_C_mu <= pd_W_mu:
                                    sepa_mea[nsc] = np.nan
                                else:
                                    sepa_mea[nsc] = (pd_C_mu - pd_W_mu)/(pd_C_sigma + pd_W_sigma)

                            if (sepa_mea[~np.isnan(sepa_mea)]).size/(n_seeds_c[nc-1]) >= min_perc_samp:
                                sepa_c_vs_c[row_i,col_i] = sepa_c_vs_c[row_i,col_i] + np.mean(sepa_mea[~np.isnan(sepa_mea)])
                                sepa_c_vs_c_N[row_i,col_i] = sepa_c_vs_c_N[row_i,col_i] + 1
            
            sepa_b_vs_b[...,feature] = sepa_c_vs_c * sepa_c_vs_c_N

            if info:
                t_end = time.time()
                print('\nMODULE 3.2: feature selection for %i th feature..Took %i' % (feature+1 , t_end-t_start/60), 'min'  )

        np.save(fm.joinpath(savepath, "sepa_b_vs_b.npy"), sepa_b_vs_b)


    #Search for Class Cluster Parameters
    # select_bands = np.load(fm.joinpath(savepath, "select_bands.npy"))
    sepa_b_vs_b = np.load(fm.joinpath(savepath, "sepa_b_vs_b.npy"))
    # select_bands = select_bands.astype(int).tolist()
    sepa_FS = np.zeros((12,12))
    for nb in range(n_features):
        sepa_FS = sepa_FS + sepa_b_vs_b[:,:,nb]

    mean_sepa_FS = np.mean(sepa_FS, axis=1)
    max_sepa_pos_samp_x_mod_FS = np.argmax(mean_sepa_FS)
    mean_sepa_max_v_FS = sepa_FS[max_sepa_pos_samp_x_mod_FS,:]
    mean_sepa_max_v_derivate_FS = np.diff(mean_sepa_max_v_FS)
    mean_sepa_max_v_derivate_FS = mean_sepa_max_v_derivate_FS/np.max(mean_sepa_max_v_derivate_FS)
    mean_sepa_max_v_derivate_FS = mean_sepa_max_v_derivate_FS * mean_sepa_max_v_FS[1 :]

    max_sepa_pos_perc_samp_FS = np.argmax(mean_sepa_max_v_derivate_FS)
    max_sepa_pos_perc_samp_FS = max_sepa_pos_perc_samp_FS + 1

    min_perc_samp = min_perc_samp_V[max_sepa_pos_perc_samp_FS]
    min_perc_samp_mod = min_perc_samp_V[max_sepa_pos_perc_samp_FS]*min_perc_samp_mod_V[max_sepa_pos_samp_x_mod_FS]
    max_mod_class = np.round(min_perc_samp_V[max_sepa_pos_perc_samp_FS]/min_perc_samp_mod)

    
    #######################################
    # MULTI FEATURE DTW SIMILARITY MATRIX #
    #######################################
    if multifeatureDTW:

        if info:
            t_start = time.time()
            print('Computing multifeature DTW ...', end='\r') 

        # select_bands = np.load(fm.joinpath(savepath, "select_bands.npy"))
        # select_bands = select_bands.astype(int).tolist()

        path = fm.check_folder(savepath, 'Multifeature')

        for b1 in range(0, n_seeds, nseeds_b):
            Seeds_B1 = load_block_multifeature(tile, b1, n_features, col_DATA, **kwargs)
            for b2 in range(0, n_seeds, nseeds_b):
                Seeds_B2 = load_block_multifeature(tile, b1, n_features, col_DATA, **kwargs)
                multidtw(Seeds_B1, Seeds_B2, b1, b2, nseeds_b, n_seeds, path, **kwargs)

        if info:
            t_end = time.time()
            print('\nMODULE 3.2: calculating multifeature DTW ...Took %i' % (t_end-t_start/60), 'min')


    #Multi feature DTW maximum distance
    path = fm.check_folder(savepath, 'Multifeature')

    DTW_max_d = 0
    for b1 in range(0, n_seeds, nseeds_b):
        for b2 in range(0, n_seeds, nseeds_b):
            with h5py.File(filename, 'r') as hf:
                max_d_block = np.nanmax(np.array(hf["DTW_matrix"][b1:b1+nseeds_b, b2:b2+nseeds_b]))
                if max_d_block > DTW_max_d:
                    DTW_max_d = max_d_block

    
    #######################
    # SIMILARITY ANALYSIS #
    #######################
    if similarity:

        simi_high = kwargs.get('simi_high', 1)      # high similarity measure
        simi_decr = kwargs.get('simi_decr', 0.001)  # decrese value of similarity measure

        min_c_vs_c = np.zeros((len(class_int_mask), len(class_int_mask)-1))
        max_c_vs_c = np.zeros((len(class_int_mask), len(class_int_mask)-1))
        mean_c_vs_c = np.zeros((len(class_int_mask), len(class_int_mask)-1))
        simi_low = np.zeros((len(class_int_mask)))

        for i, nc in enumerate(class_int_mask):
            c_r = np.delete(class_int_mask, i).tolist()
            for n, nc1 in enumerate(c_r):
                simi_c_W, simi_c_C = load_block_DTW_multi(seed_class_mask, DTW_max_d, nc, nc1, savepath)

                min_c_s = np.zeros((n_seeds_c[nc-1]))
                max_c_s = np.zeros((n_seeds_c[nc-1]))
                for nsc in range(n_seeds_c[nc-1]):
                    simi_c_C_s = simi_c_C[:,nsc]
                    simi_c_C_s = simi_c_C_s[~np.isnan(simi_c_C_s)]
                    simi_c_C_s = sorted(simi_c_C_s, reverse=True)
                    simi_c_C_s = simi_c_C_s[0:math.ceil(n_seeds_c[nc-1]*min_perc_samp_mod*min_perc_samp)]
                    simi_c_W_s = simi_c_W[:,nsc]
                    simi_c_W_s = sorted(simi_c_W_s, reverse=True)
                    simi_c_W_s = simi_c_W_s[0:math.ceil(n_seeds_c[nc-1]*min_perc_samp_mod*min_perc_samp)]
                    pd_C_mu, pd_C_sigma = scipy.stats.distributions.norm.fit(simi_c_C_s)
                    pd_W_mu, pd_W_sigma = scipy.stats.distributions.norm.fit(simi_c_W_s)
                    if pd_C_mu <= pd_W_mu:
                        min_c_s[nsc] = np.nan
                    else:
                        a = scipy.stats.norm(pd_C_mu, pd_C_sigma).pdf(np.arange(0, 1, simi_decr))
                        b = scipy.stats.norm(pd_W_mu, pd_W_sigma).pdf(np.arange(0, 1, simi_decr))
                        for int_mu in np.int64(np.arange(np.floor(pd_W_mu*(1/simi_decr)), (math.ceil(pd_C_mu*(1/simi_decr))+1), 1000*simi_decr)):                  
                            if(round(b[int_mu-1],1)-round(a[int_mu-1],1) <= 0):
                                min_c_s[nsc] = int_mu*simi_decr
                                break
                            else:
                                min_c_s[nsc] = np.nan
                
                        for int_mu in np.flipud(np.int64(np.arange(np.floor(pd_W_mu*(1/simi_decr)), (math.ceil(pd_C_mu*(1/simi_decr))+1), 1000*simi_decr))):
                            if(round(a[int_mu-1],1)-round(b[int_mu-1],1) <= 0):
                                max_c_s[nsc] = int_mu*simi_decr
                                break
                            else:
                                max_c_s[nsc] = np.nan

                min_c_vs_c[i,n] = np.mean(min_c_s[~np.isnan(min_c_s)])
                max_c_vs_c[i,n] = np.mean(max_c_s[~np.isnan(max_c_s)])
                mean_c_vs_c[i,n] = min_c_vs_c[i,n] #mean([min_c_vs_c(nc,nc1) max_c_vs_c(nc,nc1)])

            simi_low[i] = np.max(mean_c_vs_c[i,:])

        np.save(fm.joinpath(savepath, "simi_low.npy"), simi_low)
           

    ###############################
    # CLASS PROTOTYPES GENERATION #
    ###############################
    if classprototypes:
        
        pass_table = np.zeros(n_classes)            # array of pass/no pass
        models_C = [None]*9                         # variable that contains the models seeds
        used_models = np.zeros(n_classes)           # array of number of model used per class
        used_samples_perc = np.zeros(n_classes)     # array of used samples per class
        used_simi = np.zeros(n_classes)             # array of used similarity per class

        for i, nc in enumerate(class_int_mask):
            max_s = 1                  # set max similarity = 1
            min_s = 0 #simi_low(nc);   # set min similarity

            while pass_table[nc-1]==0:
                _, dist_simi_c = load_block_DTW_multi(seed_class_mask, DTW_max_d, nc, nc, savepath)

                count_simi_c = (dist_simi_c > max_s)                    # check class seed with a similarity major then the threshold
                mean_simi_c = np.empty((n_seeds_c[nc-1])) * np.nan      # initializate the similarity mean value

                # compute the mean similarity value per seed for each accepted other seed
                for nsc in range(n_seeds_c[nc-1]):
                    mean_simi_c[nsc] = np.mean(dist_simi_c[count_simi_c[:,nsc],nsc])
                
                # form a matrix with [seed ID | number of accepted seeds | mean similarity for accepted seeds]
                simi_order = np.column_stack([np.arange(0,n_seeds_c[nc-1],1), np.sum(count_simi_c, axis=0), mean_simi_c])
                
                # order the seeds
                simi_order = simi_order[np.argsort(-simi_order[:, 0])]
                simi_order = np.array(simi_order[np.argsort(-simi_order[:, 0])], dtype=int)
                #simi_order = sorted(simi_order, key=lambda x : x[0], reverse=True)

                models = []  # initialize the models
                
                for nsc in range(n_seeds_c[nc-1]):
                    n_mod = len(models) #number of exist models
                    
                    if n_mod == 0:  # if the number of models is zero, just insert the initial seed
                        models.append(simi_order[nsc,0])
                        
                    else: # else check if any model can accept the new seed
                        simi = np.zeros((n_mod,3)) #initialize the similarity matrix
                        
                        # for each model check if all seed can accept the new one
                        for nm in range(n_mod):
                            seed_int = models[nm] # get seed ID interval
                            # form a matrix with [model ID | acceptance value | mean similarity between new seed and model seeds]
                            simi[nm,:] = [nm, 
                                        np.sum((dist_simi_c[simi_order[nsc,0],seed_int] > max_s)*1)>=(np.ceil(np.size(seed_int)*1)),
                                        np.mean(dist_simi_c[simi_order[nsc,0],seed_int])]
                            
                        # sort the similarity matrix to get the most similar model
                        simi = np.array(simi[np.argsort(-simi[:,2])], dtype=int)
                        
                        if simi[0,1]==1: # if the first model can accept the new seed, insert it 
                            models[simi[0,0]] = list(flatten([models[simi[0,0]], simi_order[nsc,0]]))
                            
                        else:            # otherwise create a new model and insert the seed
                            models.append(simi_order[nsc,0])
                
                n_mod  = np.size(models,0) # get number of models
                # delete models with a percentage of seed lower than the threshold
                for nm in range(n_mod):
                    if np.size(models[nm]) < math.ceil(n_seeds_c[nc-1]*min_perc_samp_mod):
                        models[nm] = []

                models = list(filter(None, models))
                
                u_models = len(models)              # get number of used models
                u_samples = np.zeros(u_models)      # initialized the percentage of used seeds
                # compute the percentage of used seeds
                for um in range(u_models):
                    u_samples[um] = np.size(models[um])
                u_samples = (np.sum(u_samples))/(n_seeds_c[nc-1])
                
                # if the pass condition are respected update the output matrixes
                if ((u_models <= max_mod_class) and (bool(u_samples >= min_perc_samp))):
                    pass_table[nc-1]          = 1
                    models_C[nc-1]            = models
                    used_models[nc-1]         = u_models
                    used_samples_perc[nc-1]   = u_samples
                    used_simi[nc-1]           = max_s
                else:
                    if ((max_s > min_s) and (max_s > simi_decr)):   # otherwise decrease the similarity threshold
                        max_s = max_s - simi_decr
                        print(max_s)
                    else:   # or if not possible put in the pass table a false value
                        pass_table[nc-1] = 2
                        

        # class prototypes creation
        models = [[[] for _ in range(len(n_features))] for _ in range(n_classes)]
        for nc in (class_int_mask):
            for nb_o, nb in enumerate(n_features):
                n_mod = np.size(models_C[nc-1])
                Seeds_FR, Seeds_F = load_Seeds_FR(tile, nb, col_DATA, **kwargs)
                m1 = Seeds_F[:,col_nCLASS]==nc
                m2 = Seeds_F[:,col_nBAND]==nb
                m3 = np.logical_and(m1, m2)
                TABLE_cb = Seeds_FR[m3,:]
                for nm in range(n_mod):
                    TABLE_cbm = TABLE_cb[models_C[nc-1][nm],:]
                    traj = np.mean(TABLE_cbm,0)
                    models[nc-1][nb_o].append(traj)

        # prototypes vs samples
        _, col = Seeds_FR.shape
        Traj1 = np.zeros((len(n_features),col)) 
        sampleVSmodels = np.zeros((n_seeds,n_classes+3)) 

        for ns in range(n_seeds):
            for n, nb in enumerate(n_features):
                Seeds_FR, Seeds_F = load_Seeds_FR(tile, nb, col_DATA, **kwargs)
                Traj1[n,:] = Seeds_FR[ns,:]
                
            sample_simi = [ns, Seeds_F[ns,col_nCLASS], 0]
            for nc in (class_int):
                if nc == 0:
                    max_simi = 0
                else:    
                    n_mod = len(models[nc-1])
                    max_simi = 0
                    for nm in range(n_mod):
                        Traj2 = models[nc-1][nm]
                        simi = ((DTW_max_d - distance_fast(Traj1, Traj2, max_step=DTW_max_samp))/DTW_max_d)
                        max_simi = np.max([max_simi, simi])
            
                sample_simi.append(max_simi)
            
            max_v = max(sample_simi[3:])
            max_p = sample_simi[3:].index(max_v)
            sample_simi[2] = max_p+1
            sampleVSmodels[ns,:] = sample_simi
        
        #confusion matrix between training samples and prototypes
        CM_S = confusion_matrix(sampleVSmodels[:,1], sampleVSmodels[:,2])            


#---------------------------------------------------------------------------------------------------#
def counterinit(c):
    global counter
    counter = c

#---------------------------------------------------------------------------------------------------#
#DTW
#TODO: DTW_matrix = np.where(DTW_matrix==1, 0, DTW_matrix), nan on diagonal of dtw matrix

def singledtw(Seeds_B_B1, Seeds_B_B2, b1, b2, nseeds_b, n_seeds, path, **kwargs):
    #SETUP OPTIONS
    info = kwargs.get('info', True)
    DTW_max_samp = kwargs.get('DTW_max_samp', 15)
    multiprocessing = kwargs.get('multiprocessing', True)

    #ALLOC VARIABLE
    DTW_matrix_B = np.zeros((nseeds_b, nseeds_b))

    if b2<b1:
        #Exploit symmetry
        filename = fm.joinpath(path, 'DTW_matrix_B.h5')

        #read block
        with h5py.File(filename, 'r') as hf:
            DTW_matrix_B = np.array(hf["DTW_matrix_B"][b2:b2+nseeds_b, b1:b1+nseeds_b])
        
        #transpose block
        DTW_matrix_B = DTW_matrix_B.transpose()

        #save block
        with h5py.File(filename, 'a') as hf:
            hf["DTW_matrix_B"][b1:b1+nseeds_b, b2:b2+nseeds_b] = DTW_matrix_B
    else:
        #Compute the DTW distance between the seeds
        if multiprocessing:
            counter = Value('i', 0)
            corecount = int( os.cpu_count()/2 - 1 ) #half to account for virtual cores
            kwargs['corecount'] = corecount

            p = Pool(corecount, initializer=counterinit, initargs=(counter,))
            results = p.map( partial(single_DTW_process, Seeds_B_B1=Seeds_B_B1, Seeds_B_B2=Seeds_B_B2, nseeds_b=nseeds_b,
                            corecount=corecount, DTW_max_samp=DTW_max_samp), 
                            range(nseeds_b) )
            p.close()
            p.join()

            DTW_matrix_B = results
            DTW_matrix_B = np.array(DTW_matrix_B)

        else:
            for ns1 in range(nseeds_b):
                Traj1 = Seeds_B_B1[ns1,:]
                for ns2 in range(nseeds_b):
                    Traj2 = Seeds_B_B2[ns2,:]
                    DTW_matrix_B[ns1,ns2] = distance_fast(Traj1, Traj2, max_step=DTW_max_samp)
            
        #SAVE OUTPUT
        filename = fm.joinpath(path, 'DTW_matrix_B.h5')
        if not os.path.isfile(filename):
            with h5py.File(filename, 'w') as hf:
                hf.create_dataset("DTW_matrix_B", (n_seeds,n_seeds), chunks=True)
                hf["DTW_matrix_B"][b1:b1+nseeds_b, b2:b2+nseeds_b] = DTW_matrix_B
        else:
            with h5py.File(filename, 'a') as hf:
                hf["DTW_matrix_B"][b1:b1+nseeds_b, b2:b2+nseeds_b] = DTW_matrix_B


def multidtw(Seeds_B1, Seeds_B2, b1, b2, nseeds_b, n_seeds, path, **kwargs):
    #SETUP OPTIONS
    info = kwargs.get('info', True)
    DTW_max_samp = kwargs.get('DTW_max_samp', 15)
    multiprocessing = kwargs.get('multiprocessing', True)

    #ALLOC VARIABLE
    DTW_matrix = np.zeros((nseeds_b, nseeds_b))

    #Compute the DTW distance between the seeds
    if multiprocessing:
        counter = Value('i', 0)
        corecount = int( os.cpu_count()/2 - 1 ) #half to account for virtual cores
        kwargs['corecount'] = corecount

        p = Pool(corecount, initializer=counterinit, initargs=(counter,))
        results = p.map( partial(multi_DTW_process, Seeds_B1=Seeds_B1, Seeds_B2=Seeds_B2, nseeds_b=nseeds_b,
                        corecount=corecount, DTW_max_samp=DTW_max_samp), 
                        range(nseeds_b) )
        p.close()
        p.join()

        DTW_matrix = results
        DTW_matrix = np.array(DTW_matrix)

    else:
        Traj1 = np.zeros((Seeds_B1.shape[2], Seeds_B1.shape[1]))
        Traj2 = np.zeros((Seeds_B2.shape[2], Seeds_B2.shape[1]))
        for ns1 in range(nseeds_b):
            for f in range(Seeds_B1.shape[2]):
                Traj1[f] = Seeds_B1[ns1,:,f]
            for ns2 in range(nseeds_b):
                for f in range(Seeds_B2.shape[2]):
                    Traj2[f] = Seeds_B2[ns2,:,f]
                DTW_matrix[ns1,ns2] = distance_fast(Traj1, Traj2, max_step=DTW_max_samp)
        
    #SAVE OUTPUT
    filename = fm.joinpath(path, 'DTW_matrix.h5')
    if not os.path.isfile(filename):
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("DTW_matrix", (n_seeds,n_seeds), chunks=True)
            hf["DTW_matrix"][b1:b1+nseeds_b, b2:b2+nseeds_b] = DTW_matrix
    else:
        with h5py.File(filename, 'a') as hf:
            hf["DTW_matrix"][b1:b1+nseeds_b, b2:b2+nseeds_b] = DTW_matrix


def single_DTW_process(ns1, Seeds_B_B1, Seeds_B_B2, nseeds_b, **kwargs):
    #SETUP OPTIONS
    info = kwargs.get('info', True)
    DTW_max_samp = kwargs.get('DTW_max_samp', 15)
    corecount = kwargs.get('corecount', 2)
   
    time.sleep(.1)
    with counter.get_lock():
        counter.value += 1
        if info:
            print ('\r(%i-core(s))Calculating DTW %i/%i     ' 
                    % (corecount, counter.value, nseeds_b), end='\r')

    DTW_matrix_B = np.zeros(nseeds_b)

    for ns2 in range(nseeds_b):
        Traj1 = Seeds_B_B1[ns1]
        Traj2 = Seeds_B_B2[ns2]
        DTW_matrix_B[ns2] = distance_fast(Traj1, Traj2, max_step=DTW_max_samp)
    
    return DTW_matrix_B


def multi_DTW_process(ns1, Seeds_B1, Seeds_B2, nseeds_b, **kwargs):
    #SETUP OPTIONS
    info = kwargs.get('info', True)
    DTW_max_samp = kwargs.get('DTW_max_samp', 15)
    corecount = kwargs.get('corecount', 2)
   
    time.sleep(.1)
    with counter.get_lock():
        counter.value += 1
        if info:
            print ('\r(%i-core(s))Calculating DTW %i/%i     ' 
                    % (corecount, counter.value, nseeds_b), end='\r')

    DTW_matrix = np.zeros(nseeds_b)

    Traj1 = np.zeros((Seeds_B1.shape[2], Seeds_B1.shape[1]))
    Traj2 = np.zeros((Seeds_B2.shape[2], Seeds_B2.shape[1]))

    for f in range(Seeds_B1.shape[2]):
        Traj1[f] = Seeds_B1[ns1,:,f]

    for ns2 in range(nseeds_b):
        for f in range(Seeds_B2.shape[2]):
            Traj2[f] = Seeds_B2[ns2,:,f]
        DTW_matrix[ns2] = distance_fast(Traj1, Traj2, max_step=DTW_max_samp)
    
    return DTW_matrix


#---------------------------------------------------------------------------------------------------#
#LOAD DATA

def load_block(tile, b, feature, col_DATA, **kwargs):
    #SETUP OPTIONS
    years = kwargs.get('years', None)
    blocksize = kwargs.get('blocksize', 200)
    outpath = kwargs.get('outpath', None)
    weekly = kwargs.get('weekly', True)
    
    Seeds_B_B = None
    min_f = []
    max_f = []
    n2 = 'NDI' + str(feature+1)

    for y in years:
        n1 = tile + '_' + y
        filename = fm.joinpath(outpath, n1, 'NDI_TimeSeries', n2, 'ts.h5')

        with h5py.File(filename, 'r') as hf:
            temp = np.array(hf["ts"][:,col_DATA:])
        #temp = fm.loadh5(fm.joinpath(outpath, n1, 'NDI_TimeSeries', n2), 'ts.h5')
        #temp = temp[:,col_DATA:]

        if weekly:
            temp = block_reduce(temp, block_size=(1,7), func=np.median, cval=np.median(temp))
        
        min_f.append(np.amin(temp))
        max_f.append(np.amax(temp))
        if Seeds_B_B is None:
            Seeds_B_B = temp[b:(b+blocksize),:]
        else:
            Seeds_B_B = np.concatenate((Seeds_B_B, temp[b:(b+blocksize),:]), axis=1)
        
    min_f = min(min_f)
    max_f = max(max_f) - min_f
    Seeds_B_B = Seeds_B_B - min_f
    Seeds_B_B = Seeds_B_B / max_f
    
    return Seeds_B_B    


def load_block_multifeature(tile, b, n_features, col_DATA, **kwargs):
    #SETUP OPTIONS
    years = kwargs.get('years', None)
    blocksize = kwargs.get('blocksize', 200)
    weekly = kwargs.get('weekly', True)

    log = int(years[-1]) - int(years[0])
    if weekly:
        Seeds_B = np.zeros((blocksize, 53*(log+1), n_features))
    else:
        Seeds_B = np.zeros((blocksize, 365*(log+1), n_features))

    for feature in range(n_features):
        Seeds_B[:,:,feature] = load_block(tile, b, feature, col_DATA, **kwargs)

    return Seeds_B


def load_Seeds_FR(tile, feature, col_DATA, **kwargs):
    #SETUP OPTIONS
    years = kwargs.get('years', None)
    outpath = kwargs.get('outpath', None)

    Seeds_FR = None
    n2 = 'NDI' + str(feature+1)

    for y in years:
        n1 = tile + '_' + y
        filename = fm.joinpath(outpath, n1, 'NDI_TimeSeries', n2, 'ts.h5')

        with h5py.File(filename, 'r') as hf:        
            if Seeds_FR is None:
                Seeds_F = np.array(hf["ts"][:,0:col_DATA])
                Seeds_FR = np.array(hf["ts"][:,col_DATA:])
            else:
                Seeds_FR = np.concatenate((Seeds_FR, np.array(hf["ts"][:,col_DATA:])), axis=1)
    
    min_f = np.min(Seeds_FR)
    Seeds_FR = Seeds_FR - min_f
    max_f = np.max(Seeds_FR)
    Seeds_FR = Seeds_FR / max_f  
    
    return Seeds_FR, Seeds_F


def load_block_DTW(seed_class_mask, feature, max_d, nc, nc1, savepath):

    mask = (seed_class_mask==nc)
    mask1 = (seed_class_mask==nc1)

    path = fm.check_folder(savepath, "Singlefeature", 'DTW_matrix_B'+str(feature+1))
    with h5py.File(filename, 'r') as hf:
        simi_c_W = np.array(hf["DTW_matrix_B"][mask1, mask])
        simi_c_C = np.array(hf["DTW_matrix_B"][mask, mask])
    
    simi_c_W = np.negative(simi_c_W-max_d) / max_d
    simi_c_C = np.negative(simi_c_C-max_d) / max_d

    return simi_c_W, simi_c_C


def load_block_DTW_multi(seed_class_mask, max_d, nc, nc1, savepath):
    
    mask = (seed_class_mask==nc)
    mask1 = (seed_class_mask==nc1)

    path = fm.check_folder(savepath, "Multifeature", 'DTW_matrix')
    with h5py.File(filename, 'r') as hf:
        simi_c_W = np.array(hf["DTW_matrix"][mask1, mask])
        simi_c_C = np.array(hf["DTW_matrix"][mask, mask])
     
    simi_c_W = np.negative(simi_c_W-max_d) / max_d
    simi_c_C = np.negative(simi_c_C-max_d) / max_d

    return simi_c_W, simi_c_C    


#---------------------------------------------------------------------------------------------------#
#DTW

def distance_fast(Traj1, Traj2, max_step):
    if Traj1.ndim == 1:
        cost_matrix, cost, alignmend_a, alignmend_b = dtw1d(Traj1, Traj2)
    else:
        Traj1 = np.ascontiguousarray(Traj1.transpose())
        Traj2 = np.ascontiguousarray(Traj2.transpose())
        cost_matrix, cost, alignmend_a, alignmend_b = dtw2d(Traj1, Traj2)
    return cost

# def distance_fast(s1, s2, window=None, max_dist=None,
#                   max_step=None, max_length_diff=None, penalty=None, psi=None):
#     """Fast C version of :meth:`distance`.
#     Note: the series are expected to be arrays of the type ``double``.
#     Thus ``numpy.array([1,2,3], dtype=numpy.double)`` or
#     ``array.array('d', [1,2,3])``
#     """
#     if dtw_c is None:
#         _print_library_missing()
#         return None
#     if window is None:
#         window = 0
#     if max_dist is None:
#         max_dist = 0
#     if max_step is None:
#         max_step = 0
#     if max_length_diff is None:
#         max_length_diff = 0
#     if penalty is None:
#         penalty = 0
#     if psi is None:
#         psi = 0
#     d = dtw_c.distance_nogil(s1, s2, window,
#                              max_dist=max_dist,
#                              max_step=max_step,
#                              max_length_diff=max_length_diff,
#                              penalty=penalty,
#                              psi=psi)
#     return d