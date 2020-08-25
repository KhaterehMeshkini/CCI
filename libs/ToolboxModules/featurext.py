import time
import numpy as np
from libs.RSdatamanager import filemanager as fm

#---------------------------------------------------------------------------------------------------#
def manager(tile, **kwargs):
    #SETUP VARIBLES
    info = kwargs.get('info', True)
    year = kwargs.get('year', None)
    savepath = fm.check_folder(kwargs.get('savepath', None), 'Features')

    #GET FEATURES
    yearts,_,_ = tile.gettimeseries(year=year, option='farming')   
    _feature(yearts, savepath, **kwargs) 

#---------------------------------------------------------------------------------------------------#
#COMPUTE INDEX  
def _feature(ts, path, **kwargs):
  
    info = kwargs.get('info',True)
    ts_length = kwargs.get("ts_legth", len(ts) ) 
    
    if info:
        print('Extracting features for each image:')
        t_start = time.time()

    #Get some information from data
    height, width = ts[0].feature('NDVI').shape

    ts = sorted(ts, key=lambda x: x.InvalidPixNum())[0:ts_length]
    totimg = len(ts)
    totfeature = 8
    
    #Compute Index Statistics   
    for idx,img in enumerate(ts):
        if info:        
            print('.. %i/%i      ' % ( (idx+1), totimg ), end='\r' )   
        
        feature = np.empty((height, width, totfeature))
        #Compute Index
        b1 = img.feature_resc('BLUE', dtype=np.float32)
        b2 = img.feature_resc('RED', dtype=np.float32)
        b3 = img.feature_resc('GREEN', dtype=np.float32)
        b4 = img.feature_resc('NIR', dtype=np.float32)
        b5 = img.feature_resc('SWIR1', dtype=np.float32)
        b6 = img.feature_resc('SWIR2', dtype=np.float32)

        feature[..., 0] = _ndi(b6,b1)
        feature[..., 1] = _ndi(b1,b2)
        feature[..., 2] = _ndi(b5,b3)
        feature[..., 3] = _ndi(b1,b5)
        feature[..., 4] = _ndi(b6,b4)
        feature[..., 5] = _ndi(b5,b2)
        feature[..., 6] = _ndi(b6,b2)
        feature[..., 7] = _ndi(b4,b2)
        
        #Manipulate features
        feature[feature>1] = 1
        feature[feature<-1] = -1
        
        #Save features
        geotransform, projection = fm.getGeoTIFFmeta( ts[0].featurepath()['B04'] )
        sp = fm.joinpath(path, str(img._metadata['tile'])+'_'+str(img._metadata['date'])+'T'+str(img._metadata['time'])+'_NDI.tif')
        fm.writeGeoTIFFD(sp, feature, geotransform, projection)
    
    if info:
        t_end = time.time()
        print('\nMODULE 1: extracting features..Took ', (t_end-t_start)/60, 'min')
    

def _ndi(b1,b2):

    denom = b1 + b2
    nom = (b1-b2)
    index = nom/denom

    return index      
