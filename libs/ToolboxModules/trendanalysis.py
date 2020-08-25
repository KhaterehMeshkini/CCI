import os
import subprocess
from libs.RSdatamanager import filemanager as fm

#---------------------------------------------------------------------------------------------------#
def manager(tile, **kwargs):
    #SETUP VARIBLES  
    info = kwargs.get('info', True)
    years = kwargs.get('years', None)
    outpath = kwargs.get('outpath', None)
    bfastpath = kwargs.get('bfastpath', None)

    #GET IMAGE INFO
    for y in years:
        name = tile + '_' + y

        featurepath = fm.check_folder(outpath, name, 'Features')
        fn = [f for f in os.listdir(featurepath) if f.endswith('.tif')]
        if len(fn)==0:
            raise IOError('Unable to find input data!')

    img = fm.readGeoTIFFD(fm.joinpath(featurepath,fn[0]), metadata=False)
    height, width, totfeatures = img.shape

    #CHECK TS DATA
    for y in years:
        for feature in range(totfeatures):
            n1 = tile + '_' + y
            n2 = 'NDI' + str(feature+1)
            tspath = fm.check_folder(outpath, n1, 'NDI_TimeSeries', n2)
            if not os.path.exists(fm.joinpath(tspath,'ts.h5')):
                raise IOError('Unable to find input data!')

    #PREPARE PARAMETERS
    height = str(height)
    width = str(width)
    startyear = str(years[0])
    endyear = str(years[-1])
    frequency = str(kwargs.get('frequency', 365))
    tile = str(tile)
    batchsize = str(kwargs.get('batchsize', 200))

    for feature in range(totfeatures):
        if info:
            print('Change detection for feature %i/%i...' % ( (feature+1), totfeatures ), end='\r')
        
        feature = str(feature+1)

        # rscript libs/ToolboxModules/callbfast.R height width startyear endyear frequency tile feature batchsize outpath bfastpath
        process = subprocess.run(['rscript', 'libs/ToolboxModules/callbfast.R', height, width, startyear, endyear, frequency, tile, feature, batchsize, outpath, bfastpath], 
                                    stdout=subprocess.PIPE, 
                                    universal_newlines=True)
                                    
        #print(process.stdout)