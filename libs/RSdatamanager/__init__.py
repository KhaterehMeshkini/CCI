# SETUP PATH FOR RS-Package
import sys, os
package_path = os.path.dirname(__file__) # Retrieve toolbox path
sys.path.insert(0,package_path) # Insert package path into $PYTHONPATH

# INITIALIZE GDAL_DATA 
import platform, os, sys

envpath = os.path.split(sys.executable)[0]
systemOS = platform.system()
if (systemOS=='Linux'):  
    envpath = os.path.split(envpath)[0] 
elif (systemOS=='Windows'):
    pass
else:
    raise Exception('I forgot about MAC!!;)')

gdalpath = [x[0] for x in os.walk(envpath) if x[0].endswith('share'+ os.sep +'gdal')]
projpath = [x[0] for x in os.walk(envpath) if x[0].endswith('share'+ os.sep +'proj')]
if (len(gdalpath) != 1):
    print('Unable to find path to GDAL supporting files: manualy set "GDAL_DATA" environment variable! ')
    
else:
    os.environ['GDAL_DATA'] = gdalpath[0]
    #os.environ['PROJ_LIB'] = projpath[0]

