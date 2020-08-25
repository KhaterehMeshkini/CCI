import sys, os, time, shutil, json
import configparser
import argparse
from os.path import abspath
main_path = os.path.dirname(os.path.abspath(__file__)) # Retrieve toolbox path
package_path = os.path.join(main_path,'libs') # Generate package path
sys.path.insert(0,package_path) # Insert package path into $PYTHONPATH
from multiprocessing import freeze_support, set_start_method #some stuff for multi-processing support
from libs.RSdatamanager import filemanager as fm

#---------------------------------------------------------------------------------------------------#
def main(datapath, **kwargs):

    from libs.RSdatamanager.Sentinel2.S2L2A import L2Atile, getTileList    
    from libs.ToolboxModules import featurext as m1
    from libs.ToolboxModules import featurets as m2
    from libs.ToolboxModules import trendanalysis as m3
    from libs.ToolboxModules import LandCoverTraining as m4 
    from libs.ToolboxModules import LCclassificationAndCD as m5

    #PREPARE SOME TOOLBOX PARAMETERS
    tilenames = kwargs['options'].get('tilenames', None)
    years = kwargs['options'].get('years', None)
    maindir = kwargs['options'].get('maindir', None)
    outpath = kwargs['options'].get('outpath', None)
    deltemp = kwargs['options'].get('deltemp', True)

    module1 = kwargs['module1'].get('run', False)
    module2 = kwargs['module2'].get('run', False)
    module3 = kwargs['module3'].get('run', False)
    module4 = kwargs['module4'].get('run', False)
    module5 = kwargs['module5'].get('run', False)

    if (module1 or module2):
        #READ DATASETS
        tiledict = getTileList(datapath)
        keys = tiledict.keys()

        for k in keys:
            if k in tilenames:
                tileDatapath = tiledict[k]
                print("Reading Tile-%s." %(k))
                tile = L2Atile(maindir, tileDatapath)

                for y in years:
                    #UPDATE OPTIONS
                    name = k + '_' + y
                    update = {
                        'year': y,
                        'savepath': fm.check_folder(outpath, name)
                    }

                    if module1:
                        #MODULE 1
                        t_mod1 = time.time()
                        options = kwargs.get('module1',{})
                        options.update( update )
                        m1.manager(tile, **options)
                        t_mod1 = (time.time() - t_mod1)/60
                        print("MOD1 TIME = %imin                                        " %( int(t_mod1) ))

                    elif module2:
                        #MODULE 2
                        t_mod2 = time.time()
                        options = kwargs.get('module2',{})
                        options.update( update )
                        m2.manager(k, **options)
                        t_mod2 = (time.time() - t_mod2)/60
                        print("MOD2 TIME = %imin                                        " %( int(t_mod2) ))

                #DELETE TILE-TEMPPATH CONTENT
                if deltemp:
                    flag = shutil.rmtree(tile.temppath())
                    if flag==None:
                        print("Temporary File Content of Tile-%s has been successfully removed!" %(k))

    elif module3:
        for k in tilenames:
            #MODULE 3
            t_mod3 = time.time()
            options = kwargs.get('module3',{})
            m3.manager(k, **options)
            t_mod3 = (time.time() - t_mod3)/60
            print("MOD3 TIME = %imin                                        " %( int(t_mod3) ))

    elif module4:
        for k in tilenames:
            #MODULE 4
            t_mod4 = time.time()
            options = kwargs.get('module4',{})
            m4.manager(k, **options)
            t_mod4 = (time.time() - t_mod4)/60
            print("MOD4 TIME = %imin                                        " %( int(t_mod4) ))

    elif module5:
        for k in tilenames:
            #MODULE 5
            t_mod5 = time.time()
            options = kwargs.get('module5',{})
            m5.manager(k, **options)
            t_mod5 = (time.time() - t_mod5)/60
            print("MOD5 TIME = %imin                                        " %( int(t_mod5) ))
           
#---------------------------------------------------------------------------------------------------#
if (__name__ == '__main__'):
    #MULTIPROCESSING INITIALIZATION
    freeze_support() #needed for windows
    set_start_method('spawn') # because the VSCode debugger (ptvsd) is not fork-safe

    #READ COMMAND ARGUMENTS
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True, metavar='config.ini')
    parser.add_argument('-m1', '--module1', action='store_true', help="run module 1")
    parser.add_argument('-m2', '--module2', action='store_true', help="run module 2")
    parser.add_argument('-m3', '--module3', action='store_true', help="run module 3")
    parser.add_argument('-m4', '--module4', action='store_true', help="run module 4")
    parser.add_argument('-m5', '--module5', action='store_true', help="run module 5")

    args = parser.parse_args()

    configfile = abspath(args.config)
    module1 = args.module1
    module2 = args.module2
    module3 = args.module3
    module4 = args.module4
    module5 = args.module5

    #READ INITIALIZATION FILE AND SETUP OPTIONS
    config = configparser.ConfigParser()
    config.read(configfile)

    datapath = fm.formatPath(config['Paths']['data_path'])

    options = {
        'tilenames': config['Data']['tilenames'].split(','),
        'years': config['Data']['years'].split(','),
        'maindir': fm.formatPath(config['Paths']['main_dir']),
        'outpath': fm.check_folder(config['Paths']['output_path']),
        'info': True,
        'deltemp': False
    }
    
    m1options = {}
    m1options.update(options)
    m1options['run'] = module1

    m2options = {}
    m2options.update(options)
    m2options['run'] = module2
    m2options['blocksize'] = int(config['Module2']['blocksize'])
    m2options['mappath'] = fm.formatPath(config['Paths']['LC_path'])

    m3options = {}
    m3options.update(options)
    m3options['run'] = module3
    m3options['batchsize'] = int(config['Module3']['batchsize'])
    m3options['frequency'] = int(config['Module3']['frequency'])
    m3options['bfastpath'] = fm.formatPath(config['Paths']['bfast_path'])

    m4options = {}
    m4options.update(options)
    m4options['run'] = module4
    m4options['blocksize'] = int(config['Module4']['blocksize'])
    m4options['n_classes'] = int(config['Module4']['n_classes'])
    m4options['multiprocessing'] = config.getboolean('Module4', 'multiprocessing') 
    m4options['weekly'] = config.getboolean('Module4', 'weekly')
    m4options['singlefeaturedtw'] = config.getboolean('Module4', 'singlefeaturedtw')
    m4options['featureselection'] = config.getboolean('Module4', 'featureselection')
    m4options['multifeatureDTW'] = config.getboolean('Module4', 'multifeatureDTW')
    m4options['similarity'] = config.getboolean('Module4', 'similarity')
    m4options['classprototypes'] = config.getboolean('Module4', 'classprototypes')
    m4options['DTW_max_samp'] = int(config['Module4']['DTW_max_samp'])
    m4options['simi_high'] = int(config['Module4']['simi_high'])
    m4options['simi_decr'] = float(config['Module4']['simi_decr'])

    m5options = {}
    m5options.update(options)
    m5options['run'] = module5
    m5options['blocksize'] = int(config['Module5']['blocksize'])
    m5options['n_classes'] = int(config['Module5']['n_classes'])
    m5options['DTW_max_samp'] = int(config['Module5']['DTW_max_samp'])
    m5options['MAX_CD'] = int(config['Module5']['MAX_CD'])

    #CALL MAIN FUNCTION
    main(	datapath = datapath,
            options = options,
			module1 = m1options,
            module2 = m2options,
            module3 = m3options,
            module4 = m4options,
            module5 = m5options
		)
