# Executor Class
'''
Description: 
    This file is the main execution file.
Class this file Contains:
    - Execute: Used to initiate the code.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
#import matplotlib 
#matplotlib.use('Agg')
#import configparser
import pandas as pd
import os, glob, ast, time

from AML01_SetLogging import LogCreator
from AML03_GeneralFunc import LevBasedPrint

logger = LogCreator.getlogger()




# ---------------------------------------------------- Executor --------------------------------------------------- #
class Executor:
    
    def __init__(self, specific_config, generic_config):
        timDic = {}
        for i in range(7):  timDic['t'+str(i)] = '-'
        timDic['t0'] = int(time.time())
        self.exetime_dict = timDic
        self.specific_config = specific_config
        self.generic_config = generic_config
    
    
    def Execute(self):
        
        # --------------------------------------<<<  Start  >>>-------------------------------------- #
        msg = 'Execution Start ' + str(self.exetime_dict['t0'])
        LevBasedPrint(msg,0); logger.info(msg)
        
        # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
#        cycleType = self.specific_config['cycleType']
        msg = 'Inside "'+self.Execute.__name__+'" function and configurations for this has been set.'
        LevBasedPrint(msg,0,1); logger.info(msg)
        
        # -------------------------------<<<  Getting Input Data  >>>-------------------------------- #
        inp = ImportData(self.specific_config, self.generic_config)
        InputDF = inp.ImportDataFromBQ()
        # LevBasedPrint(str(inp), 0)
        msg = 'Input dataframe is having a shape of {}'.format(InputDF.shape)
        LevBasedPrint(msg,0); logger.info(msg)
        msg = 'Getting Input Data    | Complete.'
        LevBasedPrint(msg,0); logger.info(msg)
        
#        self.exetime_dict['t1'], timeConsum = int(time.time()), int(time.time()) - self.exetime_dict['t0']
#        TimeCataloging(self.generic_config, 'ImportInput', timeConsum, First = 'On')
#        
        
    

    
