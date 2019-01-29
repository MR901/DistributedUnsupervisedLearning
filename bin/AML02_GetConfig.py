# Initial Setup
'''
Description: 
    This file is to be used for initializing main execution file.
Function this file Contains:
    - GetBackSomeDirectoryAndGetAbsPath: Used to get absolute path based on the provided relative path.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
import configparser

# --------------------------------------- GetBackSomeDirectoryAndGetAbsPath --------------------------------------- #
class getConfig:
    def __init__(self):
        self.configSingleJsonFilePath = ''
        self.configSingleIniFilePath = '../config/ICLSSTA_Clustering_Config.ini'
        self.configMultiIniFilePath1 = ''
        self.configMultiIniFilePath2 = ''

    def loadSingleIniConfig(self):
        '''
        load single ini config file
        '''
        config = configparser.ConfigParser()
        config.read(self.configSingleIniFilePath)
        return config
    
    def loadMultiIniConfig(self):
        '''
        load multiple ini config file
        '''
        
        return 0
    
    def loadSingleJsonConfig(self):
        '''
        load single json config file
        '''
        
        return 0
        

    
    # ------------------------------------------------------------------------------------------- #



# ----------------------------------------------------------------------------------------------------------------- #
