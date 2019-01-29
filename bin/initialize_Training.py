# General Functions
'''
Description: 
    This file is to be used for initializing main execution file.
Function this file Contains:
    - GetBackSomeDirectoryAndGetAbsPath: Used to get absolute path based on the provided relative path.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
from AML01_SetLogging import logger
from AML02_GetConfig import getConfig
from AML03_GeneralFunc import LevBasedPrint
from AML__Executor import Executor
# --------------------------------------- GetBackSomeDirectoryAndGetAbsPath --------------------------------------- #

def main(specific_config, generic_config):
    try:        
        Executor(specific_config, generic_config).Execute()
    
    except Exception as ex:
        msg = 'Failed to run automate main file ' + str(ex)
        LevBasedPrint(msg,0); logger.critical(msg)
        raise Exception(msg)


if __name__ == "__main__":
    
    ## initiating logger
    logger.initializeLogger()
    msg = "Logger has been Initialized"
    LevBasedPrint(msg,0); logger.info(msg)
        
    config_varied = {
            'cycleType': 'TrainTest', ## 'TrainTest', 'GllTest'
            'SubID': '''['ecl1']''',
            'TimeFrameBasedBinLengthHr': '12',
            ## i.e. size of the grouping methodology, another way to put this is that the traffic will be grouped based on the timestamps, and here this value represent those group size/ period size 
            }
    msg = "Specific Configuration has been Loaded"
    LevBasedPrint(msg,0); logger.info(msg)
    
    config_generic = getConfig().loadSingleIniConfig()
    msg = "Generic Configuration has been Loaded"
    LevBasedPrint(msg,0); logger.info(msg)
    
    main(config_varied, config_generic)
    
    
    # ------------------------------------------------------------------------------------------- #



# ----------------------------------------------------------------------------------------------------------------- #

