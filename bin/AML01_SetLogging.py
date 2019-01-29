# Initial Setup
'''
Description: 
    This file is to be used for initializing main execution file.
Function this file Contains:
    - GetBackSomeDirectoryAndGetAbsPath: Used to get absolute path based on the provided relative path.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
import logging, time, os

# --------------------------------------- GetBackSomeDirectoryAndGetAbsPath --------------------------------------- #  

defaultLogConfig = {
        'TagName': 'Trial',
        'SetLevel': 'DEBUG', ## 'DEBUG', 'INFO', 'WARN', 'ERROR'
        'LogsSavingPath': '../logs/logger.log'  ## datetime.datetime.now().strftime('%Y-%m-%d') + '.log'
        }


# --------------------------------------------------- LogCreator -------------------------------------------------- #
class LogCreator:
    codeLogger = None
    
    def __init__(self, logConfig):
        self.logTagName = logConfig['TagName']
        logLevelMap = {"DEBUG" : logging.DEBUG,"WARN"  : logging.WARN,"INFO"  : logging.INFO,"ERROR" : logging.ERROR}
        self.logLevel = logLevelMap[logConfig['SetLevel']]
        self.logFilePath = logConfig['LogsSavingPath']
        try:
            ## create directory if it doesn't exist
            directory = '/'.join(self.logFilePath.split('/')[:-1]) 
            if not os.path.exists(directory): 
                os.makedirs(directory)
        except OSError as e:
            if e.errno:
                raise
    
    def initiateLogger(self):
        try:
            #Create and configure logger
            fh = logging.FileHandler(filename= self.logFilePath, mode= 'a') #'w', 'a'
            fh.setLevel(self.logLevel)
            
            formatter = logging.Formatter('%(asctime)s: %(levelname)s : %(name)s : %(message)s')
            formatter.converter = time.gmtime
            fh.setFormatter(formatter)
            
            #Creating an object
            LogCreator.codeLogger = logging.getLogger(self.logTagName)
            
            #Setting the threshold of logger
            LogCreator.codeLogger.setLevel(self.logLevel)
            LogCreator.codeLogger.addHandler(fh)
            
            print(']] Logger has been initialized, Path: "{}" is used.'.format(self.logFilePath))
            
        except AttributeError as a:
            print (a)
    
    def getLogger(self):
        try:
            if LogCreator.codeLogger is None:
                #print(loggerName, type(LogCreator.codeLogger), LogCreator.codeLogger)
                self.initiateLogger() # LogCreator.initiateLogger()
                return LogCreator.codeLogger
            else:
                return LogCreator.codeLogger
        except Exception as ex:
            print(ex)
    # ------------------------------------------------------------------------------------------- #


# ------------------------------------------------------ Logger --------------------------------------------------- #
class logger:
    __instance = None
    
    def create_instance(self, logConfig):
        try:
            logger.logCre = LogCreator(logConfig).getLogger()
        except:
            raise
    
    def getLoggerInstance():
        if logger.__instance == None:
            logger.__instance = logger()  ###to initiate in second call
        return logger.__instance
    
    @staticmethod
    def initializeLogger(logConfig = defaultLogConfig):# = defaultLogConfig):
        logger.getLoggerInstance().create_instance(logConfig)
    
    @staticmethod
    def debug(msg):
        logger.getLoggerInstance().logCre.debug(msg)
    
    @staticmethod
    def info(msg):
        logger.getLoggerInstance().logCre.info(msg)
    
    @staticmethod
    def warn(msg):
        logger.getLoggerInstance().logCre.warn(msg)
    
    @staticmethod
    def error(msg):
        logger.getLoggerInstance().logCre.error(msg)
    
    @staticmethod
    def critical(msg):
        logger.getLoggerInstance().logCre.critical(msg)
    # ------------------------------------------------------------------------------------------- #



# ----------------------------------------------------------------------------------------------------------------- #
##Test messages 
#logger.debug("Harmless debug Message") 
#logger.info("Just an information") 
#logger.warning("Its a Warning") 
#logger.error("Did you try to divide by zero") 
#logger.critical("Internet is down") 
