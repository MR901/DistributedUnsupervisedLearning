# General Functions
'''
Description: 
    This file provide some function that are for general use cases.
Function this file Contains:
    - GetBackSomeDirectoryAndGetAbsPath: Used to get absolute path based on the provided relative path.
    - TimeCataloging: To generate a REPORT file for the runtime/crashes of parts of the code.
    - CreateKey: Used to combine columns to return combined value which cann be used as KEY.
    - AddRecommendation: Used to add messages to the recommendation file. When all recommendation are 
                followed delete this file.
    - LevBasedPrint: It is used to print statement based on the level. i.e.  function level.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
import time, os, sys
import pandas as pd

# --------------------------------------------- GenCodeObjectFromFile --------------------------------------------- #
def GenCodeObjectFromFile(filePath):
    '''
    filePath = 'AML01_SetLogging.py'
    codeAsString = open(filePath, "rb").read()
    codeObject = compile(codeAsString, filePath, 'exec')
    exec(codeObject) ## to be placed inside 
    '''
    codeAsString = open(filePath, "rb").read()
    codeObject = compile(codeAsString, filePath, 'exec')
    return codeObject



# --------------------------------------- GetBackSomeDirectoryAndGetAbsPath --------------------------------------- #
def GetBackSomeDirectoryAndGetAbsPath(RelPath, msg = False):
    '''
    DirToMoveTo = ../A/B/
    CurrentAbsDir = /X/Y/Z  ##abs path
    
    returns /X/Y/Z, /X/Y/A/B/
    i.e. returns original and new path
    '''
    curr = str(os.getcwd())
    curr0 = curr
    path = RelPath
    DirToGoTo = path.split('/')
    for dirspli in DirToGoTo:
        if dirspli == '..':
            curr = '/'.join(curr.split('/')[0:-1])
        else:
            curr += '/' + dirspli
    if msg is True: print('Current directory where code is executed :', curr0)
    if msg is True: print('New directory path which was mentioned :', curr)
    return curr0, curr


# ------------------------------------------------- TimeCataloging ------------------------------------------------ #

def TimeCataloging(config, Key, Value, First = 'Off'):
    '''
    To generate a REPORT file for the runtime/crashes of parts of the code.
    '''
    if First == 'On':
        ExecTime = time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime())
        TimeConsumedReport = {
                'ExecutionTime': ExecTime,
                'ExecTimestamp': int(time.time()),
                'ImportInput': '-',
                'ImportBlKeys': '-',
                'ImportFeedbackData': '-',
                'CombineDataStrems': '-',
                'ComputeSizeOfThisIteration': '-',
                'AdaptiveKeySelection': '-',
                'BlacklistingKeys': '-',
                'UpdatingBlacklistLogs': '-',
                'WholeExecutionTime': '-'
            }
    ## Creating a DataFrame Containing Execution Time Results
    ExecTimePath = config('LogPaths','ExecutionTimeTaken')
    col = ['ExecutionTime', 'ExecTimestamp', 'ImportInput', 'ImportBlKeys', 'ImportFeedbackData', 'CombineDataStrems', 'ComputeSizeOfThisIteration',
           'AdaptiveKeySelection', 'BlacklistingKeys', 'UpdatingBlacklistLogs', 'WholeExecutionTime']
    if(os.path.exists(ExecTimePath) is False):
        tempDF = pd.DataFrame(TimeConsumedReport, columns = col, index = [0]) #TimeConsumedReport.keys()
    else:
        tempDF = pd.read_csv(ExecTimePath)
        if First == 'On':
            tempDF = tempDF.append(TimeConsumedReport, ignore_index=True)
    ## Updating Entries
    try:
        tempDF.iloc[(len(tempDF)-1), tempDF.columns.get_loc(Key)] = Value
    except:
        print('Passed Key Doesn\'t Exist in Present Structure')
    ## Saving Locally
    tempDF.to_csv(ExecTimePath, index=False)
    if Key == 'WholeExecutionTime':
        return tempDF.iloc[len(tempDF)-1,:].to_dict()


# --------------------------------------------------- CreateKey --------------------------------------------------- #

def CreateKey(DF, Key_ColToUse):
    '''
    Use to combine columns to generate a key which is seperated by '|'
    eg. Key_ColToUse = sid, bin & IP ==> return sid|Bin|IP 
    '''
    df = DF.copy()
    for col_ind in range(len(Key_ColToUse)):
        I1 = df.index.tolist()
        I2 = df[Key_ColToUse[col_ind]].astype('str').tolist()
        if col_ind == 0:
            df.index = I2
        else:
            df.index = [ "|".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ] #, I3[ind]
    return df.index


# ---------------------------------------------- AddRecommendation ------------------------------------------------ #

def AddRecommendation(msgToAdd, config):
    '''
    Used for adding recommendations inside a single recommendation File
    Delete this file after recommendationhas been followed.
    '''
    filePath = config('LogPaths', 'RecommendationFile')
    _, absPathRecommFile = GetBackSomeDirectoryAndGetAbsPath(filePath)
    NewDf = pd.DataFrame({'Recommendation': msgToAdd}, columns=['Recommendation'], index=[0])
    
    if os.path.exists(absPathRecommFile):
        df = pd.read_csv(filePath)
        if msgToAdd not in list(df['Recommendation'].unique()):
            print('New Recommendation has been added')
            df = pd.concat([df,NewDf], ignore_index=True, sort=False)
        else:
            print('This recommendation is already present, hence not adding.')
    else:
        print('First Recommendation has been added')
        df = NewDf.copy()
    df.to_csv(absPathRecommFile, index = False)


# ------------------------------------------------- LevBasedPrint ------------------------------------------------- #

def LevBasedPrint(txt, level=0, StartOrEnd=0):
    '''
    Use to print statement based on levels
    '''
    if StartOrEnd != 0:
        ## expecting '\t' len = 8 spaces
        print('', '+'+'-'*(112 - 8*level),sep= '\t'*level)
    if len(txt) != 0: print('', txt,sep= '\t'*level + '|'+' ')


# ----------------------------------------------------------------------------------------------------------------- #
#def TimeCataloging(config_clust, Key, Value, First = 'Off'):
#    if First == 'On':
#        CurrTime = time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime())
#        TimeConsumedReport = {
#                'CurrentTime': CurrTime,
#                'ExecTimestamp': int(time.time()),
#                'Task': '-',
#                'PlotDayWisetraffic': '-',
#                'DataImportFromBQ': '-',
#                'DataImportToPython': '-',
#                'DataPreProcessing': '-',
#                'DataExploration': '-',
#                'DimenTransfAndClustering': '-',
#                'SerialDataClustering': '-',
#                'TranfomAndMovingResults': '-',
#                'AlgoClustersEvaluation': '-',
#                'OutputTransfAndEnsemEval': '-',
#                'WholeExecutionTime': '-'
#            }
#    ## Creating a DataFrame Containing Execution Time Results
#    ExecTimePath = config_clust['input']['ExecutionTimeTakenData']
#    col = ['CurrentTime', 'ExecTimestamp', 'Task', 'PlotDayWisetraffic', 'DataImportFromBQ', 'DataImportToPython',
#           'DataPreProcessing', 'DataExploration', 'DimenTransfAndClustering', 'SerialDataClustering', 
#           'TranfomAndMovingResults', 'AlgoClustersEvaluation', 'OutputTransfAndEnsemEval', 'WholeExecutionTime']
#    if(os.path.exists(ExecTimePath) is False):
#        tempDF = pd.DataFrame(TimeConsumedReport, columns = col, index = [0]) #TimeConsumedReport.keys()
#    else:
#        tempDF = pd.read_csv(ExecTimePath)
#        if First == 'On':
#            tempDF = tempDF.append(TimeConsumedReport, ignore_index=True)
#    ## Updating Entries
#    try:
#        tempDF.iloc[(len(tempDF)-1), tempDF.columns.get_loc(Key)] = Value
#    except:
#        print('Passed Key Doesn\'t Exist in Present Structure')
#    ## Saving Locally
#    tempDF.to_csv(ExecTimePath, index=False)
#    if Key == 'WholeExecutionTime':
#        return tempDF.iloc[len(tempDF)-1,:].to_dict()
#
#def MoveTrainedModelsToDesiedDir(config):
#    '''
#    Move ALL files present in the directory, (Any format except 'py')
#    '''
#    FilesFormatToIgnore = ['py']
#    MovingFrom, MovingTo = config['input']['ModelsSaving_TempDir'], config['input']['ModelsSaving_dir']
#    AllFiles = glob.glob(('{}*').format(MovingFrom))
#    AllFilesToMove = [ ele for ele in AllFiles if ele.split('.')[-1] not in FilesFormatToIgnore ]
#    ## Moving the files to other Directory
#    for file in AllFilesToMove:
#        Current = file
#        NewDestination = MovingTo + file.split(MovingFrom)[-1]
#        #print(Current, 'moving to', NewDestination)
#        os.rename(Current, NewDestination)
#    print('||  Generated Model Files have been moved to SavedModels directory.')