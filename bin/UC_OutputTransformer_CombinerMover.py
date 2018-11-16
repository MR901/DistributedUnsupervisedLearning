
import pandas as pd
import glob, os, ast

def ConcatTrainTestFile(Pattern, config_clust, seperator = ',', path = 'relative', msg = 'OFF'):
    '''
    This Function Combine (concatinate) test and Train Files And Return the absolute or relative path of the file that has been created.
    Doesn't import file from cache/ram but load from local storage  
    Combining Selected Files
    
    '''

    FilesOfTheAlgoPairToUseInAdaptive = ast.literal_eval(config_clust['MovingOutputFile']['DimClustAlgoPair'])
    MovingFromDirectory = config_clust['MovingOutputFile']['DirToMoveFrom']
    
    if(msg == 'ON'): print('')
    Files_Train = sorted(glob.glob(MovingFromDirectory + Pattern + '_Train_*.csv'))
    Files_Test = sorted(glob.glob(MovingFromDirectory + Pattern + '_Test_*.csv'))
    CombinedFilesName = []
    if len(Files_Train) == len(Files_Test):
        for fil_ind in range(len(Files_Train)):
    #         print(Files_Train[fil_ind])
            if(Files_Train[fil_ind].split('Train')[1] == Files_Test[fil_ind].split('Test')[1]):
                df_train = pd.read_csv(Files_Train[fil_ind], sep = seperator)   ### Different Seperator is present in Dimension Transformation File
                df_test = pd.read_csv(Files_Test[fil_ind], sep = seperator) 
                tempDF = pd.concat([df_train, df_test], ignore_index=True, sort=False).sample(frac=1).reset_index(drop=True)
                SavingFileName = Files_Train[fil_ind].split('Train_')[0] + 'TrainTest_' + Files_Train[fil_ind].split('Train_')[1]
                tempDF.to_csv(SavingFileName, sep = seperator, index=False)
                del(tempDF)
                if path == 'absolute':
                    CombinedFilesName.append(SavingFileName)
                elif(path == 'relative'):
                    CombinedFilesName.append(SavingFileName.split('/')[len(SavingFileName.split('/')) - 1])
        if(msg == 'ON'): print('Both Train and Test files Are Combined')

    elif (len(Files_Test) == 0):
        for fil_ind in range(len(Files_Train)):
            df_train = pd.read_csv(Files_Train[fil_ind], sep = seperator)
            SavingFileName = Files_Train[fil_ind].split('Train_')[0] + 'TrainTest_' + Files_Train[fil_ind].split('Train_')[1]
            df_train.to_csv(SavingFileName, sep = seperator, index=False)
            del(df_train)
            if path == 'absolute':
                CombinedFilesName.append(SavingFileName)
            elif(path == 'relative'):
                CombinedFilesName.append(SavingFileName.split('/')[len(SavingFileName.split('/')) - 1])

    elif (len(Files_Train) == 0):
        for fil_ind in range(len(Files_Test)):
            df_test = pd.read_csv(Files_Test[fil_ind], sep = seperator)
            SavingFileName = Files_Test[fil_ind].split('Test_')[0] + 'TrainTest_' + Files_Test[fil_ind].split('Test_')[1]
            df_test.to_csv(SavingFileName, sep = seperator, index=False)
            del(df_test)
            if path == 'absolute':
                CombinedFilesName.append(SavingFileName)
            elif(path == 'relative'):
                CombinedFilesName.append(SavingFileName.split('/')[len(SavingFileName.split('/')) - 1])
        if(msg == 'ON'): print('Both Train and Test files Are Combined')
    else:
        if(msg == 'ON'): print('No Train or Test File Exists')

    if(msg == 'ON'): 
        print('Files which are combined :')
        # [ print('  '+file) for file in CombinedFilesName ]
        print([ '  '+file for file in CombinedFilesName ])

    return CombinedFilesName

# ConcatTrainTestFile('DataDimensionTransformation', config_clust, seperator = '|', path = 'relative')
# ConcatTrainTestFile('*ModelData', config_clust, seperator = ',', path = 'relative')



def GenerateCombinedFile(config, msg = 'OFF'):
    '''
    Use to Combine Train and Test files with some Serial files too , outlier and conceptual drift
    '''
    MovingFromDirectory = config['MovingOutputFile']['DirToMoveFrom']
    FilePathBeLike = 'relative'
    
    AllFilesToMove = []
    ## Generating Combined Files of DataDimensionTransformation
    ConcatTrainTestFile('DataDimensionTransformation', config, '|', FilePathBeLike, msg)   ## Not to be moved Therefore not added to the list
    ## Generating Combined Files and Getting the Files That are to be moved
    AllFilesToMove = AllFilesToMove + ConcatTrainTestFile('*ModelData', config, ',', FilePathBeLike, msg)
    
    ### Adding Outlier File to the list of the file that are to be moved to create Cluster
    SerialClustFiles = glob.glob(MovingFromDirectory + '*Serial_{PairToTake}.csv'.format(PairToTake='*'))
    for fil in SerialClustFiles:
        if config['aim']['DevelopOutlierCluster'] in ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y']:
            df = pd.read_csv(fil)
        if config['aim']['DevelopConceptualDriftCluster'] in ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y']: ## club with or
            df = pd.read_csv(fil)
        SavingFileName = fil.split('Serial_')[0] + fil.split('Serial_')[1]
        df.to_csv(SavingFileName, index=False)
        if FilePathBeLike == 'absolute':
            AllFilesToMove.append(SavingFileName)
        elif(FilePathBeLike == 'relative'):
            AllFilesToMove.append(SavingFileName.split('/')[len(SavingFileName.split('/')) - 1])
    
    return AllFilesToMove


def MoveFileToAdaptDir(config_clust):
    '''
    Selection of which files to move is done by GenerateCombinedFile
    '''
    PrintMsg = 'OFF'
    MovingFromDirectory = config_clust['MovingOutputFile']['DirToMoveFrom']
    MovingToTheDirectory = config_clust['MovingOutputFile']['DirToMoveTo']
    
    ## Creating / Generating Combined Result Files
    AllFilesToMove = GenerateCombinedFile(config_clust, msg = PrintMsg)
    
    ## Removing all the Previous csv File From The Directory
    FilesToRemove = glob.glob(('{}*.csv').format(MovingToTheDirectory + ''))
    FilesToRemove = [ fil for fil in FilesToRemove if fil.split('/')[-1].split('_')[0] not in ['FinalResultFile', 'CombinedAll']]
    [ os.unlink(file) for file in FilesToRemove ] 
    
    ## Taking input on whether to move the files or not
    GenerateACombinedFileAndMove = config_clust['MovingOutputFile']['MoveCombinedTrainAndTestFile']
    if GenerateACombinedFileAndMove in ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y']:
        if(PrintMsg == 'ON'): print(AllFilesToMove)
        for file in AllFilesToMove:
            Current = MovingFromDirectory + file
            NewDestination = MovingToTheDirectory + file
            os.rename(Current, NewDestination)
            if(PrintMsg == 'ON'): print ('Files which has been moved/will be used in adaptive algorithm:', file)
    
    ## Creating / Generating Combined Result Files # Creating another Copy so that the Clubbed Results can Also be Evaluated
    GenerateCombinedFile(config_clust, msg = PrintMsg)
    
    print('Files Has been Moved To the ModelData Directory.')
# MoveFileToAdaptDir_1(config_clust)
