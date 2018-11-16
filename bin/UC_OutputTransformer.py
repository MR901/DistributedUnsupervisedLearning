
import pandas as pd, glob, os, re, numpy as np
import time
import itertools as it
from UC_ClusterEvaluation import ClustersEvaluation
import glob, re, sys, ast


########################################
def CreateKey(DF, Key_ColToUse):
    df = DF.copy()
    for col_ind in range(len(Key_ColToUse)):
        I1 = df.index.tolist()
        I2 = df[Key_ColToUse[col_ind]].astype('str').tolist()
        if col_ind == 0:
            df.index = I2
        else:
            df.index = [ "|".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ] #, I3[ind]
    return df.index


def GenerateClustModelFileFromAnomFile(config):
    '''
    Use to Generate Bins on the Score Columns of the Anomaly File 
    Hence Converting Anomaly Files To ClusterModelFile
    Also Removes the Previous Anomaly File
    '''
    OutputDatasetDir = str(config['MovingOutputFile']['DirToMoveTo'])
    AnomalityBinsCnt = int(config['OutputTransformer']['NoOfAnomalousBins'])
    ThreshNoAnomality = int(config['OutputTransformer']['ThresholdForNoAnomalous'])

    AnomModFileNames = glob.glob(('{}*.csv').format(OutputDatasetDir + '/AnomalyModelData_'))
    if(len(AnomModFileNames)!=0): 
        print('Anomality File Loaded for Bin Transformation:')
    for FileIndex in range(len(AnomModFileNames)):
        # print('\t' + AnomModFileNames[FileIndex])
        tempDF = pd.read_csv(AnomModFileNames[FileIndex])
        tempDF.drop(['cluster'], axis = 1, inplace = True)
        # display(tempDF.head())
        Model_Name = re.search(pattern='AnomalyModelData_(\\S*).csv', string=AnomModFileNames[FileIndex]).group(1)
        if AnomalityBinsCnt == 4:
            tempDF['cluster'] = np.where(tempDF['score'] >= ThreshNoAnomality, 'NoAnomaly', pd.cut(np.where(tempDF['score'] >= ThreshNoAnomality, 0, tempDF['score']), AnomalityBinsCnt, labels=['VeryHighlyAnomalyScore', 'HighlyAnomalyScore', 'MediumAnomalyScore', 'LowAnomalyScore']))
        else:
            tempDF['cluster'] = np.where(tempDF['score'] >= ThreshNoAnomality, 'NoAnomaly', pd.cut(np.where(tempDF['score'] >= ThreshNoAnomality, 0, tempDF['score']), AnomalityBinsCnt))
        print('\t>>Saving "' + AnomModFileNames[FileIndex] + '" file having score transformed to bins')
        ## Used Again --- remove at later stage
        IdentifierColumns = [col for col in list(tempDF.columns) if col not in list(tempDF.filter(like='cluster', axis=1).columns)]
        ## Selecting Some Specific Columns
        tempDF = tempDF[IdentifierColumns + ['cluster']]
        tempDF.drop(['score'], axis = 1, inplace = True)
        tempDF.to_csv(OutputDatasetDir + '/ClusterModelData_' + Model_Name + '.csv', index=False)
        ## Removing the previous File
        os.unlink(AnomModFileNames[FileIndex])
        # display(tempDF.head())



def GenerateSingleResultFileOfClusterModel(config_clust):
    '''
    Combine Multiple ClusterModelData files into a single file
    '''
    OutputDatasetDir = str(config_clust['MovingOutputFile']['DirToMoveTo'])
    
    ## Getting the Keys Present in Conceptual Drift Data base To be removed From Analysis
    KeyFormat = ast.literal_eval(config_clust['DataProcessing_General']['KeyFormat'])  ## Used To Create Key
    ConceptDriftFilePath = config_clust['input']['ConceptualDriftDatabase']
    KeyFormat = KeyFormat + ['RecentHit_TimeStamp']
    if os.path.exists(ConceptDriftFilePath):
        tempDF = pd.read_csv(ConceptDriftFilePath)
        KeysInConceptDrift = CreateKey(tempDF, KeyFormat).tolist()
    else:
        KeysInConceptDrift = []
    # print('Keys that have exhibited Conceptual Drift :', KeysInConceptDrift)

    ## Ensembling Cluster Results
    ## ClustModFileNames --- Includes Clubbed Anomaly and Clustering Results but not the outlier results
    ## Importing All Clubbed on Clustering + Anomaly(clubbed) Result file
    ClustModFileNames = sorted(glob.glob(('{}*.csv').format(OutputDatasetDir + '/*ClusterModelData_')))
    DF = pd.DataFrame()
    print('\nFile Loaded :')
    for FileIndex in range(len(ClustModFileNames)):
        #print('\t File: ' + ClustModFileNames[FileIndex])
        tempDF = pd.read_csv(ClustModFileNames[FileIndex])
        ini_shape = tempDF.shape
        #print(len(tempDF['KEY'].unique()))

        ## Removing Keys Present in Conceptual Drift   
        tempDF.index = tempDF['KEY'].tolist() 
        #tempDF.drop_duplicates(keep = 'first', inplace = True)
        tempDF.drop(index= KeysInConceptDrift, inplace=True, errors= 'ignore')

        ## Creating a new Index That Will be used For Joining
        tempDF.reset_index(drop=True, inplace=True)
        tempDF.index = CreateKey(tempDF, ['KEY', 'RecentHit_TimeStamp', 'Probable_IsBot'])

        ## Getiing Model Name
        #Model_Name = re.search(pattern='\\S*ModelData_(\\S*).csv', string=ClustModFileNames[FileIndex]).group(1)
        Model_Name = (ClustModFileNames[FileIndex].split('__')[-2] + '__' + ClustModFileNames[FileIndex].split('__')[-1]).split('.')[0]
        print('\tFile : ' + Model_Name)
        print('\tInitial Shape: ' + str(ini_shape) + '\t---->>> \tFinal Shape: ' + str(tempDF.shape))
        # print(len(tempDF.index.unique()))
        tempDF[Model_Name + '_ClusterName'] = Model_Name + '_' + tempDF['cluster'].astype(str)

        IdentifierColumns = [col for col in list(tempDF.columns) if col not in list(tempDF.filter(like='ClusterName', axis=1).columns)]
        IdentifierColumns.remove('cluster')
        tempDF = tempDF[IdentifierColumns + [Model_Name + '_ClusterName']]
        # display(tempDF.head())
        if FileIndex == 0:
            DF = pd.DataFrame.copy(tempDF[IdentifierColumns + [ i for i in tempDF.filter(like='ClusterName', axis=1).columns ]])
        else:
            tempDF = tempDF[['KEY'] + [ i for i in tempDF.filter(like='ClusterName', axis=1).columns ]]
            tempDF = tempDF[[ i for i in tempDF.filter(like='ClusterName', axis=1).columns ]]

            DF = pd.merge(DF, tempDF, how='left', left_on = ['KEY'], right_on = ['KEY'], left_index= True, right_index= True, suffixes=('', '_y') ).fillna(value=Model_Name + '_NA')
    #         DF = DF.join(tempDF, how='left')
    #         DF = pd.merge(DF, tempDF, how='left').fillna(value=Model_Name + '_NA')  ### Memory Error
            #DF = DF.set_index('KEY').join(tempDF.set_index('KEY'), how='left').reset_index().fillna(value=Model_Name + '_NA')
            print('\tCombined DF Shape:', DF.shape)
            #display(DF.head())
        del(tempDF)
    #print(DF.columns)
    DF = DF.reset_index(drop=True)
    
    return DF


# def ChunkWiseLeftJoin(DF1, DF2):
#     '''
#     https://stackoverflow.com/questions/47386405/memoryerror-when-i-merge-two-pandas-data-frames
#     '''


def GenerateMultipleEnsembleClusterEvalReport(DF, config_clust):
    '''
    To Generate Multiple Ensembles and the Cumpute multiple Eval Results
    '''
    ## Generate Multiple Ensemble Based Cluster Evaluation of the TrainTest_ModelName file that are present
    DF = DF.copy()
    
    IdentifierColumns = [ col for col in list(DF.columns) if col not in list(DF.filter(like='ClusterName', axis=1).columns) ]
    ColCluster_li = [ col for col in DF.columns if col not in IdentifierColumns ]
    
    ## Generating all possible combination from the ModelData Directory, that are possible in While
    AllCombinations_li = []
    for i in range(1,1+len(ColCluster_li)):
        AllCombinations_li.append(list(it.combinations(ColCluster_li,i)))
    AllCombinations_li
    
    ClustEvalResultFileName = time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime()) + '___' + 'EnsembledModClusEvalReport.csv' 
    print(ClustEvalResultFileName)
    for li in AllCombinations_li:
        for comb in li:
            ## Selecting some selected columns that are to be ensembled from the All Joined DataFrame
            tempDF = DF[IdentifierColumns + list(comb)].copy()
            # print(tempDF.columns)
            print('\nModels Used in Ensembling:', '|&|'.join(list(comb)))
            ## Ensemble Results By Joining Multiple File Results By '__'
            tempDF['Model_ClusterName'] = tempDF[tempDF.filter(like='ClusterName', axis=1).columns].apply(lambda x: ('__').join(x), axis=1)
            tempDF = tempDF[IdentifierColumns + ['Model_ClusterName']]
            tempDF = tempDF.rename(columns={'RecentHit_TimeStamp':'MostRecentHitTimeStamp', 'Model_ClusterName': 'cluster'})

            ### Evaluating the Cluster Ensemble Files
            #path = 'data/keyModelsData/ClusterModelData_TrainTest_PCA_KMeans.csv'
            #ClustersEvaluation(config_clust, 'SingleFile', None, (path, 'path'))
            ClustersEvaluation(config_clust, 'SingleFile', ClustEvalResultFileName, (tempDF, '|&|'.join(list(comb))))        

            del(tempDF)

    print('Cluster Evaluation File Written Complete')
    EnsembleResultsEvalDF = pd.read_csv(config_clust['input']['FigSavingLoc_dir'] + ClustEvalResultFileName).set_index(['Algorithm'])
    
    return EnsembleResultsEvalDF
# GenerateMultipleEnsembleClusterEvalReport(DF, config_clust)



def AddSequentialFilesAsClusters(DF, config):
    '''
    To Add additional Cluster in the Existent Data Frame, these can be Outlier and Conceptual Drift Clusters Key
    '''
    Df = DF.copy()
    OutputDatasetDir = str(config['MovingOutputFile']['DirToMoveTo'])
    
    ### Load SequentialAdditionModelData File
    SeqModFileNames = glob.glob(('{}*.csv').format(OutputDatasetDir + '/SequentialAdditionModelData_'))

    for SeqAddFile in SeqModFileNames:
        print('File Loaded :', SeqAddFile)

        tempDF = pd.read_csv(SeqAddFile)
        # Model_Name = re.search(pattern='ClusterModelData_(\\S*).csv', string=OutliersFileName[0]).group(1)
        ModelName = SeqAddFile.split('ModelData__')[-1].split('.')[0]
        ModelName = ModelName.split('_With_')[0]
        print('Working On:', ModelName)

        tempDF[ModelName + '_ClusterName'] = ModelName + '_' + tempDF['cluster'].astype(str)

        # IdentifierColumns = [col for col in list(tempDF.columns) if col not in list(tempDF.filter(like='ClusterName', axis=1).columns)]
        # tempDF = tempDF[IdentifierColumns + [Model_Name + '_ClusterName']]

        tempDF.drop(['cluster'], axis = 1, inplace = True)
        tempDF.drop(['Probable_IsBot'], axis = 1, inplace = True)
        tempDF = tempDF.rename(columns={'RecentHit_TimeStamp':'MostRecentHitTimeStamp', [ i for i in tempDF.filter(like='ClusterName', axis=1).columns ][0]: 'cluster'})

        prevshap = Df.shape
        #Df = pd.concat([Df, tempDF.head()], ignore_index=True)#.sample(frac =1).reset_index(drop =True)
        Df = Df.append(tempDF, ignore_index=True).sample(frac =1).reset_index(drop =True)
        print('Shape has changed from :', prevshap, ' + ', tempDF.shape, '\t---->>>\t', Df.shape)

    return Df
# DF = AddSequentialFilesAsClusters(DF)


def KeyFormatConverter(df, config_clust):
    '''
    To be used When Key Format Needs to Converted at a later Stage
    '''
    DF = df.copy()
    ### Checking if Key Format is like this "SID__Date__IP". If it is convert it to "SID|Bin_x|IP" here using heuristic method.
    if len(DF['KEY'][0].split('__')) > 1:

        CurrentTimeStamp = int(time.time())
        BinsSize = int(float(config_clust['env']['BinSizeBasedOnPeriod_inhr']) *3600)

        IP_li = [elem.split('__')[2] for elem in DF['KEY'].tolist()]
        SIDs_li = [elem.split('__')[0] for elem in DF['KEY'].tolist()]

        UpdatedKeyList = []
        for ind in range(len(DF['MostRecentHitTimeStamp'])):
            BinToTake = int(CurrentTimeStamp - DF['MostRecentHitTimeStamp'].tolist()[ind])/ BinsSize
            UpdatedKeyList.append(SIDs_li[ind] + '|' + 'Bin_' + str(BinToTake) + '|' + IP_li[ind])
        # UpdatedKeyList

        DF['UpdatedKey'] = UpdatedKeyList
        print("Total Length", len(DF['UpdatedKey']))
        print("Total Length When New Unique Updated Keys are Considered", len(DF['UpdatedKey'].unique()))
        ## Now because of this new transform there will be a gradient b/w orignal key and new key 
        if len(DF['UpdatedKey'].value_counts()[DF['UpdatedKey'].value_counts() > 1]) > 0:  ## checking if there are duplicate observation for updated Key
            TestView = DF['UpdatedKey'].value_counts()[DF['UpdatedKey'].value_counts() > 1].index[0]
            df = DF.loc[DF['UpdatedKey'] == TestView]
            print('Originally there were multiple key from this IP')
            print(df)
            print('Since this function work only to make data Consistent in a heuristic way, The scope of this function could only be extended to select the newer behaviour and drop older behaviours.')
            print(df.loc[df.groupby(['UpdatedKey'], sort=False)['MostRecentHitTimeStamp'].transform(max) == df['MostRecentHitTimeStamp']])

            ## Now Removing the duplicate observation for the UpdatedKey
            DF = DF.loc[DF.groupby(['UpdatedKey'], sort=False)['MostRecentHitTimeStamp'].transform(max) == DF['MostRecentHitTimeStamp']].reset_index(drop = True)

        DF = DF.drop(['KEY'], axis=1)
        DF = DF.rename(columns={'UpdatedKey':'KEY'})
        DF = DF[['KEY', 'MostRecentHitTimeStamp', 'cluster']]

    return DF


def CombineAndWriteFinalCombinedResult(config_clust):
    '''
    '''
    print('Combining Final Train and Test Result Files')
    ### Variant with Configurable Setting using config_clust.ini
    dirToWriteFileTo = config_clust['MovingOutputFile']['DirToMoveTo']
    FinalDataFrame = str(config_clust['Output']['CombinedClusterModelResultFile'])
    
    TrainTest_ResultFile = dirToWriteFileTo + 'FinalResultFile_TrainTest.csv'
    GlTest_ResultFile = dirToWriteFileTo + 'FinalResultFile_GlTest.csv'
    
    if config_clust['aim']['CombineTrainTestAndGlTestData'] == 'True':
        TrainDF = pd.read_csv(TrainTest_ResultFile)

        ## Dropping some certain index to stop overlapping of the data
        if os.path.exists(GlTest_ResultFile):
            print('Combining Train and Test Result File')
            TakeIndexData = [ True if certBin.split('|')[1].split('Bin_')[1] != '0' else False for certBin in TrainDF['KEY'] ]
            TrainDF = TrainDF.loc[TakeIndexData, :].reset_index(drop=True)
            TestDF = pd.read_csv(GlTest_ResultFile)
            FinalFile = TrainDF.append(TestDF, ignore_index = True, sort = False)
        else:
            print('Assigning final result file to only Train')
            FinalFile = TrainDF.copy()
    else:
        ## Since in this pipeline Train set file isn't to be used therefore if GlTest file exist write it as result
        
        if ((os.path.exists(GlTest_ResultFile)==True) & (config_clust['aim']['Task'] == 'GlTest')):
            FinalFile = pd.read_csv(GlTest_ResultFile)
        else:
            print('Training has Occured and Model has been Developed, And the results will be developed using the predict setting')
            return 0
        
    FinalFile.reset_index(drop=True, inplace=True)
    FinalFile.to_csv(FinalDataFrame, index=False)


def OutputTransformer(config_clust):
    """
    params:
    OutputDatasetDir: Directory with files with model-cluster-name.csv as the file name. e.g., model1_cluster_1.csv . Each line within file will contain (IP, score). e.g., 10.6.8.7,0.7
    
    Models which results in clusters should have "ClusterModelData_" as prefix. e.g. ClusterModelData_ModelName.csv
    Models which results in scores should have "AnomalyModelData_" as prefix. e.g. AnomalyModelData_ModelName.csv
    
    **ModelData** is an essential text that needs to be present in the output file name to enable File Evaluation
    """
    Task = config_clust['aim']['Task']  
    dirToWriteFileTo = config_clust['MovingOutputFile']['DirToMoveTo']
    
    ### Working on anomaly Result file -- to club score ranges into bins
    GenerateClustModelFileFromAnomFile(config_clust)

    ## Generate a Combined ClusterModelData File
    DF = GenerateSingleResultFileOfClusterModel(config_clust)
    print(DF.columns)
    
    ## Generating Cluster Evaluation Metric Table 
    if config_clust['TriggerTheseFunctions']['GenerateMultipleEnsembleClusterEvalReport'] != 'False':
        EnsembleResultsEvalDF = GenerateMultipleEnsembleClusterEvalReport(DF, config_clust)
    else:
        EnsembleResultsEvalDF = None

    ## Ensemble Results By Joining Multiple File Results By '__'
    DF['Model_ClusterName'] = DF[DF.filter(like='ClusterName', axis=1).columns].apply(lambda x: ('__').join(x), axis=1)
    print(DF.columns)
    DF = DF[['KEY', 'RecentHit_TimeStamp', 'Model_ClusterName']]
    DF = DF.rename(columns={'RecentHit_TimeStamp':'MostRecentHitTimeStamp', 'Model_ClusterName': 'cluster'})

    ## Adding Outlier and Conceptual Drift Result Sequentially
    DF = AddSequentialFilesAsClusters(DF, config_clust)

    ### Checking the KEY format and converting that if required
    DF = KeyFormatConverter(DF, config_clust)
    
#     if(Task == 'TrainTest'):
#         try:
#             print('Removing FinalResultFile_GlTest.csv')
#             os.unlink(dirToWriteFileTo + 'FinalResultFile_GlTest.csv')
#         except:
#             print() ## first iteration File not present
        
    ## Saving the Results Offline to be used by other Module
    DF.to_csv(dirToWriteFileTo + 'FinalResultFile_' + Task + '.csv', index=False)
    print('Combined File for '+ Task + ' has been written.')
    CombineAndWriteFinalCombinedResult(config_clust)
    ## end combined file presence Other case

    ## Returning Ensembled Results and Ensemble Result Evaluation 
    return DF, EnsembleResultsEvalDF