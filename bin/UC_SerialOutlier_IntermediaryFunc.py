### Contains $ func
'''
:: Executor of Sub Algorithms -- outlier detection
:: Combiner for all the Result File into a Ssingle File
:: Ensembling of the Results and Finding the Optimum Therehold That Should be Used
:: Seperating the Outlier and Inliers Dataset
'''


import ast, os, glob, time
import pandas as pd


################### Generating each Result File by Running The Mentioned Algorithms

## importing Sub Parts of the Module
from UC_SerialOutlierSub_AnomalySklearn import OutlierDetection_Sklearn
from UC_SerialOutlierSub_ZScore import OutlierDetection_ZScore
from UC_SerialOutlierSub_MatrixDecomposition import OutlierDetectionAlgo_MatrixDecomposition

def SerOutRemAlgoRunner(TrainDF, TestDF, config_clust):
    '''
    This Chunk is Used To Run and Generate Multiple Algorithm that are to be used to seperated Inliers and Outliers
    '''
    if(TrainDF is not None):
        train_df = TrainDF.copy()
    else:
        train_df = None
    if(TestDF is not None):
        test_df = TestDF.copy()
    else:
        test_df = None
    
    # df = None
    # TRdf, TEdf = OutlierDetection_ZScore(train_df, df, 'ObsFeaturesCntZScoreOutside', 'ObsFeaturesCntZScoreOutside_ParamConfig1', config_clust)
    
    ## Removing Previous Iteration Files
    PreviousIterationFiles = glob.glob(config_clust['MovingOutputFile']['DirToMoveFrom'] + ('SerialAnomaly__' + '*.{FileType}').format(FileType='csv'))
    [ os.unlink(path) for path in PreviousIterationFiles ]
    
    ## Generating Result Files
    OutRemAlgoParam = ast.literal_eval(config_clust['DataProcessing_Outlier']['OutlierRemovalAlgorithmsToUse'])
    for OutAlgo in OutRemAlgoParam:
        # print(OutAlgo + ':', type(OutRemAlgoParam[OutAlgo]))
        if(type(OutRemAlgoParam[OutAlgo]) is dict):
            for SubOutAlgo in OutRemAlgoParam[OutAlgo]:
                if(OutAlgo == 'Sklearn'):
                    algo = SubOutAlgo
                    algo_config = OutRemAlgoParam[OutAlgo][SubOutAlgo]
                    print(algo, algo_config)
                    OutlierDetection_Sklearn(train_df, test_df, algo, algo_config, config_clust)
                elif(OutAlgo == 'ZScore'):
                    algo = SubOutAlgo
                    algo_config = OutRemAlgoParam[OutAlgo][SubOutAlgo]
                    print(algo, algo_config)
                    OutlierDetection_ZScore(train_df, test_df, algo, algo_config, config_clust)
        else:
            if(OutAlgo == 'MatrixDecomposition'):
                algo_config = OutRemAlgoParam[OutAlgo]
                print(OutAlgo, algo_config)
                OutlierDetectionAlgo_MatrixDecomposition(train_df, test_df, OutAlgo, algo_config, config_clust)
    

# df = None
# SerOutRemAlgoRunner(DF, df, config_clust)
# SerOutRemAlgoRunner(df, DF, config_clust)


################### Generating The Combined Result Files From Multiple Algorithm
import glob, re, sys, ast

def CreateKey(DF, Key_ColToUse):
    df = DF.copy()
    for col_ind in range(len(Key_ColToUse)):
        I1 = df.index.tolist()
        I2 = df[Key_ColToUse[col_ind]].astype('str').tolist()
        if col_ind == 0:
            df.index = I2
        else:
            df.index = [ "||".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ] #, I3[ind]
    return df.index

def GenerateCombinedResultsDF(PatternInFileName, config):
    '''
    This Function Combines the multiple dataset results into a single file i.e. bring all results into a file.
    In doing so also removes cases that has been marked as conceptually drifted observation
    '''

    ## Getting the Keys Present in Conceptual Drift Data base To be removed From Analysis
    KeyFormat = ast.literal_eval(config['DataProcessing_General']['KeyFormat'])  ## Used To Create Key
    ConceptDriftFilePath = config['input']['ConceptualDriftDatabase']
    if os.path.exists(ConceptDriftFilePath):
        tempDF = pd.read_csv(ConceptDriftFilePath)
        KeysInConceptDrift = CreateKey(tempDF, KeyFormat).tolist()
    else:
        KeysInConceptDrift = []
    if len(KeysInConceptDrift) != 0: print('Keys that have exhibited Conceptual Drift :', KeysInConceptDrift)

    ## Joining the Datsets
    MovingFromDirectory = config['MovingOutputFile']['DirToMoveFrom']
    FilesList = sorted(glob.glob(MovingFromDirectory + (PatternInFileName + '*.{FileType}').format(FileType='csv')))
    DF = pd.DataFrame()
    print('File Loaded :')
    for FileIndex in range(len(FilesList)):
        tempDF = pd.read_csv(FilesList[FileIndex])
        print('\n\tShape:'+ str(tempDF.shape) + '\t file :' + FilesList[FileIndex])

        ## Changing the index to Key
        tempDF.index = CreateKey(tempDF, KeyFormat)
        ## Dropping repetitive & keys present in conceptual Drift DB
        tempDF.drop_duplicates(keep = 'first', inplace = True)
        tempDF.drop(index= KeysInConceptDrift, inplace=True, errors= 'ignore')

        ## Dropping Columns which are Predict
        tempDF.drop(tempDF.filter(like='_Predict', axis=1).columns, axis = 1, inplace = True)

        Model_Name = re.search(pattern= PatternInFileName+'(\\S*).csv', string= FilesList[FileIndex]).group(1)
        #tempDF[Model_Name + '_Score'] = Model_Name + '_' + tempDF['cluster'].astype(str)  ## Already There

        ## Used Again --- remove at later stage
        ScoreColumns = list(tempDF.filter(like='__OutlierScore', axis=1).columns)
        IdentifierColumns = [col for col in list(tempDF.columns) if col not in ScoreColumns ]
        #IdentifierColumns.remove('cluster')

        tempDF = tempDF[IdentifierColumns + [Model_Name + '__OutlierScore']]
        # tempDF["Model-Cluster-Name"] = Model_Name + "_" + tempDF["cluster"].astype(str)

        if FileIndex == 0:
            DF = pd.DataFrame.copy(tempDF[IdentifierColumns + ScoreColumns])
        else:
            DF.index = CreateKey(DF, KeyFormat)
            X = [ i for i in tempDF.index if i not in DF.index ]
            if(len(X)!=0): print('\tKeys not present in the previous File\n\t', X)
            tempDF = tempDF[IdentifierColumns + ScoreColumns]
            DF = pd.merge(DF, tempDF,  how='left', on=IdentifierColumns).reset_index(drop=True).fillna(value=Model_Name + '_NA')
    #         DF = DF.join(tempDF, how='left', on= IdentifierColumns).reset_index().fillna(value=Model_Name + '_NA')
            # DF["Model-Cluster-Name"] = DF["Model-Cluster-Name_x"] + "-" + DF["Model-Cluster-Name_y"]
        #del(tempDF)

    print('total Shape is :', DF.shape)
    ### Security Check 
    if(DF.shape != (0,0)):
        if ((len(DF[DF.loc[:,IdentifierColumns].duplicated()]) > 0) ): # | (len(tempDF)!=len(DF))
            print('There is Repeatation in Observation On Identifiers Columns')
            sys.exit(1)
    else:
        DF = None

    return DF

# trainDF = GenerateCombinedResultsDF('SerialAnomaly__Train_', config_clust)
# display(trainDF.head())
# testDF = GenerateCombinedResultsDF('SerialAnomaly__Test_', config_clust)
# display(testDF.head())



################### Ensemble and Plotting
from UC_DataProcessing_GenMiniFunc import DataFrameScaling
import matplotlib.pyplot as plt
import sys

def PlottingDecidingThresholdForSerialOutlier_ScoreSummation(NoOfDataPts, Threshold_Score, DF, config):
    NoOfObsToPlot = NoOfDataPts
    # Threshold_Score = 4
    
    if(len(DF) < NoOfObsToPlot):
        NoOfObsToPlot = len(DF)
    PlotDF = DF.sample(n=NoOfObsToPlot, random_state = 12345).copy()
    fig_width = 22
    fig_height = 7
    XPlt = range(NoOfObsToPlot)
    Columns = PlotDF.filter(like = '__OutlierScore').columns
    StackYPlt = [0 for i in range(NoOfObsToPlot)]

    fig = plt.figure(figsize=(fig_width, fig_height))
    for col in Columns:
#         print(col)
        tempYPlt = PlotDF[col][0:NoOfObsToPlot].tolist()

        plt.bar(XPlt, tempYPlt, bottom = StackYPlt, align='edge', width=1, label = col, ls='solid')

        StackYPlt = [ StackYPlt[ind] + tempYPlt[ind] for ind in range(len(StackYPlt)) ] 

    ObsLessThanThreshold = sum([ 1 if elem < Threshold_Score else 0 for elem in StackYPlt ])
    ObsMoreThanThreshold = sum([ 1 if elem >= Threshold_Score else 0 for elem in StackYPlt ])

    plt.axhline(Threshold_Score, color='black', ls = 'dotted', alpha = 0.9, lw = 4, label='Threshold_Score')

    plt.ylabel('Total Scores', fontsize = 13)
    plt.xlabel('Index', fontsize = 13)
    # plt.title('Total Outlier Score ')
    # # plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    # plt.yticks(np.arange(0, NoOfObsToPlot + 1, 10))
    # # plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    plt.legend(loc = 8, ncol = (len(Columns)+1)//2 + 1, fontsize= 'medium')
    plt.grid()

    plt.axhline(0, color='black') ## Margin Lines
    plt.axvline(0, color='black') ## Margin Lines
    
#     print(plt.axis())
    v = plt.axis()  #returns [xmin, xmax, ymin, ymax]
    plt.axis([0, NoOfObsToPlot, v[2], v[3]]) 
    plt.margins(tight = True)

    plt.grid(True, color='black', alpha=0.2)

    plt.gcf().text(x=0.105, y=0.95, s = 'Plotting Outlier Score generated from : Ensembling models', fontsize=18, fontweight = 'bold')
    plt.gcf().text(x=0.12, y=0.9, s = 'Threshold = ' + str(Threshold_Score)+ ' ::: # of Observation Less than Threhold = '+ str(ObsLessThanThreshold) +' :: # of Observation More & Equal than Threshold =  '+str(ObsMoreThanThreshold), fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
    fig.savefig(config['input']['ClustFileSavingLoc_dir'] + 'DecidingSerialOutlierThreshold__{}.png'.format(time.time()), bbox_inches="tight")

def DoSummationOfColumns_Score(df):
    # %%timeit # 7.89 s +- 252 ms per loop (mean +- std. dev. of 7 runs, 1 loop each)
    # DF = trainDF.copy()
    # totSumScore = []
    # for index, row in DF.filter(like = '_Score').iterrows():
    #     totSumScore.append(sum([ row[col] for col in row.index ]))
    # print(totSumScore[0:10])

    # %%timeit # 7.59 s +- 987 ms per loop (mean +- std. dev. of 7 runs, 1 loop each)
    # DF = trainDF.copy()
    # totSumScore = []
    # Cols = DF.filter(like = '_Score').columns
    # for ind in range(len(DF)):
    #     templi = []
    #     for col in Cols:
    #         templi.append(DF[col][ind])
    #     totSumScore.append(sum(templi))
    # print(totSumScore[0:10])

    #%%timeit # 103 ms +- 7.93 ms per loop (mean +- std. dev. of 7 runs, 10 loops each)
    DF = df.copy()
    totSumScore = [0 for i in range(len(DF))]
    for col in DF.filter(like = '__OutlierScore').columns:
        tempYPlt = DF[col].tolist()  ### INCREASES THE SPEED
        totSumScore = [ totSumScore[ind] + tempYPlt[ind] for ind in range(len(DF)) ] 
    #print(totSumScore[0:10])
    ## inline nested loop [ x + y for x in [1,2,3] for y in [3,1,4] ]
    
    return totSumScore
    
def EnsemblingSerialoutlierResults(train_DF, test_DF, config_clust, Explicit_DataPrep = None):
    if(train_DF is not None):
        trainDF = train_DF.copy()
    else:
        trainDF = None
    if(test_DF is not None):
        testDF = test_DF.copy()
    else:
        testDF = None
    
    Methodology = config_clust['DataProcessing_Outlier']['EnsembleMethodology']
    if Explicit_DataPrep is not None:
        ModelSpecificDataPreparation = Explicit_DataPrep # 'Normalized' 'Standard' 'Standard_Median'
    else:
        ModelSpecificDataPreparation = Methodology.split('::')[1]
    EnsemblingMethodology = Methodology.split('::')[0]

    if ModelSpecificDataPreparation != 'Nil':
        print('Data Scaling')
        FeatureScalingID = 'SerAnomRemovalFromMainPipeline__Ensembling_' + EnsemblingMethodology+ModelSpecificDataPreparation
        if(trainDF is not None):
            DataToScale = trainDF.loc[:, sorted(trainDF.filter(like='__OutlierScore').columns)]
            ## Handling the Error:: TypeError: Object of type 'int64' is not JSON serializable
            for col in DataToScale.dtypes[DataToScale.dtypes != 'float64'].index:
                DataToScale[col] = DataToScale[col].astype('float64')
            trainDF.loc[:, sorted(trainDF.filter(like='__OutlierScore').columns)] = DataToScale.loc[:, sorted(DataToScale.filter(like='__OutlierScore').columns)]
            ## large changes made for newer version
            FeatureToIgno = [ col for col in trainDF.columns if col not in sorted(trainDF.filter(like='__OutlierScore').columns) ]
            trainDF, _ = DataFrameScaling(trainDF, FeatureToIgno, config_clust, FeatureScalingID, Explicit_Scaler = ModelSpecificDataPreparation)

        ## Test Should Not Change the Stored Values
        if(testDF is not None):
            DataToScale = testDF.loc[:, sorted(testDF.filter(like='__OutlierScore').columns)]
            ## Handling the Error:: TypeError: Object of type 'int64' is not JSON serializable
            for col in DataToScale.dtypes[DataToScale.dtypes != 'float64'].index:
                DataToScale[col] = DataToScale[col].astype('float64')
            testDF.loc[:, sorted(testDF.filter(like='__OutlierScore').columns)] = DataToScale.loc[:, sorted(DataToScale.filter(like='__OutlierScore').columns)]
            FeatureToIgno = [ col for col in testDF.columns if col not in sorted(testDF.filter(like='__OutlierScore').columns) ]
            testDF, _ = DataFrameScaling(testDF, FeatureToIgno, config_clust, FeatureScalingID, ModelSpecificDataPreparation, 'GlTest')

    print('\nEmsembling Outliers Score')
    if EnsemblingMethodology == 'ScoreSummation':
        ThresholdForRemoval = ast.literal_eval(config_clust['DataProcessing_Outlier']['ObsRemoval'])
        
        if(trainDF is not None):
            trainDF['TotalScore'] = DoSummationOfColumns_Score(trainDF)
            if ThresholdForRemoval['Contamination'] is not None:
                ThresScore = trainDF['TotalScore'].quantile([ThresholdForRemoval['Contamination']]).tolist()[0]  ## percentile inverse ## min value in distribution at 0, max at 1, median at 0.5
            elif ThresholdForRemoval['ThresholdScore'] is not None:
                ThresScore = ThresholdForRemoval['ThresholdScore']
            else:
                print('Threshold Not Defined')
                sys.exit(1)
            if config_clust['TriggerTheseFunctions']['PlotForDecidingThresholdOfSerialOutlier'] != 'False': 
                PlottingDecidingThresholdForSerialOutlier_ScoreSummation(1000, ThresScore, trainDF.loc[:, sorted(trainDF.filter(like='__OutlierScore').columns)], config_clust)
        
        
        if(testDF is not None):
            testDF['TotalScore'] = DoSummationOfColumns_Score(testDF)
            if ThresholdForRemoval['Contamination'] is not None:
                ThresScore = testDF['TotalScore'].quantile([ThresholdForRemoval['Contamination']]).tolist()[0]
            elif ThresholdForRemoval['ThresholdScore'] is not None:
                ThresScore = ThresholdForRemoval['ThresholdScore']
            else:
                print('Threshold Not Defined')
                sys.exit(1)
            if config_clust['TriggerTheseFunctions']['PlotForDecidingThresholdOfSerialOutlier'] != 'False': 
                PlottingDecidingThresholdForSerialOutlier_ScoreSummation(1000, ThresScore, testDF.loc[:, sorted(testDF.filter(like='__OutlierScore').columns)], config_clust)
        
    return trainDF, testDF
    
# df = None
# EnsemblingSerialoutlierResults(trainDF, df, config_clust, Explicit_DataPrep = 'Normalized')
# EnsemblingSerialoutlierResults(trainDF, df, config_clust, Explicit_DataPrep = 'Standard')
# EnsemblingSerialoutlierResults(trainDF, df, config_clust, Explicit_DataPrep = 'Standard_Median')


################### Seperate Outlier and Inlier Datasets

def SeperateInOutDF(DF_m, DF_s, config):
    main_DF = DF_m.copy()
    score_DF = DF_s.copy()
    
    AllFeatures = ast.literal_eval(config['DataProcessing_General']['AllFeaturesToUtilize'])#[ i for i in config['DataProcessing_General']['AllFeaturesToUtilize'].split("'") if len(i) > 2 ]
    FeatureToIgnore = ast.literal_eval(config['DataProcessing_General']['FeatureToIgnore'])

#     main_DF['SID'] = main_DF['SID'].astype(str)
#     score_DF['SID'] = score_DF['SID'].astype(str)
#     main_DF.index = main_DF[['SID', 'BinsBackFromCurrent', 'apidata__zpsbd6']].apply(lambda x: ('|').join(x), axis=1)
#     score_DF.index = score_DF[['SID', 'BinsBackFromCurrent', 'apidata__zpsbd6']].apply(lambda x: ('|').join(x), axis=1)
    KeyFormat = ast.literal_eval(config['DataProcessing_General']['KeyFormat'])
    main_DF.index = CreateKey(main_DF, KeyFormat).tolist()
    score_DF.index = CreateKey(score_DF, KeyFormat).tolist()
    main_DF = main_DF.sort_index().reset_index(drop=True)
    score_DF = score_DF.sort_index().reset_index(drop=True)
    
    ## Checking if order is Same
    if(len([ main_DF.apidata__zpsbd6[ind] for ind in range(len(main_DF)) if main_DF.RecentHit_TimeStamp[ind] != score_DF.RecentHit_TimeStamp[ind] ]) != 0) | (len(main_DF) != len(score_DF)):
        sys.exit(1)
    #return main_DF, score_DF
    
    ThresholdForRemoval = ast.literal_eval(config['DataProcessing_Outlier']['ObsRemoval'])
    if ThresholdForRemoval['Contamination'] is not None:
        ThresScore = score_DF['TotalScore'].quantile([ThresholdForRemoval['Contamination']]).tolist()[0]  ## percentile inverse ## min value in distribution at 0, max at 1, median at 0.5
    elif ThresholdForRemoval['ThresholdScore'] is not None:
        ThresScore = ThresholdForRemoval['ThresholdScore']
    else:
        print('Threshold Not Defined')
        sys.exit(1)

    InlierDF = main_DF.loc[[ True if(score_DF['TotalScore'][ind]>=ThresScore) else False for ind in range(len(score_DF)) ],:] 
    OutlierDF = main_DF.loc[[ True if(score_DF['TotalScore'][ind]<ThresScore) else False for ind in range(len(score_DF)) ],:]

    ## This will Add the column in the results
    # main_DF['Outlier_Predict'] = score_DF['TotalScore']
    OutlierDF['Outlier_Predict'] = score_DF.loc[[ True if(score_DF['TotalScore'][ind]<ThresScore) else False for ind in range(len(score_DF)) ],'TotalScore']

    InlierDF = InlierDF.reset_index(drop = True)
    OutlierDF = OutlierDF.reset_index(drop = True)
    
    return InlierDF, OutlierDF

def WriteOutlierAndInlierOutlierFile(TrainDF_Main, TrainDF_Score, TestDF_Main, TestDF_Score, config):
    
    ## Getting the Keys Present in Conceptual Drift Data base To be removed From Analysis
    KeyFormat = ast.literal_eval(config['DataProcessing_General']['KeyFormat'])  ## Used To Create Key
    ConceptDriftFilePath = config['input']['ConceptualDriftDatabase']
    if os.path.exists(ConceptDriftFilePath):
        tempDF = pd.read_csv(ConceptDriftFilePath)
        KeysInConceptDrift = CreateKey(tempDF, KeyFormat).tolist()
    else:
        KeysInConceptDrift = []
    if len(KeysInConceptDrift) != 0: print('Keys that have exhibited Conceptual Drift :', KeysInConceptDrift)
    ## dropping Keys present in Conceptual Drift Database
    for tempDF in [TrainDF_Main, TrainDF_Score, TestDF_Main, TestDF_Score]:
        if tempDF is not None:
            tempDF.index = CreateKey(tempDF, KeyFormat)
            tempDF.drop(index= KeysInConceptDrift, inplace=True, errors= 'ignore')
            tempDF.reset_index(drop=True, inplace = True)
    
    
    ## Sending Data To Seperate Data
    if(type(TrainDF_Main) == pd.core.frame.DataFrame):
        TrainDF_Main = pd.DataFrame.copy(TrainDF_Main)
        TrainDF_Score = pd.DataFrame.copy(TrainDF_Score)
        TrainInlier, TrainOutlier = SeperateInOutDF(TrainDF_Main, TrainDF_Score, config)
    else:
        TrainInlier, TrainOutlier = None, None
    
    TestInlier, TestOutlier = None, None
    if(type(TestDF_Main) == pd.core.frame.DataFrame):
        TestDF_Main = pd.DataFrame.copy(TestDF_Main)
        TestDF_Score = pd.DataFrame.copy(TestDF_Score)
        TestInlier, TestOutlier = SeperateInOutDF(TestDF_Main, TestDF_Score, config)        
    
    print('DataFrame has been Seperated')

    return TrainInlier, TrainOutlier, TestInlier, TestOutlier

# trainMain, trainScore, testDF_Main, testDF_Score = DF.copy(), trainDF.copy(), None, None
# InlierTrainDF, OutlierTrainDF, InlierTestDF, OutlierTestDF = WriteOutlierAndInlieroutlierFile(trainMain, trainScore, testDF_Main, testDF_Score, config)
# print(len(InlierTrainDF), len(OutlierTrainDF))
