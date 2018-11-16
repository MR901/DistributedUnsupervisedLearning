## Outlier Removal - ZScore
import pandas as pd
import ast, time
from UC_DataProcessing_GenMiniFunc import DataFrameScaling
from UC_SerialOutlierSub_AnomalySklearn import WriteSerialAnomalyOutputFile, PlotingOutlierScoreOrError

def ComputingObsZScoreSumationScore(DF):
    OutlierZScoreObsScore = []
    for index, row in DF.iterrows():
        ObsData = dict(row)
        TotZscoreInObs = []
        for feature in ObsData:
            TotZscoreInObs.append(ObsData[feature])
        OutlierZScoreObsScore.append(sum(TotZscoreInObs))
    return OutlierZScoreObsScore

def ComputingObsFeaturesCntZScoreOutside(DF,ThresholdZScoreRange):
    OutlierZScoreFeatureCount = []
    for index, row in DF.iterrows():
        ObsData = dict(row)
        NoOfFeatureOutsideZRange = 0
        for feature in ObsData:
            if((ObsData[feature] < ThresholdZScoreRange[0]) | (ThresholdZScoreRange[1] < ObsData[feature])):
                NoOfFeatureOutsideZRange += 1
                # print(index, ObsData[feature])
        OutlierZScoreFeatureCount.append(NoOfFeatureOutsideZRange)
        #print('Observation present at index', index, 'is having Zscore outside the defined range with', NoOfFeatureOutsideZRange, 'of feature.')
    return OutlierZScoreFeatureCount

def RunAlgoToGetScore(DF, AlgoVariant, params):
    if(AlgoVariant == 'ObsZScoreSumationScore'):
        tempScore_li = ComputingObsZScoreSumationScore(DF)
    elif(AlgoVariant == 'ObsFeaturesCntZScoreOutside'):
        ThresholdZScoreRange = params['ZScoreRangeForOutlierRemoval']
        print("Z Score Range for Outlier Removal:", ThresholdZScoreRange)
        tempScore_li = ComputingObsFeaturesCntZScoreOutside(DF, ThresholdZScoreRange)
    return tempScore_li


def OutlierDetection_ZScore(train_df, test_df, AlgoVariant, WhichParamsToUse, config_clust):
    """
    This is constructed in a way to handle the both formats of the Data, 'TrainTest' and 'GLTest'
    These Configuration based Action selection is ignored and Dataset is treated in a general way.

    GraphScoreThreshold is applied after the Scale has been Inversed
    AlgoVariant: 'ObsZScoreSumationScore', 'ObsFeaturesCntZScoreOutside'
    ### 
    """

    Algo = 'ZScore' + '_' + AlgoVariant
    config = config_clust

    FeatureToIgnore = [ i for i in config_clust['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]

    if(type(train_df) == pd.core.frame.DataFrame):
        trainDF = pd.DataFrame.copy(train_df)
        AllFeature = list(trainDF.columns)
    else:
        trainDF = None
    if(type(test_df) == pd.core.frame.DataFrame):
        testDF = pd.DataFrame.copy(test_df)
        AllFeature = list(testDF.columns)
    else:
        testDF = None
    ColToAnalysis = [ i for i in AllFeature if i not in FeatureToIgnore ]
    
    AlgoScoreInverseDict = ast.literal_eval(config['DataProcessing_Outlier']['InverseScoreScaleDirGivenOutlierHavingLeastScore'])
    ScoreScaleInverse = AlgoScoreInverseDict[AlgoVariant]['InverseScale']
    GraphScoreThreshold = AlgoScoreInverseDict[AlgoVariant]['GraphScoreThreshold'] ## Not Much Used Case ::: First Score is inversed then this value is used 
    
    # ## Remove Outlier outside the z score range of
    # ZScoreStr = config['data_processing']['ZScoreRangeForOutlierRemoval']
    # ZScoreRangeToConsiderToLabelOutlier = int(re.sub(r'[^\d\-]', ' ', ZScoreStr).split()[0]), int(re.sub(r'[^\d\-]', ' ', ZScoreStr).split()[1])

    ## Getting the Configuration for the Algorithm
    #WhichParamsToUse = ast.literal_eval(config['DataProcessing_Outlier']['OutlierRemovalStage1Algorithm'])['ZScore'][AlgoVariant] ## will be extractd From the executor
    params =  ast.literal_eval(config['DataProcessing_Outlier'][WhichParamsToUse])

    ## Data Scaling --- ## Training Model, 'Standard', 'Standard_Median' Scaling is Doing the Trick Here
    ScalerToUse = 'Standard_Median'
    if(ScalerToUse != 'Nil'):
        FeatureScalingID = 'SerAnomRemovalFromMainPipeline__' + Algo + '__Scale'
        if(trainDF is not None):
            trainDF, _ = DataFrameScaling(trainDF, FeatureToIgnore, config, FeatureScalingID, Explicit_Scaler = ScalerToUse)
        ## Test Should Not Change the Stored Values
        if(testDF is not None):
            testDF, _ = DataFrameScaling(testDF, FeatureToIgnore, config, FeatureScalingID, Explicit_Scaler = ScalerToUse, Explicit_Task = 'GlTest')
            testDF_transformed = testDF[FeatureToIgnore].reset_index(drop=True)


    print('Series Anomaly Removal Using', Algo)
    ModelName = config['input']['ModelsSaving_dir'] + 'SerAnomRemovalFromMainPipeline__' + Algo

    if(trainDF is not None):
        print('Developing and Saving Model :: Training Section :: On provided Training Data')
        TempScoreList = RunAlgoToGetScore(trainDF.loc[:, ColToAnalysis], AlgoVariant, params)
        trainDF[Algo+'_Score'] = [ -1 * elem if(ScoreScaleInverse is True) else elem for elem in TempScoreList ]

    if(testDF is not None):
        print('Using Saved Model :: Predict Section :: On provided Test Data')

        TempScoreList = RunAlgoToGetScore(testDF.loc[:, ColToAnalysis], AlgoVariant, params)
        testDF[Algo+'_Score'] = [ -1 * elem if(ScoreScaleInverse is True) else elem for elem in TempScoreList ]


    ## Converting the Format to a Standard Variant of Serial Anomality Detection
    FeatureToKeep = FeatureToIgnore + [Algo+'_Score']
    if(type(trainDF) == pd.core.frame.DataFrame):
        trainDF = trainDF[FeatureToKeep]
        WriteSerialAnomalyOutputFile(trainDF, 'Train_'+Algo, config)
        if config['TriggerTheseFunctions']['PlotingOutlierScoreOrError'] != 'False':
            PlotingOutlierScoreOrError(trainDF[Algo+'_Score'], config, ScoreThreshold = GraphScoreThreshold, title = Algo + ' Scores in Trainset')
    else:
        trainDF = None
    if(type(testDF) == pd.core.frame.DataFrame):
        testDF = testDF[FeatureToKeep]
        WriteSerialAnomalyOutputFile(testDF, 'Test_'+Algo, config)
        if config['TriggerTheseFunctions']['PlotingOutlierScoreOrError'] != 'False':
            PlotingOutlierScoreOrError(testDF[Algo+'_Score'], config, ScoreThreshold = GraphScoreThreshold, title = Algo + ' Scores in Testset')
    else:
        testDF = None

    return trainDF, testDF ## Serial Standard Version


#### Older Variant
# def GetOutlierIndex(dtFrame, FeatureName, zscore = (-5,5)):
#     series = minmax_scale(dtFrame[FeatureName], feature_range=(1e-100, 1))
#     zscore_li = mstats.zscore(series)
#     print("Feature Name:", FeatureName, "\t Min, Max Z-Score:",round(min(zscore_li),2), round(max(zscore_li),2))
#     Out_Index = [ind for ind in range(len(zscore_li)) if (zscore_li[ind] > zscore[1] or zscore_li[ind] < zscore[0]) ]
#     #     print("Index which seems like outlier", Out_Index)
#     return Out_Index
    
    
# def SeperateOutlierAndInlinerDF(df, config):
    
#     ## Remove Outlier outside the z score range of
#     ZScoreStr = config['data_processing']['ZScoreRangeForOutlierRemoval']
#     ZScoreRangeToConsiderToLabelOutlier = int(re.sub(r'[^\d\-]', ' ', ZScoreStr).split()[0]), int(re.sub(r'[^\d\-]', ' ', ZScoreStr).split()[1])
#     print("Z Score Range for Outlier Removal:", ZScoreRangeToConsiderToLabelOutlier)
    
#     ## Outliers to be removed from the features
#     AllFeature = [ i for i in config['data_processing']['AllFeaturesToUtilize'].split("'") if len(i) > 2 ]
#     FeatureToIgnore = [ i for i in config['data_processing']['FeatureToIgnore'].split("'") if len(i) > 2 ]

#     Col = [ i for i in AllFeature if i not in FeatureToIgnore ]
    
#     AllList = []
#     for c in Col:
#         AllList = AllList + GetOutlierIndex(df, c, ZScoreRangeToConsiderToLabelOutlier)
#     # print(AllList)
#     ## See Value_counts
#     # print(set(map(lambda x  : (x , list(AllList).count(x)) , AllList)))

#     OutlierByNoOfVariable = pd.Series(AllList).value_counts().tolist()
#     OutlierIndexInMainDF = pd.Series(AllList).value_counts().index.tolist()
#     print("\nOutliers observation index, when the Z-score value for a variable outside {} range is considered as outlier".format(ZScoreRangeToConsiderToLabelOutlier))
#     for OutlierInNoOfVar in range(df.shape[1]):
#         OutlierIndex = [OutlierIndexInMainDF[ind] for ind in range(len(OutlierByNoOfVariable)) if OutlierByNoOfVariable[ind] > OutlierInNoOfVar]
#         if len(OutlierIndex) > 0:
#             print('\nObservation Index in Main DF which are considered as Outlier by {ThisMany} variable/s \n'.format(ThisMany = OutlierInNoOfVar + 1),  OutlierIndex)

#     OutliersBadIndex = df.index.isin(pd.Series(AllList).unique().tolist())
#     OutlierDF = df[OutliersBadIndex]
#     df = df[~OutliersBadIndex].reset_index(drop = True)
    
#     return df, OutlierDF