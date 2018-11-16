
## Executor for Data Preprocessing

import pandas as pd, os
from UC_DataProcessing_HandMissingFunc import HandlingMissingValues
from UC_DataProcessing_DataSplittingFunc import CreateBotSignatureDB, FilterBotSignatureDataForMixing, SplittingDataset
from UC_SerialOutlier_IntermediaryFunc import SerOutRemAlgoRunner, GenerateCombinedResultsDF, EnsemblingSerialoutlierResults, WriteOutlierAndInlierOutlierFile
from UC_DataProcessing_UserDefinedTransfFunc import UserDefinedTransformation
from UC_DataProcessing_GenMiniFunc import DataFrameScaling

def DataPreProcess_1(input_raw_df, config_clust):
    """
    This Function that is responsible for handling Data Preparations:
    1. Data Cleaning and subsetting
    2. Handling Missing Values
    3. Splitting the Dataset into Train and Test datasets
    4. Handling and Seperating Outlier Cases
    5. Transforming Data using User Defined Functions
    6. Combining Outlier Train and Test Datasets
    
    """
    #    input_raw_df = pd.read_csv("data/ipClusterDevelopmentData/RawExtractedData.csv", sep = '|', encoding="utf-8")
    DF = pd.DataFrame.copy(input_raw_df)
    def CustLen(DT):
        try:
            return len(DT)
        except:
            return 'None'
    IniDF_len = CustLen(DF)
    print('Input DF Shape: ', DF.shape)
    
    ##### 0. Removing the previous file if present for the conceptual drift 
    ConceptDriftFilePath = config_clust['input']['ConceptualDriftDatabase']
    if os.path.exists(ConceptDriftFilePath):
        os.unlink(ConceptDriftFilePath)  # To be used only once when the code is Run for the first time 
    ## Previous Iteration SerialOutlierFile are also removed in 4.
    
    
    ##### 1,2. Handling Missing Values and Data Cleaning
    DF = HandlingMissingValues(DF, config_clust)
    
    
    
    ##### 3. Splitting the Dataset into Train and Test datasets
    ## 3.1. Saving Bot Signatures to a DB 
    BotSigDF = CreateBotSignatureDB(DF, config_clust)
    print('Full Bot Signature DB Length :', CustLen(BotSigDF))

    ## 3.2. Extracting a filtered DF containg Bot Signatures 
    DF_ForBotMixing = FilterBotSignatureDataForMixing(BotSigDF, config_clust)
    print('Filtered Bot Signature DB Length :', CustLen(DF_ForBotMixing))

    ## 3.3. Performing Dataset Seperation 
    TrainDF, TestDF, ObservationMixed = SplittingDataset(DF, DF_ForBotMixing, config_clust)
    print('Trainset Length :', CustLen(TrainDF))
    print('Testset Length :', CustLen(TestDF))
    # display(TrainDF.head())
    
    
    
    ##### 4. Handling and Seperating Outlier Cases
    ## 4.1. Developing Serial Outlier Algorithm Results Files
    SerOutRemAlgoRunner(TrainDF, TestDF, config_clust)
    print('Structure of Original Data :', CustLen(TrainDF), CustLen(TestDF))

    ## 4.2. Combining Results File Into a Single File
    train__DF = GenerateCombinedResultsDF('SerialAnomaly__Train_', config_clust)
    test__DF = GenerateCombinedResultsDF('SerialAnomaly__Test_', config_clust)
    print('Structure of Data Containing Score :', CustLen(train__DF), CustLen(test__DF))

    ## 4.3. Generating Threshold Selection Plot and Ensembling Outlier Results
    train__DF, test__DF = EnsemblingSerialoutlierResults(train__DF, test__DF, config_clust, Explicit_DataPrep = 'Standard_Median')
    print('Structure of Data Containing Score :', CustLen(train__DF), CustLen(test__DF))

    ## 4.4. Seperating Outlier and Inliers Results
    InlierTrainDF, OutlierTrainDF, InlierTestDF, OutlierTestDF = WriteOutlierAndInlierOutlierFile(TrainDF, train__DF, TestDF, test__DF, config_clust) # (trainMain, trainScore, testDF_Main, testDF_Score, config_clust)
    
    del(TrainDF, train__DF, TestDF, test__DF)
    print('Final Structure After Data Has been seperated :', CustLen(InlierTrainDF), CustLen(OutlierTrainDF), CustLen(InlierTestDF), CustLen(OutlierTestDF))
    
    
    
    ##### 5. Transforming Data using User Defined Functions
    for df in [InlierTrainDF, OutlierTrainDF, InlierTestDF, OutlierTestDF]:
        print('Dataset Length :', CustLen(df))
        df = UserDefinedTransformation(df, config_clust)
    
    
    
    ##### 6. Combining Outlier Train and Test Datasets
    if OutlierTrainDF is None:
        OutlierDF = OutlierTestDF.sample(frac =1).reset_index(drop =True).copy()
    elif OutlierTestDF is None:
        OutlierDF = OutlierTrainDF.sample(frac =1).reset_index(drop =True).copy()
    else:
        OutlierDF = OutlierTrainDF.append(OutlierTestDF, ignore_index=True, sort = False).sample(frac =1).reset_index(drop =True)
    print('Combined Outlier Dataset Length :',CustLen(OutlierDF))
    
    
    
    ##### 7. Inlier Train and Test Dataset Scaling for Global Use
    ScalerToUse = config_clust['DataProcessing_General']['GlobalFeatureScaling']
    FeatureToIgnore = [ i for i in config_clust['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]

    if ScalerToUse != 'Nil':
        FeatureScalingID = 'GlobalDataScalingOfDataset__' + ScalerToUse
        if(InlierTrainDF is not None):
            InlierTrainDF, _ = DataFrameScaling(InlierTrainDF, FeatureToIgnore, config_clust, FeatureScalingID, ScalerToUse)

        if(InlierTestDF is not None):
            InlierTestDF, _ = DataFrameScaling(InlierTestDF, FeatureToIgnore, config_clust, FeatureScalingID, ScalerToUse, 'GlTest')


    
    ## Generating Confusion Matrix
    # GenerateCorrelationPlot(data, config_clust)      #------------------------------------->
    
    print('\n'+'*'*27+'  Final Dataset Stucture After DataPreProcessing  '+'*'*27)
    print('Initial Dataset Length before DataPreProcessing :', IniDF_len)
    print('Additional Bot Signature that were added to Initial Dataset :', ObservationMixed)
    print('')
    print('Final Training Dataset Length :', CustLen(InlierTrainDF))
    print('Final Testing Dataset Length :', CustLen(InlierTestDF))
    print('Final Outlier Dataset Length :', CustLen(OutlierDF))
    try:
        cncp = pd.read_csv(ConceptDriftFilePath)
    except:
        cncp = []
    print('Current Conceptual Drift Dataset Length :', CustLen(cncp))
    print('*'*104)
    return InlierTrainDF, InlierTestDF, OutlierDF

