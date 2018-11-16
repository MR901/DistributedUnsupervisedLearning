
import json, os, ast
import pandas as pd

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


def WriteOutputFile(Dataset, ModelClassType, DataSection, AlgosComb, config):
    '''
    Use To Create Results File in the standard format of the Results
    '''
    DF = pd.DataFrame.copy(Dataset)
    DimRedAlgo = AlgosComb['DimensionTransformation'][0]
    DimRedAlgo_ParamName = AlgosComb['DimensionTransformation'][1]
    ClustAlgo = AlgosComb['AnomalyClustering'][0]
    ClustAlgo_ParamName = AlgosComb['AnomalyClustering'][1]
    KeyToBeCreatedOn = ast.literal_eval(config['DataProcessing_General']['KeyFormat'])
    
    Predict = DF.filter(like='Predict').columns
    Score = DF.filter(like='Score').columns
    # DF['SID'] = DF['SID'].astype(str)
    # DF['KEY'] = DF[['SID', 'Date', 'apidata__zpsbd6']].apply(lambda x: ('__').join(x), axis=1) 
    # DF['KEY'] = DF[['SID', 'BinsBackFromCurrent', 'apidata__zpsbd6']].apply(lambda x: ('__').join(x), axis=1)
    DF['KEY'] = CreateKey(DF, KeyToBeCreatedOn)
    
    print('Writing {} File'.format(ModelClassType))
    if ModelClassType == 'AnomalyModelData':
        DF = DF.loc[:, ['KEY', 'RecentHit_TimeStamp', 'isBotHits'] + [Score[0], Predict[0]]]
        DF = DF.rename(columns={'KEY': 'KEY', 'RecentHit_TimeStamp': 'RecentHit_TimeStamp', 'isBotHits': 'Probable_IsBot', Score[0]: 'score', Predict[0]: 'cluster'})  #inplace
    elif ModelClassType == 'ClusterModelData':
        DF = DF.loc[:, ['KEY', 'RecentHit_TimeStamp', 'isBotHits', Predict[0]]]
        DF = DF.rename(columns={'KEY': 'KEY', 'RecentHit_TimeStamp': 'RecentHit_TimeStamp', 'isBotHits': 'Probable_IsBot', Predict[0]: 'cluster'})
        #DF = DF.loc[:, ['KEY', 'RecentHit_TimeStamp', Predict[0]]]
        #DF = DF.rename(columns={'KEY': 'KEY', 'RecentHit_TimeStamp': 'RecentHit_TimeStamp', Predict[0]: 'cluster'})
    else:
        DF = DF.loc[:, ['KEY', 'RecentHit_TimeStamp', 'isBotHits', Predict[0]]]
        DF = DF.rename(columns={'KEY': 'KEY', 'RecentHit_TimeStamp': 'RecentHit_TimeStamp', 'isBotHits': 'Probable_IsBot', Predict[0]: 'cluster'})
    
    DF.to_csv(config['input']['ClustFileSavingLoc_dir'] + ModelClassType + '_' + DataSection + '__' + DimRedAlgo + '_With_' + DimRedAlgo_ParamName + '__' + ClustAlgo + '_With_' + ClustAlgo_ParamName + '.csv', index=False)
              
    print('Result File Saved Locally')
    

def OutlierBinCreator(Series, Explicit_Task, config):
    '''
    This Function is Used To Create Bin based on Outlier Score ### Can be customized for Ourlier and conceptual drift both
    '''
    NoOfBinsToCreateBasedOnOutlierScore = int(config['DataProcessing_Outlier']['NoOfClusteToUseWithOutlier'])
    
    Series = Series.copy()
    
    if NoOfBinsToCreateBasedOnOutlierScore == 1:
        li = [ '-1' for i in range(len(Series))]
    
    LabelsForOut = [ str(-i-1) for i in range(NoOfBinsToCreateBasedOnOutlierScore)]
    
    OutlierBinSizesFileName = config['input']['dataset_dir'] + 'OutlierBinSizesInformation.txt'
    if Explicit_Task == 'TrainTest':
        print('Saving the Parameter :: For Outlier Bin Creator ::', Explicit_Task)
        ObservationsGroup = pd.cut(Series, NoOfBinsToCreateBasedOnOutlierScore)
        GroupsFormed = ObservationsGroup.unique()
        print('These Catergories are formed and used for the Outlier Score Grouping', GroupsFormed)
        Ranges = [ (i.right, i.left) for i in GroupsFormed.tolist() ]
    
        ## Writing to a External File For Later Use
        file = open(OutlierBinSizesFileName, 'w+')
        data = json.dumps(Ranges)
        file.write(data)
        file.close()
    
    if Explicit_Task == 'GlTest':
        print('Using the Saved Parameter :: For Outlier Bin Creator ::', Explicit_Task)
        # if(os.path.exists(OutlierBinSizesFileName) == True):
        file = open(OutlierBinSizesFileName, 'r')
        Ranges = json.load(file)
        file.close()
    
    li = Series.tolist()
    
    for i in range(NoOfBinsToCreateBasedOnOutlierScore):
        if(i == 0):
            li = [ LabelsForOut[i] if(float(li[ind]) >= float(Ranges[i][1])) else li[ind] for ind in range(len(li)) ]
        elif(i == (NoOfBinsToCreateBasedOnOutlierScore-1)):
            li = [ LabelsForOut[i] if(float(li[ind])<float(Ranges[i][0])) else li[ind] for ind in range(len(li))  ]
        else:
            li = [ LabelsForOut[i] if((float(li[ind])<float(Ranges[i][0]))&(float(li[ind])>=float(Ranges[i][1]))) else li[ind] for ind in range(len(li)) ]
    # (type(li[ind]) is not str)
    
    li = [ 'SeqOut' + i for i in li ]
    print('Completed')
    return li


def CreateAdditionalClusters(outlier_df, config):
    '''
    Creating Cluster Bin for the Observation which were removed from the pipeline
    '''
    #### Creating cluster information for the sequential 
    ## Creating Bins for Outlier_Predict continous Scale 
    ser = outlier_df['Outlier_Predict'].copy()
    if config['aim']['Task'] == 'TrainTest':
        outlier_df['Outlier_Predict'] = OutlierBinCreator(ser, 'TrainTest', config)
    if config['aim']['Task'] == 'GlTest':
        outlier_df['Outlier_Predict'] = OutlierBinCreator(ser, 'GlTest', config)
    AlgoCombination = {'DimensionTransformation' : ('Outlier', 'NA'), 'AnomalyClustering': ('Outlier', 'NA')}
    WriteOutputFile(outlier_df, 'SequentialAdditionModelData', 'Serial', AlgoCombination, config)
    
    ## Creating Bins for ConceptualDrift_Predict
    if(os.path.exists(config['input']['ConceptualDriftDatabase'])):
        ConcptDrftDB = pd.read_csv(config['input']['ConceptualDriftDatabase'])
        AlgoCombination = {'DimensionTransformation' : ('ConceptualDrift', 'NA'), 'AnomalyClustering': ('ConceptualDrift', 'NA')}
        if config['DataProcessing_General']['NoOfClusteToUseWithConceptualDriftObs'] == '1':
            ConcptDrftDB['ConcDft_Predict'] = 'CptDft-1'
        else:
            ConcptDrftDB['ConcDft_Predict'] = 'CptDft-1'
        WriteOutputFile(ConcptDrftDB, 'SequentialAdditionModelData', 'Serial', AlgoCombination, config)
    
    