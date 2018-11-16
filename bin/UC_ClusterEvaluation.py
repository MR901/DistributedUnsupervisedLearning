from sklearn import metrics
import pandas as pd
import time
import glob, os, ast
import math

def CustomEntropy(labels_true, labels_pred, roundOffTo = 5):
    '''
    formula provided in unknown unknown paper is used
    log to the base 2 is used 
    labels_true need to be in binary format for this i.e. 0 = human and 1 = bot
    '''
    lab_true = [ int(i) for i in labels_true ]
    lab_pred = labels_pred#.copy() #[ int(i) for i in labels_pred ] 

    partitions = pd.Series(lab_pred).unique()  ##by algorithms

    Total_CriticalClass = sum(lab_true)

    entropy = 0
    for p in partitions:
        CriticalClassInThisPartition = sum([ lab_true[ind] for ind in range(len(lab_pred)) if lab_pred[ind] == p ])
        temp = CriticalClassInThisPartition/Total_CriticalClass
        #print('printing temp from Entropy:', temp)
        if(temp != 0):
            entropy -= temp * math.log2(temp)  ## lim x-->0 x*logx = 0
    
    return round(entropy, roundOffTo)

def ComputingClusterEvalMetric(X, labels_true, labels_pred):
    
    RoundOffTo = 5
    
    ## Calculating Adjusted Rand index  # consensus measure
    try:
        ES = CustomEntropy(labels_true, labels_pred, RoundOffTo)
    except Exception as e: 
        print('CustomEntropy Error: ', e)
        ES = None
    
    ## Calculating Adjusted Rand index  # consensus measure
    try:
        ARI =  round(metrics.adjusted_rand_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Adjusted Rand index Error: ', e)
        ARI = None

    ## Calculating Adjusted Mutual Information Based Scores  # consensus measure
    try:
        AMIS = round(metrics.adjusted_mutual_info_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Adjusted Mutual Information Based Scores Error: ', e)
        AMIS = None
    try:
        NMIS = round(metrics.normalized_mutual_info_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Normalized Mutual Information Based Scores Error: ', e)
        NMIS = None

    ## Calculating Homogenity, Completeness and V-measure
    try:
        HS = round(metrics.homogeneity_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Homogenity Error: ', e)
        HS = None
    try:
        CS = round(metrics.completeness_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Completeness Error: ', e)
        CS = None
    try:
        VMS = round(metrics.v_measure_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('V-Measure Error: ', e)
        VMS = None
    #HS_CS_VMS = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)

    ## Calculating Fowlkes-Mallows Scores
    try:
        FMS = round(metrics.fowlkes_mallows_score(labels_true, labels_pred), RoundOffTo)
    except Exception as e: 
        print('Fowlkes-Mallows Scores Error: ', e)
        FMS = None

    if(X is not None):
        ## Calculating Silhouette Coefficient
        try:
            SCS = round(metrics.silhouette_score(X, labels_pred, metric='euclidean', sample_size= 25000), RoundOffTo)
            #print("printing Temp Silhouette Coefficient: ", SCS)
            #print(type(SCS))
        except Exception as e: 
            print('Silhouette Coefficient Error: ', e)
            SCS = None

        ## Calculating Calinski-Harabaz Index 
        try:
            CHI = round(metrics.calinski_harabaz_score(X, labels_pred), RoundOffTo)
        except Exception as e: 
            print('Calinski-Harabaz Index Error: ', e)  ## there is no error is Anomaly algorithm was there
            CHI = None
    else:
        SCS = None
        CHI = None

    ClusterEvaluationScore = {
        'Timestamp': time.strftime('%y/%m/%d %Hhr:%Mmin(%Z)', time.gmtime()), 
        'Algorithm': '---', 
        'NoOfCluster': len(pd.Series(labels_pred).unique()), 
         ## Below Metric Do require True Label
        'Cust_EntropyScore': ES,
        'AdjustedRandIndex': ARI, 
        'AdjustedMutualInfoScore': AMIS, 
        'NormalizedMutualInfoScore': NMIS, 
        'HomogenityScore': HS, 
        'CompletenessScore': CS, 
        'V-measureScore': VMS, 
        'FowlkesMallowsScore': FMS, 
        ## Below Metric Doesn't Require True Label
        'SilhouetteCoefficient': SCS, 
        'CalinskiHarabazScore': CHI, 
        }
    return ClusterEvaluationScore

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

def ClustersEvaluation(config_clust, ResultSummerization = 'MultipleFiles', SavingSingleFileName = None, ExplicitMentionedInformation = (None,None)):
    '''
    ClustersEvaluation(config_clust)  ---> will by default start evaluating clusters in the default directory (config_clust['input']['FigSavingLoc_dir'])
    ResultSummerization = ('SingleFile', default = 'MultipleFiles')  ---> to generate either a single or multiple results files
    SavingSingleFileName = (default='ClustersEvaluation_Report.csv') if 'SingleFile' configuration is used should in which file should the result be appended.
    ExplicitMentionedInformation = (None,None) ---> Takes two kind of configuration ()[0] can be a dataframe or a path to a csv file containing Cluster Results, and the second argument should be the name that is to be used in result Dataframe
    '''
    ## Report will be saved in the directory below
    ReportDirectory = config_clust['input']['FigSavingLoc_dir']
    
    # ExplicitMentionedInformation = (Information, WhatIsIt) eg ('Path/path/file.csv', path), (dataframe, 'NameToUseInAlgorithm')
    if(ExplicitMentionedInformation[0] is not None):
        ## ExplicitMentionedFilePath should be a file containing information about labels_true, labels_pred and X
        if(type(ExplicitMentionedInformation[0]) == pd.core.frame.DataFrame):
            df = ExplicitMentionedInformation[0].copy()
            AlgoName = ExplicitMentionedInformation[1]
        elif(type(ExplicitMentionedInformation[0]) == str):
            df = pd.read_csv(ExplicitMentionedInformation[0])
            AlgoName = ExplicitMentionedInformation[0].split('/')[-1].split('.')[0]
        
        # X ---> won't be computed, as not possible with ensemble results
        X = None
        # print(df.head())
        
        ## Converting isBotHits To Category
        # Results in Warning >> df[df['NoOfCluster'] > 2]['NoOfCluster'] = '1'
        # Results in Warning >> df['Probable_IsBot'].loc[df['Probable_IsBot'] > 0] = '1'
        df['Probable_IsBot'] = ['1' if i > 0 else '0' for i in df['Probable_IsBot']]
        
        labels_true = df['Probable_IsBot'].tolist()
        labels_pred = df['cluster'].tolist()

        ClusterEvalScores = ComputingClusterEvalMetric(X, labels_true, labels_pred)  #---------------->
        ClusterEvalScores['Algorithm'] = AlgoName
        
        df = pd.DataFrame(ClusterEvalScores, columns = ClusterEvalScores.keys(), index = [0])
        
    else:
        ## Looping over all the report that contain meaning full observations
        ## Evaluating All The Files Combination present in the directory below Though will also evaluate TrainTest of the only the one which are Generated
        Directory = config_clust['input']['ClustFileSavingLoc_dir']
        df = pd.DataFrame()
        ## iterating over kind of DataSet
        DatasetType_li = ['Train', 'Test', 'TrainTest']
        for dst in DatasetType_li:
            print('\nWorking on Dataset Type: '+ dst)

            ## iterating over kind of DataDimensionTransformation based Dataset
            FiltDimensionCases = glob.glob(Directory + ('*DataDimensionTransformation_{}__*.csv'.format(dst)))
            
            ### Data For X will be Gathered From the directory mentioned below
            FeatureToIgnore = ast.literal_eval(config_clust['DataProcessing_General']['FeatureToIgnore'])
            KeyToBeMadeFrom = ast.literal_eval(config_clust['DataProcessing_General']['KeyFormat'])
            for fil_dim in FiltDimensionCases:
                print(fil_dim)
                PatternToLookInModelData = fil_dim.split('DataDimensionTransformation_')[-1].split('.')[0]  ## Will Include Test and Train based
                PatternToLookInModelData = PatternToLookInModelData.split('__')[-1]
                print(PatternToLookInModelData)

                df_DimTransf = pd.read_csv(fil_dim, sep='|')
                
                AllFeature = list(df_DimTransf.columns)
                FeaturesOfInterest = [ j for j in AllFeature if j not in FeatureToIgnore ]
                
                ## Generating Key
                df_DimTransf['KEY'] = CreateKey(df_DimTransf, KeyToBeMadeFrom)
                # df_DimTransf[['SID', 'BinsBackFromCurrent', 'apidata__zpsbd6']].apply(lambda x: ('|').join(x), axis=1)
                
                ## Subsetting the DataFrame
                ColumnsToPreserve = ['KEY'] + [ col for col in FeatureToIgnore if col not in KeyToBeMadeFrom ] + FeaturesOfInterest
                df_DimTransf = df_DimTransf[ColumnsToPreserve] 
                #print(ColumnsToPreserve)

                ## iterating over kind of ModelData based Dataset
                '''
                Data For labels_true and labels_pred will be gathered From the directory mentioned below 
                -------i.e. only those cluster files will be evaluated that are moved to the model directory
                '''
                FiltClustCases = glob.glob(Directory + ('*ModelData_{datatype}__{pattern}*.csv'.format(datatype = dst, pattern = PatternToLookInModelData)))
                for fil_mod in FiltClustCases:
                    # print(fil_mod)
                    AlgoPair = fil_mod.split('ModelData_')[-1].split('.')[0]
                    # print('Set UnderConsideration: ' + AlgoPair)
                    
                    df_LabDat = pd.read_csv(fil_mod)
                    
#                     new_df = pd.merge(df_DimTransf, df_LabDat,  how='left', left_on=['KEY','RecentHit_TimeStamp'], right_on = ['KEY','RecentHit_TimeStamp']) 
                    new_df = pd.merge(df_DimTransf, df_LabDat,  how='outer', on=['KEY','RecentHit_TimeStamp'])
                    
                    ## Converting isBotHits To Category
                    # Results in Warning >> df[df['NoOfCluster'] > 2]['NoOfCluster'] = '1'
                    # Results in Warning >> df['Probable_IsBot'].loc[df['Probable_IsBot'] > 0] = '1'
                    labels_true = ['1' if i > 0 else '0' for i in new_df['isBotHits']]
                    labels_pred = new_df['cluster'].tolist()
                    X = new_df[FeaturesOfInterest] ## Filter will work though is not Good
                    
                    ClusterEvalScores = ComputingClusterEvalMetric(X, labels_true, labels_pred)  #---------------->
                    ClusterEvalScores['Algorithm'] = AlgoPair

                    ##Creating a DataFrame Containing All Result
                    if(len(df.columns) == 0):
                        df = pd.DataFrame(ClusterEvalScores, columns = ClusterEvalScores.keys(), index = [0])
                    else:
                        df = df.append(ClusterEvalScores, ignore_index=True)


                    print('\t\t\t\t\t\t==> Entry Added')
                    
    #print('\nPrinting Eval Results')
    #print(df)
    
    ## Writing into a CSV
    if(ResultSummerization == 'SingleFile'):
        if(SavingSingleFileName is None): 
            SavingSingleFileName = 'ClustersEvaluation_Report.csv'
        FileName = ReportDirectory + SavingSingleFileName
        if(os.path.exists(FileName) == False):
            df = pd.DataFrame(ClusterEvalScores, columns = ClusterEvalScores.keys(), index = [0])
            df.to_csv(FileName, index=False)
        else:
            DF = pd.read_csv(FileName)
            df = DF.append(df, ignore_index=True)
            df = df.drop_duplicates()
            df.to_csv(FileName, index=False)
    elif(ResultSummerization == 'MultipleFiles'):
        SavingName = ReportDirectory + time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime()) + '_ClustersEvaluation_Report.csv'
        print('Entries Saved at:', SavingName)
        df.to_csv(SavingName, index=False)
    
    return df

