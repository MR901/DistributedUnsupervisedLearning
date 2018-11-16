import json, sys, time, os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec

from scipy.stats import mstats
from sklearn.preprocessing import minmax_scale

######################################################### Algorithm to Preserve and Use offline Computer Stats for Data Scaling
def DataFrameScaling(dataframe, ColumnToIgnore, configuration, FeatureScale_LocID, Explicit_Scaler = None, Explicit_Task = None):
    '''
    Since A custom variant of Data Scaling is Used and this Model is to be preserved, to be used as for predict and also to be used as a way to point conceptual/data drift if it occurs.

    Each time a Scaling is to be done the basic information of the dataset at that pt will be saved with a unique location ID.
    The Dataset properties for Scaling are to be saved/added when Model is to be Trained and just to be read when the predict will be use.
    
    Scaling Settings: 'Normalized', 'Standard', 'Standard_Median', 'Nil'
    FeatureScale_LocID: is used to id the location where which data is to be used
    Explicit_Scaler: if mentioned Use the mentioned Scaler in place of the one mentioned in config file
    '''

    config = configuration  ## See if copy can be made
    AllFeature = dataframe.columns
    temp_df = dataframe.loc[:,[ col for col in AllFeature if col not in ColumnToIgnore ]].copy()
    new_df =  dataframe.loc[:, ColumnToIgnore].copy()
    
    Scaler = config['DataProcessing_General']['GlobalFeatureScaling']
    if Explicit_Scaler is not None:
        Scaler = Explicit_Scaler
    if Explicit_Task is None:
        Explicit_Task = config['aim']['Task']

    ScalingInformationFileName = config['input']['ModelsSaving_dir'] + config['input']['TrainTestDataScalingInfoFile']
    RemovedIndexFromDF = []
    
    ## Dataset Information is to be preserved
    if(Explicit_Task == 'TrainTest'):
        print('>>> Scaling While Saving Parameters, "' + Scaler + '"  <<<')
        ## Computing Measures used for Scaling
        ScalerOffResultFile = {}
        temp_dict ={}
        for col in temp_df.columns:
            temp_dict[col] = {'Min': temp_df[col].min(),
                              'Median': temp_df[col].median(), 
                              'Max': temp_df[col].max(), 
                              'Mean': temp_df[col].mean(), 
                              'Std': temp_df[col].std()}

        ScalerOffResultFile[FeatureScale_LocID] = temp_dict
        InfoForScaling = temp_dict

        ### Preserving the Information Locally i.e saving model data 
        #if(os.path.exists(ScalingInformationFileName) == True):
        #    # Try reading the key data
        #    file = open(ScalingInformationFileName, 'r')
        #    data = json.load(file)
        #    file.close()
        #    if(FeatureScale_LocID in list(data.keys())):
        #        # overwrite that key information
        #        #ActionItem = 'a'
        #        data[FeatureScale_LocID] = InfoForScaling
        #    else:
        #        # append that information
        #        #ActionItem = 'a'
        #        data[FeatureScale_LocID] = InfoForScaling
        #else:
        #    #do something else
        #    #ActionItem = 'w' # can be 'a' too
        #    data[FeatureScale_LocID] = InfoForScaling
        if(os.path.exists(ScalingInformationFileName) == True):
            file = open(ScalingInformationFileName, 'r')
            data = json.load(file)
            file.close()
            data[FeatureScale_LocID] = InfoForScaling
        else:
            data = ScalerOffResultFile
            

        ## Preserving the Information Locally i.e saving model data 
        file = open(ScalingInformationFileName, 'w+')
        DictToWrite = json.dumps(data)
        file.write(DictToWrite)
        file.close()

    elif(Explicit_Task == 'GlTest'):
        print('<<< Scaling Using Saved Parameters, "' + Scaler + '" >>>') 
        if(os.path.exists(ScalingInformationFileName) == True):
            # Try reading the key data
            file = open(ScalingInformationFileName, 'r')
            data = json.load(file)
            file.close()
            if(FeatureScale_LocID in list(data.keys())):
                ## read information for that key
                InfoForScaling = data[FeatureScale_LocID]
            else:
                print('Key is not present in the saved Scaling model file. Exiting')
                sys.exit(1)
        else:
            print('Scaling Model Saved Data Information file doesn\'t exits. Exiting')
            sys.exit(1)
    else:
        print('config[\'aim\'][\'Task\'] is indefinate. Exiting.')
        sys.exit(1)


    # Variable 'InfoForScaling' has distribution information
    
    if(Scaler == 'Nil'):
        print('No Scaling Done')
        return temp_df
    
    ## Using the Information received from 'InfoForScaling' for Scaling the Dataset
    print('Scaling Feature\n+', '-'*100)
    for col in temp_df.columns:
        print('|\t', col)
        li = list(temp_df[col])
        if(Scaler == 'Normalized'):
            new_df[col] = [ (elem - InfoForScaling[col]['Min']) / (InfoForScaling[col]['Max'] - InfoForScaling[col]['Min']) for elem in li ] 
        elif(Scaler == 'Standard'):
            new_df[col] = [ (elem - InfoForScaling[col]['Mean']) / InfoForScaling[col]['Std'] for elem in li ] 
        elif(Scaler == 'Standard_Median'):
            new_df[col] = [ (elem - InfoForScaling[col]['Median']) / InfoForScaling[col]['Std'] for elem in li ] 
        else:
            print('This Passed Scaler is Not Defined')
            continue
            #print('Scaler to Use Undefined')
    print('+', '-'*100)
    #display(temp_df.head(10))
    #display(new_df.head(10))
    
    ################################################
    ## Additional Extension --- Highlight observation whose values lies outside from that of TrainSet --- mark as outlier (Conceptual Drift)
    
    #### Conceptual Drift ---->  Currently only with 'normalized'
    ConcpDftDF = None
    if((Explicit_Task == 'GlTest') & (Scaler == 'Normalized')):
        IndexOutsideRange = []
        print('Checking Conceptual Drift\n+', '-'*100)
        for col in temp_df.columns:
            ValOutsideRange = [ True if((obs < 0)|(obs > 1)) else False for obs in new_df[col] ]
            print('|\tFeature "'+col+'" contains '+ str(sum(ValOutsideRange)) + ' Observation Outside the Range') 
            IndOutRange = [ ind for ind in range(len(new_df[col])) if((new_df[col][ind] < 0)|(new_df[col][ind] > 1)) ]
            [ IndexOutsideRange.append(ind) for ind in IndOutRange ]
        print('|\tIndex Which Are Outside The Defined Range :', IndexOutsideRange, '\n+', '-'*100)

        RemovedIndexFromDF = IndexOutsideRange
        ConcpDftDF = new_df.iloc[IndexOutsideRange,:].reset_index(drop=True)
        new_df = new_df.iloc[[ i for i in range(len(new_df)) if i not in IndexOutsideRange ], :].reset_index(drop=True)
        
        if len(ConcpDftDF) > 0:
            ConceptDriftFilePath = config['input']['ConceptualDriftDatabase']
            # if os.path.exists(ConceptDriftFilePath):   ### Present in Data Pre Processing
            #     os.remove(ConceptDriftFilePath)  # To be used only once when the code is Run for the first time 

            if os.path.exists(ConceptDriftFilePath):
                tempDF = pd.read_csv(ConceptDriftFilePath)
                ConcpDftDF = ConcpDftDF.append(tempDF, ignore_index=True, sort = False)#.sample(frac =1)#.reset_index(drop =True)

            ### With Index as in geral drop_duplicate doesn't seems to work 
            # ConcpDftDF['SID'] = ConcpDftDF['SID'].astype(str)
            # ConcpDftDF.index = ConcpDftDF[['SID', 'BinsBackFromCurrent', 'apidata__zpsbd6']].apply(lambda x: ('|').join(x), axis=1)
            if(len(ConcpDftDF.columns) >= 3):
                I1 = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
                I2 = ConcpDftDF[ConcpDftDF.columns[1]].astype('str')
                I3 = ConcpDftDF[ConcpDftDF.columns[2]].astype('str')
                ConcpDftDF.index = [ "||".join([I1[ind], I2[ind], I3[ind]]) for ind in range(len(I1)) ]
            elif(len(ConcpDftDF.columns) >= 2):
                I1 = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
                I2 = ConcpDftDF[ConcpDftDF.columns[1]].astype('str')
                ConcpDftDF.index = [ "||".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ]
            else:
                ConcpDftDF.index = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
            # ConcpDftDF.drop_duplicates(subset = FeatureToIgnore, keep = 'first', inplace = True)#.reset_index(drop=True)
            ConcpDftDF.drop_duplicates(keep = 'first', inplace = True)#.reset_index(drop=True)
            ConcpDftDF.reset_index(drop=True, inplace=True)
            ConcpDftDF.to_csv(ConceptDriftFilePath,index=False)
            ## ConcpDftDF.groupby(by= FeatureToIgnore).first()
            print('Conceptual Drift based observations were present and they have been saved.')
            # print(ConcpDftDF.shape)
            # display(ConcpDftDF)#.head()
    
    return new_df, ConcpDftDF #, RemovedIndexFromDF

# def DataFrameNormalization(temp_df):
#         return (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())




######################################################### Generating Correlation Plots
def GenerateCorrelationPlot(DF, config):
    '''
    Use to Generate Confusion Matrix and Storing the confusion matrix
    '''
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 14))
    plt.title('Pearson Correlation of Features')
    # Draw the heatmap using seaborn
    sns.heatmap(DF.corr(method='pearson'),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)
    plt.show()
    f.savefig(config['input']['ClustFileSavingLoc_dir'] + 'CorrelationPlot__{}.png'.format(time.time()), bbox_inches="tight")

######################################################### General Pandas Series Stats Computing
def GeneralStats(df_series):
    '''
    This Function uses 'pandas series' to compute and print various statistics on the series
    # https://docs.scipy.org/doc/scipy-0.13.0/reference/stats.mstats.html
    '''
    
    print("\nGeneral Statistics")
    series = minmax_scale(df_series, feature_range=(1e-10, 1))
    print("Zscore per point", [mstats.zmap(i, series) for i in series] [0:4] + ["....."])
    print("Zscore series", mstats.zscore(series)[0:4] )

    print("Describing Series", mstats.describe(series) )

    print("Trimmed Min", mstats.tmin(series) )
    print("Trimmed Max", mstats.tmax(series) )
    print("Geometric Mean", mstats.gmean(series) )
    print("Harmonic Mean", mstats.hmean(series) )

    