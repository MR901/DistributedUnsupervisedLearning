import os, sys, time, ast
import pandas as pd
from sklearn.model_selection import train_test_split


######################################################### Saving the Bot Signature DB
## Balancing the Dataset --- Saving Some
def CreateBotSignatureDB(DF, config):
    '''
    Storing Bot Signature Observations To a DB
    This SampleDF can also be Arranged and selected in away to select only those Bot which has made large no. Of Hits
    '''
    DF = DF.copy()
    ## Extracting Some Observation From this Set
    DB_loc = config['input']['BotSignatureDatabase']
    DB_ObsToPreservePerIter = float(config['BotSignatureDB']['FracBotSigToPreservePerIteration'])
    DB_MaxLimit = int(config['BotSignatureDB']['LimitBotSignatureObsSizeTo'])
    FeatureToIgnore = [ i for i in config['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]
    
    ### Improve this fundamental further to includetime for filtering---------------------------------
    tempSampleDF = DF.loc[DF['isBotHits']>0,:].sample(frac = DB_ObsToPreservePerIter).reset_index(drop=True)
    
    if(os.path.exists(DB_loc) is True):
        SampleDF = pd.read_csv(DB_loc)
    else:
        SampleDF = pd.DataFrame(columns = tempSampleDF.columns)
    #print('Initial Shape :', SampleDF.shape, '\nAdded DataFrame Shape :', tempSampleDF.shape)

    ## Adding the Observation to the DataFrame
    SampleDF = SampleDF.append(tempSampleDF, ignore_index=True, sort = False).sample(frac =1).reset_index(drop =True)
    SampleDF['BinsBackFromCurrent'] = 'Bin_XX'
    ### with index as general drop_duplicate doesn't seems to work
    SampleDF['SID'] = SampleDF['SID'].astype(str)
    SampleDF.index = SampleDF[['SID', 'BinsBackFromCurrent', 'apidata__zpsbd6']].apply(lambda x: ('|').join(x), axis=1)
    SampleDF = SampleDF.drop_duplicates(subset = FeatureToIgnore, keep = 'first').reset_index(drop=True)
    # SampleDF.duplicated()
    SampleDF.reset_index(drop=True, inplace=True)
    
    if len(SampleDF) >= DB_MaxLimit:
        SampleDF = SampleDF.sample(n = DB_MaxLimit).reset_index(drop=True)
    #print('Final Shape :', SampleDF.shape)
    SampleDF.to_csv(DB_loc, index = False)
    
    print('>>> Bot Signatures saved Successfully in a External DB')
    return SampleDF

######################################################### Getting the filtered bot Signature DB
def FilterBotSignatureDataForMixing(BotDF, config):
    BotDF = BotDF.copy()
    curr_time = float(time.time())
    TimeDiffToConsider_inhr = float(config['BotSignatureDB']['BotDBObsShouldBeOlderToBeMixedWithCurr_inhr'])
    
    ## Selecting Those Bot Signature That are Older than the provided Time
    NotRecentObsIndex = [ True if(curr_time - i)/(60*60) > TimeDiffToConsider_inhr else False for i in BotDF['RecentHit_TimeStamp'] ]
    print(sum([1 if i is True else 0 for i in NotRecentObsIndex ]))
    BotDF = BotDF[NotRecentObsIndex].reset_index(drop =True)
    
    print('>>> Filtered Bot Signature has been extracted sucessfully to be mixed with Streaming Data.')
    return BotDF


######################################################### Mixing Bot Signature with the Stream Data
def SplittingDataset(DF, DF_BotSigMix, config):
    '''
    Splitting the Dataset based on the configuration  And Mixing With Bot Signature DBs To Balance Datasets
    
    ## mixing the Dataset is Also Done here only
    '''
    DF = DF.copy()
    DF_BotSigMix = DF_BotSigMix.copy()
    SplitSettings = config['aim']['Task']
    DB_MixWithCurrentIteration = ast.literal_eval(config['BotSignatureDB']['MixWithCurrentIteration'])
    FeatureToIgnore = ast.literal_eval(config['DataProcessing_General']['FeatureToIgnore'])
    
    ## Defining Train Test sizes
    if SplitSettings == 'TrainTest':
        SplitRatio = config['aim']['TrainTest_SplitRatio']
        TrainSize = float(SplitRatio.split(':')[0])
        TestSize = float(SplitRatio.split(':')[1])
    elif SplitSettings == 'GlTest':
        TrainSize = 0
        TestSize = 100
    else:
        print('Undefined/Error in config/>Aim/>Task')
        sys.exit(1)

    TotalSize = TrainSize + TestSize
    TrainSizeFrac = TrainSize/ TotalSize
    TestSizeFrac = TestSize / TotalSize
    
    ## Mixing Bot Signature Observation
    if DB_MixWithCurrentIteration['ObservationsFromDB'] is not None:
        BotSigObsCnt = DB_MixWithCurrentIteration['ObservationsFromDB']
    elif DB_MixWithCurrentIteration['FracObserToCurrIterObs'] is not None:
        BotSigObsCnt = int(len(DF) * DB_MixWithCurrentIteration['FracObserToCurrIterObs'])
    else:
        print('config/>BotSignatureDB/>MixWithCurrentIteration is having wrong configration')
        sys.exit(1)
    
    if len(DF_BotSigMix) < BotSigObsCnt:
        BotSigObsCnt = len(DF_BotSigMix)
    
    if len(DF_BotSigMix) != 0:
        SampleToAdd = DF_BotSigMix.sample(n=BotSigObsCnt).reset_index(drop =True)
    else:
        SampleToAdd = DF_BotSigMix ## which is empty
    
    print('# of Observation present for this Iteration :', len(DF))
    print('# of Bot Signature added to the Dataset :', len(SampleToAdd))
    
    DF = DF.append(SampleToAdd, ignore_index=True, sort =False).sample(frac =1).reset_index(drop =True)
    TotalNoOfObservation = len(DF) #+ len(DF_BotSigMix)
    
    ### Security Check 
    if len(DF[DF.loc[:,FeatureToIgnore].duplicated()]) > 0:
        print('There is Repeatation in Observation On Identifiers Columns')
        # sys.exit(1)
        
    print('Test Size Fraction ', TestSizeFrac)
    if TestSizeFrac == 0.00:
        print('Testset size is assign as Null')
        Trainset, Testset = DF.sample(frac=1).reset_index(drop=True), None
        TrSh, TeSh = Trainset.shape, None
    elif TestSizeFrac == 1.00:
        print('Trainset size is assign as Null')
        Trainset, Testset = None, DF.sample(frac=1).reset_index(drop=True)
        TrSh, TeSh = None, Testset.shape
    else:
        Trainset, Testset = train_test_split(DF, test_size=TestSizeFrac)
        Trainset, Testset = Trainset.reset_index(drop=True), Testset.reset_index(drop=True)
        TrSh, TeSh = Trainset.shape, Testset.shape
    print('Trainset Shape After Splitting Dataset: ', TrSh)
    print('Testset Shape After Splitting Dataset: ', TeSh)
    
    #if len(Trainset) == 0:
    #    Trainset = None
    #    print('Trainset --> None')
    #else:
    #    Trainset = Trainset.reset_index(drop =True)
    #if len(Testset) == 0:
    #    Testset = None
    #    print('Testset --> None')
    #else:
    #    Testset = Testset.reset_index(drop =True)
    
    print('>>> Datset Split Successfully')
    return Trainset, Testset, len(SampleToAdd)
