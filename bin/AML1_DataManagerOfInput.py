
import pandas as pd, glob, os
import time
from datetime import datetime, date, timedelta

def exec_bq(query, projectid):
    return pd.io.gbq.read_gbq(query, project_id=projectid, index_col=None, col_order=None, reauth=False, private_key=None) #, verbose=True deprecated


def GetTableNames(config):
    TableSid = [ i for i in config['env']['sid'].split("'") if len(i) > 2 ]
    if config['env']['UseStaticOrDynamicCurrentDay'] == 'static':
        TableDate = [ i for i in config['env']['date'].split("'") if len(i) == 6 ]
        TableDateWindow = config['env']['StaticDataGrabWindow_indays']

        if TableDateWindow != '-':
            CurrentDate = date(2000 + int(TableDate[0][4:6]), int(TableDate[0][2:4]), int(TableDate[0][0:2]))
            format = '%d%m%y'
            TableDate = [ (CurrentDate + timedelta(days=i)).strftime(format) for i in range(int(TableDateWindow)) ]
    
        TableToInclude = ''
        for i in range(len(TableSid)):
            for j in range(len(TableDate)):
                TableToInclude += '\n\tTABLE_QUERY([ss-production-storage.Citadel_Stream],\'table_id like "' + TableSid[i] + '_' + TableDate[j] + '_%"\'),'
        
    if config['env']['UseStaticOrDynamicCurrentDay'] == 'dynamic':
        CurrentTime = datetime(time.gmtime().tm_year, time.gmtime().tm_mon, time.gmtime().tm_mday, time.gmtime().tm_hour, time.gmtime().tm_min, time.gmtime().tm_sec) ## UTC
        TableDateWindow = int(config['env']['DynamicDataGrabWindow_inhr'])
        # TableDateTime = "".join(["%02d" % CurrentTime.day, "%02d" % CurrentTime.month, "%02d" % (CurrentTime.year % 100)])
        # TableDateTime = CurrentTime.strftime(format = '%d%m%y')
        
        TableDateToTake = []
        while TableDateWindow >= -1:  ## -1 to even include the current hour table
            tempDate = CurrentTime - timedelta(days = 0, hours = TableDateWindow, minutes = 0)
            TableDateToTake.append(tempDate.strftime(format = '%d%m%y_%H'))
            TableDateWindow -= 1
        # print(TableDateToTake)
        TableToInclude = ''
        for i in range(len(TableSid)):
            for j in range(len(TableDateToTake)):
                TableToInclude += '\n\t[ss-production-storage.Citadel_Stream.' + TableSid[i] + '_' + TableDateToTake[j] + '],'
        # print(TableToInclude)
    
    return TableToInclude


def GrabAnySizeDatafromGoogleBQ(config):
    ## Configurations
    BQ_QueryFile = config['input']['DataImportQuery']
    LimitToStartWith = config['env']['bq_LimitToStart']
    LimitDecreaseFactor = float(config['env']['bq_LimitDecreaseFactor'])
    
    ## Selecting the Tables to include
    TableToInclude = GetTableNames(config)
    print(TableToInclude)
    
    ## Getting the string that will be used to create bins for grouping based on a certain TimePeriod
    BinSizeBasedOnPeriod_inhr = float(config['env']['BinSizeBasedOnPeriod_inhr'])
    GroupsToInclude = ''
    for i in range(120): ##even if the bin size is as small as an hour, BQ has a limitation of accessing upto a max of 1000 Table, so this is the max possible limit 
        ll_insec = int(i*BinSizeBasedOnPeriod_inhr *3600)
        ul_insec = int((i+1)*BinSizeBasedOnPeriod_inhr *3600 - 1)
        GroupsToInclude += '\n\tWHEN (CurrentTimeStamp - CurrentHitTimeStamp) BETWEEN {low} AND {upp} THEN "Bin_{WhichBin}"'.format(low= ll_insec,upp= ul_insec, WhichBin= i)
    # print(GroupsToInclude)
    
    
    ## Reading the query from an external file
    ### Removed the internal query editing option for easiness 
    print('Read from a locally saved Query File')
    queryfile = open(BQ_QueryFile, 'r')
    query = queryfile.read()
    queryfile.close()
    
    ## looping over the limit and offset to grab the maximum possible bite in terms of observation that can be gathered
    ## AP
    # start = int(LimitToStartWith)  # should be equal to the maximum number of observation that you want to extract
    # stop = -1
    # step = -int(start/LimitDecreaseFactor)
    # limit = int(start/LimitDecreaseFactor)  ## util which pt to try to gather the data
    ## GP
    start = int(LimitToStartWith)  # should be equal to the maximum number of observation that you want to extract
    ratio = 1/LimitDecreaseFactor
    limit = 1000  ## util which pt to try to gather the data ## Hardcoded
    length = 1000
    # query='''SELECT 1 limit {lim} offset {off}'''
    
    DF = pd.DataFrame()
    ##AP
    # for i in [i for i in range(start,stop, step) if i >= limit]:
    ##GP
    for i in [ int(start * ratio ** (n - 1)) for n in range(1, length + 1) if start * ratio ** (n - 1) > limit ]:
        if DF.shape == (0, 0):
            try:
                offcurr = 0
                while offcurr < start:
                    print('\nSetting used in extracting data from BQ:\tNo. of obs. extracted per cycle (limit) = ' + str(i) + '\tOffset = ' + str(offcurr))
                    QueryToUse = query.format(BinToUse = GroupsToInclude, TableToInclude = TableToInclude, lim = str(i), off = str(offcurr))
                    tempDF = exec_bq(QueryToUse, config['bq_cred']['project_id'])
                    DF = DF.append(tempDF, ignore_index = True)
                    offcurr += i

            except Exception as error:
                print('\nAn exception was thrown!\nLimit used: ' + str(i) + '\n' + str(error))
                
                ## Handling a specific error
                if 'was not found in location US' in str(error):
                    print('Handling this Error')
                    ## remove the last table as it is not available
                    TableToInclude = ',\n\t'.join(TableToInclude.split(',\n\t')[:-1])
                    QueryToUse = query.format(BinToUse = GroupsToInclude, TableToInclude = TableToInclude, lim = str(i), off = str(offcurr))
                    DF = exec_bq(QueryToUse, config['bq_cred']['project_id'])
                    
                

    return DF


def ImportData_1(config_clust):
    """
    Extracts any size data from any SID of any number of days.
    
    Works in Two Configuration(config_clust['aim']['Task']), namely 'TrainTest' & 'GlTest'
    'TrainTest' is for models training purpose where This Dataset is split later too make dataset size adequate for training uing sampling
    'GlTest' is purely for prediction purpose, i.e. it will be used as testset only and will consume saved model to provide labels to observations
    """
    
    SettingToUse = config_clust['aim']['Task']
    GlTestDataSize = int(config_clust['env']['NoOfBinToBeUsedInGlTest'])
    
    if(SettingToUse == 'TrainTest'):
        FileLocalSavingName = config_clust['input']['dataset_dir'] + config_clust['input']['RawDataStorName_TrainTest']
    elif(SettingToUse == 'GlTest'):
        FileLocalSavingName = config_clust['input']['dataset_dir'] + config_clust['input']['RawDataStorName_GlTest']
        config_clust['env']['DynamicDataGrabWindow_inhr'] = str(int(config_clust['env']['BinSizeBasedOnPeriod_inhr'])*GlTestDataSize + 1)

    if (os.path.exists(FileLocalSavingName) == False) | (config_clust['env']['GetNewCopy'] in ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y']):
        DF = GrabAnySizeDatafromGoogleBQ(config_clust)
        if(SettingToUse == 'GlTest'):
            DF.drop(DF[DF.BinsBackFromCurrent != 'Bin_0'].index, inplace=True)
            DF.reset_index(drop=True, inplace=True)
        DF.to_csv(FileLocalSavingName, index=False, sep='|', encoding='utf-8')
        print('Data extracted from BQ and saved locally to the File: ', FileLocalSavingName)
    else:
        DF = pd.read_csv(FileLocalSavingName, sep='|', encoding='utf-8')
        print('Data Loaded From the File: ', FileLocalSavingName)
    print('Data Shape: ', DF.shape)
    return DF