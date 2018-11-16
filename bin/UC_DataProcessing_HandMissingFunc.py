import pandas as pd
import ast

def HandlingMissingValues(DF_HavingMissingValues, config_clust):
    """
    Missing Values are Handled In Two ways
    1: Dropping the Variable having large amount of Missing Values/ Selecting only Necessary Variable
    2: Removing Observations that are having missing values
    """
    print('Handling Missing Values')
    print('+', '-'*100)
    
    DF = pd.DataFrame.copy(DF_HavingMissingValues)
    col_ini = DF.columns.tolist()
    
    print('|\tDF Shape BEFORE handling missing values: ', DF.shape)    
    
    missing_data = DF.isnull().sum(axis=0).reset_index().copy()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data = missing_data.loc[missing_data['missing_count'] > 0]
    missing_data = missing_data.sort_values(by='missing_count')
    print('|\tColumns having missing Values: ')
    for index, row in missing_data.iterrows():
        print('|\t\t\"' + row['column_name'] + '\" contains '+ str( row['missing_count'] ) + ' missing values.')
    
    ## 1.##  Selecting ONLY some specific features
    print('|\n|\tRemoving Columns that have either too much missing value or are not useful.')
    AllFeature = ast.literal_eval(config_clust['DataProcessing_General']['AllFeaturesToUtilize'])
    DF = DF.filter(items=AllFeature, axis=1)
    col_dropped = [ col for col in col_ini if col not in DF.columns ]
    print('|\t\tColumns that are dropped:', col_dropped)
    print('|\tDF Shape after filtering the columns: ', DF.shape)
    
    ## Ignoring the feature in processing
    #FeatureToIgnore = [ i for i in config_clust['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]
    
    
    ## 2.##  Handling Missing Values
    print('|\n|\tRemoving observation that have missing values in it.')
    
    
    missing_data = DF.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data = missing_data.loc[missing_data['missing_count'] > 0]
    missing_data = missing_data.sort_values(by='missing_count')
    print('|\tColumns having missing Values: ')
    for col_name in missing_data['column_name']:
        DF = DF[~DF[col_name].isnull() == True]
    print('|\tDF Shape AFTER handling missing values: ', DF.shape)
    
    print('+', '-'*100)
    
    return DF

# input_raw_df.drop(['ZScoreAvgAvgTimeDiffBWHits'], axis = 1).shape
# DataForEDA = input_raw_df.drop(['ZScoreAvgAvgTimeDiffBWHits'], axis = 1).dropna(axis=0, how='any').reset_index(drop=True)
