import matplotlib 
matplotlib.use('Agg')

import configparser
import pandas as pd
import os, glob, ast, time

from UC_BQDataImport import ImportData_1
from UCorCS_DailySIDTrafficStatus import UnderstandEnvironmentData
from UC_DataProcessing_Executor import DataPreProcess_1
from UC_DataExploration import DataExploration_1
from UC_DataProcessing_GenMiniFunc import GenerateCorrelationPlot
from UC_DataDimensionProcessing import DimensionReduction_1
from UC_DataClustering import ClusteringApplied_1
from UC_DataClustering_NonInliner import CreateAdditionalClusters
from UC_OutputTransformer_CombinerMover import MoveFileToAdaptDir
from UC_ClusterEvaluation import ClustersEvaluation
from UC_OutputTransformer import OutputTransformer


def TimeCataloging(config_clust, Key, Value, First = 'Off'):
    if First == 'On':
        CurrTime = time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime())
        TimeConsumedReport = {
                'CurrentTime': CurrTime,
                'Task': '-',
                'PlotDayWisetraffic': '-',
                'DataImportFromBQ': '-',
                'DataImportToPython': '-',
                'DataPreProcessing': '-',
                'DataExploration': '-',
                'DimenTransfAndClustering': '-',
                'SerialDataClustering': '-',
                'TranfomAndMovingResults': '-',
                'AlgoClustersEvaluation': '-',
                'OutputTransfAndEnsemEval': '-',
                'WholeExecutionTime': '-'
            }
    ## Creating a DataFrame Containing Execution Time Results
    ExecTimePath = config_clust['input']['ExecutionTimeTakenData']
    col = ['CurrentTime', 'Task', 'PlotDayWisetraffic', 'DataImportFromBQ', 'DataImportToPython', 'DataPreProcessing', 
           'DataExploration', 'DimenTransfAndClustering', 'SerialDataClustering', 'TranfomAndMovingResults',
           'AlgoClustersEvaluation', 'OutputTransfAndEnsemEval', 'WholeExecutionTime']
    if(os.path.exists(ExecTimePath) is False):
        tempDF = pd.DataFrame(TimeConsumedReport, columns = col, index = [0]) #TimeConsumedReport.keys()
    else:
        tempDF = pd.read_csv(ExecTimePath)
        if First == 'On':
            tempDF = tempDF.append(TimeConsumedReport, ignore_index=True)
    
    ## Updating Entries
    try:
        tempDF.iloc[(len(tempDF)-1), tempDF.columns.get_loc(Key)] = Value
    except:
        print('Passed Key Doesn\'t Exist in Present Structure')
    ## Saving Locally
    tempDF.to_csv(ExecTimePath, index=False)
    
    if Key == 'WholeExecutionTime':
        return tempDF.iloc[len(tempDF)-1,:].to_dict()


def execute_clust():
    
    t0 = int(time.time())
    print('Execution Start ' + str(t0))
    
    ## Waiting till GlTest is occuring
    if os.path.exists('FlagRaised_GlTestOccurring_DontRunTrainTest'):
        print('Flag: "FlagRaised_GlTestOccurring_DontRunTrainTest" is raised therfore waiting till it completes.')
        while(os.path.exists('FlagRaised_GlTestOccurring_DontRunTrainTest')):
            time.sleep(1)
    ## Raising Flag So That Predict Doesn't Occur While This Is Running
    pd.DataFrame().to_csv('FlagRaised_TrainingOccurring_DontRunGlTest')
    
    config_clust = configparser.ConfigParser()
    config_clust.read('conf/ICLSSTA_Clustering_Config.ini')
    
    ## Plotting Traffic Data on This SID, day wise
    if config_clust['TriggerTheseFunctions']['UnderstandEnvironmentData'] != 'False': ## To Run Code Below Or Not
        UnderstandEnvironmentData(config_clust)
    t1 = int(time.time())
    TimeCataloging(config_clust, 'Task', config_clust['aim']['Task'], First = 'On')
    TimeCataloging(config_clust, 'PlotDayWisetraffic', t1 - t0)
    
    
    ## Importing Data From BQ
    input_raw_df = ImportData_1(config_clust)
    print(input_raw_df.shape)
    t2 = int(time.time())
    TimeCataloging(config_clust, 'DataImportFromBQ', t2 - t1)

    
    ## Loading Desired Data
    SettingToUse = config_clust['aim']['Task']
    if(SettingToUse == 'TrainTest'):
        FileLocalSavingName = config_clust['input']['dataset_dir'] + config_clust['input']['RawDataStorName_TrainTest']
    elif(SettingToUse == 'GlTest'):
        FileLocalSavingName = config_clust['input']['dataset_dir'] + config_clust['input']['RawDataStorName_GlTest']
    input_raw_df = pd.read_csv(FileLocalSavingName, sep = '|', encoding="utf-8")
    print(input_raw_df.shape)
    t3 = int(time.time())
    TimeCataloging(config_clust, 'DataImportToPython', t3 - t2)
    
    
    ## Data Preprocessing
    train_processed_raw_df, test_processed_raw_df, outlier_df = DataPreProcess_1(input_raw_df, config_clust)
    t4 = int(time.time())
    TimeCataloging(config_clust, 'DataPreProcessing', t4 - t3)

    
    if config_clust['TriggerTheseFunctions']['DataExploration'] != 'False': 
        print('Initiating Data Exploration Mode')
        DataExploration_1(input_raw_df, config_clust)
        #DataExploration_1(TrainDF, config_clust)
    
    if config_clust['TriggerTheseFunctions']['GenerateCorrelationPlot'] != 'False':
        GenerateCorrelationPlot(train_processed_raw_df, config_clust)
        GenerateCorrelationPlot(outlier_df, config_clust)
    t5 = int(time.time())
    TimeCataloging(config_clust, 'DataExploration', t5 - t4)
    
    
    ## Applying Dimension transformation & Clustering
    PreviousIterationFiles = glob.glob(config_clust['MovingOutputFile']['DirToMoveFrom'] + ('DataDimensionTransformation_' + '*.{FileType}').format(FileType='csv'))
    [ os.unlink(path) for path in PreviousIterationFiles ]
    PreviousIterationFiles = glob.glob(config_clust['MovingOutputFile']['DirToMoveFrom'] + ('*ModelData_' + '*.{FileType}').format(FileType='csv'))
    [ os.unlink(path) for path in PreviousIterationFiles ]
    DimRedClustAlgoDict = ast.literal_eval(config_clust['AnomalyClusterConfiguration']['DataTransfRedNClustAlgo'])
    for DimRed in DimRedClustAlgoDict.keys():
        print('Data Dimension transformation Algo Used : ', DimRed[0], '\t\tWith Params : ', DimRed[1])
        train_dimen_transf_df, test_dimen_transf_df = DimensionReduction_1(train_processed_raw_df, test_processed_raw_df, DimRed[0], DimRed[1], config_clust) 
        for ClustAlgo in DimRedClustAlgoDict[DimRed]:
            AlgoCombination = {'DimensionTransformation' : (DimRed[0], DimRed[1]), 
                               'AnomalyClustering': (ClustAlgo[0], ClustAlgo[1])}
            print('|\t\tData Segmentation Algo Used : ', ClustAlgo[0], '\t\tWith Params : ', ClustAlgo[1])
            TrainDF, TestDF = ClusteringApplied_1(train_dimen_transf_df, test_dimen_transf_df, outlier_df, AlgoCombination, config_clust)
    t6 = int(time.time())
    TimeCataloging(config_clust, 'DimenTransfAndClustering', t6 - t5)
    
    
    ## Creating Additional Serial Cluster Files
    CreateAdditionalClusters(outlier_df, config_clust)
    t7 = int(time.time())
    TimeCataloging(config_clust, 'SerialDataClustering', t7 - t6)
    
    
    ## Combine and Move train&test files
    MoveFileToAdaptDir(config_clust)
    t8 = int(time.time())
    TimeCataloging(config_clust, 'TranfomAndMovingResults', t8 - t7)
    
    
    ## Individual Cluster Algo Result Evaluation
    if config_clust['TriggerTheseFunctions']['ClustersEvaluation'] != 'False':
        df = ClustersEvaluation(config_clust, 'MultipleFiles', None, (None,None))
        print(df.set_index(['Algorithm'])) # display 
    t9 = int(time.time())
    TimeCataloging(config_clust, 'AlgoClustersEvaluation', t9 - t8)
    
    
    ## Generating Final Transformed and Ensembled Result files additionally Ensembling Cluster Evaluation.
    Output_Keysets_Df, EnsembleEval_DF = OutputTransformer(config_clust) 
    t10 = int(time.time())
    TimeCataloging(config_clust, 'OutputTransfAndEnsemEval', t10 - t9)
    TimeConsumedReport = TimeCataloging(config_clust, 'WholeExecutionTime', t10 - t0)
    
    
    print('Time Taken')
    print('|\t Task :', TimeConsumedReport['Task'])
    print('|\t Plotting Day Wise Traffic Dist. :', TimeConsumedReport['PlotDayWisetraffic'], ' sec')
    print('|\t Importing Data From BQ to Local :', TimeConsumedReport['DataImportFromBQ'], ' sec')
    print('|\t Importing Data From local to python :', TimeConsumedReport['DataImportToPython'], ' sec')
    print('|\t PreProcessing the Data :', TimeConsumedReport['DataPreProcessing'], ' sec')
    print('|\t Data Exploration :', TimeConsumedReport['DataExploration'], ' sec')
    print('|\t Dimension Transformation and Clustering :', TimeConsumedReport['DimenTransfAndClustering'], ' sec')
    print('|\t Creating Additional Serial Clusters :', TimeConsumedReport['SerialDataClustering'], ' sec')
    print('|\t Transforming and Moving Result Files :', TimeConsumedReport['TranfomAndMovingResults'], ' sec')
    print('|\t Individual Clustering Model Evaluation :', TimeConsumedReport['AlgoClustersEvaluation'], ' sec')
    print('|\t Output Tansformation and Ensembling Evaluation :', TimeConsumedReport['OutputTransfAndEnsemEval'], ' sec')
    print('|\t\n|\t Whole Execution Time :', TimeConsumedReport['WholeExecutionTime'], ' sec')
    
    
    ## Removing Flag, So That Predict can Run
    os.unlink('FlagRaised_TrainingOccurring_DontRunGlTest')
    
    print('completed main '+str(int(time.time())))
    
def main():
    execute_clust()

"""run every hour""";
if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(ex)

        