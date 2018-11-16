## Outlier - Matrix Decomposition
import os, joblib, time, ast
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from UC_DataProcessing_GenMiniFunc import DataFrameScaling
from UC_SerialOutlierSub_AnomalySklearn import WriteSerialAnomalyOutputFile, PlotingOutlierScoreOrError

def ComputingReconstructionError_MatrixDecomposition(DF, PCA_Mod, ErrorCompFunc = 'Modulas'):
    '''
    Alternative to this can be a supervised Trained Model -- To Pace Up things by some amount
    '''
    tempDF = DF.copy()
    Model = PCA_Mod
    ### Reobtaining Original Dataset from the transformed dataset
    ## Way - 1 ## https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # mu = np.mean(X, axis=0)
    # nComp = 2
    # Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
    # Xhat += mu
    # print(Xhat[0:10,]) ## Reobtained Original Dataset

    ## WAy -2
    # Reconstructed_DF = pd.DataFrame(pca.inverse_transform(Transf_DF))
    # print(Reconstructed_DF.head(10))  ## Reobtained Original Dataset

    # print(pca.components_[:nComp,:])
    # print('')
    # print(pca.explained_variance_)
    # print('')
    # print(pca.explained_variance_ratio_)
    
    
    ## Computing Reconstruction Error Assocaited with each observations 
    TotReconstructionErrorAssociatedWithObs = []
    for obs in range(len(tempDF)):  ## Computing Reconstruction error for each observation
        TotErrorWithDiffReconstruction = []
        for ExplainedVariance_Ind in range(len(Model.explained_variance_ratio_)):
            ConstDiff = []
            # print("\n", ExplainedVariance_Ind)

            mu = np.mean(np.array(tempDF), axis=0)
            nComp = ExplainedVariance_Ind
            Xhat = np.dot(Model.transform(tempDF)[:,:nComp], Model.components_[:nComp,:])
            Xhat += mu
            Reconstructed_DF =  pd.DataFrame(Xhat)
            # Reconstructed_DF = pd.DataFrame(Model.inverse_transform(Transf_DF.iloc[:,:(ExplainedVariance_Ind+1)]))

            # print("Explained Variance Ratio:", Model.explained_variance_ratio_[ExplainedVariance_Ind])
            for var in range(tempDF.shape[1]):
                # print("Original Value:", X.iloc[obs, var])
                # print("Reobtained Value:", pd.DataFrame(Model.inverse_transform(pca.transform(X))).iloc[obs, var])
                diff = tempDF.iloc[obs, var] - Reconstructed_DF.iloc[obs, var]
                ConstDiff.append(diff)
                # print("Difference:", diff)
            # print("Constructin Difference:", ConstDiff)
            if ErrorCompFunc == 'Modulas':
                ConstDiff = [abs(elem) for elem in ConstDiff]
                # print(ConstDiff)
                # print(sum(ConstDiff) * Model.explained_variance_ratio_[ExplainedVariance_Ind])
                ErrorThisReconstruction = sum(ConstDiff) * Model.explained_variance_ratio_[ExplainedVariance_Ind]
            elif ErrorCompFunc == 'RMSE':
                ConstDiff = [(elem**2) for elem in ConstDiff]
                # print(ConstDiff)
                # print(sum(ConstDiff)**0.5 * Model.explained_variance_ratio_[ExplainedVariance_Ind])
                ErrorThisReconstruction = sum(ConstDiff)**0.5 * Model.explained_variance_ratio_[ExplainedVariance_Ind]
            TotErrorWithDiffReconstruction.append(ErrorThisReconstruction)
        # print("\nError With Different Reconstuctions:", TotErrorWithDiffReconstruction)
        TotReconstructionErrorAssociatedWithObs.append(sum(TotErrorWithDiffReconstruction))
        # print("Total Error associated with this Observation:", sum(TotErrorWithDiffReconstruction), "\n")
    # print("\nReconstruction Error Associated with Observations: ", TotReconstructionErrorAssociatedWithObs[0:15])
    return TotReconstructionErrorAssociatedWithObs



def OutlierDetectionAlgo_MatrixDecomposition(train_df, test_df, Algo, WhichParamsToUse, config):
    '''
    Function is Developed With PCA as the Basis to detect Outliers.
    
    FuncToUse = 'Modulas', 'RMSE'
    # Computing PCA From SCRATCH
    https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
    https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    '''
    start_time = time.time()
    print(start_time)
    
    #ScalerToUse = 'Nil', FuncToUse = 'Modulas'
    params =  ast.literal_eval(config['DataProcessing_Outlier'][WhichParamsToUse])
    ScalerToUse = params['ScalerToUse']
    FuncToUse = params['FuncToUse']
    ScalerToUse = 'Normalized' ## OverWriting
    FuncToUse = 'Modulas'
    
    AlgoScoreInverseDict = ast.literal_eval(config['DataProcessing_Outlier']['InverseScoreScaleDirGivenOutlierHavingLeastScore'])
    ScoreScaleInverse = AlgoScoreInverseDict[Algo]['InverseScale']
    GraphScoreThreshold = AlgoScoreInverseDict[Algo]['GraphScoreThreshold'] ## Not Much Used Case ::: First Score is inversed then this value is used 
    
#     Algo = 'MatrixDecomposition'
    FeatureToIgnore = [ i for i in config['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]
    
    ## Trainset
    if(type(train_df) == pd.core.frame.DataFrame):
        trainDF = pd.DataFrame.copy(train_df)
        AllFeature = list(trainDF.columns)
        ## Copying the feature to a new DF which are to be ignored in dimension tranformation-- not shuffle
        trainDF_transformed = trainDF[FeatureToIgnore].reset_index(drop=True)
    else:
        trainDF = None
    
    ## Testset
    if(type(test_df) == pd.core.frame.DataFrame):
        testDF = pd.DataFrame.copy(test_df)
        AllFeature = list(testDF.columns)
        ## Copying the Identifiers Fetures
        testDF_transformed = testDF[FeatureToIgnore].reset_index(drop=True)
    else:
        testDF = None
    
    ColToAnalysis = [ i for i in AllFeature if i not in FeatureToIgnore ]
    
    
    ## PCA 
    Model = PCA()
    
    ## Data Scaling
    if(ScalerToUse != 'Nil'):
        FeatureScalingID = 'SerAnomRemovalFromMainPipeline__' + Algo
        if(trainDF is not None):
            trainDF, _ = DataFrameScaling(trainDF, FeatureToIgnore, config, FeatureScalingID, Explicit_Scaler = ScalerToUse)
        ## Test Should Not Change the Stored Values
        if(testDF is not None):
            testDF, _ = DataFrameScaling(testDF, FeatureToIgnore, config, FeatureScalingID, Explicit_Scaler = ScalerToUse, Explicit_Task = 'GlTest')
            testDF_transformed = testDF[FeatureToIgnore].reset_index(drop=True)
    
    end_time1 = time.time()
    print(end_time1, 'time taken:', (end_time1-start_time)/60)
    
    print('Series Anomaly Removal Using', Algo)
    ModelName = config['input']['ModelsSaving_dir'] + 'SerAnomRemovalFromMainPipeline__' + Algo
    
    ## Training Model
    if(trainDF is not None):
        print('Developing and Saving Model :: Training Section :: On provided Training Data')
        tempDF = pd.DataFrame(Model.fit_transform(trainDF[ColToAnalysis]))
        trainDF_transformed = trainDF_transformed.join(tempDF, rsuffix='_y')
        trainDF_transformed.rename(columns=dict(zip(tempDF.columns, 'TransfVar_' + tempDF.columns.astype('str'))), inplace=True)
        TempScoreList = ComputingReconstructionError_MatrixDecomposition(tempDF, Model, FuncToUse)
        trainDF_transformed[Algo+'_Score'] = [ -1 * elem if(ScoreScaleInverse is True) else elem for elem in TempScoreList ]
        ## Saving the model locally
        joblib.dump(Model, ModelName)
    
    
    ## Using Developed Model 
    if(testDF is not None):
        print('Using Saved Model :: Predict Section :: On provided Test Data')
        ## Loading the locally saved model
        Model = joblib.load(ModelName)
        tempDF = pd.DataFrame(Model.transform(testDF[ColToAnalysis]))
        testDF_transformed = testDF_transformed.join(tempDF, rsuffix='_y')
        testDF_transformed.rename(columns=dict(zip(tempDF.columns, 'TransfVar_' + tempDF.columns.astype('str'))), inplace=True)
        TempScoreList = ComputingReconstructionError_MatrixDecomposition(tempDF, Model, FuncToUse)
        testDF_transformed[Algo+'_Score'] = [ -1 * elem if(ScoreScaleInverse is True) else elem for elem in TempScoreList ]
    
    
    end_time2 = time.time()
    print(end_time2, 'time taken:', (end_time2-end_time1)/60)
    
    
    ## Converting the Format to a Standard Variant of Serial Anomality Detection
    FeatureToKeep = FeatureToIgnore + [Algo+'_Score']
    if(type(trainDF) == pd.core.frame.DataFrame):
        trainDF_transformed = trainDF_transformed[FeatureToKeep]
        WriteSerialAnomalyOutputFile(trainDF_transformed, 'Train_'+Algo, config)
        if config['TriggerTheseFunctions']['PlotingOutlierScoreOrError'] != 'False':
            PlotingOutlierScoreOrError(trainDF_transformed[Algo+'_Score'].tolist(), config, ScoreThreshold = GraphScoreThreshold, title = Algo + ' Scores in Trainset')
    else:
        trainDF_transformed = None
    if(type(testDF) == pd.core.frame.DataFrame):
        testDF_transformed = testDF_transformed[FeatureToKeep]
        WriteSerialAnomalyOutputFile(testDF_transformed, 'Test_'+Algo, config)
        if config['TriggerTheseFunctions']['PlotingOutlierScoreOrError'] != 'False':
            PlotingOutlierScoreOrError(testDF_transformed[Algo+'_Score'].tolist(), config, ScoreThreshold = GraphScoreThreshold, title = Algo + ' Scores in Testset')
    else:
        testDF_transformed = None
        
    end_time3 = time.time()
    print(end_time3, 'time taken:', (end_time3-end_time2)/60)
    
    return trainDF_transformed, testDF_transformed ## Serial Standard Version