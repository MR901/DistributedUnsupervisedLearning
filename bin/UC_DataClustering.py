### Serial Anomaly, Anomaly and Clustering Groups Generation
import pandas as pd
import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
from pandas.plotting import table
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, ast, time, os, glob
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, Birch
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN

from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, jaccard_similarity_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from UC_DataProcessing_GenMiniFunc import DataFrameScaling
from UC_DataClustering_NonInliner import OutlierBinCreator, WriteOutputFile

def VisualizeClusters(PlotDF, DimRedAlgo, ClusterAlgo, config, ax={'ax1': 0,'ax2': 1,'ax3': 2}, extra_color=False):
    # Plotting Static 3D Plot using the first three variables only
    ax1 = ax['ax1']  ## First axis to take   Data.iloc[:,2]
    ax2 = ax['ax2']  ## Second axis to take
    ax3 = ax['ax3']  ## Third axis to take
    
    FeatureToIgnore = ast.literal_eval(config['DataProcessing_General']['FeatureToIgnore'])
    IndextoStart = len(FeatureToIgnore)
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca()
    
    Cluster = PlotDF.filter(like='Predict').iloc[:, 0].fillna(-999).astype(object)  ### converting to int so that if cluster_Predict result is in float
    # centers = kmeans.cluster_centers_[:,0:3
    try:
        if ClusterAlgo == 'MeanShift':
            extra_color = 'True'
        if extra_color in ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y']:
            colors_list = list(colors._colors_full_map.values())
            color = colors_list
        else:
            #         color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'] 
            color = [
             'b', 'y', 'm', 'r', 'g', 'c', 'aqua', 'sienna', 'lime', 'steelblue', 'hotpink', 'gold',
             'yellow1', 'wheat1', 'violetred1', 'turquoise1', 'tomato1',
             'thistle1', 'tan1', 'steelblue1', 'springgreen1', 'snow3', 'slategray2', 'slateblue2',
             'skyblue2', 'sienna1', 'sgilightblue', 'sgilightgray', 'sgiolivedrab', 'sgisalmon',
             'sgislateblue', 'sgiteal', 'sgigray32', 'sgibeet', 'seagreen2', 'salmon2', 'royalblue2',
             'rosybrown2', 'red1', 'raspberry', 'purple2', 'plum1', 'peachpuff1', 'palevioletred1',
             'paleturquoise2', 'palegreen1', 'orchid1', 'orangered1', 'orange1', 'olivedrab1', 'olive',
             'navajowhite1', 'mediumvioletred', 'mediumpurple1', 'maroon2', 'limegreen', 'lightsalmon4',
             'lightpink1', 'lightcoral', 'indianred1', 'green1', 'gold2', 'firebrick1', 'dodgerblue2',
             'deeppink1', 'deepskyblue1', 'darkseagreen1', 'darkorange1', 'darkolivegreen1', 'darkgreen',
             'darkgoldenrod2', 'crimson', 'chartreuse2', 'cadmiumorange', 'burntumber', 'brown2', 'blue2',
             'antiquewhite4', 'aquamarine4', 'banana', 'bisque4', 'k']

        # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        # https://www.webucator.com/blog/2015/03/python-color-constants-module/
        #print (PlotDF.columns, PlotDF.shape)
        print(PlotDF.filter(like = 'Predict').columns)
        plt.subplot(243)
        for clust in np.sort(Cluster.unique()).tolist():
            ys = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax2]].values
            zs = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax3]].values
            plt.scatter(ys, zs, c=color[np.sort(Cluster.unique()).tolist().index(clust)], marker='o', alpha=0.5)
            # plt.scatter(ys_bot, zs_bot, color = 'red', marker = 'o', alpha = 0.7)
            # plt.scatter(ys_human, zs_human, color = 'black', marker = 'o', alpha = 0.2)
        plt.grid(True, color='black', alpha=0.2)
        plt.title('YZ plane', fontsize=15)
        plt.xlabel('Y')
        plt.ylabel('Z')

        plt.subplot(244)
        for clust in np.sort(Cluster.unique()).tolist():
            xs = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax1]].values
            zs = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax3]].values
            plt.scatter(xs, zs, c=color[np.sort(Cluster.unique()).tolist().index(clust)], marker='o', alpha=0.5)
            # plt.scatter(xs_bot, zs_bot, color = 'red', marker = 'o', alpha = 0.7)
            # plt.scatter(xs_human, zs_human, color = 'black', marker = 'o', alpha = 0.2)
        plt.grid(True, color='black', alpha=0.2)
        plt.title('XZ plane', fontsize=15)
        plt.xlabel('X')
        plt.ylabel('Z')

        plt.subplot(248)
        for clust in np.sort(Cluster.unique()).tolist():
            xs = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax1]].values
            ys = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax2]].values
            plt.scatter(xs, ys, c=color[np.sort(Cluster.unique()).tolist().index(clust)], marker='o', alpha=0.5)
            # plt.scatter(xs_bot, ys_bot, color = 'red', marker = 'o', alpha = 0.7)
            # plt.scatter(xs_human, ys_human, color = 'black', marker = 'o', alpha = 0.2)
        plt.grid(True, color='black', alpha=0.2)
        plt.title('XY plane', fontsize=15)
        plt.xlabel('X')
        plt.ylabel('Y')

        ax = fig.add_subplot(121, projection='3d')
        for clust in np.sort(Cluster.unique()).tolist():
            xs = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax1]].values
            ys = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax2]].values
            zs = PlotDF[PlotDF.filter(like='Predict').iloc[:, 0] == clust].loc[:, PlotDF.columns[IndextoStart + ax3]].values
            ax.scatter(xs, ys, zs, zdir='z', c=color[np.sort(Cluster.unique()).tolist().index(clust)], alpha=0.7, marker='o', label='Cluster_' + str(clust))
            # ax.scatter(xs_bot, ys_bot, zs_bot, zdir='z', c= 'red', marker='o', alpha = 0.7, label = 'Bot')
            # ax.scatter(xs_human, ys_human, zs_human, zdir='z', c= 'black', marker='o', alpha = 0.2, label = 'Human')
            #    centers = KMeans.cluster_centers_[:,0:3]
    #    for i,j,k in centers:
    #        ax.scatter(i,j,k, s=500,c='Black',marker='+')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title(('Visualization of Cluster on three dimensions,\n Developed by {model}.').format(model=DimRedAlgo + ClusterAlgo), fontsize=15)
        plt.legend(loc='lower right', frameon=True)
        # plt.axis([0,1,0,100])
        # plt.axhline(0, color='black')
        # plt.axvline(0, color='black')
        # plt.margins(1,1)
        plt.grid(True, color='black', alpha=0.2)

        plt.subplot(247)
        sns.countplot(x=Cluster, alpha=0.5)
        plt.title('Observations Count in each Clust')
        plt.tight_layout()
        plt.show()
        fig.savefig(config['input']['FigSavingLoc_dir'] + time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime()) + '__VisualizingClusters_' + DimRedAlgo + '_' + ClusterAlgo + '.png')


        PlotDF_col = PlotDF.filter(like='Predict').iloc[:, 0].fillna(-999).astype(object).reset_index()
        ClustModelName = PlotDF.filter(like='Predict').columns[0] + 'ed Cluster Name'
        PlotDF_col.columns = ['# of Observations', ClustModelName]
        PlotDF_col = PlotDF_col.groupby(ClustModelName).aggregate('count').reset_index()
        PlotDF_col = PlotDF_col.set_index(PlotDF_col.index).T

        fig = plt.figure(figsize=(20, 2), dpi=150)# no visible frame
    #     ax = fig.add_subplot(515, frame_on=False)
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False) # hide the x axis
        ax.yaxis.set_visible(False)
        table(ax, PlotDF_col, loc='center')
        plt.show()
        fig.savefig(config['input']['FigSavingLoc_dir'] + time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime()) + '__ObservationInEachCluster_' + DimRedAlgo + '_' + ClusterAlgo + '.png')
    
    except Exception as e:
        print('Error :', str(e))
        print('Error in Plotting Graph. Total No. of Clusters that are present :', len(Cluster.unique()))



def PredictFunction_Alternate(x, y):
    '''
    To Be updated when Distributed Supervided Learning is Added to This Module -- Evaluation Cross Validation Mmodel Optimization
    -- Confusio mAtrix 
    '''
    X = x.copy()
    y = y.copy()
    
    print('|\t\tPredict Alternate : X_Shape, y_Shape :', X.shape, y.shape)
    Mod_Class = RandomForestClassifier()
    Mod_Class.fit(X, y)
    
    Label_Pred = Mod_Class.predict(X).tolist()
    Label_True = y.tolist()
    
    print('|\tAccuracy :', accuracy_score(Label_True, Label_Pred))
    print('|\tMatthews CorrCoef :', matthews_corrcoef(Label_True, Label_Pred))
    print('|\tJaccard Similarity :', jaccard_similarity_score(Label_True, Label_Pred))
    print('|\tConfusion Matrix\n', confusion_matrix(Label_True, Label_Pred).ravel())
#     print('Confusion Matrix\n\tTP:', tp, '\tFP:', fp, '\n\tFN:', fn, '\t\tTN:', tn)
    print('|\tClassification Report\n', classification_report(Label_True, Label_Pred))
    #print(regr.feature_importances_)
    
    return Mod_Class


def ClusteringApplied_1(train_df, test_df, outlier_df, AlgosComb, config):
    """
    This Function is used to generate Clusters of data points using clustering and anomaly based algorithm.
    """
    StartTime = time.time()
    
    try:
        if(type(train_df) == pd.core.frame.DataFrame):
            TrainDF = pd.DataFrame.copy(train_df)
            AllFeature = list(TrainDF.columns)
        else:
            TrainDF = None
        if(type(test_df) == pd.core.frame.DataFrame):
            TestDF = pd.DataFrame.copy(test_df)
            AllFeature = list(TestDF.columns)
        else:
            TestDF = None
        FeatureToIgnore = ast.literal_eval(config['DataProcessing_General']['FeatureToIgnore'])
        FeatureToUse = [ j for j in AllFeature if j not in FeatureToIgnore ]

        DimRedAlgo = AlgosComb['DimensionTransformation'][0]
        ClustAlgo = AlgosComb['AnomalyClustering'][0]
        ExtraColor=config['AnomalyClusterConfiguration']['Visual_Extra_Color']

        ## Getting the Configuration for the Algorithm
        DimRedAlgo_ParamName = AlgosComb['DimensionTransformation'][1]
        # DimRedAlgo_params =  ast.literal_eval(config['DataProcessing_Outlier'][DimRedAlgo_ParamName])
        ClustAlgo_ParamName = AlgosComb['AnomalyClustering'][1]
        ClustAlgo_params =  ast.literal_eval(config['AnomalyClusterConfiguration'][ClustAlgo_ParamName])

        ### Defining Models and their property
        AnomalyClusteringModels_dict = {
            'IsolationForest': {'ModelType': 'AnomalyModelData', 'Model': IsolationForest(), 'DataTypeBoundation': 'Nil', 
                                'fit': True, 'fit_predict': False, 'predict': True, 'DecisionFunction': True}, 
            'EllipticEnvelope': {'ModelType': 'AnomalyModelData', 'Model': EllipticEnvelope(), 'DataTypeBoundation': 'Normalized', 
                                 'fit': True, 'fit_predict': False, 'predict': True, 'DecisionFunction': True}, 
            'LocalOutlierFactor': {'ModelType': 'ClusterModelData', 'Model': LocalOutlierFactor(), 'DataTypeBoundation': 'Nil', 
                                   'fit': True, 'fit_predict': True, 'predict': False, 'DecisionFunction': False}, 
            'OneClassSVM': {'ModelType': 'AnomalyModelData', 'Model': OneClassSVM(), 'DataTypeBoundation': 'Nil', 
                            'fit': True, 'fit_predict': False, 'predict': True, 'DecisionFunction': True}, 

            'KMeans': {'ModelType': 'ClusterModelData', 'Model': KMeans(), 'DataTypeBoundation': 'Nil', 
                                'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False}, 
            'MiniBatchKMeans': {'ModelType': 'ClusterModelData', 'Model': MiniBatchKMeans(), 'DataTypeBoundation': 'Nil', 
                                 'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False}, 
            'AffinityPropagation': {'ModelType': 'ClusterModelData', 'Model': AffinityPropagation(), 'DataTypeBoundation': 'Nil', 
                                   'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False}, 
            'MeanShift': {'ModelType': 'ClusterModelData', 'Model': MeanShift(), 'DataTypeBoundation': 'Nil', 
                            'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False},
            'Birch': {'ModelType': 'ClusterModelData', 'Model': Birch(), 'DataTypeBoundation': 'Nil', 
                                'fit': True, 'fit_predict': True, 'predict': True, 'DecisionFunction': False}, 
            'SpectralClustering': {'ModelType': 'ClusterModelData', 'Model': SpectralClustering(), 'DataTypeBoundation': 'Nil', 
                                 'fit': True, 'fit_predict': True, 'predict': False, 'DecisionFunction': False}, 
            'AgglomerativeClustering': {'ModelType': 'ClusterModelData', 'Model': AgglomerativeClustering(), 'DataTypeBoundation': 'Nil', 
                                   'fit': True, 'fit_predict': True, 'predict': False, 'DecisionFunction': False}, 
            'DBSCAN': {'ModelType': 'ClusterModelData', 'Model': DBSCAN(), 'DataTypeBoundation': 'Nil', 
                            'fit': True, 'fit_predict': True, 'predict': False, 'DecisionFunction': False},
        }

        Model = AnomalyClusteringModels_dict[ClustAlgo]['Model']
        Model.set_params(**ClustAlgo_params)
        ModelSpecificDataPreparation = AnomalyClusteringModels_dict[ClustAlgo]['DataTypeBoundation']
        FeatureScalingID = 'AnomalyClusteringModelsImplementation__' + DimRedAlgo + '_With_' + DimRedAlgo_ParamName + '__' + ClustAlgo + '_With_' + ClustAlgo_ParamName  + '___Data' + ModelSpecificDataPreparation
        ModelType = AnomalyClusteringModels_dict[ClustAlgo]['ModelType']

        ## Normalizing the Dataset for the algorithm which requires non negative dataset
        if ModelSpecificDataPreparation != 'Nil':
            if(TrainDF is not None):
                TrainDF, _ = DataFrameScaling(TrainDF, FeatureToIgnore, config, FeatureScalingID, ModelSpecificDataPreparation)
            ## Test Should Not Change the Stored Values
            if(TestDF is not None):
                TestDF, _ = DataFrameScaling(TestDF, FeatureToIgnore, config, FeatureScalingID, ModelSpecificDataPreparation, 'GlTest')

        print('\nSeries Anomaly Removal Using', ClustAlgo)
        ModelName = config['input']['ModelsSaving_dir'] + FeatureScalingID 

        ## Training Model
        if(TrainDF is not None):
            print('Developing and Saving Model :: Training Section :: On provided Training Data')
            if(AnomalyClusteringModels_dict[ClustAlgo]['fit_predict'] == True):
                TrainDF[ClustAlgo + '_Predict'] = pd.DataFrame( Model.fit_predict(TrainDF[FeatureToUse]) )  
            elif((AnomalyClusteringModels_dict[ClustAlgo]['fit'] == True) & (AnomalyClusteringModels_dict[ClustAlgo]['predict'] == True)):
                Model.fit(TrainDF[FeatureToUse]) 
                TrainDF[ClustAlgo + '_Predict'] = pd.DataFrame(Model.predict(TrainDF[FeatureToUse])) 
            else:
                print('Some Error is present')

            if(AnomalyClusteringModels_dict[ClustAlgo]['predict'] == False): ## Simply Traning the Model based on the fit_predict data
                ## [1]
                X = TrainDF[FeatureToUse]
                y = TrainDF[ClustAlgo + '_Predict']
                Mod = PredictFunction_Alternate(X,y)
                ## Saving the model used for predict function locally
                joblib.dump(Mod, ModelName+'_AddedExtPredict')
            ### Check Which one is more accurate [1] or [2]

            ## Generating Score if it can be generated
            if(AnomalyClusteringModels_dict[ClustAlgo]['DecisionFunction'] == True):
                TrainDF[ClustAlgo + '_Score'] = pd.DataFrame( Model.decision_function(TrainDF[FeatureToUse]) ) 
                if(AnomalyClusteringModels_dict[ClustAlgo]['predict'] == False):
                    ## [2] considering General Boundary as 0, if less tham this value --> anomaly 
                    TrainDF[ClustAlgo + '_PredBasedOnScore'] = [ 1 if elem >= 0 else -1 for elem in TrainDF[ClustAlgo + '_Score'] ]

            ## Saving the Result Locally
            WriteOutputFile(TrainDF, ModelType, 'Train', AlgosComb, config)
            ## Saving the PreProcessed Result image, first three dimension
            if config['TriggerTheseFunctions']['VisualizeClusters'] != 'False':
                VisualizeClusters(TrainDF, DimRedAlgo, ClustAlgo, config, {'ax1': 0,'ax2': 1,'ax3': 2}, ExtraColor)
            ## Saving the model locally
            joblib.dump(Model, ModelName)

        ## Using Developed Model 
        if(TestDF is not None):
            print('Using Saved Model :: Predict Section :: On provided Test Data')
            ## Loading the locally saved model
            Model = joblib.load(ModelName)
            if(AnomalyClusteringModels_dict[ClustAlgo]['predict'] == True):
                TestDF[ClustAlgo + '_Predict'] = pd.DataFrame( Model.predict(TestDF[FeatureToUse]) )  
            elif(AnomalyClusteringModels_dict[ClustAlgo]['predict'] == False):
                ## using the model used for predict function locally
                Model_pred = joblib.load(ModelName+'_AddedExtPredict')
                X = TestDF[FeatureToUse]
                TestDF[ClustAlgo + '_Predict'] = Model_pred.predict(X)

            if(AnomalyClusteringModels_dict[ClustAlgo]['DecisionFunction'] == True):
                TestDF[ClustAlgo + '_Score'] = pd.DataFrame( Model.decision_function(TestDF[FeatureToUse]) ) 
                if(AnomalyClusteringModels_dict[ClustAlgo]['predict'] == False):
                    TestDF[ClustAlgo + '_PredBasedOnScore'] = [ 1 if elem >= 0 else -1 for elem in testDF[ClustAlgo + '_Score'] ]
            ## Saving the Result Locally
            WriteOutputFile(TestDF, ModelType, 'Test', AlgosComb, config)
            ## Saving the PreProcessed Result image, first three dimension
            if config['TriggerTheseFunctions']['VisualizeClusters'] != 'False':
                VisualizeClusters(TestDF, DimRedAlgo, ClustAlgo, config, {'ax1': 0,'ax2': 1,'ax3': 2}, ExtraColor)
    except Exception as e:
        print('Error :', str(e))
        
    EndTime = time.time()
    print('Time Taken :', (EndTime - StartTime)/60, ' min.')
    return TrainDF, TestDF