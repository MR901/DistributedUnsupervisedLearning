
# import matplotlib 
# matplotlib.use('Agg')

import numpy as np
import joblib, matplotlib.pyplot as plt, seaborn as sns
from mpl_toolkits.mplot3d import axes3d

import pandas as pd, ast, time
from UC_DataProcessing_GenMiniFunc import DataFrameScaling
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, LatentDirichletAllocation, FastICA, TruncatedSVD

import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans

def DetNoOfClusters(DF, AlgoToSelect, config_clust):
    # Using the elbow method to find the optimal number of clusters 
    try:
        data = DF.sample(n=100000)
    except ValueError:
        data = DF.sample(frac=1)

    FeatureToIgnore = [ i for i in config_clust['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]
    wcss = []
    for clust_cnt in range(1, 17):
        kmeans = KMeans(n_clusters=clust_cnt, init='k-means++', random_state=42)
        kmeans.fit(data[[ i for i in data.columns if i not in FeatureToIgnore ]])
        wcss.append(kmeans.inertia_)

    width = 20
    height = 7
    fig = plt.figure(figsize=(width, height))
    plt.subplot(121)
    plt.plot(range(1, 17), wcss, color='k', linewidth=2, linestyle='-', marker='o', markerfacecolor='black', markersize=10)
    plt.title('The Elbow Method computed over ' + AlgoToSelect)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    #     plt.xticks(NoOfFeature)
    plt.xticks(np.arange(start=0, stop=17, step=1))
    plt.yticks(np.arange(start=0, stop=501, step=100))
    #     plt.axis([0,1,0,100])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    #     plt.margins(1,1)
    plt.grid(True, color='black', alpha=0.2)
    
    # Using the dendrogram to find the optimal number of clusters
    plt.subplot(122)
    sch.dendrogram(sch.linkage(data[[ i for i in data.columns if i not in FeatureToIgnore ]].iloc[0:1000, :], method='ward'), orientation='right')
    # Methods:'single' 'complete' 'average' 'weighted'  'centroid' 'median' 'ward'
    plt.title('Dendrogram computed over ' + AlgoToSelect)
    plt.xlabel('Euclidean distances')
    plt.ylabel('IPs')
    plt.show()
    # if config_clust['aim']['PaceMode'] == 'Off':
    fig.savefig(config_clust['input']['FigSavingLoc_dir'] + time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime()) + '__OptimunClusters_' + AlgoToSelect + '.png')



def DimenRed_Visual(DF, AlgoToSelect, config_clust):
    # Plotting Static 3D Plot using the first three variables only 
    PlotData = DF.copy()
    FigSav_dir = config_clust['input']['FigSavingLoc_dir']
    FeatureToIgnore = [ i for i in config_clust['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]
    IndextoStart = len(FeatureToIgnore)
    
    fig = plt.figure(figsize=(20, 10))
    xs_bot = PlotData.loc[(PlotData['isBotHits'] > 0, PlotData.columns[IndextoStart + 0])].values
    ys_bot = PlotData.loc[(PlotData['isBotHits'] > 0, PlotData.columns[IndextoStart + 1])].values
    zs_bot = PlotData.loc[(PlotData['isBotHits'] > 0, PlotData.columns[IndextoStart + 2])].values
    
    xs_human = PlotData.loc[(PlotData['isBotHits'] == 0, PlotData.columns[IndextoStart + 0])].values
    ys_human = PlotData.loc[(PlotData['isBotHits'] == 0, PlotData.columns[IndextoStart + 1])].values
    zs_human = PlotData.loc[(PlotData['isBotHits'] == 0, PlotData.columns[IndextoStart + 2])].values
    
    plt.subplot(243)
    plt.scatter(ys_bot, zs_bot, color='red', marker='o', alpha=0.8) # 0.7
    plt.scatter(ys_human, zs_human, color='black', marker='o', alpha=0.05) # 0.2
    plt.grid(True, color='black', alpha=0.2)
    plt.title('YZ plane', fontsize=15)
    plt.xlabel('Y')
    plt.ylabel('Z')
    
    plt.subplot(244)
    plt.scatter(xs_bot, zs_bot, color='red', marker='o', alpha=0.8)
    plt.scatter(xs_human, zs_human, color='black', marker='o', alpha=0.05)
    plt.grid(True, color='black', alpha=0.2)
    plt.title('XZ plane', fontsize=15)
    plt.xlabel('X')
    plt.ylabel('Z')
    
    plt.subplot(248)
    plt.scatter(xs_bot, ys_bot, color='red', marker='o', alpha=0.8)
    plt.scatter(xs_human, ys_human, color='black', marker='o', alpha=0.05)
    plt.grid(True, color='black', alpha=0.2)
    plt.title('XY plane', fontsize=15)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(xs_bot, ys_bot, zs_bot, zdir='z', c='red', marker='o', alpha=0.8, label='Bot')
    ax.scatter(xs_human, ys_human, zs_human, zdir='z', c='black', marker='o', alpha=0.05, label='Human')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('top 3 feature Visualization', fontsize=15)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, color='black', alpha=0.2)
    
    plt.subplot(247)
    ## Converting isBotHits To Category
    # Results in Warning >> df[df['NoOfCluster'] > 2]['NoOfCluster'] = '1'
    # Results in Warning >> PlotData['isBotHits'].loc[PlotData['isBotHits'] > 0] = '1'
    PlotData['isBotHits'] = ['1' if i > 0 else '0' for i in PlotData['isBotHits']]
    
    sns.countplot(x=PlotData['isBotHits'].astype('str'), alpha=0.5)
    plt.title('Observations Count in each Clust')
    plt.tight_layout()
    plt.show()
    
    # if config_clust['aim']['PaceMode'] == 'Off':
    fig.savefig(FigSav_dir + time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime()) + '__DataDimTransformedUsing_' + AlgoToSelect + '.png')



def DimensionReduction_1(TrainData, TestData, AlgoToUse, ParamsToUse, config):
    '''
    This Function will work on Transforming the Dimensions and Visualizing the Traffic over the top 3 axis
    '''
    FeatureToIgnore = [ i for i in config['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]
    if TrainData is not None:
        TrainDF = TrainData.copy()
        AllFeature = TrainDF.columns
        TrainDF_transf = TrainDF[FeatureToIgnore].reset_index(drop=True)
    else:
        TrainDF = None
    if TestData is not None:
        TestDF = TestData.copy()
        AllFeature = TestDF.columns
        TestDF_transf = TestDF[FeatureToIgnore].reset_index(drop=True)
    else:
        TestDF = None
    FeatureToScale = [ i for i in AllFeature if i not in FeatureToIgnore ]

    ### Defining Models and their property
    DimensionTransformModels_dict = {
        'PCA': {'Model': PCA(), 'DataTypeBoundation': 'Nil', 'fit': True, 'fit_transform': True, 'transform': True }, 
        'IncPCA': {'Model': IncrementalPCA(), 'DataTypeBoundation': 'Nil', 'fit': True, 'fit_transform': True, 'transform': True }, 
        'KerPCA': {'Model': KernelPCA(), 'DataTypeBoundation': 'Nil', 'fit': True, 'fit_transform': True, 'transform': True }, 
        'LDA': {'Model': LatentDirichletAllocation(), 'DataTypeBoundation': 'Normalized', 'fit': True, 'fit_transform': True, 'transform': True }, 
        'ICA': {'Model': FastICA(), 'DataTypeBoundation': 'Normalized', 'fit': True, 'fit_transform': True, 'transform': True },  
        'TrunSVD': {'Model': TruncatedSVD(), 'DataTypeBoundation': 'Nil', 'fit': True, 'fit_transform': True, 'transform': True }, 

        'MiniBatchSparsePCA': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html#sklearn.decomposition.MiniBatchSparsePCA
        'SparsePCA': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA
        'DictionaryLearning': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning
        'MiniBatchDictionaryLearning': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html#sklearn.decomposition.MiniBatchDictionaryLearning
        'FactorAnalysis': {}, # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis
        'NMF': {} # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF
        }

    ## Getting the Configuration for the Algorithm
    params =  ast.literal_eval(config['AnomalyClusterConfiguration'][ParamsToUse])

    Model = DimensionTransformModels_dict[AlgoToUse]['Model']
    Model.set_params(**params)
    ModSpecDataPrep = DimensionTransformModels_dict[AlgoToUse]['DataTypeBoundation']


    ##### Scaling Data for Specific Model Requirement
    if ModSpecDataPrep != 'Nil':
        FeatureScalingID = 'DataDimensionTransformation__' + ModSpecDataPrep
        if(TrainDF is not None):
            TrainDF, _ = DataFrameScaling(TrainDF, FeatureToIgnore, config, FeatureScalingID, ModSpecDataPrep)
        if(TestDF is not None):
            TestDF, _ = DataFrameScaling(TestDF, FeatureToIgnore, config, FeatureScalingID, ModSpecDataPrep, 'GlTest')
            TestDF_transf = TestDF[FeatureToIgnore].reset_index(drop=True)

        print('Transforming Dimensions Using :', AlgoToUse)
    ModelName = config['input']['ModelsSaving_dir'] + 'DataDimensionTransformation__' + AlgoToUse   

    ## Training Model
    if(TrainDF is not None):
        print('Developing and Saving Model :: Training Section :: On provided Training Data')

        if(DimensionTransformModels_dict[AlgoToUse]['fit_transform'] == True):
            tempDF = pd.DataFrame(Model.fit_transform(TrainDF.loc[:, FeatureToScale]))
            TrainDF_transf = TrainDF_transf.join(tempDF, rsuffix='_y')
            # Trainset_transformed.columns = Trainset_transformed.columns.astype(str)  # Column name being numeric
            TrainDF_transf.rename(columns=dict(zip(tempDF.columns, AlgoToUse + '_var_' + tempDF.columns.astype('str'))), inplace=True)
        elif((DimensionTransformModels_dict[AlgoToUse]['fit'] == True) & (DimensionTransformModels_dict[AlgoToUse]['transform'] == True)):
            Model.fit(TrainDF.loc[:, FeatureToScale])
            tempDF = pd.DataFrame(Model.transform(TrainDF.loc[:, FeatureToScale]))
            TrainDF_transf = TrainDF_transf.join(tempDF, rsuffix='_y')
            TrainDF_transf.rename(columns=dict(zip(tempDF.columns, AlgoToUse + '_var_' + tempDF.columns.astype('str'))), inplace=True)
        else:
            print('Some Error is present')
        ## Saving the model locally
        joblib.dump(Model, ModelName)

    ## Using Developed Model 
    if(TestDF is not None):
        print('Using Saved Model :: Predict Section :: On provided Test Data')
        ## Loading the locally saved model
        Model = joblib.load(ModelName)

        if(DimensionTransformModels_dict[AlgoToUse]['transform'] == True):
            tempDF = pd.DataFrame(Model.transform(TestDF.loc[:, FeatureToScale]))
            TestDF_transf = TestDF_transf.join(tempDF, rsuffix='_y')
            TestDF_transf.rename(columns=dict(zip(tempDF.columns, AlgoToUse + '_var_' + tempDF.columns.astype('str'))), inplace=True)
        else:
            print('Transform Setting not available with Dimension Transformation')



    ## ## Creating a local copy of the DataFrame, Traffic Visual for first three dimension
    if(type(TrainDF) == pd.core.frame.DataFrame):
        TrainDF_transf.to_csv((config['input']['dataset_dir'] + 'DataDimensionTransformation_Train__' + AlgoToUse + '_With_'+ ParamsToUse + ".csv"), index = False, sep = '|', encoding="utf-8")
        if config['TriggerTheseFunctions']['DimensionReductionTraffic3DVisual'] != 'False': DimenRed_Visual(TrainDF_transf, AlgoToUse, config)
        if config['TriggerTheseFunctions']['DeterminNoOfPossibleClusters'] != 'False': DetNoOfClusters(TrainDF_transf, AlgoToUse, config)
    else:
        TrainDF_transf = None
    if(type(TestDF) == pd.core.frame.DataFrame):
        TestDF_transf.to_csv((config['input']['dataset_dir'] + 'DataDimensionTransformation_Test__' + AlgoToUse + '_With_'+ ParamsToUse + ".csv"), index = False, sep = '|', encoding="utf-8")
        if config['TriggerTheseFunctions']['DimensionReductionTraffic3DVisual'] != 'False': DimenRed_Visual(TestDF_transf, AlgoToUse, config)
        if config['TriggerTheseFunctions']['DeterminNoOfPossibleClusters'] != 'False': DetNoOfClusters(TestDF_transf, AlgoToUse, config)
    else:
        TestDF_transf = None
    
    return TrainDF_transf, TestDF_transf


# config = config_clust
# AlgoToUse = 'ICA'
# ParamsToUse = 'FastICA_ParamConfig'
# DimensionReduction_1(train_processed_raw_df, test_processed_raw_df, AlgoToUse, ParamsToUse, config_clust)