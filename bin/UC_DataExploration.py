# uncompyle6 version 3.0.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.5.2 (default, Nov 23 2017, 16:37:01) 
# [GCC 5.4.0 20160609]
# Embedded file name: /home/ubuntu/users/mohit/Project-X/project-x_new/UC_DataProcessing.py
# Compiled at: 2018-03-02 08:57:53
import pandas as pd
from sklearn.preprocessing import minmax_scale
import re, sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mstats

from UC_DataProcessing_GenMiniFunc import GeneralStats

# def GeneralStats(df_series):
#     # https://docs.scipy.org/doc/scipy-0.13.0/reference/stats.mstats.html
#     from scipy.stats import mstats
    
#     print("\nGeneral Statistics")
#     series = minmax_scale(df_series, feature_range=(1e-10, 1))
#     print("Zscore per point", [mstats.zmap(i, series) for i in series] [0:4] + ["....."])
#     print("Zscore series", mstats.zscore(series)[0:4] )

#     print("Describing Series", mstats.describe(series) )

#     print("Trimmed Min", mstats.tmin(series) )
#     print("Trimmed Max", mstats.tmax(series) )
#     print("Geometric Mean", mstats.gmean(series) )
#     print("Harmonic Mean", mstats.hmean(series) )
    

def GetOutlierIndex(dtFrame, FeatureName, zscore = (-5,5)):
    series = minmax_scale(dtFrame[FeatureName], feature_range=(1e-100, 1))
    zscore_li = mstats.zscore(series)
    print("Feature Name:", FeatureName, "\t Min, Max Z-Score:",round(min(zscore_li),2), round(max(zscore_li),2))
    Out_Index = [ind for ind in range(len(zscore_li)) if (zscore_li[ind] > zscore[1] or zscore_li[ind] < zscore[0]) ]
    #     print("Index which seems like outlier", Out_Index)
    return Out_Index

def Visual(var_x, var_y, Var, config):
    
    print('UniVariate Plot')
    # plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    sns.set_style("darkgrid") #whitegrid, darkgrid, dark
    # df = sns.load_dataset('iris')

    # Cut the window in 2 parts
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

    # Add a graph in each part
    sns.boxplot(var_x, ax=ax_box)
    sns.distplot(var_x, ax=ax_hist)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    # plt.show()
    
    print('-'*80)
    print('BiVariate Plot')
    
    sns.jointplot(x= var_x, y= var_y)
    plt.show()
    #f.savefig(config['input']['ClustFileSavingLoc_dir'] + 'VariablePlot__{}__{}.png'.format(Var, time.time()), bbox_inches="tight")
    
def GenerateGraph(X_series, Y_series, transf, config, VariablePlotted = ""):
        
    import inspect
    FuncUsed = re.sub(' },', '|||', re.sub(', \'', '|||', re.sub('\'\: ', '|||', inspect.getsource(transf['Actual'])))).split('|||')
    ActFunc = FuncUsed[2]
    UserDefFunc = FuncUsed[4]
    
    print('\n')
    
    xa = minmax_scale(X_series.apply( transf['Actual'] ) , feature_range=(1e-100, 1))
    ya = Y_series
    print('*'*90)
    print('General Distribution')
    print('*'*90)
    print('Plotting Data point for the feature:', VariablePlotted)
    print('Transformation that has been used for this feature:', ActFunc)
    print('Values for this feature is  Scaled between:', '[1e-100, 1]')
    Visual(xa, ya, VariablePlotted, config)      #------------------------------------->

    import sys
    PythonVer = float(str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + str(sys.version_info[2]))
    if PythonVer >= 3:
        ## Power Transformation
        try:
            from sklearn.preprocessing import PowerTransformer
            PT = PowerTransformer(method='box-cox', standardize=False)

            ### Applying Power Transformation
            X = minmax_scale(X_series.apply( lambda x: (x) ) , feature_range=(1e-10, 1))
            X_2d = X.reshape(-1, 1)
            X_trans = PT.fit_transform(X_2d)
            xa = X_trans.reshape(-1,)
            ya = Y_series
            print('*'*90)
            print('Using Box-Cox To Find a suitable Transformation')
            print("Best Lambda Value Obtained on using Box-Cox is", round(PT.lambdas_[0], 3))
            print('*'*90)
            print('Plotting Data point for the feature:', VariablePlotted)
            print('Transformation that has been used for this feature:', 'PowerTransformation')
            print('Values for this feature is  Scaled between:', '[1e-10, 1]')
            Visual(xa, ya, VariablePlotted, config)      #------------------------------------->
        except:
            print('Error With Power Transformer')
        
    else:
        print("Graph using Power Transformation isn't plotted as function isn't working in python 2")

    ## Applying other Transformation
    xa = minmax_scale(X_series.apply( transf['UserDefined'] ) , feature_range=(1e-100, 1))
    ya = Y_series
    print('*'*90)
    print('Applying user Defined Transformation')
    print('*'*90)
    print('Plotting Data point for the feature:', VariablePlotted)
    print('Transformation that has been used for this feature:', UserDefFunc)
    print('Values for this feature is  Scaled between:', '[1e-100, 1]')
    Visual(xa, ya, VariablePlotted, config)      #------------------------------------->
    
    
def DataExploration_1(DatFram, config):

    #tranformations = {'x': , '1/x': lambda x: 1/x, 'x**2': lambda x: x**2, 'x**3': lambda x: x**3, 'log(x)': lambda x: np.log(x)}
    
    Transformations = { 'D_UzmaToD_UA': {'Actual': lambda x: x, 'UserDefined': lambda x: (x)**0.001 },
                           'HitsToD_Uzmc': {'Actual': lambda x: x, 'UserDefined': lambda x: (x)**0.001 },
                           'D_PageVisitedToHits': {'Actual': lambda x: x, 'UserDefined': lambda x: -x**3 },
                           'PageActToD_PageVisit': {'Actual': lambda x: x, 'UserDefined': lambda x: np.exp(-0.0001*np.log(x + 1e-1)) }, # np.exp(-100*np.log(x + 1e-1))
                           'BrowsrActToD_BrowsrUsed': {'Actual': lambda x: x, 'UserDefined': lambda x: (x)**0.00001 },
                           'AvgMedianTimeDiffBWHits': {'Actual': lambda x: x, 'UserDefined': lambda x: np.log10(1/(x + 1e-1)) },
                           #'AvgAvgTimeDiffBWHits': {'Actual': lambda x: x, 'UserDefined': lambda x: np.log10(1/(x + 1e-1)) },
                           #'HitsToAvgAvgTimeDiffBWHits': {'Actual': lambda x: x, 'UserDefined': lambda x: (x)**0.001 },
                           'StandDeviatAvgTimeDiffBWHits': {'Actual': lambda x: x, 'UserDefined': lambda x: np.log10(1/(x + 1e-1)) },
                           # 'HitsToAvgAvgTimeDiffBWHits': {'Actual': lambda x: x, 'UserDefined': lambda x: np.log10(1/(x + 1e-1)) },
                           'AvgHitsPerUnitTime': {'Actual': lambda x: x, 'UserDefined': lambda x: (x)**0.001 },
                           'DiffOfAvgTimeDiffBWHitsWhnGrpIPAndIPUzma': {'Actual': lambda x: x, 'UserDefined': lambda x: (1/(x + 1e-1)) },
                      }
    
    ## Remove Outlier outside the z score range of
    ZScoreStr = '[-5, 5]' #config['DataProcessing_Outlier']['ZScoreRangeForOutlierRemoval']
    ZScoreRangeToConsiderToLabelOutlier = int(re.sub(r'[^\d\-]', ' ', ZScoreStr).split()[0]), int(re.sub(r'[^\d\-]', ' ', ZScoreStr).split()[1])
    print("Z Score Range for Outlier Removal:", ZScoreRangeToConsiderToLabelOutlier)
    
    for var in Transformations:
        print("X"*95)
        print("\nPlotting Graphs For Variable:", var)
        print(Transformations[var])
        # for transf in Transformations[var]:
            
        # Without Treating Outlier
        print("\nPlotting Graph Including Outliers")
        ## Printing General Stats
        PlotDF = DatFram.copy()
        GeneralStats(PlotDF[var])
        GenerateGraph(PlotDF[var], PlotDF['isBotHits'], Transformations[var], config, var)      #------------------------------------->

        # After Treating Outlier
        print("\nPlotting Graph Excluding Outliers")
        PlotDF = DatFram.copy()

        print("Initial DataFrame Shape:", PlotDF.shape)
        ## Remove Outlier outside the z score range of
        ZScoreStr = '[-5, 5]' #config['DataProcessing_Outlier']['ZScoreRangeForOutlierRemoval']
        ZScoreRangeToConsiderToLabelOutlier = int(re.sub(r'[^\d\-]', ' ', ZScoreStr).split()[0]), int(re.sub(r'[^\d\-]', ' ', ZScoreStr).split()[1])
        print("Z Score Range for Outlier Removal:", ZScoreRangeToConsiderToLabelOutlier)

        IndexOfOutlier = GetOutlierIndex(PlotDF, var, ZScoreRangeToConsiderToLabelOutlier)      #------------------------------------->
        print("Outliers Index:", IndexOfOutlier)
        # ZScoreAtTheseOutlierIndex = mstats.zscore(minmax_scale(PlotDF[var].apply(lambda x: x ) , feature_range=(1e-100, 1)))[IndexOfOutlier]
        # print("Z-score at these Outliers Index:", ZScoreAtTheseOutlierIndex)

        PlotDF = PlotDF.iloc[[ind for ind in PlotDF.index if ind not in IndexOfOutlier]].reset_index(drop = True)
        print("No. of Outlier Cases Removed From dataframe:", len(IndexOfOutlier))
        print("Final DataFrame Shape:", PlotDF.shape)

        GenerateGraph(PlotDF[var], PlotDF['isBotHits'], Transformations[var], config, var)      #----------------------------->

# def DataExploration_1(input_processed_raw_df):
#     """
#     """
#     pass