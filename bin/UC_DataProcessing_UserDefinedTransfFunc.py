import pandas as pd
import numpy as np

########################################################## Transforming Data To User Defined Functions
def UserDefinedTransformation(DF, config):
    '''
    Transforming Dimension based on Some Functions
    '''
    if DF is not None:
        DF = DF.copy()
    else:
        return None
    
    AllFeature = [ i for i in config['DataProcessing_General']['AllFeaturesToUtilize'].split("'") if len(i) > 2 ]
    FeatureToIgnore = [ i for i in config['DataProcessing_General']['FeatureToIgnore'].split("'") if len(i) > 2 ]
    FeatureToTransform = [ i for i in AllFeature if i not in FeatureToIgnore ]
    
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
    
    if config['DataProcessing_General']['TransformFeatureToUserDefinedFunction'] in ['Yes', 'yes', 'y', 'Y', 'True', 'true', 't', 'T']:
        for Var in FeatureToTransform:
            DF[Var] = DF[Var].apply( Transformations[Var]['UserDefined'] )
    print('>>> Data has been Transformed according to User Defined Function')
    return DF
## Doing Data Transformation
# for df in [InlierTrainDF, OutlierTrainDF, InlierTestDF, OutlierTestDF]:
#     df = UserDefinedTransformation(df, config_clust)

