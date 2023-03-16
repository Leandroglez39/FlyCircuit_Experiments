import pandas as pd
import pyreadr
#import rpy2
#import rpy2.robjects as robjects

if __name__ == '__main__':
  
    #import rpy2.robjects as robjects
    #from rpy2.robjects import pandas2ri
    #readRDS = robjects.r['readRDS']
    #df = readRDS('A_str.rds')
    #df = pandas2ri.rpy2py_dataframe(df)

   

    result = pyreadr.read_r('./data/A_str.rds')

    # done! let's see what we got
    print(result.keys()) # let's check what objects we got: there is only None
    df1 = result[None] # extract the pandas data frame for the only object available