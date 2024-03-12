def RG_Gfit(true,pred,residual='Yes'):
    import numpy as np
    import pylab
    import matplotlib.pyplot as plt
    import sklearn.metrics
    import math
    import statistics
    from scipy.stats import pearsonr
    '''
    Statistics used in Nieto's paper to evaluate the model performance
    (Evaluating different metrics from the thermal-based two-source energy balance model for monitoring grapevine water stress) 
    are adopted in this repository.
    Except fot the statistics, the residual plot is an option in this function.
    
    The input information is:
    param true: the observation data.
    param pred: the predictions gained from models.
    param residual: if the input is 'Yes', residual plot will be presented. Otherwise, no residual plot.
    
    6 statistics (outputs) are:
    n: the number of cases.
    mse: the mean square error.
    rmse: the root mean square error.
    bias: the mean bias computed as the observed minus the predicted.
    r: the Pearson correlation coefficient between the observed and the predicted.
    d: the Wilmott's index of agreement.
    '''
    
    # residual plot (optional)
    if residual=='Yes':
        plt.figure(figsize=(5, 2.5))
        plt.scatter(range(1, 1+len(true)),true-pred)
        plt.plot(range(1, 1+len(true)),[0]*len(true),'r--')
        plt.xlim(0, len(true)+1)
        plt.ylabel("Residual")
        plt.show()
    else:
        pass
    
    # The number of cases used for validation
    n = len(true)
    # Mean square error
    mse = sklearn.metrics.mean_squared_error(true,pred)
    # Root mean square error
    rmse = math.sqrt(mse)
    # Bias
    bias = statistics.mean(true-pred)
    # Pearson correlation coefficient - r
    r, _ = pearsonr(true,pred)
    # Willmott's index of agreement - d
    tmp_1 = (true-pred)**2
    tmp_1 = tmp_1.sum()
    tmp_2 = true.mean()
    tmp_2 = (np.abs(true-tmp_2)+np.abs(pred-tmp_2))**2
    d = 1-(tmp_1/tmp_2.sum())
    
    print('Number of cases: %.0f' % n)
    print('Root mean square error: %.3f' % rmse)
    print('Bias: %.3f' % bias)
    print('Pearsons correlation: %.3f' % r)
    print('Willmott\'s index of agreement: %.3f' % d)
    print('Mean square error: %.3f' % mse)
    
    return(n,mse,rmse,bias,r,d)
