# File for utility functions.
import numpy as np
# This function generates random partition sizes
def generate_partition_sizes(K, n, mean=1, variance=0, options=None, minsize=64):
    numattempt = 0
    
    # K: Number of partitions; n: Total number of samples; the rest: Parameters
    if options == 'uniform':
        nk = [int(n/K)]*K
        nk[-1] = n - sum(nk[:-1])
    
    # This option tries its best to generate a few data points with much larger sample sizes.
    # Might fail sometimes. Important to double check.
    elif options == 'lopsided':
        mean = 0
        vals = np.random.normal(mean, variance, (K,))
        vals = np.abs(vals)
        nvals = vals / np.sum(vals) * (n - K*minsize) 
        vals[np.argmin(vals)] *= 2
        nk = [int(nval+minsize) for nval in nvals]
        nk[-1] = n - sum(nk[:-1])

    # By default, try to use a normal distribution to determine partition sizes, and normalize the sum.
    # Note that the user is responsible for giving reasonable parameters. 
    else:
        vals = np.random.normal(mean, variance, (K,))
        vals = np.maximum(0, vals)
#         print(vals)
        nvals = vals / np.sum(vals) * (n - K*minsize) 
        nk = [int(nval+minsize) for nval in nvals]
        nk[-1] = n - sum(nk[:-1])
        
    if numattempt >= 1000:
        print("Failed to generate a good data partition distribution. Proceed at your own risk.")
#     print("Generated: ", nk)
    return nk