import numpy as np
import cryptolytic.util.date as date


def pearson(a, b):
    asum = np.sum(a)
    bsum = np.sum(b)
    print(asum)
    aavg = asum / len(a)
    bavg = bsum / len(b)
    print(aavg)
    adiff = a - aavg
    bdiff = b - bavg
    print(adiff)
    ans = ( 
        np.sum(np.multiply(adiff, bdiff)) / 
        np.sqrt((np.sum(adiff)**2 * np.sum(bdiff)**2))
    )
    return ans



