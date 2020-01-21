import numpy as np
import cryptolytic.util.date as date


def get_by_time(df, start, end):
    q = (df.index > start) & (df.index < end)
    return df[q]


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

def denoise(signal, repeat):
    "repeat: how smooth to make the graph"
    copy_signal = np.copy(signal)
    for j in range(repeat):
        for i in range(3, len(signal)):
            # set previous timestep to be between the timestep i and i - 2
            copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
    return copy_signal

