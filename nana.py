import numpy as np
from scipy.signal import find_peaks

def helloworld():
    print("Hello world!")

def get_started(num_of_peaks, xs, ys):
    indxs, properties = find_peaks(ys)
    return indxs[np.argsort(-ys[indxs])[:num_of_peaks]]

def check_inputs(xs):
    for i in range(len(xs)-1):
        if xs[i] > xs[i+1]:
            print("check_inputs(): input xs is badly ordered. Use reorder_inputs to reorder")
            return False
    return True

def reorder_inputs(xs,ys):
    i = np.argsort(xs)
    return xs[i], ys[i]

def design_matrix(xlist):
    return (np.vstack((xlist**0,xlist**1,0.5*xlist**2))).T

def fit_parabola(xs, ys, index):
    return np.linalg.solve(design_matrix(xs[index-1:index+2]), ys[index-1:index+2])

def refine_peak(xs,ys,index):
    b,m,q = fit_parabola(xs, ys, index)
    x_peak = -m / q
    return x_peak, 0.5 * q * (x_peak) ** 2 + m * (x_peak) + b

def refine_peaks(xs, ys, indices):
    foo = lambda i: refine_peak(xs,ys,i)
    return list(map(foo,indices))



