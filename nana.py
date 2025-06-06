from scipy.signal import find_peaks
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy.signal
from astropy import units as u
from scipy.interpolate import CubicSpline
from math import cos, sin, radians

def star(kic_id):
    search_result = lk.search_lightcurve(kic_id, mission='Kepler')


    lc_collection = search_result.download_all()


    lc = lc_collection.stitch()


    delta_f = (1/(lc.time[-1] - lc.time[0]).value) 


    sampling_time= np.median(np.diff(lc.time.value))





    return (lc, delta_f, sampling_time)





f_avoid = 3.5 / 372.5


lc_exptime = (6.52 * 270) / (60 * 60 * 24)





def check_inputs(xs):


    """


    


    Checks whether the input array `xs` is sorted in ascending order.





    Args:


        xs (numpy.ndarray or list): The input array to check.





    Returns:


        bool: `True` if `xs` is sorted in ascending order, otherwise `False`.


    


    """


    for i in range(len(xs)-1):


        if xs[i] > xs[i+1]:


            print("check_inputs(): input xs is badly ordered. Use reorder_inputs to reorder")


            return False


    return True





def reorder_inputs(xs,ys):


    """


    


    Reorders the input arrays `xs` and `ys` in ascending order of `xs`.





    Args:


        xs (numpy.ndarray): The x-axis values 


        ys (numpy.ndarray): The y-axis values


        


    Returns:


        tuple of numpy arrays (sorted xs, sorted ys)





    Bugs:


        `xs` or `ys` must be NumPy array


        `xs` and `ys`must be same length


        


    """


    i = np.argsort(xs)


    return xs[i], ys[i]





#xlist is the teh index left to highest peak, highest peak, and the index right to teh highest peak


def design_matrix(xlist): 


    """


    


    Constructs a design matrix for quadratic curve fitting.





    Args:


        xs (numpy.ndarray): The x-axis values 





    Returns:


        3 x 3 design matrix of numpy arrays





    Bugs:


        xlist must be an numpy array





    Note:


        Note the factor of 0.5 that Hogg likes and no one else


        Also assumes xlist is ordered


    


    """


    return (np.vstack((xlist**0,xlist**1,0.5*xlist**2))).T





def fit_parabola(xs, ys, index):


    """





    Fits a quadratic function to three consecutive data points. Solves for coefficients (b,m,q) in the quadratic


    f(x) = b + m*x + (1/2) * q * x^2





    Args:


        xs (numpy.ndarray): The x-axis values 


        ys (numpy.ndarray): The y-axis values


        index (int): The index of peak


        





    Returns:


        tuple: (b, m, q)





    Bugs: 


        index-1` or `index+2` can't be out of bounds


        ``xs` or `ys` must be NumPy array


        


    


    """


    return np.linalg.solve(design_matrix(xs[index-1:index+2]), ys[index-1:index+2])





def refine_peak(xs, ys, index):


    """





    Refines the peak position using quadratic fit





    Args:


        xs (numpy.ndarray): The x-axis values 


        ys (numpy.ndarray): The y-axis values


        index (int): The index of peak


    


    Returns:


        3-tuple: (x position of refined peak, y position of refined peak, and second derivative (q))





    Bugs:


        Must be synchronized with the design matrix


        


        


    """


    b,m,q = fit_parabola(xs, ys, index)


    x_peak = -m / q


    return x_peak, 0.5 * q * (x_peak) ** 2 + m * (x_peak) + b, q


    


def refine_peaks(xs, ys, indices):


    """





    Refines the peak position for a set of indices using quadratic fit





    Args:


        xs (numpy.ndarray): The x-axis values 


        ys (numpy.ndarray): The y-axis values


        indices (numpy array): indices of the peaks (this should be the output of get_started()





    Returns:


        three  numpy arrays (array of refined x positions, array of refined y positions, and the second derivatives)


    





    """


    foo = lambda i: refine_peak(xs,ys,i)


    xs_refined, ys_refined, second_derivatives = zip(*list(map(foo,indices)))


    return np.array(xs_refined), np.array(ys_refined), np.array(second_derivatives)








def folding_freq(delta_f, fs, ps, sampling_time, makeplots=False):


    """


    ##bugs:


    - assumes fs are ordered


    """


    fc_guess = 1. / sampling_time


    


    IA = fs < 0.5 * fc_guess


    fsA, psA = fs[IA], ps[IA]


    fsA,psA  = fsA[2:-2],  psA[2:-2]


    cs = CubicSpline(fs, ps, extrapolate=False)


    


    small, tiny = 20 * delta_f, 0.25 * delta_f


    fc_candidates = np.arange(fc_guess - small, fc_guess + small, tiny)


    foos_c = np.array([np.nansum(psA * cs(fc - fsA)) for fc in fc_candidates])


    fc_index = get_filtered_peaks(1,fc_candidates, foos_c)


    fc, _, _ = refine_peaks(fc_candidates, foos_c, fc_index)


    fc = fc[0]


    


    if makeplots:


        plt.plot(fc_candidates, foos_c)


        plt.axvline(fc_guess)


        plt.axvline(fc, color = 'red', alpha = 0.5)


        plt.title(fc)


        plt.show()


    return fc





def find_min_and_refine(xs,ys):


    indxs, properties = find_peaks(-ys)


    min_index =  indxs[np.argsort(ys[indxs])[:1]]


    refined_x, refined_y, second_derivative = refine_peaks(xs, ys, min_index)


    return refined_x[0], refined_y[0]





def get_filtered_peaks(num_of_peaks, xs, ys): 


    '''


    ##bugs:


    - realizes on global variable f_avoid


    '''


    indxs, properties = find_peaks(ys)


    indices = indxs[np.argsort(-ys[indxs])]





    filtered = []


    for index in indices:


        if all(abs(xs[index] - xs[i]) >= (f_avoid) for i in filtered):


            filtered.append(index)


            if(len(filtered) >= num_of_peaks):


                break


    return np.array(filtered)





def integral_design_matrix(ts, om, T):


    """


    ##bugs:


    - assumes all data points have the same exposure time, `T`


    - not numerically stable when `om * T` is small


    """


    return np.vstack([


        np.ones_like(ts),


        (+ np.sin(om * (ts + T/2)) - np.sin(om * (ts - T/2))) / (om * T),


        (- np.cos(om * (ts + T/2)) + np.cos(om * (ts - T/2))) / (om * T)


    ]).T





def weighted_least_squares(A, b, weights):


    ATA = A.T @ (A * weights[:, np.newaxis])


    ATb = A.T @ (b * weights)


    return A @ np.linalg.solve(ATA, ATb)





def weighted_least_squares_new(A, b, weights):


    ATA = A.T @ (A * weights[:, np.newaxis])


    ATb = A.T @ (b * weights)


    trace = np.trace(ATA)


    det = np.linalg.det(ATA)


    return np.linalg.solve(ATA, ATb), ATA





def integral_chi_squared(om, ts, ys, ws, T):


    A = integral_design_matrix(ts, om, T)


    return np.sum(ws * (ys - weighted_least_squares(A, ys, ws))**2)





def region_and_freq(indices, folding_freq, f_min, unrefined_freq, unrefined_power, t_fit, flux_fit, weight_fit):


    


    regions, best_freqs, best_chi2s = [], [], []





    for i in range(len(indices)):


        fas, __, __ = refine_peaks(unrefined_freq, unrefined_power, indices)


        A, B, C = fas, folding_freq-fas, folding_freq+fas


        


        fine_freqsA = np.arange(A[i] - 5 * f_min, A[i] + 5 * f_min, 0.2 * f_min)  


        chi2_fineA = np.array([integral_chi_squared(2. * np.pi * f, t_fit, flux_fit, weight_fit, lc_exptime) for f in fine_freqsA])


        best_freqA, best_chi2A = find_min_and_refine(fine_freqsA,chi2_fineA)


    


        fine_freqsB = np.arange(B[i]- 5 * f_min, B[i] + 5 * f_min, 0.2 * f_min)  


        chi2_fineB = np.array([integral_chi_squared(2. * np.pi * f, t_fit, flux_fit, weight_fit, lc_exptime) for f in fine_freqsB])


        best_freqB, best_chi2B = find_min_and_refine(fine_freqsB,chi2_fineB)


    


        fine_freqsC = np.arange(C[i] - 5 * f_min, C[i] + 5 * f_min, 0.2 * f_min)  


        chi2_fineC = np.array([integral_chi_squared(2. * np.pi * f, t_fit, flux_fit, weight_fit, lc_exptime) for f in fine_freqsC])


        best_freqC, best_chi2C = find_min_and_refine(fine_freqsC,chi2_fineC)


        


        #print(best_chi2A, best_chi2B, best_chi2C)





        if (best_chi2A <= best_chi2B and best_chi2A <= best_chi2C):


            regions.append("A")


            best_freqs.append(best_freqA)


            best_chi2s.append(best_chi2A)


            


        elif (best_chi2B < best_chi2A and best_chi2B < best_chi2C):


            #print("here")


            regions.append("B")


            best_freqs.append(best_freqB)


            best_chi2s.append(best_chi2B)


            


        elif (best_chi2C < best_chi2A and best_chi2C < best_chi2B):


            #print("herec")


            regions.append("C")


            best_freqs.append(best_freqC)


            best_chi2s.append(best_chi2C)








    return(regions, best_freqs, best_chi2s)





def check_coherence(ts, ys, weights, final_freq):


    '''assumes a lot of thing about the data


    '''





    oms = np.array([(f * 2 * np.pi) for f in final_freq])


    


    a_early, a_late, b_early, b_late, invvars = [], [], [], [], []


    


    ts_median = np.median(ts)


    


    cases = [(ts < ts_median, "early"),


             (ts > ts_median, "late")]


    


    for om in oms:


        


        for I, name in cases:


            A = integral_design_matrix(ts[I], om, lc_exptime)


            pars, ___ = weighted_least_squares_new(A, ys[I], weights[I])


            a,b = pars[1], pars[2]


            


            if name == "early":


                a_early.append(pars[1])


                b_early.append(pars[2])


            elif name == "late":


                a_late.append(pars[1])


                b_late.append(pars[2])





    return a_early, a_late, b_early, b_late





def change_in_phase_and_amp(a_early, a_late, b_early, b_late, ts):





    rates_of_phases = []


    rates_of_amps = []


    ts_median = np.median(ts)


    delta_t = np.median(ts[ts>ts_median]) - np.median(ts[ts < ts_median])





    





    for a_earl_grey, a_latte, b_earl_grey, b_latte in zip(a_early, a_late, b_early, b_late):


    





        delta_r = [a_latte - a_earl_grey, b_latte - b_earl_grey]


        vector_r = [0.5 * (a_latte + a_earl_grey), 0.5 * (b_latte + b_earl_grey)]





        cross_z = delta_r[0] * vector_r[1] - delta_r[1] * vector_r[0]


        phase = (1 / delta_t) * (cross_z / np.dot(vector_r, vector_r))


        rates_of_phases.append(phase)


        


        amplitude = (1/(delta_t)) * (np.dot(delta_r, vector_r) / np.dot(vector_r, vector_r))


        rates_of_amps.append(amplitude)





    return (rates_of_phases, rates_of_amps)





def sharpness(second_derivatives, y_news):


    sharps = []


    for second_derivative, y_new in zip(second_derivatives, y_news):


        sharpness = (-second_derivative/y_new)**(1/2)


        sharps.append(sharpness)


    return sharps





def null_chi_squared(ts, ys, weights):


    


    a0 = np.sum(weights * ys) / np.sum(weights)


    null_chisq = np.sum(weights * (ys - a0) ** 2)


    return null_chisq





def mask_vals(lc): 





    t_clean = np.ma.filled(lc.time.value, np.nan)


    flux_clean = np.ma.filled(lc.flux.value, np.nan)


    sigma_clean = np.ma.filled(lc.flux_err.value, np.nan)





    mask = np.isfinite(t_clean) & np.isfinite(flux_clean) & np.isfinite(sigma_clean)





    t_fit = t_clean[mask]


    flux_fit = flux_clean[mask]


    sigma_fit = sigma_clean[mask]


    weight_fit = 1 / sigma_fit**2


    


    return(t_fit, flux_fit,weight_fit)





def pg_full(f_min, f_max, lc):


    


    frequency_grid_full = np.arange(f_min, f_max, f_min)/(u.day)


    


    pg_full = lc.to_periodogram(


        method='lombscargle',


        normalization='psd',


    frequency=frequency_grid_full


    )





    power_full = pg_full.power.value


    freq_full = pg_full.frequency.to(1/u.day).value 





    return(freq_full, power_full)





def pg_mini(f_min, f_max, lc):





    frequency_grid_mini = np.arange(f_min, f_max / 3, f_min) / (u.day)


    


    pg_mini = lc.to_periodogram(


        method='lombscargle',


        normalization='psd',


        frequency=frequency_grid_mini


    )


    


    power_mini = pg_mini.power.value


    freq_mini = pg_mini.frequency.to(1/u.day).value 





    return(freq_mini, power_mini)
