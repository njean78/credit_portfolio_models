
# general formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i

import math
from scipy import stats
import pylab
import numpy as np
#import matplotlib.pyplot as plt

def loss_funtion(loss, default_probability, correlation, loss_given_default):
    norm_dist = stats.norm(0,1)
    scale_factor = 1.0/math.sqrt(correlation)
    inv1 = math.sqrt(1.0-correlation)*norm_dist.ppf(loss/loss_given_default)
    inv2 = norm_dist.ppf(default_probability)
    return norm_dist.cdf(scale_factor*(inv1 - inv2))

def loss_density(loss, default_probability, correlation, loss_given_default):
    norm_dist = stats.norm(0,1)
    scale_factor = math.sqrt(1.0 - correlation)/math.sqrt(correlation)
    inv1 = math.sqrt(1.0-correlation)*norm_dist.ppf(loss/loss_given_default)
    inv2 = norm_dist.ppf(default_probability)
    inv3 = norm_dist.ppf(loss/loss_given_default)
    loss_factor = (1.0/(2.0*correlation))*( inv1-inv2)**2.0 + 0.5*(inv3**2)
    return scale_factor*math.exp(-loss_factor)

def loss_var(percentile, default_probability, correlation):
    norm_dist = stats.norm(0,1)
    inv1 = norm_dist.ppf(default_probability)
    return norm_dist.cdf((inv1+math.sqrt(correlation)*norm_dist.ppf(percentile))/ math.sqrt(1.0-correlation))

def expected_shortfall(percentile, default_probability, correlation, loss_given_default):
    norm_dist = stats.norm(0,1)
    scale_factor = 1.0/(1.0-percentile)
    means = np.array([0.0, 0.0])
    lower = np.array([-100.0, -100.0])
    inv1 = norm_dist.ppf(default_probability)
    inv2 = norm_dist.ppf(percentile)
    higher = np.array([inv1, -inv2])
    covariance = np.matrix([[1.0, math.sqrt(correlation)], [math.sqrt(correlation), 1.0]])
    return scale_factor*loss_given_default*stats.mvn.mvnun(lower, higher, means, covariance)[0]

def main():
    default_probability = 0.05
    correlation = 0.2 
    loss_given_default = 1.0
    x_list = pylab.arange(0.0, 1.00, 0.01)
    loss_f_values = [loss_funtion(i, default_probability, correlation, loss_given_default) for i in x_list]
    loss_d_values = [loss_density(i, default_probability, correlation, loss_given_default) for i in x_list]


    pylab.xlabel("loss in percentage")
    pylab.ylabel("function values")
    pylab.plot(x_list, loss_f_values, 'b')
    pylab.plot(x_list, loss_d_values, 'g')
    pylab.show()

if __name__ == "__main__":
    main()
