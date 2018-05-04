### ASRF model (asymptotic single risk factor)
## ansatz : 
## 1 - homogeneous portfolio (pd, lgd)
## 2 - infinite number of obligors 

# general loss formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i
import math
from scipy import stats
import numpy as np
import sys

def loss_function(loss, default_probability, correlation):
    """
    loss distribution for the vasicek model (pg 34 Hibbeln book)
    """
    norm_dist = stats.norm(0,1)
    scale_factor = 1.0/math.sqrt(correlation)
    inv1 = math.sqrt(1.0-correlation)*norm_dist.ppf(loss)
    inv2 = norm_dist.ppf(default_probability)
    return norm_dist.cdf(scale_factor*(inv1 - inv2))

def loss_density(loss, default_probability, correlation):
    """
    probability density function for the vasicek model (pg 34 Hibbeln book)
    """
    norm_dist = stats.norm(0,1)
    scale_factor = math.sqrt(1.0 - correlation)/math.sqrt(correlation)
    inv1 = math.sqrt(1.0-correlation)*norm_dist.ppf(loss)
    inv2 = norm_dist.ppf(default_probability)
    inv3 = norm_dist.ppf(loss)
    loss_factor = (1.0/(2.0*correlation))*( inv1-inv2)**2.0 + 0.5*(inv3**2)
    return scale_factor*math.exp(-loss_factor)

def loss_var(percentile, default_probability, correlation):
    """
    VaR function for the vasicek model (pg 35 Hibbeln book)
    """
    norm_dist = stats.norm(0,1)
    inv1 = norm_dist.ppf(default_probability)
    return norm_dist.cdf((inv1+math.sqrt(correlation)*norm_dist.ppf(percentile))/ math.sqrt(1.0-correlation))

def loss_var_mc(percentile, default_probability, correlation):
    """
    VaR function for the vasicek model (pg 35 Hibbeln book)
    """
    norm_dist = stats.norm(0,1)
    inv1 = norm_dist.ppf(default_probability)
    
    samples = [norm_dist.cdf((inv1+math.sqrt(correlation)*np.random.randn())/ math.sqrt(1.0-correlation)) for i in range(10000)]
    samples.sort()
    return samples[int(math.floor(percentile*10000))]

def expected_shortfall(percentile, default_probability, correlation, loss_given_default):
    """
    Expected Shortfall function for the vasicek model (pg 35 Hibbeln book). 
    The bivariate cumualtive distribution is integrated using the mvn.mvnun library function.
    """
    norm_dist = stats.norm(0,1)
    scale_factor = 1.0/(1.0-percentile)
    means = np.array([0.0, 0.0])
    lower = np.array([-100.0, -100.0])
    inv1 = norm_dist.ppf(default_probability)
    inv2 = norm_dist.ppf(percentile)
    higher = np.array([inv1, -inv2])
    covariance = np.matrix([[1.0, math.sqrt(correlation)], [math.sqrt(correlation), 1.0]])
    return scale_factor*loss_given_default*stats.mvn.mvnun(lower, higher, means, covariance)[0]

def main(x_list, default_probability=0.05, correlation=0.2, loss_given_default=1.0):
    """
    return 2 lists of values or raises an Exception
    """
    loss_f_values = [loss_function(i/loss_given_default, default_probability, correlation) for i in x_list]
    loss_d_values = [loss_density(i/loss_given_default, default_probability, correlation) for i in x_list]
    return loss_f_values, loss_d_values
    

def usage():
    print " python asrf_model.py <default_probability> <correlation> <lgd> [<x_min> <x_max> <x_step>]"
    exit

if __name__=="__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 7:
        usage() 
    else:
        if len(sys.argv) == 4:
             pd, corr, lgd  = sys.argv[1:4]
             x_min, x_max, x_step = (0.0, 1.00, 0.01)
        else:
            pd, corr, lgd, x_min, x_max, x_step  = sys.argv[1:7]
        import pylab
        x_list = pylab.arange(float(x_min), float(x_max), float(x_step))
        loss_f_values, loss_d_values = main(x_list, float(pd), float(corr), float(lgd))
        pylab.suptitle('ASFR model', fontsize=12)
        pylab.xlabel("loss in percentage")
        pylab.ylabel("function values")
        pylab.plot(x_list, loss_f_values, 'b--', label='loss distribution')
        pylab.plot(x_list, loss_d_values, 'g-' , label='loss density')
        pylab.legend()
        pylab.show()
    
