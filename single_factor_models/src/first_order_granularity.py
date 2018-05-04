### First order granularity adjustment to the ASRF model (Gordy)
## ansatz : 
## 1 - homogeneous portfolio (pd, lgd)
## 2 - finite number of obligors  

# general formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i

import math
from scipy import stats
import numpy as np
import sys

def z(alpha, pd, corr):
    """
    z function (pg 80 Hibbeln book)
    """
    norm_dist = stats.norm(0,1)
    return (norm_dist.ppf(pd)+ 
            math.sqrt(corr)*norm_dist.ppf(alpha))/math.sqrt(1-corr)

def var_adjustment(alpha,  pd, corr, elgd, num_of_credits):
    """
    VaR first order granularity adjustment (pg 82 Hibbeln book)
    """
    norm_dist = stats.norm(0,1)
    z_value = z(alpha, pd, corr)
    cum_z = norm_dist.cdf(z_value)
    density_z = norm_dist.pdf(z_value)
    beta = norm_dist.ppf(alpha)*(1-2*corr)- norm_dist.ppf(pd)*math.sqrt(corr)
    beta = beta/math.sqrt(corr*(1-corr))
    beta = beta*cum_z*(1-cum_z)/density_z
    return -elgd*(1.0/2.0)*(1.0/num_of_credits)*(beta-1.0+2*cum_z)

def es_adjustment(alpha,  pd, corr, elgd, num_of_credits):
    """
    ES first order granularity adjustment (pg 111 Hibbeln book)
    """
    norm_dist = stats.norm(0,1)
    z_value = z(alpha, pd, corr)
    cum_z = norm_dist.cdf(z_value)
    density_z = norm_dist.pdf(z_value)
    beta = (1.0/2.0)*(1.0/num_of_credits)
    beta*= norm_dist.pdf(norm_dist.ppf(1.0-alpha))/(1.0-alpha)
    beta*= math.sqrt(1.0-corr)/math.sqrt(corr)
    beta*= cum_z/density_z
    beta*= elgd*(1-cum_z)
    return beta

def main(x_list, percentile, num_of_credits, pd, lgd):
    var_values = [var_adjustment(percentile, pd, i, lgd, num_of_credits) 
                  for i in x_list]
    es_values = [es_adjustment(percentile, pd, i, lgd, num_of_credits) 
                 for i in x_list]
    return var_values, es_values

def usage():
    print " python _model.py <percentile> <number_of_credits> <default_probability> <lgd> [<x_min> <x_max> <x_step>]"
    exit    

if __name__=="__main__":
    """
    var and es granularity adjustment as a function of the correlation 
    """
    #percentile = 0.99 
    #num_of_credits = 40
    #default_probability = 0.01
    #loss_given_default = 1.0
    if len(sys.argv) != 5 and len(sys.argv) != 8:
        usage() 
    else:
        if len(sys.argv) == 5:
             percentile, num_of_credits, pd, lgd  = sys.argv[1:5]
             x_min, x_max, x_step = (0.01, 1.00, 0.01)
        else:
            percentile, num_of_credits, pd, lgd, x_min, x_max, x_step = sys.argv[1:8]
        import pylab
        x_list = pylab.arange(float(x_min), float(x_max), float(x_step))
        var_values, es_values = main(x_list, float(percentile), int(num_of_credits), 
                                     float(pd), float(lgd))
        pylab.suptitle('First Order Granularity Adjustment', fontsize=12)
        pylab.xlabel("correlation")
        pylab.ylabel("function values")
        pylab.plot(x_list, var_values, 'b--', label='var adjustment')
        pylab.plot(x_list, es_values, 'g-' , label='es adjustment')
        pylab.legend()
        pylab.show()
    
