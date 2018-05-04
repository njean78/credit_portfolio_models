# general formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i


import math
from scipy import stats
import pylab
import numpy as np


def z(alpha, pd, corr):
    norm_dist = stats.norm(0,1)
    return (norm_dist.ppf(pd)+ 
            math.sqrt(corr)*norm_dist.ppf(alpha))/math.sqrt(1-corr)
def varinf(alpha,  pd, corr):
    norm_dist = stats.norm(0,1)
    a = norm_dist.ppf(pd)+ math.sqrt(corr)*norm_dist.ppf(alpha)
    a = a/math.sqrt(1-corr)
    return norm_dist.cdf(a)

def var_adjustment(alpha,  pd, corr, elgd, num_of_credits):
    norm_dist = stats.norm(0,1)
    z_value = z(alpha, pd, corr)
    cum_z = norm_dist.cdf(z_value)
    density_z = norm_dist.pdf(z_value)
    beta = norm_dist.ppf(alpha)*(1-2*corr)- norm_dist.ppf(pd)*math.sqrt(corr)
    beta = beta/math.sqrt(corr*(1-corr))
    beta = beta*cum_z*(1-cum_z)/density_z
    return -elgd*(1.0/2.0)*(1.0/num_of_credits)*(beta-1.0+2*cum_z)

def es_adjustment(alpha,  pd, corr, elgd, num_of_credits):
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

def main():
    num_of_credits = 40
    default_probability = 0.01
    correlation = 0.2 
    loss_given_default = 1.0
    
    x_list = pylab.arange(0.05, 1.0, 0.05)

    loss_f_values = [var_adjustment(0.99, default_probability, i, 
                                    loss_given_default, num_of_credits) 
                     for i in x_list]
    es_values = [es_adjustment(0.99, default_probability, i, 
                                 loss_given_default, num_of_credits) 
                   for i in x_list]
    pylab.xlabel("loss in percentage")
    pylab.ylabel("loss function values")
    pylab.plot(x_list, loss_f_values, 'b')
    pylab.plot(x_list, es_values, 'g')
    pylab.show()


if __name__ == "__main__":
    main()

