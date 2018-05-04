# general formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i


import math
from scipy import stats, integrate
import pylab
import numpy as np
#import matplotlib.pyplot as plt

def factorial(a):
    if a == 0:
        return 1
    else:
        return a*factorial(a-1)

def combinatorial(a,b):
    return factorial(a)/(factorial(b)*factorial(a-b))



def single_name_probability(loss, default_probability, correlation):
    norm_dist = stats.norm(0,1)
    inv1 = norm_dist.ppf(default_probability)
    inv2 = math.sqrt(correlation)*loss
    numerator = norm_dist.ppf(default_probability) - math.sqrt(correlation)*loss
    denominator = math.sqrt(1-correlation)
    return norm_dist.cdf(numerator/denominator)

def k_default_probability(k, x, num_of_credits, default_probability, correlation):
    norm_dist = stats.norm(0,1)
    p_x = single_name_probability(x, default_probability, correlation)
    return (p_x**k)*((1-p_x)**(num_of_credits-k))*norm_dist.pdf(x)
    
def loss_function(loss, num_of_credits, default_probability, correlation, loss_given_default):
    result = 0.0
    for k in range(int(math.floor(loss*num_of_credits/loss_given_default)+1)):
        factor = combinatorial(num_of_credits,k)
        result = result + factor*integrate.quad( lambda x : k_default_probability(k, x, num_of_credits, default_probability, correlation), -20.0, 20.0)[0]
    return result

def get_loss(y, var_values):
     for (loss, var_value) in var_values:
         if var_value >= y:
             return loss

def expected_shortfall(es_percentile, var_values, num_of_credits, default_probability, correlation, loss_given_default):
    result = 0.0
    cum = 0
    var_value = get_loss(es_percentile, var_values)
    for k in xrange(int(math.floor(var_value*num_of_credits/loss_given_default))+1, int(math.floor(num_of_credits/loss_given_default))+1):
        factor = combinatorial(num_of_credits,k)
        prob = factor*integrate.quad( 
            lambda x : k_default_probability(k, x, num_of_credits, default_probability, correlation), -20.0, 20.0)[0]
        cum += prob
        result+= k*prob    
    if cum:
        es_value = (result/num_of_credits-var_value*(cum-(1.0-es_percentile)) )/(1.0-es_percentile)
        return (es_value, es_percentile)
    else: 
        return (1.0, es_percentile)

def main():
    num_of_credits = 40
    default_probability = 0.01
    correlation = 0.2 
    loss_given_default = 1.0
    x_list = pylab.arange(0.0, 0.30, 0.005)
    y_list = pylab.arange(0.8, 0.999, 0.001)
    loss_f_values = [loss_function(i, num_of_credits, default_probability, correlation, loss_given_default) for i in x_list]
    es_values = [expected_shortfall(i, zip(list(x_list), list(loss_f_values)), num_of_credits, default_probability, correlation, loss_given_default) for i in y_list]
    
    pylab.xlabel("loss in percentage")
    pylab.ylabel("loss function values")
    pylab.plot(x_list, loss_f_values, 'b')
    pylab.plot(zip(*es_values)[0], zip(*es_values)[1], 'r')
    pylab.show()


if __name__ == "__main__":
    main()

