### Binomial Approximation
## ansatz : 
## 1 - homogeneous portfolio (pd, lgd)
## 2 - finite number of obligors  
from single_factor_models.src.common import binomial_factor
import math
from scipy import stats, integrate
import numpy as np
import sys

### simple mathematical functions
def k_default_probability(k, x, num_of_credits, default_probability, correlation):
    norm_dist = stats.norm(0,1)
    p_x = single_name_probability(x, default_probability, correlation)
    return (p_x**k)*((1-p_x)**(num_of_credits-k))*norm_dist.pdf(x)

def get_var_loss(y, var_values):
     for (loss, var_value) in var_values:
         if var_value >= y:
             return loss

def single_name_probability(loss, default_probability, correlation):
    """
    conditional default probability (pg 30) Hibbeln book
    """
    norm_dist = stats.norm(0,1)
    numerator = norm_dist.ppf(default_probability) - math.sqrt(correlation)*loss
    denominator = math.sqrt(1-correlation)
    return norm_dist.cdf(numerator/denominator)

### biomial var and es
def binomial_var(loss, num_of_credits, default_probability, correlation, loss_given_default):
    """
    analytical integral of the binomial approximation (pg 32-33) Hibbeln book
    """
    result = 0.0
    for k in range(int(math.floor(loss*num_of_credits/loss_given_default)+1)):
        factor = binomial_factor(num_of_credits,k)
        result = result + factor*integrate.quad( lambda x : k_default_probability(k, x, num_of_credits, default_probability, correlation), 
                                                 -20.0, 20.0)[0]
    return result

def binomial_es(es_percentile, var_values, num_of_credits, default_probability, correlation, loss_given_default):
    """
    analytical integral of the binomial approximation (pg 32-33) Hibbeln book
    """
    result = 0.0
    cum = 0
    var_value = get_var_loss(es_percentile, var_values)
    for k in xrange(int(math.floor(var_value*num_of_credits/loss_given_default))+1, int(math.floor(num_of_credits/loss_given_default))+1):
        factor = binomial_factor(num_of_credits,k)
        prob = factor*integrate.quad( 
            lambda x : k_default_probability(k, x, num_of_credits, default_probability, correlation), -20.0, 20.0)[0]
        cum += prob
        result+= k*prob    
    if cum:
        es_value = (result/num_of_credits-var_value*(cum-(1.0-es_percentile)) )/(1.0-es_percentile)
        return (es_value, es_percentile)
    else: 
        return (1.0, es_percentile)

def main(x_list, num_of_credits, default_probability, correlation, loss_given_default):
    var_values = [binomial_var(i, num_of_credits, default_probability, correlation, loss_given_default) for i in x_list]
    y_list = pylab.arange(0.8, max(var_values), 0.005)
    es_values = [binomial_es(i, zip(list(x_list), list(var_values)), num_of_credits, default_probability, correlation, loss_given_default) 
                 for i in y_list]
    return var_values, es_values

def usage():
    print " python _model.py <number_of_credits> <default_probability> <correlation> <lgd> [<x_min> <x_max> <x_step>]"
    exit  

if __name__=="__main__":
    """
    var and es  as a function of the percentile 
    """
    #correlation = 0.2 
    #num_of_credits = 40
    #default_probability = 0.01
    #loss_given_default = 1.0
    if len(sys.argv) != 5 and len(sys.argv) != 8:
        usage() 
    else:
        if len(sys.argv) == 5:
            num_of_credits, pd, correlation, lgd  = sys.argv[1:5]
            x_min, x_max, x_step = (0.0, 0.30, 0.005)
        else:
            num_of_credits, pd,  correlation, lgd, x_min, x_max, x_step = sys.argv[1:8]
        import pylab
        x_list = pylab.arange(float(x_min), float(x_max), float(x_step))
        var_values, es_values = main(x_list, int(num_of_credits), 
                                     float(pd), float(correlation), float(lgd))
        pylab.suptitle('Binomial Approximation', fontsize=12)
        pylab.xlabel("percentile")
        pylab.ylabel("loss")
        pylab.plot(x_list, var_values, 'b--', label='var loss')
        pylab.plot(zip(*es_values)[0], zip(*es_values)[1], 'g-' , label='es loss')
        pylab.legend()
        pylab.show()

