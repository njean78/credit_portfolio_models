# general formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i
# importing es and var for infinite_obligors cases
from single_factor_models.src.asrf_model import expected_shortfall as es_infinite_obligors
from single_factor_models.src.asrf_model import loss_function as var_infinite_obligors
# importing es and var for binomial cases
from single_factor_models.src.binomial_approximation import binomial_var
from single_factor_models.src.binomial_approximation import binomial_es
# importing es and var finite obligors adjustment
from single_factor_models.src.first_order_granularity import var_adjustment
from single_factor_models.src.first_order_granularity import es_adjustment
import pylab
import sys
import time
import numpy as np
import math


def main_var(num_of_credits, default_probability, correlation, loss_given_default, x_list):
    # binomial portfolio
    start_time = time.time()
    loss_binomial = [binomial_var(i, num_of_credits, default_probability, 
                                  correlation, loss_given_default) 
                     for i in x_list]
    binomial_time = time.time()
    # homogeneous + infinte portfolio
    loss_infinite_obligors = [var_infinite_obligors(i, default_probability, correlation) 
                              for i in x_list]
    homogeneous_time = time.time()
    # first order correction
    loss_first_order_correction = [var_adjustment(i, default_probability, correlation, loss_given_default, num_of_credits) 
                                   for i in loss_infinite_obligors]
    first_order_time = time.time()
    # summing homogenous+infinite result to the first order adjustment
    loss_first_adjusted = [(i+j) for (i,j) in zip(x_list, loss_first_order_correction)]
    # printing the results
    pylab.xlabel("var percentile in % ")
    pylab.ylabel("var value")
    pylab.plot((1.0-x_list)*100.0, loss_binomial, 'b', label='binomial, time in ms: %s'%
               math.ceil((binomial_time-start_time)*1000))
    pylab.plot((1.0-x_list)*100.0, loss_infinite_obligors, 'g', 
               label='infinite obligors, time in ms: %s' %math.ceil((homogeneous_time-binomial_time)*1000))
    pylab.plot((1.0-np.array(loss_first_adjusted))*100.0, loss_infinite_obligors, 'r', label='first order adjustment %s' %math.ceil((first_order_time - homogeneous_time)*1000))
    pylab.xlim([70, 100])
    pylab.ylim([0.8, 1.0])
    pylab.legend(loc=(0,0))
    pylab.show()

def main_es(num_of_credits, default_probability, correlation, loss_given_default, x_list, y_list):
    # binomial portfolio
    loss_binomial = [binomial_var(i, num_of_credits, default_probability, 
                              correlation, loss_given_default) 
                     for i in x_list]
    start_time = time.time()
    es_binomial_values = [binomial_es(i, zip(x_list, loss_binomial), num_of_credits, default_probability, 
                                      correlation, loss_given_default) 
                     for i in y_list]
    binomial_time = time.time()
    # homogeneous portfolio
    es_infinite_obligors_values = [es_infinite_obligors(i, default_probability, correlation, loss_given_default) 
                    for i in y_list]
    homogeneous_time = time.time()
    # first order correction
    es_first_order_correction = [
        es_adjustment(i, default_probability, correlation, 
                       loss_given_default, num_of_credits) 
        for i in y_list]

    es_first_adjusted = [
         (i+j)
         for (i,j) in zip(es_infinite_obligors_values, es_first_order_correction )]
    first_order_time = time.time()
    pylab.xlabel("es percentile in % ")
    pylab.ylabel("es value")
    pylab.plot((1.0-np.array(zip(*es_binomial_values)[0]))*100.0,zip(*es_binomial_values)[1],'b', 
               label='binomial, time in ms: %s'% math.ceil((binomial_time-start_time)*1000) )
    pylab.plot((1.0-np.array(es_infinite_obligors_values))*100.0, y_list,  'g',
               label='infinite obligors, time in ms: %s' %math.ceil((homogeneous_time-binomial_time)*1000))
    pylab.plot((1.0-np.array(es_first_adjusted))*100.0, y_list, 'r',
               label='first order adjustment %s' %math.ceil((first_order_time - homogeneous_time)*1000))
    pylab.xlim([70, 100])
    pylab.ylim([0.8, 1.0])
    pylab.legend(loc=(0,0))
    pylab.show()

if __name__ == "__main__":
    num_of_credits = 40
    default_probability = 0.01
    correlation = 0.2 
    loss_given_default = 1.0
    x_list = pylab.arange(0.0, 0.30, 0.01)
    y_list = pylab.arange(0.900, 0.99, 0.01)
    if len(sys.argv) !=2:
        print "Usage portfolio.py [es|var]"
    else:
        if sys.argv[1] == "var": 
            main_var(num_of_credits, default_probability, correlation, loss_given_default, x_list)
        elif sys.argv[1] == "es": 
            main_es(num_of_credits, default_probability, correlation, loss_given_default, x_list, y_list)
        else:
            print "wrong argument " + sys.argv[1]
