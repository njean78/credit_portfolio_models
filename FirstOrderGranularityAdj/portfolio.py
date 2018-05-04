# general formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i

from HomogeneousPortfolio.portfolio import expected_shortfall as es_hp
from HomogeneousPortfolio.portfolio import loss_funtion as var_hp
from FirstOrderGranularityAdj.binomial_approximation import loss_function as var_binomial
from FirstOrderGranularityAdj.binomial_approximation import expected_shortfall as es_binomial
from FirstOrderGranularityAdj.first_order import var_adjustment, es_adjustment
import pylab
import sys

def main_var(num_of_credits, default_probability, correlation, loss_given_default, x_list):
    # binomial portfolio
    loss_binomial = [var_binomial(i, num_of_credits, default_probability, 
                              correlation, loss_given_default) 
                     for i in x_list]
    # homogeneous portfolio
    loss_hp = [var_hp(i, default_probability, correlation, loss_given_default) 
               for i in x_list]
    # first order correction
    loss_first_order_correction = [
         var_adjustment(i, default_probability, correlation, 
                        loss_given_default, num_of_credits) 
        for i in loss_hp]

    loss_first_adjusted = [
        (i+j)
        for (i,j) in zip(x_list, loss_first_order_correction)]
    return x_list, loss_binomial, loss_hp, loss_first_adjusted
    #pylab.plot(x_list, loss_binomial, 'b')
    #pylab.plot(x_list, loss_hp, 'g')
    #pylab.plot(loss_first_adjusted, loss_hp, 'r')#'c'
    #pylab.show()

def main_es(num_of_credits, default_probability, correlation, loss_given_default, x_list, y_list):
    # binomial portfolio
    loss_binomial = [var_binomial(i, num_of_credits, default_probability, 
                              correlation, loss_given_default) 
                     for i in x_list]
    es_binomial_values = [es_binomial(i, zip(x_list, loss_binomial), num_of_credits, default_probability, 
                                      correlation, loss_given_default) 
                     for i in y_list]
    # homogeneous portfolio
    es_hp_values = [es_hp(i, default_probability, correlation, loss_given_default) 
                    for i in y_list]
    # first order correction
    es_first_order_correction = [
        es_adjustment(i, default_probability, correlation, 
                       loss_given_default, num_of_credits) 
        for i in y_list]

    es_first_adjusted = [
         (i+j)
         for (i,j) in zip(es_hp_values, es_first_order_correction )]
    return es_binomial_values, es_hp_values, es_first_adjusted, y_list
    #pylab.plot(zip(*es_binomial_values)[0], zip(*es_binomial_values)[1], 'b')
    #pylab.plot(es_hp_values, y_list, 'g')
    #pylab.plot(es_first_adjusted, y_list, 'r')#'c'
    #pylab.show()

if __name__ == "__main__":
    num_of_credits = 40
    default_probability = 0.01
    correlation = 0.2 
    loss_given_default = 1.0
    x_list = pylab.arange(0.0, 0.30, 0.002)
    y_list = pylab.arange(0.900, 0.999, 0.001)
    if len(sys.argv) !=2:
        print "Usage portfolio.py [es|var]"
    else:
        if sys.argv[1] == "var": 
            main_var(num_of_credits, default_probability, correlation, loss_given_default, x_list)
        elif sys.argv[1] == "es": 
            main_es(num_of_credits, default_probability, correlation, loss_given_default, x_list, y_list)
        else:
            print "wrong argument " + sys.argv[1]
