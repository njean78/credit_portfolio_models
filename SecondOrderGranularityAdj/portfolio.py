# general formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i


from HomogeneousPortfolio.portfolio import expected_shortfall as es_hp
from HomogeneousPortfolio.portfolio import loss_funtion as var_hp
from FirstOrderGranularityAdj.binomial_approximation import loss_function as var_binomial
from FirstOrderGranularityAdj.binomial_approximation import expected_shortfall as es_binomial
from FirstOrderGranularityAdj.first_order import var_adjustment, es_adjustment
from SecondOrderGranularityAdj.second_order import var_adjustment as var_adjustment_second
from SecondOrderGranularityAdj.second_order import es_adjustment as es_adjustment_second

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

    # second order correction
    loss_second_order_correction = [
        var_adjustment_second(i, default_probability, correlation,
                              num_of_credits) 
        for i in loss_hp]
    loss_second_adjusted = [
        (i+j)
        for (i,j) in zip(loss_first_adjusted, loss_second_order_correction)]
    data = [{'name': 'binomial', 'data' : zip(x_list, loss_binomial)},
            {'name': 'asrf'    , 'data' : zip(x_list, loss_hp)},
            {'name': 'first order adj' ,
             'data' : zip(loss_first_adjusted, loss_hp)},
            {'name': 'second order adj' ,
             'data' : zip(loss_second_adjusted, loss_hp)},]
    return data

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
    # second order correction
    es_second_order_correction = [
        es_adjustment_second(i, default_probability, correlation,
                             num_of_credits) 
        for i in y_list]
    es_second_adjusted = [
        (i+j)
        for (i,j) in zip(es_first_adjusted, es_second_order_correction)]

    data = [{'name': 'binomial', 'data' : es_binomial_values},
            {'name': 'asrf'    , 'data' : zip(es_hp_values, y_list)},
            {'name': 'first order adj' ,
             'data' : zip(es_first_adjusted, y_list)},
            {'name': 'second order adj' ,
             'data' : zip(es_second_adjusted, y_list)},]
    return data
            
