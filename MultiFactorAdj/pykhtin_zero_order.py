import math
from scipy import stats
import pylab
import numpy as np
import sys, time

class memoized(object):
    def __init__(self, func):
       self.func = func
       self.cache = []
    def __call__(self, *args):
       try:
           if self.cache and all([abs(cache_value-arg_value) < 0.00000001
                                  for (cache_value,arg_value) in zip(self.cache, args[0])]):
               return self.cache[-1]
           else:
               value = self.func(*args)
               self.cache = args[0]+ [value]
               return value
       except TypeError:
          # uncachable -- for instance, passing a list as an argument.
          # Better to not cache than to blow up entirely.
          return self.func(*args)
    def __repr__(self):
       """Return the function's docstring."""
       return self.func.__doc__
    def __get__(self, obj, objtype):
       """Support instance methods."""
       return functools.partial(self.__call__, obj)

# corr is normally a function of pd
@memoized
def mem_c_value(key, pd, corr, alpha):
    norm_dist = stats.norm(0,1)
    factor = norm_dist.ppf(pd) + corr*norm_dist.ppf(alpha)
    result = norm_dist.cdf(factor/math.sqrt(1.0-corr**2))
    return result

def c_value(pd, corr, alpha):
    """c_i issuer dependent. Corr is equivalent to r_i in pykhtin original paper"""
    return mem_c_value([pd, corr], pd, corr, alpha)

@memoized
def b_value(key, sector, extra_corr_matrix, pd_list, lgd_list, weight_list,
            intra_corr_list, alpha):
    result = 0
    issuer_args = zip(pd_list, lgd_list, weight_list, intra_corr_list)
    c_values = np.array([weight*lgd*c_value(pd, corr, alpha) for (pd, lgd, weight, corr) in issuer_args])
    return np.dot(c_values, np.array(extra_corr_matrix[sector]))

def lagr_mult(extra_corr_matrix, pd_list, lgd_list, weight_list,
              intra_corr_array, alpha):
    """ function to calculate the normalization factor called lambda in the orginal paper """
    sector_indices = range(len(extra_corr_matrix))
    issuer_args = zip(pd_list, lgd_list, weight_list, intra_corr_array)
    c_array = np.array([weight*lgd*c_value(pd, corr, alpha) for (pd, lgd, weight, corr) in issuer_args])
    c_elements = np.dot(extra_corr_matrix,c_array)
    return math.sqrt(sum([c_el**2 for c_el in c_elements]))

def multifactor_corr_array(extra_corr_matrix, pd_list, lgd_list,
                           weight_list, intra_corr_list, alpha):
    lambda_value = lagr_mult(extra_corr_matrix, pd_list, lgd_list, weight_list,
                             intra_corr_list, alpha)
    sector_indeces = range(len(extra_corr_matrix))
    b_value_array = np.array([ b_value([sector], sector, extra_corr_matrix, pd_list, lgd_list,
                                       weight_list, intra_corr_list, alpha)/lambda_value for sector in sector_indeces])


    return np.dot(extra_corr_matrix.T, b_value_array)

def loss(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
         alpha):
    norm_dist = stats.norm(0,1)
    result = 0
    m_corr = multifactor_corr_array(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
    a_array = intra_corr_list*m_corr
    factors = np.array([(norm_dist.ppf(pd)- a*norm_dist.ppf(1-alpha))/math.sqrt(1-(a**2)) for (pd, a) in zip(pd_list, a_array)])

    return sum(weight_list*lgd_list*norm_dist.cdf(factors))

def expected_shortfall(extra_corr_matrix, pd_list, lgd_list, weight_list,
                       intra_corr_list, alpha):
    norm_dist = stats.norm(0,1)

    result = 0
    m_corr = multifactor_corr_array(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
    a_array = intra_corr_list*m_corr
    num_issuers = len(pd_list)
    lower  = np.array([-100.0, -100.0] )
    means = np.array([0.0, 0.0])
    higher1 = norm_dist.ppf(pd_list)
    higher2_el = -norm_dist.ppf(alpha)
    cum_array = np.array([stats.mvn.mvnun(lower, np.array([higher1_el, higher2_el]),
                                          means, np.array([[1.0,a], [a ,1.0]]))[0]
                          for ( higher1_el, a) in zip( higher1, a_array)])
    return sum(weight_list*lgd_list*cum_array)/(1.0-alpha)

def correl_function(rho):
    k = (1+math.sqrt(1-rho**2))/2
    return np.array([[math.sqrt(1.0-k), math.sqrt(k)] , [  math.sqrt(k), math.sqrt(1.0-k)]])

def main_loss(pd, lgd, vol_lgd, r_intra, alpha, x_list):
    weights = np.array([0.5, 0.5])
    weights2 = np.array([0.3, 0.7])
    weights3 = np.array([0.2, 0.8])
    a = time.time()
    loss_values = [loss(correl_function(rho), pd, lgd, weights, r_intra, alpha)
                   for rho in x_list]
    b = time.time()
    print 'equally weighted time:', b-a
    loss_values2 = [loss(correl_function(rho), pd, lgd, weights2, r_intra, alpha)
                   for rho in x_list]
    c = time.time()
    print '0.3 weighted time:', c-b
    loss_values3 = [loss(correl_function(rho), pd, lgd, weights3, r_intra, alpha)
                    for rho in x_list]
    d = time.time()
    print '0.2 weighted time: ', d-c
    pylab.xlabel("correlation")
    pylab.ylabel("loss function values")
    pylab.plot( x_list,loss_values, 'b',  label=' 0.5-0.5 weight')
    pylab.plot( x_list,loss_values2, 'g', label=' 0.3-0.7 weight')
    pylab.plot( x_list,loss_values3, 'r', label=' 0.2-0.8 weight')
    pylab.suptitle('Two factors model', fontsize=12)
    pylab.legend( loc=0, ncol=2, borderaxespad=0.)
    pylab.show()

def main_es(pd, lgd, vol_lgd, r_intra, alpha, x_list):
    weights = np.array([0.5, 0.5])
    weights2 = np.array([0.3, 0.7])
    weights3 = np.array([0.2, 0.8])
    a = time.time()
    es_values = [expected_shortfall(correl_function(rho), pd, lgd, weights, r_intra, alpha)
                   for rho in x_list]
    b = time.time()
    print 'equally weighted time:', b-a
    es_values2 = [expected_shortfall(correl_function(rho), pd, lgd, weights2, r_intra, alpha)
                   for rho in x_list]
    c = time.time()
    print '0.3 weighted time:', c-b
    es_values3 = [expected_shortfall(correl_function(rho), pd, lgd, weights3, r_intra, alpha)
                    for rho in x_list]
    d = time.time()
    print '0.2 weighted time:', d-c
    pylab.xlabel("correlation")
    pylab.ylabel("es function values")
    pylab.plot( x_list,es_values, 'b',  label=' 0.5-0.5 weight')
    pylab.plot( x_list,es_values2, 'g', label=' 0.3-0.7 weight')
    pylab.plot( x_list,es_values3, 'r', label=' 0.2-0.8 weight')
    pylab.suptitle('Two factors model', fontsize=12)
    pylab.legend( loc=0, ncol=2, borderaxespad=0.)
    pylab.show()

if __name__ == "__main__":
    pd = np.array([ 0.005, 0.005 ])
    lgd = np.array([ 0.4, 0.4])
    vol_lgd = np.array([0.2, 0.2])
    r_intra = np.array([0.5, 0.5])
    x_list = pylab.arange(0.001, 1.0, 0.05)
    alpha =0.999
    if len(sys.argv) !=2:
        print "Usage pykhtin_zero_order.py [es|loss]"
    else:
        if sys.argv[1] == "loss":
            main_loss(pd, lgd, vol_lgd, r_intra, alpha, x_list)
        elif sys.argv[1] == "es":
            main_es(pd, lgd, vol_lgd, r_intra, alpha, x_list)
        else:
            print "wrong argument " + sys.argv[1]
