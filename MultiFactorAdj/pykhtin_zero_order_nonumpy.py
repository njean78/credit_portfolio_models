import math
from scipy import stats
import pylab
import numpy as np
import sys, time
# cholesky decomposition
# inter_corr = np.matrix(in_corr)
# num_sectors =  inter_corr.shape[0]
# np.linalg.cholesky(inter_corr).T

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

class Memoize(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    def __init__(self, func):
       self.func = func
       self.cache = {}
    def __call__(self, *args):
       try:
          return self.cache[args[0]]
       except KeyError:
          value = self.func(*args)
          self.cache[args[0]] = value
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



@memoized
def mem_c_value(key, pd, lgd, weight, corr, alpha):
    norm_dist = stats.norm(0,1)
    factor = norm_dist.ppf(pd) + corr*norm_dist.ppf(alpha)
    result = weight*lgd*norm_dist.cdf(factor/math.sqrt(1.0-corr**2))
    return result

def c_value(pd, lgd, weight, corr, alpha):
    """c_i issuer dependent. Corr is equivalent to r_i in pykhtin original paper"""
    return mem_c_value([pd, weight, corr], pd, lgd, weight, corr, alpha)

@memoized
def b_value(key, sector, extra_corr_matrix, pd_list, lgd_list, weight_list,
            intra_corr_list, alpha):
    lambda_value = lagr_mult(extra_corr_matrix, pd_list, lgd_list, weight_list,
                             intra_corr_list, alpha)
    result = 0
    issuer_args = zip(pd_list, lgd_list, weight_list, intra_corr_list)
    for issuer in range(len(pd_list)):
        # I love pattern matching!
        (pd, lgd, weight, corr) = issuer_args[issuer]
        result += c_value(pd, lgd, weight, corr, alpha)*extra_corr_matrix[sector][issuer]
    return  result/lambda_value

def lagr_mult(extra_corr_matrix, pd_list, lgd_list, weight_list,
              intra_corr_matrix, alpha):
    """ function to calculate the normalization factor called lambda in the orginal paper """
    result = 0
    sector_indices = range(len(extra_corr_matrix))
    for sector in sector_indices:
        c_sum = c_times_alpha(extra_corr_matrix[sector], pd_list, lgd_list,
                              weight_list, intra_corr_matrix, alpha)
        result += c_sum**2
    return math.sqrt(result)

def c_times_alpha(extra_corr_row, pd_list, lgd_list, weight_list,
                  intra_corr_list, alpha):
    value = 0
    issuer_args = zip(pd_list, lgd_list, weight_list, intra_corr_list)
    issuer_indeces = range(len(pd_list))
    for issuer in issuer_indeces:
        (pd, lgd, weight, corr) = issuer_args[issuer]
        value += extra_corr_row[issuer]*c_value(pd, lgd, weight, corr, alpha)
    return value

def multifactor_corr(issuer, extra_corr_matrix, pd_list, lgd_list,
                     weight_list, intra_corr_list, alpha):
    result = 0
    sector_indeces = range(len(extra_corr_matrix))
    for sector in sector_indeces:
        b_val = b_value([sector], sector, extra_corr_matrix, pd_list, lgd_list,
                        weight_list, intra_corr_list, alpha)
        result += extra_corr_matrix[sector][issuer]*b_val
    return result

def loss(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
         alpha):
    norm_dist = stats.norm(0,1)
    result = 0
    for issuer in range(len(pd_list)):
        a = intra_corr_list[issuer]*multifactor_corr(issuer, extra_corr_matrix,
                                                     pd_list, lgd_list, weight_list,
                                                     intra_corr_list, alpha)
        factor = norm_dist.ppf(pd_list[issuer])- a*norm_dist.ppf(1-alpha)
        factor/=math.sqrt(1-(a**2))
        result += weight_list[issuer]*lgd_list[issuer]*norm_dist.cdf(factor)
    return result

def expected_shortfall(extra_corr_matrix, pd_list, lgd_list, weight_list,
                       intra_corr_list, alpha):
    norm_dist = stats.norm(0,1)
    lower  = np.array([-100.0, -100.0])
    means = np.array([0.0, 0.0])
    result = 0
    for issuer in range(len(pd_list)):
        a = intra_corr_list[issuer]*multifactor_corr(
            issuer, extra_corr_matrix, pd_list, lgd_list, weight_list,
            intra_corr_list, alpha)
        higher = np.array([norm_dist.ppf(pd_list[issuer]),-norm_dist.ppf(alpha)])
        covariance = np.matrix([[1.0,a], [a ,1.0]])
        result += weight_list[issuer]*lgd_list[issuer]*stats.mvn.mvnun(lower, higher, means, covariance)[0]
    return result/(1.0-alpha)

def correl_function(rho):
    k = (1+math.sqrt(1-rho**2))/2
    return [[math.sqrt(1.0-k), math.sqrt(k)] , [  math.sqrt(k), math.sqrt(1.0-k)]]

def main_loss(pd, lgd, vol_lgd, r_intra, alpha, x_list):
    weights = [0.5, 0.5]
    weights2 = [0.3, 0.7]
    weights3 = [0.2, 0.8]
    a = time.time()
    loss_values = [loss(correl_function(rho), pd, lgd, weights, r_intra, alpha)
                   for rho in x_list]
    b = time.time()
    print 'equally weighted ', b-a
    loss_values2 = [loss(correl_function(rho), pd, lgd, weights2, r_intra, alpha)
                   for rho in x_list]
    c = time.time()
    print '0.3 weighted ', c-b
    loss_values3 = [loss(correl_function(rho), pd, lgd, weights3, r_intra, alpha)
                    for rho in x_list]
    d = time.time()
    print '0.2 weighted ', d-c
    pylab.xlabel("correlation")
    pylab.ylabel("loss function values")
    pylab.plot( x_list,loss_values, 'b')
    pylab.plot( x_list,loss_values2, 'g')
    pylab.plot( x_list,loss_values3, 'r')
    pylab.show()

def main_es(pd, lgd, vol_lgd, r_intra, alpha, x_list):
    weights = [0.5, 0.5]
    weights2 = [0.3, 0.7]
    weights3 = [0.2, 0.8]
    a = time.time()
    es_values = [expected_shortfall(correl_function(rho), pd, lgd, weights, r_intra, alpha)
                   for rho in x_list]
    b = time.time()
    print 'equally weighted ', b-a
    es_values2 = [expected_shortfall(correl_function(rho), pd, lgd, weights2, r_intra, alpha)
                   for rho in x_list]
    c = time.time()
    print '0.3 weighted ', c-b
    es_values3 = [expected_shortfall(correl_function(rho), pd, lgd, weights3, r_intra, alpha)
                    for rho in x_list]
    d = time.time()
    print '0.2 weighted ', d-c
    pylab.xlabel("correlation")
    pylab.ylabel("es function values")
    pylab.plot( x_list,es_values, 'b')
    pylab.plot( x_list,es_values2, 'g')
    pylab.plot( x_list,es_values3, 'r')
    pylab.show()

if __name__ == "__main__":
    pd = [ 0.005, 0.005 ]
    lgd = [ 0.4, 0.4]
    vol_lgd = [0.2, 0.2]
    r_intra = [0.5, 0.5]
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
