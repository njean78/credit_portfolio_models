import math
from scipy import stats
import pylab
import numpy as np
from Models.HomogeneousPortfolio.portfolio import loss_var
from Models.MultiFactorAdj.pykhtin_zero_order_old import (multifactor_corr,
                                                      correl_function,
                                                      expected_shortfall,
                                                      memoized, Memoize)
import sys, time
_cached_c_i = {}
def mem_c_i(i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha):
    global _cached_c_i
    pd_index = int(10000*pd_list[i])
    lgd_index = int(10000*lgd_list[i])
    c_index = str(pd_index)+'-'+str(lgd_index)
    if c_index in _cached_c_i.keys():
        return _cached_c_i[c_index]
    else:
        result = intra_corr_list[i]*multifactor_corr(i, extra_corr_matrix,
                                                     pd_list, lgd_list, weight_list,
                                                     intra_corr_list, alpha)
        _cached_c_i[c_index] = result
        return result

_cached_deta_cumul = {}
def deta_cumul(i,j, pd_list, lgd_list, weight_list, alpha, rho, c_corr_list):
    global _cached_deta_cumul
    pd_i = pd_list[i]
    pd_j = pd_list[j]
    lgd_index_i = int(10000*lgd_list[i])
    lgd_index_j = int(10000*lgd_list[j])
    pd_index = "-".join([str(int(10000*pd_i)),str(int(10000*pd_j)),
                         str(int(10000*lgd_index_i)), str(int(10000*lgd_index_j))]
                        )
    if pd_index in _cached_deta_cumul.keys():
        return _cached_deta_cumul[pd_index]
    else:
        norm_dist = stats.norm(0,1)
        c_i = c_corr_list[i]
        c_j = c_corr_list[j]
        rho_value = rho[i][j]
        p_i = _p(pd_i, c_i, alpha)
        p_j = _p(pd_j, c_j, alpha)
        result = norm_dist.ppf(p_j) - rho_value*norm_dist.ppf(p_i)
        result = result / math.sqrt(1.0 - rho_value**2)
        result = norm_dist.cdf(result)
        _cached_deta_cumul[pd_index] = result
        return result

_cached_looped_deta_cumul = {}
def looped_deta_cumul(i, pd_list, lgd_list, weight_list, alpha, rho, c_corr_list):
    global _cached_looped_deta_cumul
    pd_index = int(10000*pd_list[i])
    lgd_index = int(10000*lgd_list[i])
    c_index = str(pd_index)+'-'+str(lgd_index)
    if c_index in _cached_looped_deta_cumul.keys():
        return _cached_looped_deta_cumul[c_index]
    else:
        issuer_indices = range(len(pd_list))
        result = 0
        for j in issuer_indices:
            deta_cumul_value = deta_cumul(i,j, pd_list, lgd_list, weight_list,
                                          alpha, rho, c_corr_list)
            c_j = c_corr_list[j]
            result+= weight_list[j]*lgd_list[j]*(deta_cumul_value - _p(pd_list[j], c_j, alpha))
        _cached_looped_deta_cumul[c_index] = result
        return result

_cached_eta_inf = {}
def mem_eta_inf(i, pd_list, lgd_list, weight_list, alpha, rho, c_corr_list):
    global _cached_eta_inf
    pd_index = int(10000*pd_list[i])
    lgd_index = int(10000*lgd_list[i])
    c_index = str(pd_index)+'-'+str(lgd_index)
    if c_index in _cached_eta_inf.keys():
        return _cached_eta_inf[c_index]
    else:
        p_i = _p(pd_list[i], c_corr_list[i], alpha)
        issuer_indices = range(len(pd_list))
        result = 0
        for j in issuer_indices:
            cum_factor = cumulative_factor(i,j, pd_list, alpha, rho, c_corr_list)
            p_j = _p(pd_list[j], c_corr_list[j], alpha)
            result+=weight_list[j]*lgd_list[j]*(cum_factor - p_i*p_j)
        _cached_eta_inf[c_index] = result
        return result

_cached_mem_loss = {}
def mem_loss(i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
             alpha):
    global _cached_mem_loss
    pd_index = int(10000*pd_list[i])
    lgd_index = int(10000*lgd_list[i])
    c_index = str(pd_index)+'-'+str(lgd_index)
    if c_index in _cached_mem_loss.keys():
        return _cached_mem_loss[c_index]
    else:
        norm_dist = stats.norm(0,1)
        a = intra_corr_list[i]*multifactor_corr(i, extra_corr_matrix,
                                                pd_list, lgd_list, weight_list,
                                                intra_corr_list, alpha)
        factor = norm_dist.ppf(pd_list[i])- a*norm_dist.ppf(1-alpha)
        factor/=math.sqrt(1-(a**2))
        result = weight_list[i]*lgd_list[i]*norm_dist.cdf(factor)
        _cached_mem_loss[c_index] = result
        return result

#redefining loss in order to cache part of it
def loss(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
         alpha):
    result = sum([ mem_loss(i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                           alpha)
                  for i in range(len(pd_list))])

    return result

# correlation dependent on the single systematic factor
def rho_function(i,j, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha):
    c_i = mem_c_i(i, extra_corr_matrix, pd_list, lgd_list, weight_list,
                  intra_corr_list, alpha)
    c_j = mem_c_i(j, extra_corr_matrix, pd_list, lgd_list, weight_list,
                  intra_corr_list, alpha)
    a = sum([extra_corr_matrix[k][i]*extra_corr_matrix[k][j]
             for k in range(len(extra_corr_matrix))])
    result = intra_corr_list[i]*intra_corr_list[j]*a - c_i*c_j
    result = result/math.sqrt((1-c_i**2)*(1-c_j**2))

    return result


# probability derivatives
def _copula_factor(pd, c_corr, alpha):
    norm_dist = stats.norm(0,1)
    factor = norm_dist.ppf(pd) - c_corr*norm_dist.ppf(1-alpha)
    factor = factor/math.sqrt(1.0-c_corr**2)
    return factor

def _p(pd, c_corr, alpha):
    norm_dist = stats.norm(0,1)
    return norm_dist.cdf(_copula_factor(pd, c_corr, alpha))

def _dp(pd, c_corr, alpha):
    norm_dist = stats.norm(0,1)
    return (-c_corr/math.sqrt(1.0-c_corr**2))*norm_dist.pdf(_copula_factor(pd, c_corr, alpha))

def _d2p(pd, c_corr, alpha):
    norm_dist = stats.norm(0,1)
    a = c_corr/(1.0-c_corr**2)
    b = norm_dist.ppf(pd) - c_corr*norm_dist.ppf(1-alpha)
    return a*b*_dp(pd, c_corr, alpha)

# moments of the distribution
def mu1(weight_list, lgd_list, pd_list, c_corr_list, alpha):
    return sum([weight*lgd*_p(pd, corr, alpha)
                for (weight, lgd, pd, corr) in zip(weight_list, lgd_list, pd_list, c_corr_list)])

def dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha):
    return sum([weight*lgd*_dp(pd, corr, alpha)
                for (weight, lgd, pd, corr) in zip(weight_list, lgd_list, pd_list, c_corr_list)])

def d2mu1(weight_list, lgd_list, pd_list, c_corr_list, alpha):
    return sum([weight*lgd*_d2p(pd, corr, alpha)
                for (weight, lgd, pd, corr) in zip(weight_list, lgd_list, pd_list, c_corr_list)])

#eta functions

@memoized
def mem_cumulative_factor(key, i,j, c_i , c_j, p_i, p_j, rho):
    norm_dist = stats.norm(0,1)
    lower  = np.array([-100.0, -100.0])
    higher = np.array([norm_dist.ppf(p_i), norm_dist.ppf(p_j)])
    means = np.array([0.0, 0.0])
    covariance = np.matrix([[1.0,rho[i][j]], [rho[i][j] ,1.0]])
    return stats.mvn.mvnun(lower, higher, means, covariance)[0]

def cumulative_factor(i,j, pd_list, alpha, rho, c_corr_list):
    norm_dist = stats.norm(0,1)
    corr = rho[i][j]
    c_i = c_corr_list[i]
    c_j = c_corr_list[j]
    p_i = _p(pd_list[i], c_i , alpha)
    p_j = _p(pd_list[j], c_j , alpha)
    key = [rho[i][j],rho[j][i],  p_i, p_j]
    return mem_cumulative_factor(key, i,j, c_i , c_j, p_i, p_j, rho)

def eta_inf(pd_list, lgd_list, weight_list, alpha, rho, c_corr_list):
    issuer_indices = range(len(pd_list))
    result = 0
    for i in issuer_indices:
        mem_value = mem_eta_inf(i, pd_list, lgd_list, weight_list, alpha, rho, c_corr_list)
        result+=weight_list[i]*lgd_list[i]*mem_value
    return result

def eta_ga(pd_list, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list):
    issuer_indices = range(len(pd_list))
    result = 0
    for i in issuer_indices:
        p_i = _p(pd_list[i], c_corr_list[i], alpha)
        cum_factor = cumulative_factor(i,i, pd_list, alpha, rho, c_corr_list)
        factor1 = (lgd_list[i]**2)*(p_i - cum_factor)
        factor2 = (pd_vol_list[i]**2)*p_i
        result+= (weight_list[i]**2)*(factor1+factor2)
    return result

# eta derivative functions

def deta_inf(pd_list, lgd_list, weight_list, alpha, rho, c_corr_list):
    issuer_indices = range(len(pd_list))
    result = 0
    for i in issuer_indices:
        c_i = c_corr_list[i]
        cumul_value = looped_deta_cumul(i, pd_list, lgd_list, weight_list, alpha, rho, c_corr_list)
        result += cumul_value*2*weight_list[i]*lgd_list[i]*_dp(pd_list[i], c_i, alpha)
    return result

def deta_ga(pd_list, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list):
    issuer_indices = range(len(pd_list))
    result = 0
    for i in issuer_indices:
        deta_cumul_value = deta_cumul(i,i, pd_list, lgd_list, weight_list,
                                      alpha, rho, c_corr_list)
        c_i = c_corr_list[i]
        result+= (weight_list[i]**2)*_dp(pd_list[i], c_i, alpha)*(
            (lgd_list[i]**2)*(1.0 - 2* deta_cumul_value)+ pd_vol_list[i]**2)
    return result
# loss : inf and ga corrections
def delta_qa_inf(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha, pd_vol_list):

    c_corr_list = [
        mem_c_i(i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
        for i in range(len(pd_list))]
    #c_corr_list = [intra_corr_list[i]*multifactor_corr(i, extra_corr_matrix,
    #                                                   pd_list, lgd_list, weight_list,
    #                                                   intra_corr_list, alpha)
    #               for i in range(len(pd_list))]
    rho = []
    a = time.time()
    for i in range(len(pd_list)):
        rho.append([rho_function(i,j, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                                 alpha) for j in range(len(pd_list))])
    norm_dist = stats.norm(0,1)
    dm1_value = dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    deta_value = deta_inf(pd_list, lgd_list, weight_list, alpha, rho, c_corr_list)
    eta_value =  eta_inf(pd_list, lgd_list, weight_list, alpha, rho, c_corr_list)
    dm2_value = d2mu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    x_value = norm_dist.ppf(1-alpha)
    result = -(deta_value-eta_value*(dm2_value/dm1_value+x_value))/(2.0*dm1_value)
    b = time.time()
    return result

def delta_qa_ga(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha, pd_vol_list, ):
    a = time.time()
    c_corr_list = [
        mem_c_i(i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
        for i in range(len(pd_list))]
    b = time.time()
    #print ' c_corr_list ' , b-a
    rho = []
    issuer_num = len(pd_list)
    for i in range(issuer_num):
        tmp_row = [0,]*i + [
            rho_function(i,i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                         alpha) ] + [0,]*(issuer_num-i-1)
        rho.append(tmp_row)
        #rho.append([rho_function(i,j, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
        #                         alpha) for j in range(len(pd_list))])
    c = time.time()
    #print ' rho ' , c-b
    norm_dist = stats.norm(0,1)
    dm1_value = dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    d = time.time()
    #print ' dm1 ' , d-c
    dm2_value = d2mu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    ee = time.time()
    #print ' dm2 ' , ee- d
    eta_value =  eta_ga(pd_list, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list)
    ff = time.time()
    #print ' eta ' ,ff- ee
    deta_value = deta_ga(pd_list, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list)
    g = time.time()
    #print ' deta ' ,g-ff
    x_value = norm_dist.ppf(1-alpha)
    result = -(deta_value-eta_value*(dm2_value/dm1_value+x_value))/(2.0*dm1_value)
    h = time.time()
    #print ' result ' ,h-g
    return result

# expected shortfall : inf and ga corrections
def delta_es_inf(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha, pd_vol_list):
    c_corr_list = [
        mem_c_i(i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
        for i in range(len(pd_list))]
    rho = []
    for i in range(len(pd_list)):
        rho.append([rho_function(i,j, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                                 alpha) for j in range(len(pd_list))])

    norm_dist = stats.norm(0,1)
    dm1_value = dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    eta_value =  eta_inf(pd_list, lgd_list, weight_list, alpha, rho, c_corr_list)
    factor = - norm_dist.pdf(norm_dist.ppf(1-alpha))/(2.0*(1-alpha))
    return factor * eta_value/dm1_value

def delta_es_ga(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha, pd_vol_list):
    c_corr_list = [
        mem_c_i(i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
        for i in range(len(pd_list))]


    issuer_num = len(pd_list)
    rho = []
    for i in range(issuer_num):
        tmp_row = [0,]*i + [
            rho_function(i,i, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                         alpha) ] + [0,]*(issuer_num-i-1)
        rho.append(tmp_row)
        #rho.append([rho_function(i,j, extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
        #                         alpha) for j in range(len(pd_list))])

    norm_dist = stats.norm(0,1)
    dm1_value = dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    eta_value =  eta_ga(pd_list, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list)
    factor = - norm_dist.pdf(norm_dist.ppf(1-alpha))/(2.0*(1-alpha))
    return factor * eta_value/dm1_value

def main_finite_loss(weight, plot_style, pd, lgd, vol_lgd, r_intra, alpha, x_list):
    M_list = [160, 40]
    weight_list = [weight[0]/M_list[0]]*M_list[0]+[weight[1]/M_list[1]]*M_list[1]
    pd_list = [pd[0]]*M_list[0] + [pd[1]]*M_list[1]
    lgd_list = [lgd[0]]*M_list[0] + [lgd[1]]*M_list[1]
    vol_lgd_list = [vol_lgd[0]]*M_list[0] + [vol_lgd[1]]*M_list[1]
    r_intra_list =  [r_intra[0]]*M_list[0] + [r_intra[1]]*M_list[1]
    weight_list = [weight[0]/M_list[0]]*M_list[0]+[weight[1]/M_list[1]]*M_list[1]
    loss_values = []
    inf_correction = []
    ga_correction = []
    for rho in x_list:
        a = time.time()
        loss_values += [loss([[correl_function(rho)[0][0]]*M_list[0]+[correl_function(rho)[0][1]]*M_list[1],
                              [correl_function(rho)[1][0]]*M_list[0]+[correl_function(rho)[1][1]]*M_list[1]],
                        pd_list, lgd_list, weight_list, r_intra_list, alpha)]
        b = time.time()
        print b-a
        inf_correction += [delta_qa_inf([[correl_function(rho)[0][0]]*M_list[0]+[correl_function(rho)[0][1]]*M_list[1],
                                     [correl_function(rho)[1][0]]*M_list[0]+[correl_function(rho)[1][1]]*M_list[1]],
                                   pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]
        c = time.time()
        print c-b

        ga_correction += [delta_qa_ga([[correl_function(rho)[0][0]]*M_list[0]+[correl_function(rho)[0][1]]*M_list[1],
                                       [correl_function(rho)[1][0]]*M_list[0]+[correl_function(rho)[1][1]]*M_list[1]],
                                      pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]
        d = time.time()
        print d-c
        global _cached_c_i
        _cached_c_i.clear()
        global _cached_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_looped_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_eta_inf
        _cached_eta_inf.clear()
        global _cached_mem_loss
        _cached_mem_loss.clear()

    loss_first_corr = [i+j+k for (i,j,k) in zip(inf_correction, loss_values, ga_correction)]
    pylab.plot( x_list,loss_values, plot_style+'--', label=' wa = '+ str(weight[0])+ ' base')
    pylab.plot( x_list, loss_first_corr, plot_style+'-', label=' wa = '+ str(weight[0]))

def correlation_matrix_multifactor(M_list, rho):
    issuer_num = sum(M_list)
    correl = []
    first_index = 0
    for bucket in M_list:
        ending_zeros = issuer_num - bucket - first_index
        issuer_correl = [0.0,]*first_index + [math.sqrt(1-rho)]*bucket + [0,]*ending_zeros
        first_index += bucket
        correl.append(issuer_correl)
    correl.append([math.sqrt(rho)]*issuer_num)
    return correl

def main_multifactor_es(alpha, x_list, m_list, plot_style):
    pd = [0.001, 0.002, 0.002, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.05]
    lgd = [0.5, 0.3, 0.5, 0.3,  0.5, 0.3, 0.5, 0.3, 0.5, 0.3]
    vol_lgd = [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1 ]
    r_intra = [0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2]
    weights = [0.1]*10
    m_list_inf = [1,]*10

    pd_list = []
    lgd_list = []
    r_intra_list = []
    weight_list = []
    vol_lgd_list = []
    for i in range(len(m_list)):
        pd_list.extend([pd[i],]*m_list[i])
        lgd_list.extend([lgd[i],]*m_list[i])
        vol_lgd_list.extend([vol_lgd[i],]*m_list[i])
        weight_list.extend([weights[i]/float((m_list[i])),]*m_list[i])
        r_intra_list.extend([r_intra[i],]*m_list[i])

    #M_lists =  [m_1, m_2, m_3]
    #weights1 = [[i/j,]*j for (i,j) in zip(exp_weights, m_1)]
    #weights2 = [[i/j,]*j for (i,j) in zip(exp_weights, m_2)]
    #weights3 = [[i/j,]*j for (i,j) in zip(exp_weights, m_3)]

    loss_values = []
    inf_correction = []
    global _cached_c_i
    global _cached_deta_cumul
    global _cached_looped_deta_cumul
    global _cached_eta_inf
    global _cached_mem_loss

    for rho in x_list:
        a = time.time()
        loss_values += [expected_shortfall(correlation_matrix_multifactor(m_list_inf, rho),
                                           pd, lgd, weights, r_intra, alpha)]
        inf_correction += [delta_es_inf(correlation_matrix_multifactor(m_list_inf, rho),
                                        pd, lgd, weights, r_intra, alpha, vol_lgd) ]
        _cached_c_i.clear()
        _cached_deta_cumul.clear()
        _cached_deta_cumul.clear()
        _cached_eta_inf.clear()
        _cached_mem_loss.clear()

    ga_correction = []
    for rho in x_list:
        ga_correction += [delta_es_ga(correlation_matrix_multifactor(m_list, rho),
                                      pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]
        _cached_c_i.clear()
        _cached_deta_cumul.clear()
        _cached_deta_cumul.clear()
        _cached_eta_inf.clear()
        _cached_mem_loss.clear()


    loss_no_ga = [i+j for (i,j) in zip(inf_correction, loss_values)]
    loss_ga = [i+j for (i,j) in zip(loss_no_ga,  ga_correction)]
    #loss_first_corr = [i+j for (i,j) in zip(inf_correction, loss_values)]
    print " value ", loss_ga
    pylab.plot( x_list, loss_no_ga, plot_style+'--', label=' wa = '+ str(m_list[0])+ '-' + str(m_list[1])+ ' base')
    pylab.plot( x_list, loss_ga, plot_style+'-', label=' wa = '+  str(m_list[0])+ '-' + str(m_list[1]))


def main_multifactor_loss(alpha, x_list, m_list, plot_style):
    pd = [0.001, 0.002, 0.002, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.05]
    lgd = [0.5, 0.3, 0.5, 0.3,  0.5, 0.3, 0.5, 0.3, 0.5, 0.3]
    vol_lgd = [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1 ]
    r_intra = [0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2]
    weights = [0.1]*10
    m_list_inf = [1,]*10

    pd_list = []
    lgd_list = []
    r_intra_list = []
    weight_list = []
    vol_lgd_list = []
    for i in range(len(m_list)):
        pd_list.extend([pd[i],]*m_list[i])
        lgd_list.extend([lgd[i],]*m_list[i])
        vol_lgd_list.extend([vol_lgd[i],]*m_list[i])
        weight_list.extend([weights[i]/float((m_list[i])),]*m_list[i])
        r_intra_list.extend([r_intra[i],]*m_list[i])

    #M_lists =  [m_1, m_2, m_3]
    #weights1 = [[i/j,]*j for (i,j) in zip(exp_weights, m_1)]
    #weights2 = [[i/j,]*j for (i,j) in zip(exp_weights, m_2)]
    #weights3 = [[i/j,]*j for (i,j) in zip(exp_weights, m_3)]

    loss_values = []
    inf_correction = []
    global _cached_c_i
    global _cached_deta_cumul
    global _cached_looped_deta_cumul
    global _cached_eta_inf
    global _cached_mem_loss

    for rho in x_list:
        a = time.time()
        loss_values += [loss(correlation_matrix_multifactor(m_list_inf, rho), pd, lgd, weights, r_intra, alpha)]
        b = time.time()
        print 'loss infinite ', (b-a), (b-a)/60.0
        inf_correction += [delta_qa_inf(correlation_matrix_multifactor(m_list_inf, rho),
                                        pd, lgd, weights, r_intra, alpha, vol_lgd) ]
        c = time.time()
        print 'first correction ', (c-b), (c-b)/60.0
        #loss_values += [loss(correlation_matrix_multifactor(m_list, rho), pd_list, lgd_list, weight_list, r_intra_list, alpha)]
        #inf_correction += [delta_qa_inf(correlation_matrix_multifactor(m_list, rho),
        #                                pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]
        _cached_c_i.clear()
        _cached_deta_cumul.clear()
        _cached_deta_cumul.clear()
        _cached_eta_inf.clear()
        _cached_mem_loss.clear()

    ga_correction = []
    for rho in x_list:
        d = time.time()
        ga_correction += [delta_qa_ga(correlation_matrix_multifactor(m_list, rho),
                                      pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]
        ee = time.time()
        print 'ga correction ' , (ee-d), (ee-d)/60.0
        _cached_c_i.clear()
        _cached_deta_cumul.clear()
        _cached_deta_cumul.clear()
        _cached_eta_inf.clear()
        _cached_mem_loss.clear()


    loss_no_ga = [i+j for (i,j) in zip(inf_correction, loss_values)]
    loss_ga = [i+j for (i,j) in zip(loss_no_ga,  ga_correction)]
    #loss_first_corr = [i+j for (i,j) in zip(inf_correction, loss_values)]
    print " value ", loss_ga
    pylab.plot( x_list, loss_no_ga, plot_style+'--', label=' wa = '+ str(m_list[0])+ '-' + str(m_list[1])+' base')
    pylab.plot( x_list, loss_ga, plot_style+'-', label=' wa = '+ str(m_list[0])+ '-' + str(m_list[1]))

# list of main functions
def main_loss(weight, plot_style, pd, lgd, vol_lgd, r_intra, alpha, x_list):
    loss_values = []
    inf_correction = []
    a = time.time()
    for rho in x_list :
        loss_values += [loss(correl_function(rho), pd, lgd, weight, r_intra, alpha)]
        inf_correction += [delta_qa_inf(correl_function(rho), pd, lgd, weight,
                                   r_intra, alpha, vol_lgd)]
        global _cached_c_i
        _cached_c_i.clear()
        global _cached_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_looped_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_eta_inf
        _cached_eta_inf.clear()
        global _cached_mem_loss
        _cached_mem_loss.clear()
    loss_first_corr = [i+j for (i,j) in zip(inf_correction, loss_values)]
    b = time.time()
    print b-a
    # plotting
    pylab.plot( x_list,loss_values, plot_style+'--', label=' wa = '+ str(weight[0])+ ' base')
    pylab.plot( x_list, loss_first_corr, plot_style+'-', label=' wa = '+ str(weight[0]))


def main_es(weight, plot_style, pd, lgd, vol_lgd, r_intra, alpha, x_list):
    loss_values = []
    inf_correction = []
    for rho in x_list:
        loss_values += [expected_shortfall(correl_function(rho), pd, lgd, weight, r_intra, alpha)]
        inf_correction += [delta_es_inf(correl_function(rho), pd, lgd, weight,
                                        r_intra, alpha, vol_lgd)]
        global _cached_c_i
        _cached_c_i.clear()
        global _cached_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_looped_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_eta_inf
        _cached_eta_inf.clear()
        global _cached_mem_loss
        _cached_mem_loss.clear()
    loss_first_corr = [i+j for (i,j) in zip(inf_correction, loss_values)]
    # plotting
    pylab.plot( x_list,loss_values, plot_style+'--', label=' wa = '+ str(weight[0])+ ' base')
    pylab.plot( x_list, loss_first_corr, plot_style+'-', label=' wa = '+ str(weight[0]))

def mult_correl(correlation, dimension):
    rho = []
    alpha = math.sqrt(correlation)
    for i in range(dimension-1):
        x = [alpha,] + [0.0 for j in range(dimension-1)]
        x[i+1] = math.sqrt(1-alpha**2)
        rho.append(x)
    return np.matrix(rho).T


if __name__ == "__main__":
    #pd = [ 0.005, 0.005 ]
    #lgd = [ 0.4, 0.4]
    #vol_lgd = [0.2, 0.2]
    #r_intra = [0.5, 0.5]
    weights = [[0.5, 0.5] , [0.3, 0.7], [0.2, 0.8], [0.7, 0.3]]
    plot_styles = ['b','c', 'g', 'm']
    pd = [ 0.001, 0.02 ]
    lgd = [ 0.4, 0.4]
    vol_lgd = [0.2, 0.2]
    r_intra = [0.5, 0.2]

    x_list = pylab.arange(0.1, 0.6, 0.1)#pylab.arange(0.05, 1.0, 0.05)##
    alpha =0.999
    if len(sys.argv) !=2:
        print "Usage pykhtin_zero_order.py [es|loss|finite_loss]"
        exit
    else:
        if sys.argv[1] == "loss":
            for (weight, plot_style) in zip(weights, plot_styles):
                main_loss(weight, plot_style, pd, lgd, vol_lgd, r_intra, alpha, x_list)
            pylab.ylabel("loss function values")
        elif sys.argv[1] == "es":
            for (weight, plot_style) in zip(weights, plot_styles):
                main_es(weight, plot_style, pd, lgd, vol_lgd, r_intra, alpha, x_list)
            pylab.ylabel("es function values")
        elif sys.argv[1] == "finite_loss":
            a = time.time()
            for (weight, plot_style) in zip(weights, plot_styles):
                main_finite_loss(weight, plot_style, pd, lgd, vol_lgd, r_intra, alpha, x_list)
            b = time.time()
            print 'tot time ', b-a
            pylab.ylabel("loss function values")
        elif sys.argv[1] == "multifactor_var":
            m_lists = [[50, 100 ]*5,
                       [10, 20 ]*5,
                       [10, 20 , 50, 50 , 100, 100, 200, 200, 500, 1000 ]]
            a = time.time()
            for (m_list, plot_style) in zip(m_lists, plot_styles):
                main_multifactor_loss(alpha, x_list, m_list, plot_style)
            b = time.time()
            print 'tot time ', b-a
            pylab.ylabel("loss multifactor function values")
        elif sys.argv[1] == "multifactor_es":
            m_lists = [[50, 100 ]*5,
                       [10, 20 ]*5,
                       [10, 20 , 50, 50 , 100, 100, 200, 200, 500, 1000 ]]
            a = time.time()
            for (m_list, plot_style) in zip(m_lists, plot_styles):
                main_multifactor_es(alpha, x_list, m_list, plot_style)
            b = time.time()
            print 'tot time ', b-a
            pylab.ylabel("es multifactor function values")
        else:
            print "wrong argument " + sys.argv[1]
            exit
        pylab.xlabel("correlation")
        pylab.legend( loc=0, ncol=2, borderaxespad=0.)
        pylab.show()
