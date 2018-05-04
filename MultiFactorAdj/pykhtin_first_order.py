import math
from scipy import stats, sparse
import pylab
import numpy as np
from HomogeneousPortfolio.portfolio import loss_var
from MultiFactorAdj.pykhtin_zero_order import (loss,
                                               multifactor_corr_array, 
                                               correl_function,
                                               expected_shortfall)
import sys, time

#returns an array of probabilities
def _p(pd_array, c_corr_array, alpha):
    norm_dist = stats.norm(0,1)
    alpha_cum = norm_dist.ppf(1-alpha) 
    num = norm_dist.ppf(pd_array) - c_corr_array*alpha_cum
    den = (1.0-c_corr_array**2)**0.5
    return norm_dist.cdf(num/den)

def _dp(pd_array, c_corr_array, alpha):
    norm_dist = stats.norm(0,1)
    alpha_cum = norm_dist.ppf(1-alpha) 
    num = norm_dist.ppf(pd_array) - c_corr_array*alpha_cum
    den = (1.0-c_corr_array**2)**0.5
    return (-c_corr_array/den)*norm_dist.pdf(num/den)

def _d2p(pd_array, c_corr_array, alpha):
    norm_dist = stats.norm(0,1)
    alpha_cum = norm_dist.ppf(1-alpha) 
    num = norm_dist.ppf(pd_array) - c_corr_array*alpha_cum
    den = (1.0-c_corr_array**2)
    return _dp(pd_array, c_corr_array, alpha)*num*c_corr_array/den

# moments of the distribution
def mu1(weight_list, lgd_list, pd_list, c_corr_list, alpha):
    return sum(weight_list*lgd_list*_p(pd_list, c_corr_list, alpha))

def dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha):
    return sum(weight_list*lgd_list*_dp(pd_list, c_corr_list, alpha))

def d2mu1(weight_list, lgd_list, pd_list, c_corr_list, alpha):
    return sum(weight_list*lgd_list*_d2p(pd_list, c_corr_list, alpha))

#eta functions

_cached_cum_factor = {}
def mem_cumulative_factor(key,pis, rho_els):
    global _cached_cum_factor
    if key in _cached_cum_factor.keys():
        return _cached_cum_factor[key]
    else:
        norm_dist = stats.norm(0,1)
        lower  = np.array([-100.0, -100.0])
        higher = np.array([norm_dist.ppf(pis[0]), norm_dist.ppf(pis[1])])
        means = np.array([0.0, 0.0])
        covariance = np.matrix([ [1.0,rho_els[0]], [rho_els[1] ,1.0]])
        result = stats.mvn.mvnun(lower, higher, means, covariance)[0]
        _cached_cum_factor[key] = result
        return result

def cumulative_factor(i,j, pd_list, pi_array, corr):
    norm_dist = stats.norm(0,1)
    rho_el1 = corr[i][j]
    rho_el2 = corr[j][i]
    p_i = pd_list[i]
    p_j = pd_list[j]
    key = "-".join([str(int(el*100000000000)) for el in [rho_el1, rho_el2, p_i, p_j]])
    return mem_cumulative_factor(key, [pi_array[i], pi_array[j]], [rho_el1, rho_el2])

_cached_eta_inf = {}
def mem_eta_inf(i, pd_list, pi_array, lgd_list, weight_list, alpha, rho):
    global _cached_eta_inf
    pd_index = int(10000*pd_list[i])
    lgd_index = int(10000*lgd_list[i])
    c_index = str(pd_index)+'-'+str(lgd_index)
    if c_index in _cached_eta_inf.keys():
        return _cached_eta_inf[c_index]
    else:
        issuer_indices = range(len(pd_list))
        corr = rho.tolist()
        cum_array = np.array([cumulative_factor(i,j, pd_list, pi_array, corr) 
                              for j in issuer_indices])

        result=sum(weight_list*lgd_list*(cum_array - pi_array*pi_array[i]))
        _cached_eta_inf[c_index] = result
        return result 

def eta_inf(pd_list, pi_array, lgd_list, weight_list, alpha, rho, c_corr_list):
    issuer_indices = range(len(pd_list))
    eta_inf_array = [mem_eta_inf(i, pd_list, pi_array, lgd_list, weight_list, alpha, rho)
                     for i in  issuer_indices]
    return sum(weight_list*lgd_list*eta_inf_array)

def get_key(rho_el, pd_el):
    return "-".join([str(int(el*100000000000)) for el in [rho_el, rho_el, pd_el, pd_el]])

def eta_ga(pd_list, pi_array, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list):
    cum_factor_array = [
        mem_cumulative_factor(get_key(rho_el, pd_el),
                              [pi_array_el, pi_array_el], [rho_el, rho_el]) 
        for (rho_el, pd_el, pi_array_el) in zip(rho, pd_list, pi_array)]

    factor = (lgd_list**2)*(pi_array - cum_factor_array)+(pd_vol_list**2)*pi_array
    return sum(factor*(weight_list**2))

# eta derivative functions


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
def looped_deta_cumul(i, pd_list, pi_array, lgd_list, weight_list, alpha, rho, c_corr_list):
    global _cached_looped_deta_cumul
    pd_index = int(10000*pd_list[i])
    lgd_index = int(10000*lgd_list[i])
    c_index = str(pd_index)+'-'+str(lgd_index)
    if c_index in _cached_looped_deta_cumul.keys():
        return _cached_looped_deta_cumul[c_index]
    else:
        norm_dist = stats.norm(0,1)
        issuer_indices = range(len(pd_list))
        rho_array = np.array(rho[i].tolist()[0])
        deta_cumul_value = norm_dist.cdf(
            norm_dist.ppf(pi_array) - rho_array*norm_dist.ppf(pi_array[i])/((1.0 - rho_array**2)**0.5))
        result = sum(weight_list*lgd_list*(deta_cumul_value - pi_array))
        # for j in issuer_indices:
        #     deta_cumul_value = deta_cumul(i,j, pd_list, lgd_list, weight_list,
        #                                   alpha, rho, c_corr_list)
        #     c_j = c_corr_list[j]
        #     result+= weight_list[j]*lgd_list[j]*(deta_cumul_value - _p(pd_list[j], c_j, alpha))
        _cached_looped_deta_cumul[c_index] = result
        return result
    

def deta_inf(pd_list, pi_array, lgd_list, weight_list, alpha, rho, c_corr_list):
    issuer_indices = range(len(pd_list))
    result = 0 
    dpi_array = _dp(pd_list, c_corr_list, alpha)
    cumul_value = np.array(
        [looped_deta_cumul(i, pd_list, pi_array, lgd_list, weight_list, alpha, rho, c_corr_list) for i in issuer_indices])
    return sum(cumul_value*2*weight_list*lgd_list*dpi_array)
    
    # for i in issuer_indices:
    #     c_i = c_corr_list[i]
    #     cumul_value = looped_deta_cumul(i, pd_list, lgd_list, weight_list, alpha, rho, c_corr_list)
    #     result += cumul_value*2*weight_list[i]*lgd_list[i]*_dp(pd_list[i], c_i, alpha)
    # return result

def deta_ga(pd_list, pi_array, dpi_array, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list):
    issuer_indices = range(len(pd_list))
    norm_dist = stats.norm(0,1)
    deta_cumul = (norm_dist.ppf(pi_array) - rho*norm_dist.ppf(pi_array))/ ((1.0 - rho**2)**0.5)
    deta_cumul = norm_dist.cdf(deta_cumul)
    return sum((weight_list**2)*dpi_array*((lgd_list**2)*(1.0 - 2* deta_cumul)+ pd_vol_list**2))
    #     c_i = c_corr_list[i]
    #     result+= (weight_list[i]**2)*_dp(pd_list[i], c_i, alpha)*(
    #         (lgd_list[i]**2)*(1.0 - 2* deta_cumul_value)+ pd_vol_list[i]**2)
    # return result
# loss : inf and ga corrections

def rho_function(c_corr_list, in_extra_corr_matrix,  intra_corr_list):
    extra_mat = sparse.coo_matrix(in_extra_corr_matrix)
    extra_mat = extra_mat.T * extra_mat
    sparse_dim = len(intra_corr_list)
    intra_corr = sparse.lil_matrix((sparse_dim, sparse_dim))
    intra_corr.setdiag(intra_corr_list) 
    extra_mat = extra_mat*intra_corr
    extra_mat =  extra_mat.T*intra_corr
    #extra_mat = np.multiply(np.multiply(np.matrix(in_extra_corr_matrix).T*np.matrix(in_extra_corr_matrix),intra_corr_list).T, intra_corr_list)
    extra_mat = extra_mat.todense()- np.matrix(c_corr_list).T * np.matrix(c_corr_list)
    squared_mat = np.matrix((1-c_corr_list**2)**-0.5).T* np.matrix((1-c_corr_list**2)**-0.5)
    try:
        return np.multiply(extra_mat, squared_mat)
    except:
        import pdb;pdb.set_trace()

def diag_rho_function(c_corr_list,  intra_corr_list):
    return (intra_corr_list**2 - c_corr_list**2)/(1-c_corr_list**2)

def delta_qa_inf(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha, pd_vol_list):
    c_corr_list = intra_corr_list*multifactor_corr_array(
        extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
    rho = rho_function(c_corr_list, extra_corr_matrix,  intra_corr_list)

    pi_array = _p(pd_list, c_corr_list, alpha)
    norm_dist = stats.norm(0,1)
    dm1_value = dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    dm2_value = d2mu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    eta_value =  eta_inf(pd_list, pi_array, lgd_list, weight_list, alpha, rho, c_corr_list)
    deta_value = deta_inf(pd_list, pi_array, lgd_list, weight_list, alpha, rho, c_corr_list)
    x_value = norm_dist.ppf(1-alpha)
    result = -(deta_value-eta_value*(dm2_value/dm1_value+x_value))/(2.0*dm1_value)
    return result

def delta_qa_ga(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha, pd_vol_list):
    c_corr_list = intra_corr_list*multifactor_corr_array(
        extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
    
    rho = diag_rho_function(c_corr_list,  intra_corr_list)
    norm_dist = stats.norm(0,1)
    dm1_value = dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    dm2_value = d2mu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    pi_array = _p(pd_list, c_corr_list, alpha)
    eta_value =  eta_ga(pd_list, pi_array, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list)
    dpi_array = _dp(pd_list, c_corr_list, alpha)
    deta_value = deta_ga(pd_list,  pi_array, dpi_array, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list)
    x_value = norm_dist.ppf(1-alpha)
    result = -(deta_value-eta_value*(dm2_value/dm1_value+x_value))/(2.0*dm1_value)
    return result

# expected shortfall : inf and ga corrections
def delta_es_inf(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha, pd_vol_list):

    c_corr_list = intra_corr_list*multifactor_corr_array(
        extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
    rho = rho_function(c_corr_list, extra_corr_matrix,  intra_corr_list)
    
    norm_dist = stats.norm(0,1)
    pi_array = _p(pd_list, c_corr_list, alpha)
    dm1_value = dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    eta_value =  eta_inf(pd_list, pi_array, lgd_list, weight_list, alpha, rho, c_corr_list)
    factor = - norm_dist.pdf(norm_dist.ppf(1-alpha))/(2.0*(1-alpha))
    return factor * eta_value/dm1_value

def delta_es_ga(extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list,
                 alpha, pd_vol_list):
    c_corr_list = intra_corr_list*multifactor_corr_array(
        extra_corr_matrix, pd_list, lgd_list, weight_list, intra_corr_list, alpha)
    rho = diag_rho_function(c_corr_list, intra_corr_list)
    norm_dist = stats.norm(0,1)
    dm1_value = dmu1(weight_list, lgd_list, pd_list, c_corr_list, alpha)
    pi_array = _p(pd_list, c_corr_list, alpha)
    eta_value = eta_ga(pd_list, pi_array, lgd_list, weight_list, alpha, pd_vol_list, rho, c_corr_list)
    factor = - norm_dist.pdf(norm_dist.ppf(1-alpha))/(2.0*(1-alpha))
    return factor * eta_value/dm1_value

# list of main functions
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
    return np.array(correl)

def extend_array(in_array, m_list):
    out_array = []
    for el,m in zip(in_array, m_list):
        out_array.extend([el]*m)
    return np.array(out_array)

def extend_matrix(in_matrix, m_list):
    rows = []
    for in_row in in_matrix:
        row = []
        for (rho_value,m) in zip(in_row, m_list):
            row.extend([rho_value]*m )
        rows.append(row)
    return np.array(rows)

def main_multifactor_loss(tag, alpha, x_list, m_list):
    pd = np.array([0.001, 0.002, 0.002, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.05])
    lgd = np.array([0.5, 0.3, 0.5, 0.3,  0.5, 0.3, 0.5, 0.3, 0.5, 0.3])
    vol_lgd = np.array([0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1 ])
    r_intra = np.array([0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2])
    weights = np.array([0.1]*10)
    m_list_inf = np.array([1,]*10)
    
    pd_list = extend_array(pd, m_list)
    lgd_list = extend_array(lgd, m_list)
    r_intra_list = extend_array(r_intra, m_list)
    weight_list = extend_array(weights/m_list, m_list)
    vol_lgd_list = extend_array(vol_lgd, m_list)

    loss_values = []
    inf_correction = []

    global _cached_deta_cumul
    global _cached_looped_deta_cumul
    global _cached_eta_inf
    global _cached_cum_factor 
    
    for rho in x_list:
        corr_matrix = correlation_matrix_multifactor(m_list_inf, rho)
        loss_values += [loss(corr_matrix, pd, lgd, weights, r_intra, alpha)]
        inf_correction += [delta_qa_inf(corr_matrix, pd, lgd, weights, r_intra, alpha, vol_lgd) ]

        _cached_deta_cumul.clear()
        _cached_deta_cumul.clear()
        _cached_eta_inf.clear()
        _cached_cum_factor.clear()
        
    ga_correction = []
    for rho in x_list:
        corr_matrix = correlation_matrix_multifactor(m_list, rho)
        ga_correction += [delta_qa_ga(corr_matrix, 
                                      pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]

        _cached_deta_cumul.clear()
        _cached_deta_cumul.clear()
        _cached_eta_inf.clear()
        _cached_cum_factor.clear()


    loss_no_ga = [i+j for (i,j) in zip(inf_correction, loss_values)] 
    loss_ga = [i+j for (i,j) in zip(loss_no_ga,  ga_correction)]
    data = [{'name': '%s base var'%tag, 'data': zip(x_list, loss_no_ga)},
            {'name': '%s var granularity adj'%tag, 'data': zip(x_list, loss_ga)}]
    #pylab.plot( x_list, loss_no_ga, plot_style+'--', label=' wa = '+ str(m_list[0])+ '-' + str(m_list[1])+' base')
    #pylab.plot( x_list, loss_ga, plot_style+'-', label=' wa = '+ str(m_list[0])+ '-' + str(m_list[1]))
    #pylab.suptitle('Ten factors model', fontsize=12)
    return data


def main_multifactor_es(tag, alpha, x_list, m_list):
    pd = np.array([0.001, 0.002, 0.002, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.05])
    lgd = np.array([0.5, 0.3, 0.5, 0.3,  0.5, 0.3, 0.5, 0.3, 0.5, 0.3])
    vol_lgd = np.array([0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1 ])
    r_intra = np.array([0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2])
    weights = np.array([0.1]*10)
    m_list_inf = np.array([1,]*10)
    
    pd_list = extend_array(pd, m_list)
    lgd_list = extend_array(lgd, m_list)
    r_intra_list = extend_array(r_intra, m_list)
    weight_list = extend_array(weights/m_list, m_list)
    vol_lgd_list = extend_array(vol_lgd, m_list)

    loss_values = []
    inf_correction = []

    global _cached_deta_cumul
    global _cached_looped_deta_cumul
    global _cached_eta_inf
    global _cached_cum_factor 
    for rho in x_list:
        corr_matrix = correlation_matrix_multifactor(m_list_inf, rho)
        loss_values += [expected_shortfall(corr_matrix, pd, lgd, weights, r_intra, alpha)]
        inf_correction += [delta_es_inf(corr_matrix, pd, lgd, weights, r_intra, alpha, vol_lgd) ]

        _cached_deta_cumul.clear()
        _cached_deta_cumul.clear()
        _cached_eta_inf.clear()
        _cached_cum_factor.clear()
        
    ga_correction = []
    for rho in x_list:
        corr_matrix = correlation_matrix_multifactor(m_list, rho)
        ga_correction += [delta_es_ga(corr_matrix, 
                                      pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]

        _cached_deta_cumul.clear()
        _cached_deta_cumul.clear()
        _cached_eta_inf.clear()
        _cached_cum_factor.clear()


    loss_no_ga = [i+j for (i,j) in zip(inf_correction, loss_values)] 
    loss_ga = [i+j for (i,j) in zip(loss_no_ga,  ga_correction)]
    #print loss_ga
    data = [{'name': '%s base es'%tag, 'data': zip(x_list, loss_no_ga)},
            {'name': '%s es granularity adj'%tag, 'data': zip(x_list, loss_ga)}]
    #pylab.plot( x_list, loss_no_ga, plot_style+'--', label=' wa = '+ str(m_list[0])+ '-' + str(m_list[1])+ ' base')
    #pylab.plot( x_list, loss_ga, plot_style+'-', label=' wa = '+  str(m_list[0])+ '-' + str(m_list[1]))
    #pylab.suptitle('Ten factors model', fontsize=12)
    return data

def main_finite_loss(tag, weight, pd, lgd, vol_lgd, r_intra, alpha, x_list):
    m_list = np.array([160, 40])
    # extend input
    weight_list = weight/m_list
    weight_list = extend_array(weight_list, m_list)
    pd_list = extend_array(pd, m_list)
    lgd_list = extend_array(lgd, m_list)
    vol_lgd_list = extend_array(vol_lgd, m_list)
    r_intra_list =  extend_array(r_intra, m_list)
    
    loss_values = []
    inf_correction = []
    ga_correction = []
    for rho in x_list:
        loss_values += [loss(extend_matrix(correl_function(rho), m_list), 
                             pd_list, lgd_list, weight_list, r_intra_list, alpha)]
        inf_correction += [delta_qa_inf(extend_matrix(correl_function(rho), m_list), 
                                        pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]

        ga_correction += [delta_qa_ga(extend_matrix(correl_function(rho), m_list), 
                                      pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list) ]


        global _cached_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_looped_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_eta_inf
        _cached_eta_inf.clear()
        global _cached_cum_factor 
        _cached_cum_factor.clear()
        #global _cached_mem_loss
        #_cached_mem_loss.clear()

    loss_first_corr = [i+j+k for (i,j,k) in zip(inf_correction, loss_values, ga_correction)]
    data = [{'name': '%s base loss'%tag, 'data': zip(x_list,loss_values)},
            {'name': '%s loss first order'%tag, 'data': zip(x_list,loss_first_corr)}]
    #pylab.plot( x_list,loss_values, plot_style+'--', label=' wa = '+ str(weight[0])+ ' base')
    #pylab.plot( x_list, loss_first_corr, plot_style+'-', label=' wa = '+ str(weight[0]))
    #pylab.suptitle('Two factor model', fontsize=12)
    return data

def main_loss(tag, weight,  pd, lgd, vol_lgd, r_intra, alpha, x_list):
    loss_values = []
    inf_correction = []
    for rho in x_list :
        loss_values += [loss(correl_function(rho), pd, lgd, weight, r_intra, alpha)]
        inf_correction += [delta_qa_inf(correl_function(rho), pd, lgd, weight, 
                                        r_intra, alpha, vol_lgd)]


        global _cached_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_looped_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_eta_inf
        _cached_eta_inf.clear()
        global _cached_cum_factor 
        _cached_cum_factor.clear()
        #global _cached_mem_loss
        #_cached_mem_loss.clear()
    loss_first_corr = [i+j for (i,j) in zip(inf_correction, loss_values)]
    data = [{'name': '%s base loss'%tag,
             'data': zip(x_list,loss_values)},
            {'name': '%s loss first order'%tag,
             'data': zip(x_list,loss_first_corr)}]
    #pylab.suptitle('Two factor model', fontsize=12)
    return data

def main_es(tag, weight, pd, lgd, vol_lgd, r_intra, alpha, x_list):
    loss_values = []
    inf_correction = []
    for rho in x_list:
        loss_values += [expected_shortfall(correl_function(rho), pd, lgd, weight, r_intra, alpha)]
        inf_correction += [delta_es_inf(correl_function(rho), pd, lgd, weight, 
                                        r_intra, alpha, vol_lgd)]


        global _cached_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_looped_deta_cumul
        _cached_deta_cumul.clear()
        global _cached_eta_inf
        _cached_eta_inf.clear()
        global _cached_cum_factor 
        _cached_cum_factor.clear()
        #global _cached_mem_loss
        #_cached_mem_loss.clear()
    loss_first_corr = [i+j for (i,j) in zip(inf_correction, loss_values)]    
    # plotting 
    #pylab.plot( x_list,loss_values, plot_style+'--', label=' wa = '+ str(weight[0])+ ' base')
    #pylab.plot( x_list, loss_first_corr, plot_style+'-', label=' wa = '+ str(weight[0]))
    data = [{'name': '%s base loss'%tag, 'data': zip(x_list,loss_values)},
            {'name': '%s loss first order'%tag, 'data': zip(x_list,loss_first_corr)}]
    return data
    #pylab.suptitle('Two factor model', fontsize=12)

def mult_correl(correlation, dimension):
    rho = []
    alpha = math.sqrt(correlation)
    for i in range(dimension-1):
        x = [alpha,] + [0.0 for j in range(dimension-1)]
        x[i+1] = math.sqrt(1-alpha**2)
        rho.append(x)
    return np.matrix(rho).T
