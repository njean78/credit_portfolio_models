
import math, random
from scipy import stats
import pylab
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from Models.HomogeneousPortfolio.portfolio import loss_var
from Models.MultiFactorAdj.pykhtin_zero_order import (loss, 
                                                      expected_shortfall)
from Models.MultiFactorAdj.pykhtin_first_order import (extend_array,
                                                       extend_matrix,
                                                       _cached_deta_cumul,
                                                       _cached_looped_deta_cumul,
                                                       _cached_eta_inf,
                                                       _cached_cum_factor,
                                                       delta_qa_inf,
                                                       delta_qa_ga,
                                                       delta_es_inf,
                                                       delta_es_ga)


import sys, time

def correl_f(pd):
    factor = 1.0-math.exp(-50*pd)
    num_factor = 1.0-math.exp(-50)
    return math.sqrt(0.185*(factor/num_factor) + 0.34 * (1- factor/num_factor))

def pd_from_correl(correl):
    factor1 = (1.0-math.exp(-50)) * (correl/100.0 - 0.34) / (0.185-0.34)
    return -(1.0/50.0)*math.log(1.0-factor1)

def extra_correl_f(sector_correl):
    correl_matrix = []
    for i in range(len(sector_correl)):
        row = [] 
        for j in range(i):
            row.append(sector_correl[j][i-j])
        row.extend(sector_correl[i])
        correl_matrix.append(row) 
    # cholesky decomposition       
    mat = np.matrix(correl_matrix)
    return mat


def extend_pd_array(pd_sectors, pd_classes, pd_values, m_list):
    out_array = []
    for el,m in zip(pd_sectors, m_list):
        for rating, weight in  pd_classes[el].items() :
            out_array.extend([pd_values[rating]]*int(m*weight/100.0))
    return np.array(out_array)

def mean_beta(weight_list, lgd_list, rc_array, corr_list, sector_correl):
    ck_list =  rc_array *weight_list*lgd_list
    numerator = 0
    denumerator = 0
    for i, row in enumerate(sector_correl.tolist()):
        for j in range(len(row)):
            if i ==0 or j ==0:
                next
            if i!=j:
                numerator+=(row[j]/100.0)*ck_list[i-1]*ck_list[j-1]
                denumerator+=ck_list[i-1]*ck_list[j-1]
    return numerator/denumerator

def main_multifactor_loss(alpha,  pd_classes, pd_values, mean_pds, pds, weights, m_list, sector_correl, lgd, vol_lgd, plot_style):
    #pd = np.array([d.uniform(0.01,0.1827) for d in pds])
    pd_keys = pds.keys()
    pd_sectors = np.array([pd_keys[d.randint(0, len(pd_keys)-1)] for d in pds])
    pd = np.array([pds[sec] for sec in pd_sectors])
    r_intra = np.array([correl_f(pd_value) for pd_value in pd])
    weight = np.array([w.uniform(0,1) for w in weights])
    #lgd = np.array([l.uniform(0.0,0.4) for l in lgd_rnd])
    w_sum = sum(weight)
    weight = weight/w_sum
    m_list_inf = np.array([1,]*10)
    extra_corr_orig = extra_correl_f(sector_correl)
    extra_corr_scaled = zip(*zip(*(np.linalg.cholesky(extra_corr_orig).T).tolist())[1:])
    extra_corr = []
    # 10 = math.sqrt(100)
    for row in extra_corr_scaled:
        extra_corr.append([el/10.0 for el in row])
    extra_corr = np.array(extra_corr)
    pd_list = extend_pd_array(pd_sectors, pd_classes, pd_values, m_list)
    lgd_list =extend_array(lgd,m_list)
    r_intra_list = extend_array(r_intra,m_list)
    weight_list = extend_array(weight/m_list,m_list)
    vol_lgd_list = extend_array(vol_lgd,m_list)
    m_extra_corr = extend_matrix(extra_corr, m_list)
    
    global _cached_deta_cumul
    global _cached_looped_deta_cumul
    global _cached_eta_inf
    global _cached_cum_factor 
    loss_value = loss(extra_corr, pd, lgd, weight, r_intra, alpha)
    #print "loss_value ", loss_value
    inf_correction  = delta_qa_inf(extra_corr, pd, lgd, weight, r_intra, alpha, vol_lgd) 
    #print "inf correction ", inf_correction
    _cached_deta_cumul.clear()
    _cached_deta_cumul.clear()
    _cached_eta_inf.clear()
    _cached_cum_factor.clear()
    rc_array = np.array([(loss_var(alpha, pd_value, corr**2)-pd_value) for (pd_value, corr) in zip(pd, r_intra)])
    dnum = (sum(lgd*rc_array*weight))**2
    hhi_value = sum((lgd*rc_array*weight)**2)/dnum
    mean_beta_value = mean_beta(weight, lgd, rc_array, r_intra, extra_corr_orig)
    ga_correction = delta_qa_ga(m_extra_corr, pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list)
    #print "ga correction ", ga_correction
    _cached_deta_cumul.clear()
    _cached_deta_cumul.clear()
    _cached_eta_inf.clear()
    _cached_cum_factor.clear()
    df_value = (loss_value+inf_correction+ga_correction-sum(lgd*weight*pd))/sum(lgd*rc_array*weight)
    return (hhi_value, mean_beta_value, df_value)



def main_multifactor_es(alpha, pd_classes, pd_values, mean_pds, pds, weights, m_list, sector_correl, lgd, vol_lgd, plot_style):
    pd = np.array([d.uniform(0.0,0.10) for d in pds])
    r_intra = np.array([correl_f(pd_value) for pd_value in pd])
    weight = np.array([w.uniform(0,1) for w in weights])
    #lgd = np.array([l.uniform(0.0,0.4) for l in lgd_rnd])
    w_sum = sum(weight)
    weight = weight/w_sum
    m_list_inf = np.array([1,]*10)
    extra_corr_orig = extra_correl_f(sector_correl)
    extra_corr_scaled = zip(*zip(*(np.linalg.cholesky(extra_corr_orig).T).tolist())[1:])
    extra_corr = []
    # 10 = math.sqrt(100)
    for row in extra_corr_scaled:
        extra_corr.append([el/10.0 for el in row])
    extra_corr = np.array(extra_corr)
    pd_list = extend_array(pd,m_list)
    lgd_list =extend_array(lgd,m_list)
    r_intra_list = extend_array(r_intra,m_list)
    weight_list = extend_array(weight/m_list,m_list)
    vol_lgd_list = extend_array(vol_lgd,m_list)
    m_extra_corr = extend_matrix(extra_corr, m_list)

    global _cached_deta_cumul
    global _cached_looped_deta_cumul
    global _cached_eta_inf
    global _cached_cum_factor 
    loss_value = expected_shortfall(extra_corr, pd, lgd, weight, r_intra, alpha)
    #print "loss_value ", loss_value
    inf_correction  = delta_es_inf(extra_corr, pd, lgd, weight, r_intra, alpha, vol_lgd) 
    #print "inf correction ", inf_correction
    _cached_deta_cumul.clear()
    _cached_deta_cumul.clear()
    _cached_eta_inf.clear()
    _cached_cum_factor.clear()
    rc_array = np.array([(loss_var(alpha, pd_value, corr**2)-pd_value) for (pd_value, corr) in zip(pd, r_intra)])
    dnum = (sum(lgd*rc_array*weight))**2
    hhi_value = sum((lgd*rc_array*weight)**2)/dnum
    mean_beta_value = mean_beta(weight, lgd, rc_array, r_intra, extra_corr_orig)
    ga_correction = delta_es_ga(m_extra_corr, pd_list, lgd_list, weight_list, r_intra_list, alpha, vol_lgd_list)
    #print "ga correction ", ga_correction
    _cached_deta_cumul.clear()
    _cached_deta_cumul.clear()
    _cached_eta_inf.clear()
    _cached_cum_factor.clear()
    df_value = (loss_value+inf_correction+ga_correction-sum(lgd*weight*pd))/sum(lgd*rc_array*weight)
    return (hhi_value, mean_beta_value, df_value)

if __name__ == "__main__":
    #pd = [ 0.005, 0.005 ]
    #lgd = [ 0.4, 0.4]
    #vol_lgd = [0.2, 0.2]
    #r_intra = [0.5, 0.5]
    
    pds = np.array([random.Random() for i in range(10)])
    weights = np.array([random.Random() for i in range(10)])
    m_list = np.array([500,]*10)
    lgd = np.array([1.0, ]*10)
    #lgd = np.array([random.Random() for i in range(10)])
    vol_lgd = np.array([0.2, ] *10)
    
    sector_correl = [[100, 50, 42, 34, 45, 46, 57, 34, 10, 31, 69],
                     [100, 87, 61, 75, 84, 62, 30, 56, 73, 66],
                     [100, 67, 83, 92, 65, 32, 69, 82, 66],
                     [100, 58, 68, 40, 8,  50, 60, 37],
                     [100, 83, 68, 27, 58, 77, 67],
                     [100, 76, 21, 69, 81, 66],
                     [100, 33, 46, 56, 66],
                     [100, 15, 24, 46],
                     [100, 75, 42],
                     [100, 62],
                     [100],
                     ]
    # bug fixing: u should use the sector pds described in hibbeln book.
    # for every sector u will have 5 pd classes. The percentage are given by the pg109 picture.
    # each sector is divided  by 5 subsectors....therefore I'll end up with 50 subsectors.
    pd_classes = {'very low': {'AAA':0.5, 'AA':1, 'A':3,'BBB': 12,'BB': 34,'B': 37,'CCC': 12.5},
                  'low': {'AAA':1, 'AA':1.5, 'A':3.5,'BBB': 16,'BB': 38,'B': 32,'CCC': 8},
                  'average': {'AAA':3, 'AA':5, 'A':12,'BBB': 31,'BB': 31,'B': 14,'CCC': 4},
                  'high': {'AAA':3, 'AA':5, 'A':32,'BBB': 38,'BB': 16,'B': 4,'CCC': 2},
                  'very high': {'AAA':5, 'AA':7, 'A':36,'BBB': 30,'BB': 18,'B': 3.5,'CCC': 0.5},
                  }

    pd_values = {'AAA' : 0.0002,
                 'AA': 0.0002,
                 'A': 0.0003
                 'BBB': 0.0007,
                 'BB' : 0.0132,
                 'B' : 0.0558,
                 'CCC' : 0.186,
                 }
    
    mean_pds = {'very low': pd_from_correl(21),
                'low': pd_from_correl(23),
                'average': pd_from_correl(25),
                'high': pd_from_correl(28),
                'very high': pd_from_correl(30)
        }
    plot_styles = ['b','c', 'g', 'm']
    num_sim = 30
    alpha =0.999
    if len(sys.argv) !=2:
        print "Usage pykhtin_zero_order.py [es|loss|finite_loss]"
        exit
    else:
        points = []
        a = time.time()
        if sys.argv[1] == "loss": 
            for i in range(num_sim):
                points+=[main_multifactor_loss(alpha, pd_classes, pd_values, mean_pds, pds, weights, m_list,sector_correl, lgd, vol_lgd, plot_styles)
                         ]

        elif sys.argv[1] == "es": 
            for i in range(num_sim):
                points+=[main_multifactor_es(alpha, pd_classes, pd_values, mean_pds, pds, weights, m_list,sector_correl, lgd, vol_lgd, plot_styles)]
        else:
            print "wrong argument " + sys.argv[1]
            exit
        fig = pylab.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('HHI')
        ax.set_ylabel('Beta')
        ax.set_zlabel('')
        ax.scatter3D(*zip(*points))
        b = time.time()
        print b-a
        # get parameters
        (x,y,z) = zip(*points)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        # equation DF = a0+a1*(1-hhi)*(1-b) + a2*(1-b)*(1-hhi)**2 + a3*(1-hhi)*(1-b)**2
        v = np.array([np.ones(len(x)),(1-x)*(1-y), (1-y)*((1-x)**2), (1-x)*((1-y)**2)])
        coefficients, residues, rank, singval = np.linalg.lstsq(v.T, z)
        print 'coefficients ', coefficients
        print 'residues ' , residues
        print 'rank ', rank
        print 'singval ', singval
        # show the graph            
        pylab.show()
        #pylab.xlabel("correlation")
        #pylab.legend( loc=0, ncol=2, borderaxespad=0.)
        #pylab.show()
