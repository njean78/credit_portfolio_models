from Models.MultiFactorAdj.pykhtin_zero_order import loss, correl_function
from Models.HomogeneousPortfolio.portfolio import loss_var
import random, math
from scipy import stats, polyfit
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import pylab
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

def correl_f(pd):
    factor = 1.0-math.exp(-50*pd)
    num_factor = 1.0-math.exp(-50)
    return math.sqrt(0.12*(factor/num_factor) + 0.24 * (1- factor/num_factor))

def avg(in_list):
    tot = len(in_list)
    return sum([i/tot for i in in_list])

def cdi(alpha,weight_list, lgd_list,  pd_list, corr_list):
    ck_list = [w*lgd*loss_var(alpha, pd, corr**2)
               for (w, lgd, pd, corr) in zip(weight_list, lgd_list, pd_list, corr_list)]
    ck_sum = sum(ck_list)
    return sum([ck**2/ck_sum**2 for ck in ck_list])


def main_basel(sim_num, rho):
    d1 = random.Random()
    d2 = random.Random()
    d3 = random.Random()
    pd_limits = [0.0, 0.1]
    lgds = [ 1.0, 1.0]
    vol_lgd = [0.2, 0.2]
    alpha =0.999
    extra_corr = correl_function(rho)
    results = []
    asympt_results = []
    x_values = []
    for i in range(sim_num):
        pds = np.array([ d1.uniform(*pd_limits), d2.uniform(*pd_limits) ])
        weight = d3.uniform(0,1)
        weights = np.array([weight, 1-weight])
        r_intra = np.array([correl_f(pds[0]), correl_f(pds[1])])
        exp_loss = sum(weights*lgds*pds)
        results.append((loss(extra_corr, pds, lgds, weights, r_intra, alpha)-exp_loss)/2)
        asympt_results.append((sum([w*lgd*loss_var(alpha, pd, corr**2)
                                   for (w, lgd, pd, corr) in zip(weights, lgds, pds, r_intra)])-exp_loss)/2)
        x_values.append(avg(pds) )
    pylab.plot( x_values, results, 'b+')
    pylab.plot( x_values, asympt_results, 'go')

def main_df(sim_num, rho):
    d1 = random.Random()
    d2 = random.Random()
    d3 = random.Random()
    pd_limits = [0.0, 0.1]
    lgds = [ 1.0, 1.0]
    vol_lgd = [0.2, 0.2]
    alpha =0.999
    extra_corr = correl_function(rho)
    results = []
    asympt_results = []
    x_values = []
    cdis = []
    for i in range(sim_num):
        pds = np.array([ d1.uniform(*pd_limits), d2.uniform(*pd_limits) ])
        weight = d3.uniform(0,1)
        weights = np.array([weight, 1-weight])
        r_intra = np.array([correl_f(pds[0]), correl_f(pds[1])])
        exp_loss = sum(weights*lgds*pds)
        results.append((loss(extra_corr, pds, lgds, weights, r_intra, alpha)-exp_loss)/2)
        asympt_results.append((sum([w*lgd*loss_var(alpha, pd, corr**2)
                                   for (w, lgd, pd, corr) in zip(weights, lgds, pds, r_intra)])-exp_loss)/2)
        cdis.append(cdi(alpha,weights, lgds,  pds, r_intra))
        x_values.append(avg(pds) )
    #pylab.plot( x_values, results, 'b+')
    #pylab.plot( x_values, asympt_results, 'go')
    pylab.ylim(ymin=0.0, ymax=1.0)
    dfs = [i/j for (i,j) in zip(results, asympt_results)]
    pylab.plot( cdis, dfs, 'b+')
    (a,b) = polyfit(cdis, dfs,1)
    print 'coefficient a = ', a , ' and b = ' ,  b


def main_3df(sim_num, rho_list):
    d1 = random.Random()
    d2 = random.Random()
    d3 = random.Random()
    pd_limits = [0.0, 0.1]
    lgds = [ 1.0, 1.0]
    vol_lgd = [0.2, 0.2]
    alpha =0.999

    fig = pylab.figure()
    ax = Axes3D(fig)
    fit_parameters = []
    y_values = []
    for rho in rho_list:
        extra_corr = correl_function(rho)
        results = []
        asympt_results = []
        x_values = []
        cdis = []
        for i in range(sim_num):
            pds = np.array([ d1.uniform(*pd_limits), d2.uniform(*pd_limits) ])
            weight = d3.uniform(0,1)
            weights = np.array([weight, 1-weight])
            r_intra = np.array([correl_f(pds[0]), correl_f(pds[1])])
            exp_loss = sum(weights*lgds*pds)
            results.append((loss(extra_corr, pds, lgds, weights, r_intra, alpha)-exp_loss)/2)
            asympt_results.append((sum([w*lgd*loss_var(alpha, pd, corr**2)
                                       for (w, lgd, pd, corr) in zip(weights, lgds, pds, r_intra)])-exp_loss)/2)
            cdis.append(cdi(alpha,weights, lgds,  pds, r_intra))
            x_values.append(avg(pds))
            #pylab.ylim(ymin=0.0, ymax=1.0)
            dfs = [i/j for (i,j) in zip(results, asympt_results)]
        (a,b) = polyfit(cdis, dfs,1)
        xcdis = [0.01*i for i in range(100)]
        #ax.plot(xs=xcdis, ys=[rho,]*len(xcdis), zs=[a*i+b for i in xcdis], zdir='z')
        fit_parameters.append((a,b))
        y_values.append(rho)
    x_values = [0.01*i for i in range(100)]
    z_values = []
    for (a,b) in fit_parameters:
        z_values .append([a*i+b for i in xcdis])
    Z = np.array(zip(*z_values))
    Y, X = np.meshgrid(y_values, x_values)
    fig = pylab.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                              linewidth=0, antialiased=False)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

if __name__ == "__main__":
    sim_num = 3000
    if len(sys.argv) !=2:
        print "Usage cespedes.py [basel|df|3df]"
    else:
        if sys.argv[1] == "basel":
            rho = 0.60
            main_basel(sim_num, rho)
        elif sys.argv[1] == "df":
            rho = 0.60
            main_df(sim_num, rho)
        elif sys.argv[1] == "3df":
            rho_list = [0.0, 0.1, 0.2, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8]
            main_3df(sim_num,rho_list)

        else:
            print "wrong argument " + sys.argv[1]
        pylab.show()
