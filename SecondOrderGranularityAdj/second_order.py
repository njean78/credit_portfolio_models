# general formula a_i = b_i x + c_i eps_i
# where b_i = sqrt(rho_i) and c_i = sqrt(1-rho_i)
# a_i = sqrt(rho_i) * x + sqrt(1-rho_i) eps_i


import math
from scipy import stats
import pylab
import numpy as np

def s(corr):
    return math.sqrt(corr)/math.sqrt(1.0-corr)

def x(alpha):
    norm_dist = stats.norm(0,1)
    return norm_dist.ppf(1.0-alpha)

def z(alpha, pd, corr):
    norm_dist = stats.norm(0,1)
    return (norm_dist.ppf(pd)-
            math.sqrt(corr)*x(alpha))/math.sqrt(1.0-corr)

def factor1(alpha, pd, corr):
    return (x(alpha)**2 -1.0 + s(corr)**2 +
            3*x(alpha)*s(corr)*z(alpha, pd, corr)+
            2*((s(corr)*z(alpha, pd, corr))**2))

def mul_factor1(alpha, pd, corr):
    norm_dist = stats.norm(0,1)
    PHI = norm_dist.cdf(z(alpha, pd, corr))
    return PHI -3*PHI*PHI + 2*PHI*PHI*PHI

def factor2(alpha, pd, corr):
    norm_dist = stats.norm(0,1)
    return s(corr)*norm_dist.pdf(z(alpha, pd, corr))*(2*x(alpha)+3*s(corr)*z(alpha, pd, corr))

def mul_factor2(alpha, pd, corr):
    norm_dist = stats.norm(0,1)
    PHI = norm_dist.cdf(z(alpha, pd, corr))
    return 1-6*PHI+6*PHI*PHI

def factor3(alpha, pd, corr):
    norm_dist = stats.norm(0,1)
    phi = norm_dist.pdf(z(alpha, pd, corr))
    return -s(corr)*s(corr)*phi

def mul_factor3(alpha, pd, corr):
    norm_dist = stats.norm(0,1)
    PHI = stats.norm(0,1).cdf(z(alpha, pd, corr))
    phi = stats.norm(0,1).pdf(z(alpha, pd, corr))
    return (z(alpha, pd, corr)-
            6*(PHI*z(alpha, pd, corr) - phi) +
            6*PHI*(PHI*z(alpha, pd, corr)-2*phi) )

def factor4(alpha, pd, corr):
    return  (-x(alpha)-3*s(corr)*z(alpha, pd, corr))

def mul_factor4(alpha, pd, corr):
    PHI = stats.norm(0,1).cdf(z(alpha, pd, corr))
    phi = stats.norm(0,1).pdf(z(alpha, pd, corr))
    return ((PHI - PHI*PHI)*(-x(alpha)-s(corr)*z(alpha, pd, corr)) - s(corr)*phi*(1-2*PHI))**2

def factor5(alpha, pd, corr):
    PHI = stats.norm(0,1).cdf(z(alpha, pd, corr))
    phi = stats.norm(0,1).pdf(z(alpha, pd, corr))
    return ((PHI-PHI*PHI)*(1.0-s(corr)*s(corr)) -
            s(corr)*phi*(1.0-2*PHI)*(x(alpha)+s(corr)*z(alpha, pd, corr)) +
            s(corr)*s(corr)*phi*(z(alpha, pd, corr)+2*(phi-PHI*z(alpha, pd, corr))))

def mul_factor5(alpha, pd, corr):
    PHI = stats.norm(0,1).cdf(z(alpha, pd, corr))
    phi = stats.norm(0,1).pdf(z(alpha, pd, corr))
    return 2*((PHI-PHI*PHI)*(x(alpha)+s(corr)*z(alpha, pd, corr))+
              s(corr)*phi*(1-2*PHI))


def var_adjustment(alpha,  pd, corr, num_of_credits):
    PHI = stats.norm(0,1).cdf(z(alpha, pd, corr))
    phi = stats.norm(0,1).pdf(z(alpha, pd, corr))
    mul1 = 1.0/(6.0*((num_of_credits*s(corr)*phi)**2))
    mul2 = num_of_credits/(8.0*((num_of_credits*s(corr)*phi)**3))
    return (mul1*(factor1(alpha,  pd, corr)*mul_factor1(alpha,  pd, corr) +
                  factor2(alpha,  pd, corr)*mul_factor2(alpha,  pd, corr) +
                  factor3(alpha,  pd, corr)*mul_factor3(alpha,  pd, corr)) -
            mul2* (factor4(alpha,  pd, corr)*mul_factor4(alpha,  pd, corr)+
                   factor5(alpha,  pd, corr)*mul_factor5(alpha,  pd, corr)))

def es_adjustment(alpha,  pd, corr, num_of_credits):
    PHI = stats.norm(0,1).cdf(z(alpha, pd, corr))
    phi = stats.norm(0,1).pdf(z(alpha, pd, corr))
    phi_x = stats.norm(0,1).pdf(x(alpha))
    mul1 = phi_x/(6.0*(1.0-alpha)*((s(corr)*phi)**2))
    mul2 = -phi_x/(8.0*(1.0-alpha)*((s(corr)*phi)**3))
    fac1 = -s(corr)*phi*(1-6*PHI+6*(PHI**2))
    fac1+= -1*(PHI-3*(PHI**2)+2*(PHI**3))*(x(alpha)-s(corr)*z(alpha, pd, corr))
    fac2= -s(corr)*phi*(1-2*PHI)
    fac2+= -(PHI-PHI**2)*(x(alpha)-s(corr)*z(alpha, pd, corr))
    return (mul1*fac1 + mul2*(fac2**2))/(num_of_credits**2)

def main():
    num_of_credits = 40
    default_probability = 0.01
    correlation = 0.2

    x_list = pylab.arange(0.9, 1.00, 0.0005 )
    loss_f_values = [var_adjustment(i, default_probability, correlation, num_of_credits)
                     for i in x_list]
    es_values = [es_adjustment(i, default_probability, correlation, num_of_credits)
                 for i in x_list]
    pylab.xlabel("loss in percentage")
    pylab.ylabel("loss function values")
    pylab.plot(loss_f_values, x_list, 'b')
    pylab.plot(es_values, x_list, 'g')
    pylab.show()

if __name__ == "__main__":
    main()
