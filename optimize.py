import numpy as np
import scipy as sp

def model_A(alpha,beta,ug2,ua):
    return alpha*(1 - np.exp(-ua/(beta*ug2)))

def model_i1(K,mu_c,mu_s,exp_coeff,ug1,ug2,ua):
    return K*(mu_c*ug1 + mu_s*ug2 + ua)**exp_coeff

def optimize_A(params,data,domain,bounds=None):

    def objective(x):
        l = data - model_A(x[0],x[1],domain[0],domain[1])
        return np.sum(l**2)/l.size

    result = sp.optimize.minimize(objective,params,method="Nelder-Mead",options={'maxiter':100000,'maxfev':100000},bounds=bounds)
    return result.x

def optimize_coeffs(params,data,domain,A_opt,bounds=None):

    def objective(x):
        l = data - (model_i1(x[0],x[1],x[2],x[3],domain[0],domain[1],domain[2])/A_opt)
        return np.sum(l**2)

    result = sp.optimize.minimize(objective,params,method="Nelder-Mead",options={'maxiter':100000,'maxfev':100000},bounds=bounds)
    return result.x
