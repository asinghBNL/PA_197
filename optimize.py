import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from labellines import labelLines

Ug2 = 900 # [V]
# Ug2 = 1500 # [V]
ua_lims  = [0e3,12e3]
ug1_lims = [-300,400]

ia_arr  = np.loadtxt("./ia_arr.csv", delimiter=",")
ig2_arr = np.loadtxt("./ig2_arr.csv", delimiter=",")

ua_domain = ia_arr[0]
ia_arr  = ia_arr[1:]
ig2_arr = ig2_arr[1:]

# plot the initial data 
for i in ia_arr:
    plt.plot(ua_domain,i,color='red')
for i in ig2_arr:
    plt.plot(ua_domain,i,color='blue')
plt.xlim(ua_lims[0],ua_lims[1])
plt.ylim(ug1_lims[0],ug1_lims[1])
plt.grid()
plt.show()

def model_A(alpha, beta, ug2, ua):
    ua = np.asarray(ua, dtype=float)
    base = alpha * (1.0 - np.exp(-ua / (beta * ug2)))
    return np.where(
        ua >= ug2,
        base,
        (ua / ug2) * base
    )

def model_Ii(K,mu_c,mu_s,exp_coeff,ug1,ug2,ua):
    drive = mu_c * ug1 + mu_s * ug2 + ua
    # Prevent invalid fractional powers of negative values
    drive = np.maximum(drive, 0.0)
    return K * drive**exp_coeff

def ug2_vs_ua(alpha,beta,K,mu_c,mu_s,exp_coeff,ug2,ua,current,ig2_true=False):
    if ig2_true:
        i_a = current/(1-model_A(alpha,beta,ug2,ua))
    else:
        i_a = current/(model_A(alpha,beta,ug2,ua))
    inv_exp = (i_a/(K))**(1/exp_coeff)
    return (inv_exp - ua - mu_s*ug2)/mu_c

def mse(l): return np.nansum(l**2)/l.size

def optimize_coeffs(params,data,domain,ia_ug1_ranges,ig2_ug1_ranges,bounds=None):
    # params : 6 of them, 2 for A and 4 for Ii
    # data   : current values, Ia [0] and Ig2 [1]
    # domain : the Ua values

    curr_min = 1
    curr_max = 7

    ia_currents  = np.array(data[0])[curr_min:curr_max]
    ig2_currents  = np.array(data[1])
    print(f"Currents To Optimize : {ia_currents}")
    # ig2_currents = np.array(data[1])

    def objective(x):
        # define a matrix of #curr x domain.size
        IA_MATRIX = np.ones((ia_currents.size,domain.size))*ia_currents.reshape(-1,1)
        # IA_MATRIX = ia_currents.reshape(-1,1)
        CALC_IA_MATRIX = model_A(x[0],x[1],Ug2,domain)*model_Ii(x[2],x[3],x[4],x[5],ia_ug1_ranges[curr_min:curr_max],Ug2,domain)
        # CALC_IA_MATRIX = ug2_vs_ua(x[0],x[1],x[2],x[3],x[4],x[5],Ug2,domain,IA_MATRIX)
        l_Ia  = (IA_MATRIX - CALC_IA_MATRIX).flatten()
        l_Ia = (l_Ia[np.isfinite(l_Ia)])

        # define a matrix of #curr x domain.size
        IA_MATRIX = np.ones((ig2_currents.size,domain.size))*ig2_currents.reshape(-1,1)
        # # IA_MATRIX = ia_currents.reshape(-1,1)
        CALC_IA_MATRIX = (1-model_A(x[0],x[1],Ug2,domain))*model_Ii(x[2],x[3],x[4],x[5],ig2_ug1_ranges,Ug2,domain)
        # # CALC_IA_MATRIX = ug2_vs_ua(x[0],x[1],x[2],x[3],x[4],x[5],Ug2,domain,IA_MATRIX)
        l_Ig2  = (IA_MATRIX - CALC_IA_MATRIX).flatten()
        l_Ig2 = (l_Ig2[np.isfinite(l_Ig2)])

        # print(l_Ia)
        # return np.concatenate((l_Ia,l_Ig2))
        l_Ia = np.concatenate((l_Ia,l_Ig2))
        return mse(l_Ia)
    result = sp.optimize.minimize(objective,params,method="Nelder-Mead",options={'disp':True,'maxiter':1e8,'maxfev':1e8,'fatol':1e-8,'xatol':1e-8},bounds=bounds)
    # result = sp.optimize.minimize(objective,params,method="BFGS")
    # result = sp.optimize.least_squares(objective,params,method="dogbox")
    return result.x

# with alpha and beta, A is variable
params_0 = [0.9596,0.7408,2.99e-10,521.793,144.766,3/2]

# Current Values -----------------------------------------
# 197
current_vals = [[0,5,10,20,30,40,60,80],[1,2,5,10,15,20]]
# 28
# current_vals = [[0.001,1,5,15,30,50,80,100,120],[-0.2,-0.1,0,0.5,2,5,10,15]]
# current_vals = [[1,5,15,30,50]]

opt_params = optimize_coeffs(params_0,current_vals,ua_domain,ia_arr,ig2_arr)

for i in range(opt_params.size):
    print(params_0[i],opt_params[i])

plt.figure(figsize=(9, 7))
for i in current_vals[0]:
    # ug2_pred = ug2_vs_ua(*params_0,Ug2,ua_domain,i)
    # plt.plot(ua_domain,ug2_pred,color="blue")
    ug2_pred = ug2_vs_ua(*opt_params,Ug2,ua_domain,i)
    plt.plot(ua_domain,ug2_pred,color="green")
    # ug2_pred = ug2_vs_ua(*opt_params,Ug2,ua_domain,i,ig2_true=True)
    # plt.plot(ua_domain,ug2_pred,color="orange")

for idx, i in enumerate(ia_arr):
    # plt.plot(ua_domain,i*np.ones_like(ua_domain),color="red",linewidth=3)
    plt.plot(ua_domain,i,'r--',linewidth=1.5,label=f"{current_vals[0][idx]} Amps")
    labelLines(plt.gca().get_lines(), align=True, zorder=1, fontsize=10)
# for idx, i in enumerate(ig2_arr):
#     # plt.plot(ua_domain,i*np.ones_like(ua_domain),color="red",linewidth=3)
#     plt.plot(ua_domain,i,'b--',linewidth=1.5,label=f"{current_vals[0][idx]} Amps")
#     labelLines(plt.gca().get_lines(), align=True, zorder=1, fontsize=10)

plt.xlabel("UA")
plt.ylabel("UG1")
plt.title("Datasheet-style IA contours")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim(ua_lims[0],ua_lims[1])
plt.ylim(ug1_lims[0],ug1_lims[1])
plt.show()
