import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

Ug2 = 1500 # [V]
ua_lims  = [0,20e3]
ug1_lims = [-600,800]

ia_arr  = np.loadtxt("./ia_arr.csv", delimiter=",")
ig2_arr = np.loadtxt("./ig2_arr.csv", delimiter=",")

ua_domain = ia_arr[0]
ia_arr  = ia_arr[1:]
ig2_arr = ig2_arr[1:]

for i in ia_arr:
    plt.plot(ua_domain,i,color='red')

for i in ig2_arr:
    plt.plot(ua_domain,i,color='blue')

plt.xlim(ua_lims[0],ua_lims[1])
plt.ylim(ug1_lims[0],ug1_lims[1])
plt.grid()
plt.show()

def model_A(alpha,beta,ug2,ua):
    return alpha*(1 - np.exp(-ua/(beta*ug2)))

def model_Ii(K,mu_c,mu_s,exp_coeff,ug1,ug2,ua):
    drive = mu_c * ug1 + mu_s * ug2 + ua
    # Prevent invalid fractional powers of negative values
    drive = np.maximum(drive, 0.0)
    return K * drive**exp_coeff

def mse(l): return np.nansum(l**2)/l.size

def optimize_coeffs(params,data,domain,ia_ug1_ranges,ig2_ug1_ranges,bounds=None):
    # params : 6 of them, 2 for A and 4 for Ii
    # data   : current values, Ia [0] and Ig2 [1]
    # domain : the Ua values

    domain_ia  = domain[0]
    domain_ig2 = domain[1]

    ia_currents  = np.array(data[0])
    ig2_currents = np.array(data[1])

    def objective(x):
        l_Ig2 = 0

        # define a matrix of #curr x domain.size
        IA_MATRIX = np.ones([ia_currents.size,domain_ia[0].size])*ia_currents.reshape(-1,1)
        CALC_IA_MATRIX = model_A(x[0],x[1],Ug2,domain_ia)*model_Ii(x[2],x[3],x[4],x[5],ia_ug1_ranges,Ug2,domain_ia)

        l_Ia  = (IA_MATRIX - CALC_IA_MATRIX)
        l_Ia = l_Ia.reshape(1,l_Ia.size)[0]
        l_Ia = mse(l_Ia[np.isfinite(l_Ia)])
        # for idx, ia in enumerate(data[0]):
        #     l_Ia  += mse(ia*np.ones(domain.size) - model_A(x[0],x[1],Ug2,domain)*model_Ii(x[2],x[3],x[4],x[5],ia_ug1_ranges[idx],Ug2,domain))
            # l_Ia  += mse(ia*np.ones(domain.size) - x[0]*model_Ii(x[1],x[2],x[3],x[4],ia_ug1_ranges[idx],Ug2,domain))
            # l_Ia  += mse(ia*np.ones(domain.size) - model_A(x[0],x[1],Ug2,domain)*model_Ii(x[2],x[3],x[4],x[5],ia_ug1_ranges[idx],Ug2,domain))

        # for idx, ig2 in enumerate(data[1]):
        #     # plt.plot(domain[:ig2_domain_limits[idx]],ig2_ug1_ranges[idx][:ig2_domain_limits[idx]])
        #     # plt.show()
            # l_Ig2 += mse(ig2*np.ones(ig2_domain_limits[idx]) - (1-model_A(x[0],x[1],Ug2,domain[:ig2_domain_limits[idx]]))*model_Ii(x[2],x[3],x[4],x[5],ig2_ug1_ranges[idx][:ig2_domain_limits[idx]],Ug2,domain[:ig2_domain_limits[idx]]))
            # l_Ig2 += (ig2*np.ones(ig2_domain_limits[idx]) - (1-model_A(x[0],x[1],Ug2,domain[:ig2_domain_limits[idx]]))*model_Ii(x[2],x[3],x[4],x[5],ig2_ug1_ranges[idx][:ig2_domain_limits[idx]],Ug2,domain[:ig2_domain_limits[idx]]))

        print(l_Ia, l_Ig2)
        return l_Ia #+ l_Ig2

    # result = sp.optimize.minimize(objective,params,method="Nelder-Mead",options={'maxiter':1000000,'maxfev':1000000},bounds=bounds)
    result = sp.optimize.minimize(objective,params,method="BFGS")
    # result = sp.optimize.least_squares(objective,params)
    return result.x

# with alpha and beta, A is variable
params_0 = [0.9596,0.7408,2.99e-10,521.793,144.766,1.5]

# Current Values -----------------------------------------
# 197
# current_vals = [[0,5,10,20,30,40,60,80],[1,2,5,10,15,20]]
# 28
current_vals = [[0.001,1,5,15,30,50,80,100,120],[-0.2,-0.1,0,0.5,2,5,10,15]]

UA_IA  = np.tile(ua_domain, (len(current_vals[0]),1))
UA_IG2 = np.tile(ua_domain, (len(current_vals[1]),1))
opt_params = optimize_coeffs(params_0,current_vals,[UA_IA,UA_IG2],ia_arr,ig2_arr)

for i in range(opt_params.size):
    print(params_0[i],opt_params[i])


for i in range(len(current_vals[0])):
    opt_ia = model_A(opt_params[0],opt_params[1],Ug2,ua_domain)*model_Ii(opt_params[2],opt_params[3],opt_params[4],opt_params[5],ia_arr[i],Ug2,ua_domain)
    # opt_ia = opt_params[0]*model_Ii(opt_params[1],opt_params[2],opt_params[3],opt_params[4],ia_arr[i],Ug2,ua_domain)

    plt.figure(i)
    plt.plot(current_vals[0][i]*np.ones(ua_domain.size),label=f"Ideal Ia : {current_vals[0][i]:.1f} A")
    plt.plot(opt_ia,label=f"Real Ia : {current_vals[0][i]:.1f} A")
    plt.legend()
    plt.grid()

plt.show()

def predict_ia(params, ug2, ua, ug1):
    alpha, beta, K, mu_c, mu_s, exp_coeff = params
    A = model_A(alpha, beta, ug2, ua)
    Ii = model_Ii(K, mu_c, mu_s, exp_coeff, ug1, ug2, ua)
    return A * Ii

# Example ranges. Adjust these to match your datasheet plot.
ua_min = np.nanmin(ua_domain)
ua_max = np.nanmax(ua_domain)

ug1_min = ug1_lims[0]
ug1_max = ug1_lims[1]

ua_grid = np.linspace(ua_min, ua_max, 300)
ug1_grid = np.linspace(ug1_min, ug1_max, 300)

UA, UG1 = np.meshgrid(ua_grid, ug1_grid)

IA_grid = predict_ia(opt_params, Ug2, UA, UG1)
# IA_grid = predict_ia(params_0, Ug2, UA, UG1)

plt.figure(figsize=(9, 7))

levels = current_vals[0]

cs = plt.contour(
    UA,
    UG1,
    IA_grid,
    levels=levels,
    linewidths=1.2,
)

plt.clabel(cs, inline=True, fontsize=8, fmt="IA = %g A")

ug1_0A_curve = ia_arr[0]
ug1_5A_curve = ia_arr[1]
ug1_10A_curve = ia_arr[2]
ug1_20A_curve = ia_arr[3]
ug1_30A_curve = ia_arr[4]
ug1_40A_curve = ia_arr[5]
ug1_60A_curve = ia_arr[6]
ug1_80A_curve = ia_arr[7]

mask = np.isfinite(ug1_0A_curve)
mask = np.isfinite(ug1_5A_curve)
mask = np.isfinite(ug1_10A_curve)
mask = np.isfinite(ug1_20A_curve)
mask = np.isfinite(ug1_30A_curve)
mask = np.isfinite(ug1_40A_curve)
mask = np.isfinite(ug1_60A_curve)
mask = np.isfinite(ug1_80A_curve)

for idx, i in enumerate(ia_arr):
    mask = np.isfinite(i)
    plt.plot(
        ua_domain[mask],
        i[mask],
        "o",
        markersize=3,
        label=f"Extracted {current_vals[0][idx]} A points",
    )

plt.xlabel("UA")
plt.ylabel("UG1")
plt.title("Datasheet-style IA contours")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
