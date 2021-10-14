import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import TC_only as TC
import imageio
import pickle

# Constants
G = 6.674e-8 #cgs
c = 3e10 #cm/s
delta_m = 2.306e-27 #g
k_B = 1.380649e-16 #erg/K
T_d = 9.28e9 #K
m_p = 1.672621911e-24 #grams
m_e = 9.1093837015e-28 #grams
sigma_T = 6.6524587e-25 #cm^2
M_solar_to_g = 2e33 #g
years_to_s = 3.154e7 #s
Mpc_in_cm = 3.086e24 #cm

# Cosmology

Omega_m_0 = 0.28
Omega_b_0 = 0.0486
Omega_CDM_0 = Omega_m_0 - Omega_b_0
Omega_r_0 = 9.3e-5
Omega_k_0 = 0.
Omega_L = 1-Omega_m_0-Omega_r_0
H_0 =  2.2e-18 #s^-1

#------------------------------
## Changes of variables

def x(a):
    return np.log(a)

def a(x):
    return np.exp(x)
    

#------------------------------
## 

# Hubble fac
def H_a(a):
    H_a = (H_0**2 * (Omega_r_0/(a)**4 + Omega_m_0 / (a)**3 + Omega_k_0 / (a)**2 + Omega_L))**0.5
    return H_a

def H(x):
    H_x = (H_0**2 * (Omega_r_0/(np.exp(x))**4 + Omega_m_0 / (np.exp(x))**3 + Omega_k_0 / (np.exp(x))**2 + Omega_L))**0.5
    return H_x
    
def H_(x):
    H_x = (H_0**2 * (Omega_r_0/(np.exp(x))**4 + Omega_m_0 / (np.exp(x))**3 + Omega_k_0 / (np.exp(x))**2 + Omega_L))**0.5
    return np.exp(x)*H_x

def dH_dx(x):
    dHdx = np.exp(x)*H_0*(-2*np.exp(-2*x)*Omega_k_0 - 3*np.exp(-3*x)*Omega_m_0-4*np.exp(-4*x)*Omega_r_0)/(2.*((Omega_r_0/(np.exp(x))**4 + Omega_m_0 / (np.exp(x))**3 + Omega_k_0 / (np.exp(x))**2 + Omega_L))**0.5) + H_(x)
    return dHdx   

# Scale fac    
def z(a):
    return (1./ a) - 1.  
    
def eta(x):
    x_r = np.linspace(-25,x,10000)
    c_H_r = c/H_(x_r)
    c_H_r[0] = 0
    return cumtrapz(c_H_r,x_r)[-1]

# Critical density
def rho_crit_a(a):
    rho_c = 3 * H_a(a)**2 / (8 * np.pi * G)
    return rho_c
    
def rho_crit(x):
    rho_c = 3 * H(x)**2 / (8 * np.pi * G)
    return rho_c

# Mean photon density
def rho_gamma_a(a):
    rho_g = (Omega_r_0/(a)**4)*rho_crit_a(a) / (Omega_r_0/(a)**4 + Omega_m_0 / (a)**3 + Omega_k_0 / (a)**2 + Omega_L)
    return rho_g
    
def rho_gamma(x):
    rho_g = (Omega_r_0/(np.exp(x))**4)*rho_crit(x) / (Omega_r_0/(np.exp(x))**4 + Omega_m_0 / (np.exp(x))**3 + Omega_k_0 / (np.exp(x))**2 + Omega_L)
    return rho_g

# Mean baryon density    
def rho_b_a(a):
    rho_b = (Omega_b_0/(a)**3)*rho_crit_a(a)/ (Omega_r_0/(a)**4 + Omega_m_0 / (a)**3 + Omega_k_0 / (a)**2 + Omega_L)
    return rho_b
    
def rho_b(x):
    rho_b = (Omega_b_0/(np.exp(x))**3)*rho_crit(x)/ (Omega_r_0/(np.exp(x))**4 + Omega_m_0 / (np.exp(x))**3 + Omega_k_0 / (np.exp(x))**2 + Omega_L)
    return rho_b
    
# Approximate background electron density (assuming all is ionized)    
def n_e_a(a):
    Y_p = 0.24 
    ne = (1-Y_p)*rho_b_a(a)/m_p + Y_p*rho_b_a(a)/(4.*m_p)
    return ne

def n_e(x):
    Y_p = 0.24 
    ne = (1-Y_p)*rho_b(x)/m_p + Y_p*rho_b(x)/(4.*m_p)
    return ne
    
# R = 4/3 rho_gamma / rho_baryon
def R_(a):
    return (4./3.) * Omega_r_0 / (Omega_b_0 * a)

def R(x):
    return (4./3.) * Omega_r_0 / (Omega_b_0 * np.exp(x))
    
# Sound speed squared (can we make a simpler approx. at radiation domination?)
#def c_s_2(a):
    #da = 0.0001
    #mu = 2./3. #check this
    #cs2 = k_B * T_b / mu * (1. - (1./3.) * (np.log(T_b(a)) - np.log(T_b(a - da))) / (np.log(a) - np.log(H(a)*a)) 
#    cs2 = (1./3.) * c #approx
#    return cs2

#Optical depth: we use an analytic approximation for early times and interpolate from pre-calculated results for later times

#These tables are x,log10(tau)
tau_dat = np.genfromtxt('tau_x.dat')
tau_prime_dat = np.genfromtxt('tau_prime_ext.dat')
tau_primeprime_dat = np.genfromtxt('tau_primeprime_ext.dat')

tau_interp = interp1d(tau_dat[:,0],tau_dat[:,1])
tau_prime_interp = interp1d(tau_prime_dat[:,0],tau_prime_dat[:,1])
tau_primeprime_interp = interp1d(tau_primeprime_dat[:,0],tau_primeprime_dat[:,1])


#Shown at the beginning of Sec 5.7 in Ma + Bertschinger (1995). ~ interaction timescale
def tau_c(a):
    tc = 1./ (a * n_e_a(a) * sigma_T)
    return tc
    
def tau_prime(x):#This is an approximation for before recombination
    #tau_p = -n_e(x)*sigma_T*c / H(x)
    tau_p = -10**tau_prime_interp(x)
    return tau_p

def tau_prime2(x):#This is an approximation for before recombination
    tau_p = -n_e(x)*sigma_T*c / H(x)
    return tau_p
    
def tau_primeprime(x): #This is a rough approximation for before recombination and is not rigorous
    #tau_pp = -tau_prime(x)
    tau_pp = 10**tau_primeprime_interp(x)
    return tau_pp



#------------------------------
## Solving Boltzmann equations post-recombination: Get the initial conditions from tight_coupling_approx
#Initial conditions


#Here we use coupled ODEs to describe the post-recombination regime. We solve this equation by describing a set of ten 1st order equations: let Y be the vector [delta_CDM, v_CDM, delta_b, v_b, Phi, Theta_0, Theta_1, Theta_2, Theta_3, Theta_4].  

#Here we set the initial conditions for [delta_CDM, v_CDM, delta_b, v_b,Phi,Theta_0,Theta_1,Theta_2,Theta_3,Theta_4].
def set_ICs(x,k,TC):
    x_start = -22
    delta_CDM, v_CDM, delta_b, v_b, Phi, Theta_0, Theta_1 = TC[-1]
    Theta_2 = -(20.*c*k)/(45.*H_(x)*tau_prime(x))*Theta_1
    Theta_3 = -(3.*c*k)/(7.*H_(x)*tau_prime(x))*Theta_2
    Theta_4 = -(4.*c*k)/(9.*H_(x)*tau_prime(x))*Theta_3
    return delta_CDM, v_CDM, delta_b, v_b, Phi, Theta_0, Theta_1, Theta_2, Theta_3, Theta_4
    
#The vector we are solving for is Y = [delta_CDM, v_CDM, delta_b, v_b, Phi, Theta_0, Theta_1, Theta_2, Theta_3,Theta_4].
#dYdx = [d/dx delta_CDM, d/dx v_CDM, d/dx delta_b, d/dx v_b, d/dx Phi, d/dx Theta_0, d/dx Theta_1, d/dx Theta_2, d/dx Theta_3, d/dx Theta_4]
def dY_dx(Y,x,k):
    dYdx = np.zeros(10)

    Psi = -Y[4] - (12. * H_0**2. /(c**2 * k**2 * (np.exp(x))**2)) * (Omega_r_0 * Y[7])
         
    dYdx[0] = c*k/H_(x) * Y[1] - 3.*dYdx[4]
    
    dYdx[1] = -Y[1] - (c*k/H_(x))*Psi
    
    dYdx[2] = c*k/H_(x) * Y[3] - 3.*dYdx[4]
    
    dYdx[3] = -Y[3] - (c*k/H_(x))*Psi + tau_prime(x)*R(x)*(3.*Y[6] + Y[3])
    
    dYdx[4] = Psi - (c**2 * k**2 / (3. * H_(x)**2)) * Y[4] + (H_0**2 / (2. * H_(x)**2))*(Omega_CDM_0/(np.exp(x))*Y[0] + Omega_b_0/(np.exp(x))*Y[2] + 4.*Omega_r_0/((np.exp(x))**2)*Y[5])
    
    dYdx[5] = - (c*k/H_(x)) * Y[6] - dYdx[4]
    
    dYdx[6] = (c*k/(3.*H_(x))) * Y[5] - (2.*c*k/(3.*H_(x))) * Y[7] + (c*k/(3.*H_(x))) * Psi + tau_prime(x) * (Y[6] + (1./3.) * Y[3])
    
    dYdx[7] = 2.*c*k/(5.*H_(x)) * Y[6] - 3.*c*k/(5.*H_(x)) * Y[8] + tau_prime(x) * (Y[7] - (1./10.) * Y[7])
    
    dYdx[8] = 3.*c*k/(7.*H_(x)) * Y[7] - 4.*c*k/(7.*H_(x)) * Y[9] + tau_prime(x) * (Y[8])
    
    dYdx[9] = c*k/H_(x) * Y[8] - c*(5./(H_(x)*eta(x))) * Y[9] + tau_prime(x) * Y[9]
    
    return dYdx
    


#------------------------------
x_start = -22
x_end = -5
x_arrs_ = np.linspace(x_start,x_end,2000)
 
k_vals = np.logspace(-6.9,11.6,num=1500,base = np.e)
db_k_array_mod = np.zeros((len(k_vals), len(x_arrs_)))
dm_k_array_mod = np.zeros((len(k_vals), len(x_arrs_)))
x_k_array_mod = np.zeros((len(k_vals), len(x_arrs_)))
M_BH = 2e38 #g
z_BBN = 4e8


for i in range(len(k_vals)):
    print(i)
    k_new = k_vals[i]/Mpc_in_cm
    x_switch = TC.TC_end(k_new,x_arrs_)
    T_C = TC.setup_for_post_TC(x_start,x_arrs_,k_new)
    ICs = set_ICs(x_arrs_[x_switch],k_new, T_C)
    
    
    
    P_R = odeint(dY_dx, ICs, x_arrs_[x_switch:], args=(k_new,))
   
    d_b_k = np.concatenate([T_C[:, 2],P_R[:, 2]])
    d_m_k = np.concatenate([T_C[:, 0],P_R[:, 0]])
    db_k_array_mod[i] = d_b_k
    dm_k_array_mod[i] = d_m_k
    x_k_array_mod[i] = x_arrs_


file_add = "DMdelta"

file = open("db_k_array_" + file_add + ".dat", "wb")
pickle.dump(db_k_array_mod, file)
file.close()

file2 = open("dm_k_array_" + file_add + ".dat", "wb")
pickle.dump(dm_k_array_mod, file2)
file2.close()


file_x = open("x_k_array_" + file_add + ".dat", "wb")
pickle.dump(x_k_array_mod, file_x)
file_x.close

file_k = open("k_array_" + file_add + ".dat", "wb")
pickle.dump(k_vals, file_k)
file_k.close


