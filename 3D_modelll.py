import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

start_time = datetime.now()

def save(name='', format='png'):
    pwd = os.getcwd()
    basePath = './pictures/'
    iPath = './pictures/{}'.format(format)
    if not os.path.exists(basePath):
        os.mkdir(basePath)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, format), format='png')
    os.chdir(pwd)

def Gauss_Hermite_mods(ghm_number, ghm_u):
    if ghm_number == 0:
        return 1
    if ghm_number == 1:
        return 2*ghm_u
    return 2*ghm_u*Gauss_Hermite_mods(ghm_number - 1, ghm_u) - 2*(ghm_number-1)*Gauss_Hermite_mods(ghm_number - 2, ghm_u)

def mt_wawe_E(z, alpha, tau):
    lenth = 400*10**(-9)
    E = np.sin((z + 70*alpha + tau) *2*np.pi/lenth )
    return E

def mt_wawe_H(z,alpha,tau):
    Z = 376.7
    H = -mt_wawe_E(z, alpha, tau)*1/Z
    return H

def create_Gauss_Hermite_emittion(cghm_x, cghm_y, cghm_z, cghm_tau, cghm_k, cghm_omega, cghm_l, cghm_m, cghm_left_boundary, cghm_right_boundary, cghm_up_boundary):
    cghm_W = cghm_right_boundary - cghm_left_boundary
    cghm_z_0 = cghm_up_boundary
    cghm_R = (cghm_z_0**2)/cghm_z_0
    cghm_psi = np.arctan(cghm_z/cghm_z_0)
    cghm_u = (2**0.5)*cghm_x/cghm_W
    cghm_v = (2**0.5)*cghm_y/cghm_W
    cghm_G_l = Gauss_Hermite_mods(cghm_l, cghm_u)
    cghm_G_m = Gauss_Hermite_mods(cghm_m, cghm_v)
    #return cghm_G_l*cghm_G_m*np.sin(-cghm_k*cghm_z - cghm_k*(cghm_x**2 + cghm_y**2)/cghm_R + (cghm_l+cghm_m + 1)*cghm_psi + cghm_omega*cghm_tau)
    return cghm_G_l * cghm_G_m * np.sin(cghm_k * cghm_z + cghm_k * (cghm_x ** 2 + cghm_y ** 2) / cghm_R - ( cghm_l + cghm_m + 1) * cghm_psi - cghm_omega * cghm_tau)

def define_init_e_field(dief_x, dief_y, dief_z, dief_tau, dief_k, dief_omega, dief_l, dief_m, dief_left_boundary, dief_right_boundary, dief_up_boundary):
    dief_e_field = create_Gauss_Hermite_emittion(dief_x, dief_y, dief_z, dief_tau, dief_k, dief_omega, dief_l, dief_m, dief_left_boundary, dief_right_boundary, dief_up_boundary)
    return dief_e_field

def define_init_h_field(dihf_x, dihf_y, dihf_z, dihf_tau, dihf_k, dihf_omega, dihf_l, dihf_m, dihf_left_boundary, dihf_right_boundary, dihf_up_boundary, dihf_impedance):
    dief_h_field = 1/dihf_impedance*define_init_e_field(dihf_x, dihf_y, dihf_z, dihf_tau, dihf_k, dihf_omega, dihf_l, dihf_m, dihf_left_boundary, dihf_right_boundary, dihf_up_boundary)
    return dief_h_field

def calculate_h_i(chi_past_h_i, chi_next_e_k, chi_previous_e_k, chi_next_e_j, chi_previous_e_j, chi_j_step, chi_k_step,
                  chi_time_step, chi_permeability, chi_magnetic_conductivity, chi_magnetic_current_x):
    Z = 376.7
    chi_h_i = chi_past_h_i - 1/Z*chi_time_step*((chi_next_e_k - chi_previous_e_k)/chi_j_step - (chi_next_e_j - chi_previous_e_j)/chi_k_step + chi_magnetic_conductivity*chi_past_h_i)
    return chi_h_i

def calculate_e_i(cei_past_e_i, cei_next_h_k, cei_previous_h_k, cei_nex_h_j, cei_previous_h_j, cei_j_step,
                  cei_k_step, cei_time_step, cei_permittivity, cei_electrical_conductivity, cei_electrical_current):
    Z = 376.7
    cei_e_i = cei_past_e_i + Z*cei_time_step*((cei_next_h_k - cei_previous_h_k)/cei_j_step -
                                                                                  (cei_nex_h_j - cei_previous_h_j)/cei_k_step - cei_electrical_conductivity*cei_past_e_i)
    return cei_e_i

# Формула парабалического затухания проводимости в слое поглощающей области
def parabolic_conduction_decay(pcd_max_conductivity, pcd_distance_to_border, pcd_layer_thickness):
    pcd_conductivity = pcd_max_conductivity*((pcd_distance_to_border)**2)/((pcd_layer_thickness)**2)
    return pcd_conductivity


x_left_board = 0
x_right_board = 10**(-6)
y_left_board = 0
y_right_board = 10**(-6)
z_down_board = 0
z_up_board = 2*10**(-6)

alpha = 10**(-8)
x_step = 2*alpha
y_step = 2*alpha
z_step = 2*alpha
tau_step = alpha # tau = c*t

x_point_number = int((x_right_board - x_left_board)/y_step) + 1
y_point_number = int((y_right_board - y_left_board)/y_step) + 1
z_point_number = int((z_up_board - z_down_board)/z_step) + 1
time_point_number = 60

electric_field_x = np.zeros((((2, x_point_number, y_point_number, z_point_number))))
electric_field_y = np.zeros((((2, x_point_number, y_point_number, z_point_number))))
electric_field_z = np.zeros((((2, x_point_number, y_point_number, z_point_number))))
magnetic_field_x = np.zeros((((2, x_point_number, y_point_number, z_point_number))))
magnetic_field_y = np.zeros((((2, x_point_number, y_point_number, z_point_number))))
magnetic_field_z = np.zeros((((2, x_point_number, y_point_number, z_point_number))))

result_x = int(3/4*x_point_number)
result_y = int(3/4*y_point_number)
result_z = z_point_number - 40

# Моды Гаусса-Эрмита
l = 0 # По оси Ох
m = 0 # По оси Оу

# Параметры излучения
c = 3*10**8
laser_length = 488*10**(-9)
omega = 2*np.pi*c/laser_length
wave_number = 2*np.pi/laser_length

# Поглощающая область
max_absorbing_electric_conductivity = 10**(-1)
max_absorbing_magnetic_conductivity = (376.7**2)*max_absorbing_electric_conductivity
width_of_the_absorbing_region = 5 # в количестве точек

for i in range(x_point_number):
    for j in range(y_point_number):
        for k in range(25, 76):
            electric_field_x[0][i][j][k] = mt_wawe_E(z_step*(k - 0.5), alpha, 0.5*tau_step)

for i in range(x_point_number):
    for j in range(y_point_number):
        for k in range(25, 75):
            magnetic_field_y[0][i][j][k] = mt_wawe_H(z_step*(k), alpha, 0)

e_x_result = np.zeros(int(time_point_number))
e_y_result = np.zeros(int(time_point_number))
e_z_result = np.zeros(int(time_point_number))
h_x_result = np.zeros(int(time_point_number))
h_y_result = np.zeros(int(time_point_number))
h_z_result = np.zeros(int(time_point_number))

Intencity = (electric_field_x[0][int(x_point_number/2)][int(y_point_number/2)]**2 + electric_field_y[0][int(x_point_number/2)][int(y_point_number/2)]**2 +
             electric_field_z[0][int(x_point_number/2)][int(y_point_number/2)]**2)**1

z = np.arange(0, z_point_number)
plt.ioff()
fig, ax = plt.subplots()
ax.plot(z, electric_field_x[0][int(x_point_number / 2)][int(y_point_number / 2)], 'rx', linestyle='solid')
ax.grid(True)
yticks = ax.get_yticks()
yy = np.arange(-1.2, 1.3, 0.2)
ylabels = []
for line in yy:
    ylabels.append('' % line)
ax.set_yticks(yy)
save('3D_mod' + str(0), format='png')
plt.show()
plt.savefig('3D_mod' + str(0))

for n in range(1,time_point_number):
    for i in range(1,x_point_number-1):
        for j in range(1, y_point_number-1):
            for k in range(1, z_point_number - 1):
                electric_conductivity = 0
                magnetic_conductivity = 0
                if (i <= width_of_the_absorbing_region or i >= x_point_number - 1 - width_of_the_absorbing_region) and (
                        j <= width_of_the_absorbing_region or j >= y_point_number - 1 - width_of_the_absorbing_region) and (
                (k <= width_of_the_absorbing_region or k >= z_point_number - 1 - width_of_the_absorbing_region)):
                    distance_to_border = min(i,x_point_number-1-i, j, y_point_number - 1 - j, k, z_point_number - 1 - k)
                    electric_conductivity = parabolic_conduction_decay(max_absorbing_electric_conductivity, distance_to_border, width_of_the_absorbing_region)
                    magnetic_conductivity = (376.7**2)*electric_conductivity
                magnetic_field_x[1][i][j][k] = calculate_h_i(magnetic_field_x[0][i][j][k], electric_field_z[0][i][j+1][k],
                                                          electric_field_z[0][i][j][k],
                                                          electric_field_y[0][i][j][k+1],
                                                          electric_field_y[0][i][j][k],  y_step, z_step, tau_step,
                                                          1, magnetic_conductivity, 0)
                magnetic_field_y[1][i][j][k] = calculate_h_i(magnetic_field_y[0][i][j][k], electric_field_x[0][i][j][k+1],
                                                          electric_field_x[0][i][j][k],
                                                          electric_field_z[0][i+1][j][k],
                                                          electric_field_z[0][i][j][k],  z_step, x_step, tau_step,
                                                          1, magnetic_conductivity, 0)
                magnetic_field_z[1][i][j][k] = calculate_h_i(magnetic_field_z[0][i][j][k], electric_field_y[0][i+1][j][k],
                                                          electric_field_y[0][i][j][k],
                                                          electric_field_x[0][i][j+1][k],
                                                          electric_field_x[0][i][j][k],  x_step, y_step, tau_step,
                                                          1, magnetic_conductivity, 0)
                electric_field_x[1][i][j][k] = calculate_e_i(electric_field_x[0][i][j][k], magnetic_field_z[1][i][j][k],
                                                          magnetic_field_z[1][i][j-1][k],
                                                          magnetic_field_y[1][i][j][k],
                                                          magnetic_field_y[1][i][j][k-1],  y_step, z_step, tau_step,
                                                          1, electric_conductivity, 0)
                electric_field_y[1][i][j][k] = calculate_e_i(electric_field_y[0][i][j][k], magnetic_field_x[1][i][j][k],
                                                          magnetic_field_x[1][i][j][k-1],
                                                          magnetic_field_z[1][i][j][k],
                                                          magnetic_field_z[1][i-1][j][k],  z_step, x_step, tau_step,
                                                          1, electric_conductivity, 0)
                electric_field_z[1][i][j][k] = calculate_h_i(electric_field_z[0][i][j][k], magnetic_field_y[1][i][j][k],
                                                          magnetic_field_y[1][i-1][j][k],
                                                          magnetic_field_x[1][i][j][k],
                                                          magnetic_field_x[1][i][j-1][k],  x_step, y_step, tau_step,
                                                          1, electric_conductivity, 0)


                electric_field_x[0][i][j][k] = electric_field_x[1][i][j][k]
                electric_field_y[0][i][j][k] = electric_field_y[1][i][j][k]
                electric_field_z[0][i][j][k] = electric_field_z[1][i][j][k]
                magnetic_field_x[0][i][j][k] = magnetic_field_x[1][i][j][k]
                magnetic_field_y[0][i][j][k] = magnetic_field_y[1][i][j][k]
                magnetic_field_z[0][i][j][k] = magnetic_field_z[1][i][j][k]

    if n % 1 == 0:
        print(electric_field_y[0][int(x_point_number/2)][int(y_point_number/2)])
        Intencity = (electric_field_x[0][int(x_point_number / 2)][int(y_point_number / 2)] ** 2 +
                     electric_field_y[0][int(x_point_number / 2)][int(y_point_number / 2)] ** 2 +
                     electric_field_z[0][int(x_point_number / 2)][int(y_point_number / 2)] ** 2) ** 1

        z = np.arange(0, z_point_number)
        plt.ioff()
        fig, ax = plt.subplots()
        ax.plot(z, electric_field_x[0][int(x_point_number / 2)][int(y_point_number / 2)], 'rx', linestyle='solid')
        ax.grid(True)
        yticks = ax.get_yticks()
        yy = np.arange(-1.2, 1.3, 0.2)
        ylabels = []
        for line in yy:
            ylabels.append('' % line)
        ax.set_yticks(yy)
        #ax.plot(z, Intencity, 'rx', linestyle='solid')
        save('3D_mod' + str(n), format='png')
        plt.show()
        plt.savefig('2D_mod' + str(n))


