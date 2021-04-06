import os
import numpy as np
import matplotlib.pyplot as plt



''' Зададим исходную падающую волну '''

def define_init_e_field(dief_x, dief_alpha, dief_t, dief_c):
    dief_e_field = np.sin((dief_x - 50*dief_alpha + dief_c*dief_t)*np.pi/(8*dief_alpha))
    return dief_e_field

def define_init_h_field(dief_x, dief_alpha, dief_t, dief_c, dief_Z):
    dief_h_field = 1/dief_Z*define_init_e_field(dief_x, dief_alpha, dief_t, dief_c)
    return dief_h_field

# Формулы для вычисления компонент магнитного поля
def calculate_C_coefficient(ccc_permeability,ccc_time_step,ccc_magnetic_conductivity):
    ccc_c = (ccc_permeability/ccc_time_step - ccc_magnetic_conductivity/2)/(ccc_permeability/ccc_time_step - ccc_magnetic_conductivity/2)
    return ccc_c

def calculate_D_coefficiend(cdc_permeability,cdc_time_step,cdc_magnetic_conductivity):
    cdc_d = 1/(cdc_permeability/cdc_time_step - cdc_magnetic_conductivity/2)
    return cdc_d

def calculate_h_x(chx_past_h_x, chx_next_e_z, chx_previous_e_z, chx_y_step, chx_time_step, chx_permeability, chx_magnetic_conductivity, chx_magnetic_current_x):
    chx_c = calculate_C_coefficient(chx_permeability, chx_time_step, chx_magnetic_conductivity)
    chx_d = calculate_D_coefficiend(chx_permeability, chx_time_step, chx_magnetic_conductivity)
    chx_h_x = chx_c*chx_past_h_x - chx_d*((chx_next_e_z - chx_previous_e_z)/chx_y_step - chx_magnetic_current_x)
    return chx_h_x

def calculate_h_y(chy_past_h_x, chy_next_e_z, chy_previous_e_z, chy_x_step, chy_time_step, chy_permeability, chy_magnetic_conductivity, chy_magnetic_current_y):
    chx_c = calculate_C_coefficient(chy_permeability, chy_time_step, chy_magnetic_conductivity)
    chx_d = calculate_D_coefficiend(chy_permeability, chy_time_step, chy_magnetic_conductivity)
    chx_h_x = chx_c*chy_past_h_x - chx_d*((chy_next_e_z - chy_previous_e_z)/chy_x_step - chy_magnetic_current_y)
    return chx_h_x

# Формулы для вычисления компонент электрического поля
def calculate_Q_coefficient(cqc_permittivity,cqc_time_step,cqc_electrical_conductivity):
    cqc_q = (cqc_permittivity/cqc_time_step - cqc_electrical_conductivity/2)/(cqc_permittivity/cqc_time_step - cqc_electrical_conductivity/2)
    return cqc_q

def calculate_F_coefficient(cfc_permittivity,cfc_time_step, cfc_electrical_conductivity):
    cfc_f = 1/(cfc_permittivity/cfc_time_step - cfc_electrical_conductivity/2)
    return cfc_f

def calculate_e_z(cez_past_e_z, cez_next_h_y, cez_previous_h_y, cez_nex_h_x, cez_previous_h_x, cez_x_step,
                  cez_y_step, cez_time_step, cez_permittivity, cez_electrical_conductivity):
    cez_q = calculate_Q_coefficient(cez_permittivity, cez_time_step, cez_electrical_conductivity)
    cez_f = calculate_F_coefficient(cez_permittivity, cez_time_step, cez_electrical_conductivity)
    cez_e_z = cez_q*cez_past_e_z + cez_f*((cez_next_h_y - cez_previous_h_y)/cez_x_step -
                                                                                  (cez_nex_h_x - cez_previous_h_x)/cez_y_step)
    return cez_e_z

# Формула парабалического затухания проводимости в слое поглощающей области
def parabolic_conduction_decay(pcd_max_conductivity, pcd_coordinate, pcd_inside_boundary, pcd_outside_boundary):
    pcd_conductivity = pcd_max_conductivity*((pcd_coordinate - pcd_inside_boundary)**2)/((pcd_outside_boundary - pcd_inside_boundary)**2)
    return pcd_conductivity

''' 
Формулы для токов Гюйгенса на поверхности рассеивателя:
Обозначим P_1 x+плосткость, P_2 x- плоскость, P_3 y+ плоскость, P_4 y- плоскость, P_5 z+ плоскость, P_6 z- плоскость. 
'''

''' Для электрических токов '''
def surface_electric_current_y_on_P_1(secyp1_e_z, secyp1_x_step):
    return - secyp1_e_z/(secyp1_x_step)

def surface_electric_current_z_on_P_1(seczp1_e_y, secyp1_x_step):
    return seczp1_e_y/(secyp1_x_step)

def surface_electric_current_y_on_P_2(secyp2_e_z, secyp2_x_step):
    return secyp2_e_z/(secyp2_x_step)

def surface_electric_current_z_on_P_2(seczp2_e_y, secyp2_x_step):
    return - seczp2_e_y/(secyp2_x_step)

def surface_electric_current_x_on_P_3(secxp3_e_z, secxp3_y_step):
    return secxp3_e_z/(secxp3_y_step)

def surface_electric_current_y_on_P_3(secyp3_e_x, secyp3_y_step):
    return - secyp3_e_x/(secyp3_y_step)

def surface_electric_current_x_on_P_4(secxp4_e_z, secxp4_y_step):
    return - secxp4_e_z/(secxp4_y_step)

def surface_electric_current_y_on_P_4(secyp4_e_x, secyp4_y_step):
    return secyp4_e_x/(secyp4_y_step)

def surface_electric_current_x_on_P_5(secxp5_e_y, secxp5_z_step):
    return - secxp5_e_y/(secxp5_z_step)

def surface_electric_current_y_on_P_5(secyp5_e_x, secyp5_z_step):
    return secyp5_e_x/(secyp5_z_step)

def surface_electric_current_x_on_P_6(secxp6_e_y, secxp6_z_step):
    return secxp6_e_y/(secxp6_z_step)

def surface_electric_current_y_on_P_6(secyp6_e_x, secyp6_z_step):
    return - secyp6_e_x/(secyp6_z_step)


''' Для магнитных токов '''
def surface_magnetic_current_y_on_P_1(smcyp1_h_z, smcyp1_x_step):
    return smcyp1_h_z/(smcyp1_x_step)

def surface_magnetic_current_z_on_P_1(smczp1_h_y, smcyp1_x_step):
    return - smczp1_h_y/(smcyp1_x_step)

def surface_magnetic_current_y_on_P_2(smcyp2_h_z, smcyp2_x_step):
    return - smcyp2_h_z/(smcyp2_x_step)

def surface_magnetic_current_z_on_P_2(smczp2_h_y, smcyp2_x_step):
    return smczp2_h_y/(smcyp2_x_step)

def surface_magnetic_current_x_on_P_3(smcxp3_h_z, smcxp3_y_step):
    return - smcxp3_h_z/(smcxp3_y_step)

def surface_magnetic_current_y_on_P_3(smcyp3_h_x, smcyp3_y_step):
    return smcyp3_h_x/(smcyp3_y_step)

def surface_magnetic_current_x_on_P_4(smcxp4_h_z, smcxp4_y_step):
    return smcxp4_h_z/(smcxp4_y_step)

def surface_magnetic_current_y_on_P_4(smcyp4_h_x, smcyp4_y_step):
    return - smcyp4_h_x/(smcyp4_y_step)

def surface_magnetic_current_x_on_P_5(smcxp5_h_y, smcxp5_z_step):
    return smcxp5_h_y/(smcxp5_z_step)

def surface_magnetic_current_y_on_P_5(smcyp5_h_x, smcyp5_z_step):
    return - smcyp5_h_x/(smcyp5_z_step)

def surface_magnetic_current_x_on_P_6(smcxp6_h_y, smcxp6_z_step):
    return - smcxp6_h_y/(smcxp6_z_step)

def surface_magnetic_current_y_on_P_6(smcyp6_h_x, smcyp6_z_step):
    return smcyp6_h_x/(smcyp6_z_step)


''' 
Зададим цилиндр и поглощающую область на границе. Пусть внутренние точки цилиндра будут с индексом 1, внешние - с индексом -1, граничные - с индексом 0, 
    точки внутри поглощающей области - с индексом 2, точки на границе поглощающей области с индексом 3 
'''
def create_grid_index(cgi_length, cgi_radius, cgi_left_center_x, cgi_left_center_y, cgi_left_center_z, cgi_problem_space):
    cgi_number_of_x_points = int((cgi_problem_space[1] - cgi_problem_space[0])/cgi_problem_space[2]) + 1
    cgi_number_of_y_points = int((cgi_problem_space[4] - cgi_problem_space[3])/cgi_problem_space[5]) + 1
    cgi_number_of_z_points = int((cgi_problem_space[7] - cgi_problem_space[6])/cgi_problem_space[8]) + 1
    cgi_absorption_layer_points = 5
    cgi_grid_array = [[[[0,0,0,-1] for i in range(cgi_number_of_x_points)] for j in range(cgi_number_of_y_points)] for k in range(cgi_number_of_z_points)]
    for k in range(cgi_number_of_z_points):
        for j in range(cgi_number_of_y_points):
            for i in range(cgi_number_of_x_points):
                if (j*cgi_problem_space[5] >= cgi_left_center_y - cgi_radius + cgi_problem_space[5]) and (j*cgi_problem_space[5] <= cgi_left_center_y + cgi_radius - cgi_problem_space[5]):
                    if (i*cgi_problem_space[2] - cgi_left_center_x)**2 + (k*cgi_problem_space[8] - cgi_left_center_z)**2 <= (cgi_radius - cgi_problem_space[2])**2:
                        cgi_grid_array[k][j][i][3] = 1
                if ((j * cgi_problem_space[5] < cgi_left_center_y - cgi_radius + cgi_problem_space[5]) and (j * cgi_problem_space[5] >= cgi_left_center_y - cgi_radius)) or \
                        ((j * cgi_problem_space[5] > cgi_left_center_y + cgi_radius - cgi_problem_space[5]) and (j * cgi_problem_space[5] <= cgi_left_center_y + cgi_radius)):
                    if ((i*cgi_problem_space[2] - cgi_left_center_x)**2 + (k*cgi_problem_space[8] - cgi_left_center_z)**2 > (cgi_radius - cgi_problem_space[2])**2) and \
                            ((i * cgi_problem_space[2] - cgi_left_center_x) ** 2 + (k * cgi_problem_space[8] - cgi_left_center_z) ** 2 <= (cgi_radius)**2):
                        cgi_grid_array[k][j][i][3] = 0
                if (i == cgi_absorption_layer_points) or (i == cgi_number_of_x_points - cgi_absorption_layer_points):
                    cgi_grid_array[k][j][i][3] = 3
                if (j == cgi_absorption_layer_points) or (j == cgi_number_of_y_points - cgi_absorption_layer_points):
                    cgi_grid_array[k][j][i][3] = 3
                if (k == cgi_absorption_layer_points) or (k == cgi_number_of_z_points - cgi_absorption_layer_points):
                    cgi_grid_array[k][j][i][3] = 3
                if (i < cgi_absorption_layer_points) or (i > cgi_number_of_x_points - cgi_absorption_layer_points):
                    cgi_grid_array[k][j][i][3] = 2
                if (j < cgi_absorption_layer_points) or (i > cgi_number_of_y_points - cgi_absorption_layer_points):
                    cgi_grid_array[k][j][i][3] = 2
                if (k < cgi_absorption_layer_points) or (k > cgi_number_of_z_points - cgi_absorption_layer_points):
                    cgi_grid_array[k][j][i][3] = 2
    return cgi_grid_array


def create_cylinder(cc_problem_space):
    cylinder_length = 0.01
    cylinder_radius = 0.0025
    cylinder_left_center_x_coordinate = 0.005
    cylinder_left_center_y_coordinate = 0.005
    cylinder_left_center_z_coordinate = 0.02
    return create_grid_index(cylinder_length, cylinder_radius, cylinder_left_center_x_coordinate, cylinder_left_center_y_coordinate, cylinder_left_center_z_coordinate, cc_problem_space)

def Gauss_Hermite_mods():
    if ghm_number == 0:
        return 1
    if ghm_number == 1:
        return ghm_u

def create_Gauss_Hermite_emittion():
    return






