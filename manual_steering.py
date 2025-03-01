"""
Manually enter steering angles or commands
Utility script for testing and debugging purposes, not actual operation.
"""


import lib.mirror as mirror
import pickle
import numpy as np
import traceback


def func_degree2(theta, phi, p00, p10, p01, p20, p11, p02):
    """
    Polynomial function of degree 2, defined by the coefficients p*
    
    Parameters:
    data : 2xN array 
        Array of points, where each column is a point in 2D space
    p00, p10, p01, p20, p11, p02: ints
        Coefficients of the polynomial function

    Returns:
    z : 1xN array 
        Array of z values for each point, computed by the polynomial function

    """
    return p00 + p10*theta+ p01*phi + p20*theta**2 + p11*theta*phi + p02*phi**2 


DISPLAY = True


mir_x_popt = np.load("params/mir_x_popt_jul8.npy")
mir_y_popt = np.load("params/mir_y_popt_jul8.npy")

# set up mirror
scale = 1 #IMPORTANT: make sure the scale is the same as the one used for mapping
sn = 'BPAA1034'
loc = 'lpd_v3'
mir = mirror.SerialMirror(sn, loc, '/dev/cu.usbmodem00000000001A1', range_scale=scale, mirrors_csv_path='mirrors.csv')
mir.enable()

try:
    while True:
        # raw_inp = input("Enter angle in form theta, phi")
        # theta, phi = raw_inp.split(",")
        # theta = float(theta.strip(" "))
        # phi = float(phi.strip(" "))
        # print("Got angles ", theta, phi)

        # mir_x = func_degree2(theta, phi, *mir_x_popt)
        # mir_y = func_degree2(theta, phi, *mir_y_popt)
        
        mir_comamand = input("Enter command in form x, y")
        mir_x, mir_y = mir_comamand.split(",")
        mir_x = float(mir_x.strip(" "))
        mir_y = float(mir_y.strip(" "))
        print("Steering to ", mir_x, mir_y)
        mir.steer((mir_x, mir_y), expect='OK')
    
except KeyboardInterrupt:
    pass
except Exception as e:
    print(traceback.format_exc())

# Cleanup
mir.disable()
