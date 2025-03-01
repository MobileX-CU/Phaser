"""
Camera settings for each Allied vision camera in the dual camera setup, 
including separate, lower exposure and gain settings for mapping.
"""

import vmbpy

dual_cam_settings = {
    "cam1" : {
        "pixel_format" : vmbpy.PixelFormat.Mono8,
        "auto_exposure" : False,
        "checkerboard_exposure_us" : 30000,
        "checkerboard_gain_db" : 28,
        "scanning_exposure_us" : 6000,
        "scanning_gain_db" : 10,
        "mapping_exposure_us" : 10000,  
        "mapping_gain_db" : 10,  
        "deploy_exposure_us" : 8000,
        "deploy_gain_db" : 20,#22,
        "white_balance" : 'Once',
        "binning" : 1,
        "reverse_y" : False,
        "reverse_x" : False,
        "pad_top" : 0,  # Padding is from [0..1] relative to the left, right, top, bottom extents
        "pad_bottom" : 0,
        "pad_left" : 0,
        "pad_right" : 0
    },
    "cam2" : {
        "pixel_format" : vmbpy.PixelFormat.Mono8,
        "auto_exposure" : False,
        "checkerboard_exposure_us" : 20000,
        "checkerboard_gain_db" : 10,
        "scanning_exposure_us" : 3500,
        "scanning_gain_db" : 4,
        "mapping_exposure_us" : 6000,  
        "mapping_gain_db" : 10,  
        "deploy_exposure_us" : 800,
        "deploy_gain_db" : 8, #18,
        "white_balance" : 'Once',
        "binning" : 1,
        "reverse_y" : False,
        "reverse_x" : False,
        "pad_top" : 0,
        "pad_bottom" : 0,
        "pad_left" : 0,
        "pad_right" : 0,
    }
}