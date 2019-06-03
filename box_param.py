"""
This records basic experimental parameters

"""

veo_105mm_scale_limit = 0.058 #mm/px
particle_diameter = 0.060 # mm
eta_estimated = 0.050 # mm
params_mid_box = {'L': 351.0, # mm
                  'l_limit': veo_105mm_scale_limit,
                  'l_particle': particle_diameter,
                  'blob_size': 100.0, # mm
                  'eta_estimated': eta_estimated
                 }

def get_box_params():
    return params_mid_box