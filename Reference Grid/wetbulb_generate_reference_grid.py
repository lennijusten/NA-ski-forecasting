# Calculate reference grid at specified resoulution
# WARNING: Long run times (5+ days)

# Author: Lennart Justen

import numpy as np
import matplotlib.pyplot as plt
import metpy.calc
from metpy.units import units
import pickle

# Grid Parameters
press_min = 51000 # Pa
press_max = 105000 # Pa
press_nlayers = 300

temp_min = -81.0 # degC
temp_max = 48.0 # degC
temp_res = 0.5

dewpoint_min = -81.0
dewpoint_max = 30.0 
dewpoint_res = 0.5

# LowRes Grid Parameters

# press_min = 51000 # Pa
# press_max = 105000 # Pa
# press_nlayers = 200

# temp_min = -81.0 # degC
# temp_max = 48.0 # degC
# temp_res = 1.0

# dewpoint_min = -81.0
# dewpoint_max = 30.0 
# dewpoint_res = 1.0

# Set range over all params
t = np.arange(temp_min, temp_max, temp_res)
td = np.arange(dewpoint_min, dewpoint_max, dewpoint_res)
p = np.linspace(press_min, press_max, press_nlayers)


tt,td,pp = np.meshgrid(t,td,p)

wetbulb = metpy.calc.wet_bulb_temperature(pp * units.Pa, tt * units.degC, td*units.degC)

with open('wetbulb_metpy_300_0.5_grid.pkl', 'wb') as f:
    pickle.dump(wetbulb, f)
