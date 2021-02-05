import numpy as np
import matplotlib.pyplot as plt
import metpy.calc
from metpy.units import units
from tqdm import tqdm
import pickle


def create_grid_depression(temp_min, temp_max, ddp_min, ddp_max, pressure, resolution=1.0):
    t = np.arange(temp_min, temp_max, resolution)
    ddp = np.arange(ddp_min, ddp_max, resolution)
    tt, ddpddp = np.meshgrid(t, ddp)
    tdtd = tt - ddpddp

    pp = np.meshgrid([pressure] * len(t), [pressure] * len(ddp))
    return t, ddp, tt, ddpddp, tdtd, pp

def calc_wetbulb_onethird(tt, tdtd):
    return (2/3)*tt+(1/3)*tdtd

# wetbulb2 = metpy.calc.wet_bulb_temperature(pp[0]*units.Pa, tt*units.degC, tdtd*units.degC)

def residual(wetbulb1, wetbulb2):
    return wetbulb2-wetbulb1*units.degC


volume = []
wetbulb1_all = []
wetbulb2_all = []
for p in np.arange(50000, 105000, 250):
    t, ddp, tt, ddpddp, tdtd, pp = create_grid_depression(-35, 50, -30, 75, p)

    wetbulb1 = calc_wetbulb_onethird(tt, tdtd)
    wetbulb2 = metpy.calc.wet_bulb_temperature(pp[0] * units.Pa, tt * units.degC, tdtd * units.degC)
    wetbulb1_all.append(wetbulb1)
    wetbulb2_all.append(wetbulb2)


    res = residual(wetbulb1, wetbulb2)
    volume.append(res)

with open('residual_stack.pkl', 'wb') as f:
   pickle.dump(volume, f)

with open('wetbulb_onethird.pkl', 'wb') as f:
   pickle.dump(wetbulb1_all, f)

with open('wetbulb_metpy.pkl', 'wb') as f:
    pickle.dump(wetbulb2_all, f)

