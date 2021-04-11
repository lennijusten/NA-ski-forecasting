# Calculate Wetbulb Temperature

![Wetbulb Temperature over NA in 2082!](wetbulb_001.png =250x)


Calculating wetbulb temperature with `metpy.calc.wet_bulb_temperature` is an extremly long and expensive computation that is not well suited to run over large ensemble members ([see the docs](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.wet_bulb_temperature.html)). 

Instead we present a method to calculate wetbulb for an evenely spaced reference grid over the entire range of values for pressure, temperature, and dewpoint respectively. The script `wetbulb_generate_reference_grid.py` allows you to specify the range of parameters and the resolution at which to compute the reference grid. 

For example, the file `wetbulb_metpy_lowres_200_1.0_grid.pkl` contains the reference grid calculated from the following parameters:
```
press_min = 51000 # Pa
press_max = 105000 # Pa
press_nlayers = 200

temp_min = -81.0 # degC
temp_max = 48.0 # degC
temp_res = 1.0

dewpoint_min = -81.0
dewpoint_max = 30.0 
dewpoint_res = 1.0
```

Once the desired reference grid has been computed and saved (this can take several days of runtime), our method can find and return the nearest reference wetbulb value for any set of parameters [temperature, pressure, dewpoint]. 

To calculate wetbulb over an entire large ensemble member use `wetbulb_calc_ensemble_from_reference_grid.py`

To calculate wetbulb over any specified region and time use `wetbulb_calc_from_reference_grid.py`

This "lookup" method is a much quicker and less expensive process that still has good accuracy. The reference grid parameters shown above  resulted in a mean absolute error between `metpy.calc.wet_bulb_temperature` and our method of <0.2 degC for an entire year (n=336). The maximum residual was 0.6 degC. By calculating a finer resolution reference grid, one can achive even better performance. 

For information on the CESM Large Ensemble see www.cesm.ucar.edu/projects/community-projects/LENS/
