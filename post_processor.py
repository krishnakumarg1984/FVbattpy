############################################ Python Module Imports #####################################################
from __future__ import division                                                                                         # Force floating-point division
from __future__ import print_function                                                                                   # Force Python 3 compatability
print("Initialising post-processor..")
from sys import exit
import cPickle as pickle
import matplotlib
if matplotlib.get_backend() != 'Qt4Agg':
        matplotlib.use('Qt4Agg')
from matplotlib.widgets import Slider, Button, RadioButtons
try:
    from pylab import *
except ImportError:
    print("PyQt4 or PyQt5 module is not installed. Find and pip install the PyQt4-4.11.4-cp27-none-win_amd64.whl wheel")
    exit('Simulation terminated')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from os import path, makedirs
from scipy import interpolate
import warnings
from time import gmtime
from calendar import timegm
from datetime import timedelta
import pandas as pd

######################################################## Admin #########################################################
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")                                            # Supress all Matplitlib user warnings
postprocessor_clock_start_time = timegm(gmtime())                                                                       # Record the start time of this post-processing script

################################################### Control ############################################################
benchmarking = False                                                                                                    # False = don't plot the COMSOL comparison data. True = Do plot it
save_images = True
COMSOL_results_folder = 'COMSOL_results'
np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf)

################################################ Figure Settings #######################################################
plt.rcParams['savefig.dpi'] = 600
image_format = 'pdf'                                                                                                    # Possible formats are: pdf / svg / eps / png

fig_titles = [()]
vlinecolor = (174/256, 177/256, 168/256)                                                                                # Colour of lines defining the electrode-separator boundaries
Python_line_colour = (132/256, 201/256, 34/256)                                                                         # Colour of lines representing Python-FiPy model data
comparison_line_colour = (125/256, 125/256, 125/256)                                                                    # Colour of lines representing the comparison dataset
plot_peripherals_color = (68/256, 68/256, 68/256)                                                                       # Colour of plot axes

time_series_plot_line_style = '-'
ChangingPlot_line_style = '-'
time_series_plot_marker_style = ''
ChangingPlot_marker_style = 'o'

plot_aspect_ratio = (12, 27)                                                                                            # (height, width)
marker_size = 6
marker_facecolor = (174/256, 177/256, 168/256)
slider_bg_colour = 'white'                                                                                              # Background colour
slider_fg_colour = (132/256, 201/256, 34/256)                                                                           # Foreground colour
title_fontsize = 18
ticksize= 16
xy_labels_fontsize = 16
ax_y_labels = '$\phi_e$ (V)'
ax_x_labels = 'Displacement (dimensionless)'
neg_elec_sep_vline = 1
neg_elec_sep_vline_label = 1.015
pos_elec_sep_vline = 2
pos_elec_sep_vline_label = 2.015
slider_max_extension = 0.001                                                                                            # Necessary to elongate the slider range slightly, so that user can "select" the last value on the slider

################################## Function to Load COMSOL Data & Generate PKL #########################################
def import_comparison_data():
    comsol_x_cell = np.linspace(0.0, 3.0, (3.0 / 0.01)+1)                                                               # Generate the x-axis co-ordinates for the comparison dataset in each domain
    comsol_x_neg = np.linspace(0.0, 0.99, (1.0 / 0.01))
    comsol_x_pos = np.linspace(0.01, 1.0, (1.0 / 0.01))

    comsol_time = (pd.read_csv(COMSOL_results_folder + '\comsol_time.csv', sep=',', header=None, dtype=float)).values
    comsol_ce = (pd.read_csv(COMSOL_results_folder + '\comsol_c_e.csv', sep=',', header=None, dtype=float)).values
    comsol_phi_e = (pd.read_csv(COMSOL_results_folder + '\comsol_phi_e.csv', sep=',', header=None, dtype=float)).values
    # NEED TO ADD COMSOL Cs data here
    comsol_phi_s_neg = (pd.read_csv(COMSOL_results_folder + '\comsol_phi_s_neg.csv', sep=',', header=None, dtype=float)).values
    comsol_phi_s_pos = (pd.read_csv(COMSOL_results_folder + '\comsol_phi_s_pos.csv', sep=',', header=None, dtype=float)).values
    comsol_flux_density_j_neg = (pd.read_csv(COMSOL_results_folder + '\comsol_j_neg.csv', sep=',', header=None, dtype=float)).values
    comsol_flux_density_j_pos = (pd.read_csv(COMSOL_results_folder + '\comsol_j_pos.csv', sep=',', header=None, dtype=float)).values
    comsol_exchange_flux_density_j0_neg = (pd.read_csv(COMSOL_results_folder + '\comsol_j0_neg.csv', sep=',', header=None, dtype=float)).values
    comsol_exchange_flux_density_j0_pos = (pd.read_csv(COMSOL_results_folder + '\comsol_j0_pos.csv', sep=',', header=None, dtype=float)).values
    comsol_overpotential_neg = (pd.read_csv(COMSOL_results_folder + '\comsol_eta_neg.csv', sep=',', header=None, dtype=float)).values
    comsol_overpotential_pos = (pd.read_csv(COMSOL_results_folder + '\comsol_eta_pos.csv', sep=',', header=None, dtype=float)).values
    comsol_uocp_neg = (pd.read_csv(COMSOL_results_folder + '\comsol_Uocp_neg.csv', sep=',', header=None, dtype=float)).values
    comsol_uocp_pos = (pd.read_csv(COMSOL_results_folder + '\comsol_Uocp_pos.csv', sep=',', header=None, dtype=float)).values

    comsol_terminal_voltage = np.subtract(comsol_phi_s_pos[:,-1], comsol_phi_s_neg[:,0])                                # Compute the terminal voltage

################################## Package Data & Serialise for Faster Loading #########################################
    packaged_csv_comp_data = [comsol_x_cell, comsol_x_neg, comsol_x_pos, comsol_time, comsol_ce, comsol_phi_e,
                              comsol_phi_s_neg, comsol_phi_s_pos, comsol_flux_density_j_neg,
                              comsol_flux_density_j_pos, comsol_exchange_flux_density_j0_neg,
                              comsol_exchange_flux_density_j0_pos, comsol_overpotential_neg, comsol_overpotential_pos,
                              comsol_uocp_neg, comsol_uocp_pos, comsol_terminal_voltage]

    with open(COMSOL_results_folder + '\\' "pre_loaded_comparison_data.pkl", "w") as pre_loaded_comparison_data:
        pickle.dump(packaged_csv_comp_data, pre_loaded_comparison_data)

    return (comsol_x_cell, comsol_x_neg, comsol_x_pos, comsol_time, comsol_ce, comsol_phi_e, comsol_phi_s_neg,
            comsol_phi_s_pos, comsol_flux_density_j_neg, comsol_flux_density_j_pos, comsol_exchange_flux_density_j0_neg,
            comsol_exchange_flux_density_j0_pos, comsol_overpotential_neg, comsol_overpotential_pos, comsol_uocp_neg,
            comsol_uocp_pos, comsol_terminal_voltage)

################################## Import the Comparison Dataset from CSV or PKL #######################################
if benchmarking:
    print("Importing COMSOL data..")
    if path.isdir(COMSOL_results_folder):
        try:                                                                                                            # Try loading the pre-built Numpy arrays of comparison data (faster)
            with open(COMSOL_results_folder + '\\' "pre_loaded_comparison_data.pkl") as pre_loaded_comparison_data:     # .pkl file won't exist if you're post-processing sim results for the 1st time
                packaged_csv_comp_data = pickle.load(pre_loaded_comparison_data)
            print("PKL loaded. Now unpacking")
            (comsol_x_cell, comsol_x_neg, comsol_x_pos, comsol_time, comsol_ce, comsol_phi_e,
             comsol_phi_s_neg, comsol_phi_s_pos, comsol_flux_density_j_neg,
             comsol_flux_density_j_pos, comsol_exchange_flux_density_j0_neg,
             comsol_exchange_flux_density_j0_pos, comsol_overpotential_neg, comsol_overpotential_pos,
             comsol_uocp_neg, comsol_uocp_pos, comsol_terminal_voltage) = packaged_csv_comp_data
        except:                                                                                                         # If the serialised results don't exist yet, load CSVs & serialise for future use (slower)
            print("No PKL available. Loading CSVs..")
            (comsol_x_cell, comsol_x_neg, comsol_x_pos, comsol_time, comsol_ce, comsol_phi_e, comsol_phi_s_neg,
             comsol_phi_s_pos, comsol_flux_density_j_neg, comsol_flux_density_j_pos, comsol_exchange_flux_density_j0_neg,
             comsol_exchange_flux_density_j0_pos, comsol_overpotential_neg, comsol_overpotential_pos, comsol_uocp_neg,
             comsol_uocp_pos, comsol_terminal_voltage) = import_comparison_data()
    else:                                                                                                               # If no CSVs or PKL file of comparison data is available, continue, but disable comparison
        print('No COMSOL results folder found. Continuing post-processing without comparison data...')
        benchmarking = False

################################## Import the Comparison Dataset from CSV or PKL #######################################
def extrap_comparison_data():                                                                                           # Could make the code below neater by looping. Lots of repetition at the moment
    comsol_phi_s_neg_extrapd = np.zeros((comsol_phi_s_neg.shape[0], (comsol_phi_s_neg.shape[1] + 1)))                   # Pre-allocate a new array with an extra column to take the extrapolated values at new x
    comsol_phi_s_neg_extrapd[:, :-1] = comsol_phi_s_neg                                                                 # Insert the original comparison dataset into all but the last column of this new, larger array

    comsol_flux_density_j_neg_extrapd = np.zeros((comsol_flux_density_j_neg.shape[0],
                                                  (comsol_flux_density_j_neg.shape[1] + 1)))                            # Pre-allocate a new, larger array (1 extra column)
    comsol_flux_density_j_neg_extrapd[:, :-1] = comsol_flux_density_j_neg                                               # Insert the original comparison dataset, leaving last column empty

    comsol_exchange_flux_density_j0_neg_extrapd = np.zeros((comsol_exchange_flux_density_j0_neg.shape[0],
                                                            (comsol_exchange_flux_density_j0_neg.shape[1] + 1)))        # Pre-allocate a new, larger array (1 extra column)
    comsol_exchange_flux_density_j0_neg_extrapd[:, :-1] = comsol_exchange_flux_density_j0_neg                           # Insert the original comparison dataset, leaving last column empty

    comsol_overpotential_neg_extrapd = np.zeros((comsol_overpotential_neg.shape[0],
                                                 (comsol_overpotential_neg.shape[1] + 1)))                              # Pre-allocate a new, larger array (1 extra column)
    comsol_overpotential_neg_extrapd[:, :-1] = comsol_overpotential_neg                                                 # Insert the original comparison dataset, leaving last column empty

    comsol_uocp_neg_extrapd = np.zeros((comsol_uocp_neg.shape[0], (comsol_uocp_neg.shape[1] + 1)))                      # Pre-allocate a new, larger array (1 extra column)
    comsol_uocp_neg_extrapd[:, :-1] = comsol_uocp_neg                                                                   # Insert the original comparison dataset, leaving last column empty

    comsol_phi_s_pos_extrapd = np.zeros((comsol_phi_s_pos.shape[0], (comsol_phi_s_pos.shape[1] + 1)))                   # Pre-allocate a new, larger array (1 extra column)
    comsol_phi_s_pos_extrapd[:, 1:] = comsol_phi_s_pos                                                                  # Insert the original comparison dataset, leaving last column empty

    comsol_flux_density_j_pos_extrapd = np.zeros((comsol_flux_density_j_pos.shape[0],
                                                  (comsol_flux_density_j_pos.shape[1] + 1)))                            # Pre-allocate a new, larger array (1 extra column)
    comsol_flux_density_j_pos_extrapd[:, 1:] = comsol_flux_density_j_pos                                                # Insert the original comparison dataset, leaving last column empty

    comsol_exchange_flux_density_j0_pos_extrapd = np.zeros((comsol_exchange_flux_density_j0_pos.shape[0],
                                                            (comsol_exchange_flux_density_j0_pos.shape[1] + 1)))        # Pre-allocate a new, larger array (1 extra column)
    comsol_exchange_flux_density_j0_pos_extrapd[:, 1:] = comsol_exchange_flux_density_j0_pos                            # Insert the original comparison dataset, leaving last column empty

    comsol_overpotential_pos_extrapd = np.zeros((comsol_overpotential_pos.shape[0],
                                                 (comsol_overpotential_pos.shape[1] + 1)))                              # Pre-allocate a new, larger array (1 extra column)
    comsol_overpotential_pos_extrapd[:, 1:] = comsol_overpotential_pos                                                  # Insert the original comparison dataset, leaving last column empty

    comsol_uocp_pos_extrapd = np.zeros((comsol_uocp_pos.shape[0], (comsol_uocp_pos.shape[1] + 1)))                      # Pre-allocate a new, larger array (1 extra column)
    comsol_uocp_pos_extrapd[:, 1:] = comsol_uocp_pos                                                                    # Insert the original comparison dataset, leaving last column empty

############################ Extrapolate for the Value at the Electrode-Sep Boundaries #################################
    for timestep_row in xrange(0, comsol_phi_s_neg.shape[0]):                                                           # Assuming all comparison arrays here are same length (no. timesteps) as this one
        comsol_phi_s_neg_extrapd[timestep_row][-1] = np.poly1d(np.polyfit(comsol_x_neg[-2:],
                                                                          comsol_phi_s_neg[timestep_row][-2:],
                                                                          polynomial_order)
                                                               )(neg_sep_x_loc)
        comsol_flux_density_j_neg_extrapd[timestep_row][-1] = np.poly1d(np.polyfit(comsol_x_neg[-2:],
                                                                                   comsol_flux_density_j_neg[timestep_row][-2:],
                                                                                   polynomial_order)
                                                                        )(neg_sep_x_loc)
        comsol_exchange_flux_density_j0_neg_extrapd[timestep_row][-1] = np.poly1d(np.polyfit(comsol_x_neg[-2:],
                                                                                             comsol_exchange_flux_density_j0_neg[timestep_row][-2:],
                                                                                             polynomial_order)
                                                                                  )(neg_sep_x_loc)
        comsol_overpotential_neg_extrapd[timestep_row][-1] = np.poly1d(np.polyfit(comsol_x_neg[-2:],
                                                                                  comsol_overpotential_neg[timestep_row][-2:],
                                                                                  polynomial_order)
                                                                       )(neg_sep_x_loc)
        comsol_uocp_neg_extrapd[timestep_row][-1] = np.poly1d(np.polyfit(comsol_x_neg[-2:],
                                                                         comsol_uocp_neg[timestep_row][-2:],
                                                                         polynomial_order)
                                                              )(neg_sep_x_loc)
        comsol_phi_s_pos_extrapd[timestep_row][0] = np.poly1d(np.polyfit(comsol_x_pos[:2],
                                                                         comsol_phi_s_pos[timestep_row][:2],
                                                                         polynomial_order)
                                                              )(pos_sep_x_loc)
        comsol_flux_density_j_pos_extrapd[timestep_row][0] = np.poly1d(np.polyfit(comsol_x_pos[:2],
                                                                                  comsol_flux_density_j_pos[timestep_row][:2],
                                                                                  polynomial_order)
                                                                       )(pos_sep_x_loc)
        comsol_exchange_flux_density_j0_pos_extrapd[timestep_row][0] = np.poly1d(np.polyfit(comsol_x_pos[:2],
                                                                                            comsol_exchange_flux_density_j0_pos[timestep_row][:2],
                                                                                            polynomial_order)
                                                                                 )(pos_sep_x_loc)
        comsol_overpotential_pos_extrapd[timestep_row][0] = np.poly1d(np.polyfit(comsol_x_pos[:2],
                                                                                 comsol_overpotential_pos[timestep_row][:2],
                                                                                 polynomial_order)
                                                                      )(pos_sep_x_loc)
        comsol_uocp_pos_extrapd[timestep_row][0] = np.poly1d(np.polyfit(comsol_x_pos[:2],
                                                                        comsol_uocp_pos[timestep_row][:2],
                                                                        polynomial_order)
                                                             )(pos_sep_x_loc)

####################### Return the Comparison Datasets with Extra Column with Extrapd. Values ##########################
    return (comsol_phi_s_neg_extrapd, comsol_flux_density_j_neg_extrapd, comsol_exchange_flux_density_j0_neg_extrapd,
            comsol_overpotential_neg_extrapd, comsol_uocp_neg_extrapd, comsol_phi_s_pos_extrapd,
            comsol_flux_density_j_pos_extrapd, comsol_exchange_flux_density_j0_pos_extrapd,
            comsol_overpotential_pos_extrapd, comsol_uocp_pos_extrapd)

###################################### Call Extrapolation Fn. from Above ###############################################
if benchmarking:
    print("Processing comparison data..")
    polynomial_order = 1                                                                                                # Order of polynomial used in the extrapolation
    neg_sep_x_loc = 1.0                                                                                                 # Define location of anode-separator interface in anode domain of length 1.0
    pos_sep_x_loc = 0.0                                                                                                 # Define location of cathode-separator interface in cathode domain of length 1.0

    (comsol_phi_s_neg_extrapd, comsol_flux_density_j_neg_extrapd, comsol_exchange_flux_density_j0_neg_extrapd,
     comsol_overpotential_neg_extrapd, comsol_uocp_neg_extrapd, comsol_phi_s_pos_extrapd,
     comsol_flux_density_j_pos_extrapd, comsol_exchange_flux_density_j0_pos_extrapd,
     comsol_overpotential_pos_extrapd, comsol_uocp_pos_extrapd) = extrap_comparison_data()

    comsol_x_neg_extrapd = np.linspace(0.0, 1.0, (1.0 / 0.01)+1)                                                        # Re-generate the x-axis co-ordinates for each electrode, with one new datapoint in each
    comsol_x_pos_extrapd = np.linspace(0.0, 1.0, (1.0 / 0.01)+1)

########################################### Load Python Model data #####################################################
print("Importing Python/FiPy model results..")
try:
    with open("sim_results_location.pkl") as sim_results_location:                                                      # Load the serialised data file containing the name of the results folder
        directory = pickle.load(sim_results_location)
except:                                                                                                                 # Except if the serialised PKL file isn't available
    print('Warning: Could not find the sim_results_location.pkl file')
    print('Post-processing terminated')
    exit()
image_save_directory = directory + "/Plots/"                                                                            # Create a directory for saving plots
try:
    with open(directory + "\sim_results.pkl", 'r+') as pickled_sim_output:                                              # Load the serialised results that were produced by the simulation
        sim_results = pickle.load(pickled_sim_output)
except:                                                                                                                 # Except if the serialised PKL results file isn't available
    print('Warning: Could not find the sim_results.pkl file')
    print('Post-processing terminated')
    exit()
settings_dict, sim_timing_data, time_independent_data = sim_results[0], sim_results[1], sim_results[2]                  # Extract the categorised, packaged results from the imported serialised sim. results
time_dependent_data = sim_results[3]

########################################### Cleaning Loaded Data #######################################################
nr_neg, nr_pos = int(settings_dict['nr_neg']), int(settings_dict['nr_pos'])
nx_neg, nx_pos = int(settings_dict['nx_neg']), int(settings_dict['nx_pos'])

########################################### Split & Categorise Datasets ################################################
time_dependent_metadata = OrderedDict()                                                                                 # Remove all non-solution variable datasets from time_dependent_data & place them in another OrderedDict
for elem in deepcopy(time_dependent_data.items()[0:10]):                                                                # Make a copy of the first ten objects in the vars_logged list - the metadata objects
    time_dependent_metadata[elem[0]] = elem[1]                                                                          # For every dict. key-value pair, create an identical key in the new dict. & assign it the same paired value
time_dependent_metadata[(time_dependent_data.items()[-1][0])] = (time_dependent_data.items()[-1][1])                    # Also copy the final item in the time_dependent_data dict. to the new array

TwoD_time_dependent_data = OrderedDict()                                                                                # Begin splitting the time_dependent_data into that which is solved in a 2D domain & that in a 1D domain
for elem in deepcopy(time_dependent_data.items()[14:18]):                                                               # Make a copy of Cs (node & facevalue) objects in the vars_logged list
    TwoD_time_dependent_data[elem[0]] = elem[1]                                                                         # For every dict. key-value pair, create an identical key in the new dict. & assign it the same paired value

for key in time_dependent_data.keys():                                                                                  # Now remove these metadata objects from the time_dependent_data dict.
    if key in ['simtime', 'current_applied', 'sweep_count', 'sweep_duration', 'res_phi_s_neg', 'res_phi_s_pos',         # These are the metadata objects, sent from the vars_logged list, that have been copied to the time_dependent_metadata dict.
               'res_Ce_sim_cell', 'res_phi_e_sim_cell', 'res_Cs_neg', 'res_Cs_pos', 'gc_object_count']:
        del time_dependent_data[key]
    elif key in['cs_neg', 'cs_neg_Facevals', 'cs_pos', 'cs_pos_Facevals']:                                              # Now remove the time-dependent datasets from the time_dependent_data dict. before renaming it as the 1D object
        del time_dependent_data[key]

OneD_time_dependent_data = time_dependent_data                                                                          # Now purged of any 2D time-dependent datasets, time_dependent_data is renamed as the object for 1D data only

##################################### Process Python/FiPy Sim. Results #################################################
print("Processing Python/FiPy model data..")
for dataset in time_independent_data:
    dataset[1] = np.array(dataset[1])                                                                                   # Convert all time-independent datasets from lists to NumPy arrays

############################################# Trim Trailing Zeros ######################################################
# Not sure this code will work as intended. Should probably use Numpy's trim function on simtime..
# Then use the new length of simtime to determine how many rows to delete from all other arrays..
# Existing code will likely only work if only the very final timestep was missed, not more than that
if time_dependent_metadata['simtime'][-1] == 0:                                                                         # Check if any zeros remain in the simtime vector, which would indicate sim. terminated earlier than expected
    for key, array in time_dependent_metadata.items():                                                                  # For arrays in the time_dependent_metadata dictionary
        time_dependent_metadata[key] = np.delete(array, -1, axis=0)                                                     # Remove the row of data for the last timestep
    for key, array in OneD_time_dependent_data.items():                                                                 # And also do so for arrays in the OneD_time_dependent_data dictionary too
        OneD_time_dependent_data[key] = np.delete(array, -1, axis=0)
    for key, array in TwoD_time_dependent_data.items():                                                                 # Cycle through the Cs node & face values - i.e. the 2D datasets
        if key in ['cs_neg', 'cs_pos']:                                                                                 # For the Cs node values, which have already been formatted into an array of a sensible shape..
            TwoD_time_dependent_data[key] = array[0:-nr_neg,:]                                                          # Trim nr_neg number of rows - those rows corresponding to the solution at the final timestep - by copying out only rows to be kept
        else:                                                                                                           # The dataset is a facevalue dataset, which consists of stacked vectors
            TwoD_time_dependent_data[key] = np.delete(array, -1, axis=0)                                                # Remove only one row - that for the final timestep - since these are all vectors (not formatted arrays)

############################# Generate Vectors of Merged Face & Node Co-ordinates ######################################
# 1D Datasets, creating the blank vectors
sim_cell_axial_mesh_coordinates = [None]*(len(time_independent_data[1][1])+len(time_independent_data[0][1]))            # Generate a blank vector of correct length to take the merged x-axis cell-level axial locations
neg_axial_mesh_coordinates = [None]*(len(time_independent_data[3][1])+len(time_independent_data[2][1]))                 # Generate a blank vector of correct length to take the merged x-axis neg. electrode-level axial locations
pos_axial_mesh_coordinates = [None]*(len(time_independent_data[5][1])+len(time_independent_data[4][1]))                 # Generate a blank vector of correct length to take the merged x-axis pos. electrode-level axial locations

# 1D Datasets, merging the co-ordinates
sim_cell_axial_mesh_coordinates[::2] = time_independent_data[1][1]                                                      # Insert face value co-ordinates into every 2nd list position from start to end of list
sim_cell_axial_mesh_coordinates[1::2] = time_independent_data[0][1]                                                     # Insert FV cell-centre/node co-ordinates into every second list position from the 2nd element to end
neg_axial_mesh_coordinates[::2] = time_independent_data[3][1]                                                           # Insert face value co-ordinates into every 2nd list position from start to end of list
neg_axial_mesh_coordinates[1::2] = time_independent_data[2][1]                                                          # Insert FV cell-centre/node co-ordinates into every second list position from the 2nd element to end
pos_axial_mesh_coordinates[::2] = time_independent_data[5][1]                                                           # Insert face value co-ordinates into every 2nd list position from start to end of list
pos_axial_mesh_coordinates[1::2] = time_independent_data[4][1]                                                          # Insert FV cell-centre/node co-ordinates into every second list position from the 2nd element to end

################################ Generate Empty Arrays to Accept Merged Values #########################################
# 1D Datasets
array_length = OneD_time_dependent_data['ce'].shape[0]                                                                  # Use the Ce dataset length (no. timesteps) to specify new array lengths for all merged datasets
data_array_definitions_merged = [('ce_merged', array_length, (len(OneD_time_dependent_data['ce'][0])                    # List of tuples containing information used to create new arrays to hold merged face-node datasets
                                                              + len(OneD_time_dependent_data['ce_Facevals'][0]))),      # New array width is the sum of the number of node values & the number of facevalues
                                 ('phi_e_merged', array_length, (len(OneD_time_dependent_data['phi_e'][0])
                                                                 + len(OneD_time_dependent_data['phi_e_Facevals'][0]))),
                                 ('phi_s_neg_merged', array_length,
                                  (len(OneD_time_dependent_data['phi_s_neg'][0])
                                   + len(OneD_time_dependent_data['phi_s_neg_Facevals'][0]))),
                                 ('phi_s_pos_merged', array_length,
                                  (len(OneD_time_dependent_data['phi_s_pos'][0])
                                   + len(OneD_time_dependent_data['phi_s_pos_Facevals'][0]))),
                                 ('flux_density_j_neg_merged',
                                  array_length, (len(OneD_time_dependent_data['flux_density_j_neg'][0])
                                                 + len(OneD_time_dependent_data['flux_density_j_neg_Facevals'][0]))),
                                 ('flux_density_j_pos_merged',
                                  array_length, (len(OneD_time_dependent_data['flux_density_j_pos'][0])
                                                 + len(OneD_time_dependent_data['flux_density_j_pos_Facevals'][0]))),
                                 ('exchange_flux_density_j0_neg_merged',
                                  array_length, (len(OneD_time_dependent_data['exchange_flux_density_j0_neg'][0])
                                                 + len(OneD_time_dependent_data['exchange_flux_density_j0_neg_Facevals'][0]))),
                                 ('exchange_flux_density_j0_pos_merged',
                                  array_length, (len(OneD_time_dependent_data['exchange_flux_density_j0_pos'][0])
                                                 + len(OneD_time_dependent_data['exchange_flux_density_j0_pos_Facevals'][0]))),
                                 ('overpotential_neg_merged',
                                  array_length, (len(OneD_time_dependent_data['overpotential_neg'][0])
                                                 + len(OneD_time_dependent_data['overpotential_neg_Facevals'][0]))),
                                 ('overpotential_pos_merged',
                                  array_length, (len(OneD_time_dependent_data['overpotential_pos'][0])
                                                 + len(OneD_time_dependent_data['overpotential_pos_Facevals'][0]))),
                                 ('uocp_neg_merged',
                                  array_length, (len(OneD_time_dependent_data['uocp_neg'][0])
                                                 + len(OneD_time_dependent_data['uocp_neg_Facevals'][0]))),
                                 ('uocp_pos_merged',
                                  array_length, (len(OneD_time_dependent_data['uocp_pos'][0])
                                                 + len(OneD_time_dependent_data['uocp_pos_Facevals'][0]))),
                                 ]
OneD_time_dependent_data_merged = OrderedDict((variable, np.zeros((array_length, array_width), dtype=float))            # Loop using the above-data to generate the new blank arrays, large-enough to hold merged node & facevalues
                                  for variable, array_length, array_width in data_array_definitions_merged)

###################################### Merge Node & Face Values, 1D Datasets ###########################################
# 1D Datasets
merged_dataset_index = 0                                                                                                # Define a counter to help determine which emtpy array to grab to insert the merged values into
for orig_dataset_index in xrange(0, len(OneD_time_dependent_data.keys()), 2):                                           # Loop through the number of 1D datasets that need their face & node values merged
    empty_array = OneD_time_dependent_data_merged.values()[merged_dataset_index]                                        # Define the corresponding empty array to hold the merged values as "empty_array"
    for row_index, row in enumerate(empty_array):                                                                       # Loop through all timesteps, & at each timestep...
        empty_array[row_index][1::2] = OneD_time_dependent_data.values()[orig_dataset_index][row_index]                 # ...insert the node values into every 2nd row element, beginning with the 2nd
        empty_array[row_index][::2] = OneD_time_dependent_data.values()[orig_dataset_index+1][row_index]                # ...insert the face values into every 2nd row element, beginning with the 1st
    merged_dataset_index += 1

####################################### Calculate Terminal Voltage #####################################################
ground_terminal_potential = OneD_time_dependent_data['phi_s_neg_Facevals'][:,0]
live_terminal_potential = OneD_time_dependent_data['phi_s_pos_Facevals'][:,-1]
terminal_voltage = np.subtract(live_terminal_potential, ground_terminal_potential)

######################################### Background Plot Settings #####################################################
if benchmarking:
    OneD_aux_data = [(sim_cell_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['ce_merged']), np.amin(comsol_ce)), max(np.amax(OneD_time_dependent_data_merged['ce_merged']), np.amax(comsol_ce))],
                 False, False, '$C_e$ vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$C_e$ (mol/$m^3$)'),
                (sim_cell_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['phi_e_merged']), np.amin(comsol_phi_e)), max(np.amax(OneD_time_dependent_data_merged['phi_e_merged']), np.amax(comsol_phi_e))],
                 False, False, '$\phi_e$ vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\phi_e$ (V)'),
                (neg_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['phi_s_neg_merged']), np.amin(comsol_phi_s_neg_extrapd)), max(np.amax(OneD_time_dependent_data_merged['phi_s_neg_merged']), np.amax(comsol_phi_s_neg_extrapd))],
                 True, False, '$\phi_s$ Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\phi_s$ Neg. (V)'),
                (pos_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['phi_s_pos_merged']), np.amin(comsol_phi_s_pos_extrapd)), max(np.amax(OneD_time_dependent_data_merged['phi_s_pos_merged']), np.amax(comsol_phi_s_pos_extrapd))],
                 False, False, '$\phi_s$ Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\phi_s$ Pos. (V)'),
                (neg_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['flux_density_j_neg_merged']), np.amin(comsol_flux_density_j_neg_extrapd)), max(np.amax(OneD_time_dependent_data_merged['flux_density_j_neg_merged']), np.amax(comsol_flux_density_j_neg_extrapd))],
                 True, False, 'Flux Density $j$ Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$j$ Neg. (N/A)'),
                (pos_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['flux_density_j_pos_merged']), np.amin(comsol_flux_density_j_pos_extrapd)), max(np.amax(OneD_time_dependent_data_merged['flux_density_j_pos_merged']), np.amax(comsol_flux_density_j_pos_extrapd))],
                 True, False, 'Flux Density $j$ Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$j$ Pos. (N/A)'),
                (neg_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['exchange_flux_density_j0_neg_merged']), np.amin(comsol_exchange_flux_density_j0_neg_extrapd)), max(np.amax(OneD_time_dependent_data_merged['exchange_flux_density_j0_neg_merged']), np.amax(comsol_exchange_flux_density_j0_neg_extrapd))],
                 True, False, 'Exchange Flux Density $j_0$ Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$j_0$ Neg. (N/A)'),
                (pos_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['exchange_flux_density_j0_pos_merged']), np.amin(comsol_exchange_flux_density_j0_pos_extrapd)), max(np.amax(OneD_time_dependent_data_merged['exchange_flux_density_j0_pos_merged']), np.amax(comsol_exchange_flux_density_j0_pos_extrapd))],
                 True, False, 'Exchange Flux Density $j_0$ Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$j_0$ Pos. (N/A)'),
                (neg_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['overpotential_neg_merged']), np.amin(comsol_overpotential_neg_extrapd)), max(np.amax(OneD_time_dependent_data_merged['overpotential_neg_merged']), np.amax(comsol_overpotential_neg_extrapd))],
                 True, False, 'Overpotential $\eta$ Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\eta$ Neg. (V)'),
                (pos_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['overpotential_pos_merged']), np.amin(comsol_overpotential_pos_extrapd)), max(np.amax(OneD_time_dependent_data_merged['overpotential_pos_merged']), np.amax(comsol_overpotential_pos_extrapd))],
                 True, False, 'Overpotential $\eta$ Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\eta$ Pos. (V)'),
                (neg_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['uocp_neg_merged']), np.amin(comsol_uocp_neg_extrapd)), max(np.amax(OneD_time_dependent_data_merged['uocp_neg_merged']), np.amax(comsol_uocp_neg_extrapd))],
                 True, False, 'Open-circuit Potential Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$U_{ocp}$ Neg. (V)'),
                (pos_axial_mesh_coordinates, [min(np.amin(OneD_time_dependent_data_merged['uocp_pos_merged']), np.amin(comsol_uocp_pos_extrapd)), max(np.amax(OneD_time_dependent_data_merged['uocp_pos_merged']), np.amax(comsol_uocp_pos_extrapd))],
                 False, False, 'Open-circuit Potential Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$U_{ocp}$ Pos. (V)'),
                (time_dependent_metadata['simtime'], [min(np.amin(comsol_terminal_voltage), np.amin(terminal_voltage)), 3.93],
                 False, False, 'Terminal Voltage vs. Time (1C CC Discharge)', 'Time (s)', 'Terminal Voltage (V)')
                ]

else:
    OneD_aux_data = [(sim_cell_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['ce_merged']), np.amax(OneD_time_dependent_data_merged['ce_merged'])],
                 False, False, '$C_e$ vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$C_e$ (mol/$m^3$)'),
                (sim_cell_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['phi_e_merged']), np.amax(OneD_time_dependent_data_merged['phi_e_merged'])],
                 False, False, '$\phi_e$ vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\phi_e$ (V)'),
                (neg_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['phi_s_neg_merged']), np.amax(OneD_time_dependent_data_merged['phi_s_neg_merged'])],
                 True, False, '$\phi_s$ Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\phi_s$ Neg. (V)'),
                (pos_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['phi_s_pos_merged']), np.amax(OneD_time_dependent_data_merged['phi_s_pos_merged'])],
                 False, False, '$\phi_s$ Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\phi_s$ Pos. (V)'),
                (neg_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['flux_density_j_neg_merged']), np.amax(OneD_time_dependent_data_merged['flux_density_j_neg_merged'])],
                 True, False, 'Flux Density $j$ Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$j$ Neg. (N/A)'),
                (pos_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['flux_density_j_pos_merged']), np.amax(OneD_time_dependent_data_merged['flux_density_j_pos_merged'])],
                 True, False, 'Flux Density $j$ Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$j$ Pos. (N/A)'),
                (neg_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['exchange_flux_density_j0_neg_merged']), np.amax(OneD_time_dependent_data_merged['exchange_flux_density_j0_neg_merged'])],
                 True, False, 'Exchange Flux Density $j_0$ Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$j_0$ Neg. (N/A)'),
                (pos_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['exchange_flux_density_j0_pos_merged']), np.amax(OneD_time_dependent_data_merged['exchange_flux_density_j0_pos_merged'])],
                 True, False, 'Exchange Flux Density $j_0$ Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$j_0$ Pos. (N/A)'),
                (neg_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['overpotential_neg_merged']), np.amax(OneD_time_dependent_data_merged['overpotential_neg_merged'])],
                 True, False, 'Overpotential $\eta$ Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\eta$ Neg. (V)'),
                (pos_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['overpotential_pos_merged']), np.amax(OneD_time_dependent_data_merged['overpotential_pos_merged'])],
                 True, False, 'Overpotential $\eta$ Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$\eta$ Pos. (V)'),
                (neg_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['uocp_neg_merged']), np.amax(OneD_time_dependent_data_merged['uocp_neg_merged'])],
                 True, False, 'Open-circuit Potential Neg. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$U_{ocp}$ Neg. (V)'),
                (pos_axial_mesh_coordinates, [np.amin(OneD_time_dependent_data_merged['uocp_pos_merged']), np.amax(OneD_time_dependent_data_merged['uocp_pos_merged'])],
                 False, False, 'Open-circuit Potential Pos. vs. Through-thickness Displacement', 'Through-thickness Displacement (dimensionless)', '$U_{ocp}$ Pos. (V)'),
                (time_dependent_metadata['simtime'], [np.amin(terminal_voltage), 3.93],
                 False, False, 'Terminal Voltage vs. Time (1C CC Discharge)', 'Time (s)', 'Terminal Voltage (V)')
                ]

# TwoD_aux_data = [(neg_radial_mesh_coordinates, [np.amin(TwoD_time_dependent_data_merged['cs_neg_merged']),
#                                                 np.amax(TwoD_time_dependent_data_merged['cs_neg_merged'])],
#                  False, False, '$C_s$ Neg. vs. Particle Depth', '$C_s$ (mol/$m^3$)', 'Particle Radius (dimensionless)',
#                   'neg'),
#                  (pos_radial_mesh_coordinates, [np.amin(TwoD_time_dependent_data_merged['cs_pos_merged']),
#                                                 np.amax(TwoD_time_dependent_data_merged['cs_pos_merged'])],
#                   False, False, '$C_s$ Pos. vs. Particle Depth', '$C_s$ (mol/$m^3$)', 'Particle Radius (dimensionless)',
#                   'pos')
#                  ]

# TwoD_aux_data = [((np.fliplr([neg_radial_node_coordinates])[0]), [np.amin(TwoD_time_dependent_data['cs_neg']),          # Flipping co-ordinate left-to-right necessary when plotting co-ordinates on y-axis
#                                                 np.amax(TwoD_time_dependent_data['cs_neg'])],
#                  False, False, '$C_s$ Neg. vs. Particle Depth', '$C_s$ (mol/$m^3$)', 'Particle Radius (dimensionless)',
#                   'neg'),
#                  ((np.fliplr([pos_radial_node_coordinates])[0]), [np.amin(TwoD_time_dependent_data['cs_pos']),          # Flipping co-ordinate left-to-right necessary when plotting co-ordinates on y-axis
#                                                 np.amax(TwoD_time_dependent_data['cs_pos'])],
#                   False, False, '$C_s$ Pos. vs. Particle Depth', '$C_s$ (mol/$m^3$)', 'Particle Radius (dimensionless)',
#                   'pos')
#                  ]

######################################### Ordered X-Y Comparison Data ##################################################
if benchmarking:
    comsol_plot_data = [(comsol_x_cell, comsol_ce),
                        (comsol_x_cell, comsol_phi_e),
                        (comsol_x_neg_extrapd, comsol_phi_s_neg_extrapd),
                        (comsol_x_pos_extrapd, comsol_phi_s_pos_extrapd),
                        (comsol_x_neg_extrapd, comsol_flux_density_j_neg_extrapd),
                        (comsol_x_pos_extrapd, comsol_flux_density_j_pos_extrapd),
                        (comsol_x_neg_extrapd, comsol_exchange_flux_density_j0_neg_extrapd),
                        (comsol_x_pos_extrapd, comsol_exchange_flux_density_j0_pos_extrapd),
                        (comsol_x_neg_extrapd, comsol_overpotential_neg_extrapd),
                        (comsol_x_pos_extrapd, comsol_overpotential_pos_extrapd),
                        (comsol_x_neg_extrapd, comsol_uocp_neg_extrapd),
                        (comsol_x_pos_extrapd, comsol_uocp_pos_extrapd),
                        (comsol_time, comsol_terminal_voltage)
                        ]

############################### Interpolate Comparison Data; Prep. for Variance Plot ###################################
# if benchmarking:
#     interpd_array_length = len(time_dependent_metadata['simtime'])
#     interpd_comparison_arrays = [('comparison_ce_interpd', interpd_array_length, len(sim_cell_axial_mesh_coordinates)), # Specify lengths & widths to hold interpolated comparison datasets
#                                  ('comparison_phi_e_interpd', interpd_array_length, len(sim_cell_axial_mesh_coordinates)),
#                                  ('comparison_phi_s_neg_interpd', interpd_array_length, len(neg_axial_mesh_coordinates)),
#                                  ('comparison_phi_s_pos_interpd', interpd_array_length, len(pos_axial_mesh_coordinates)),
#                                  ('comparison_flux_density_j_neg_interpd', interpd_array_length, len(neg_axial_mesh_coordinates)),
#                                  ('comparison_flux_density_j_pos_interpd', interpd_array_length, len(pos_axial_mesh_coordinates)),
#                                  ('comparison_exchange_flux_density_j0_neg_interpd', interpd_array_length, len(neg_axial_mesh_coordinates)),
#                                  ('comparison_exchange_flux_density_j0_pos_interpd', interpd_array_length, len(pos_axial_mesh_coordinates)),
#                                  ('comparison_overpotential_neg_interpd', interpd_array_length, len(neg_axial_mesh_coordinates)),
#                                  ('comparison_overpotential_pos_interpd', interpd_array_length, len(pos_axial_mesh_coordinates)),
#                                  ('comparison_uocp_neg_interpd', interpd_array_length, len(neg_axial_mesh_coordinates)),
#                                  ('comparison_uocp_pos_interpd', interpd_array_length, len(pos_axial_mesh_coordinates)),
#                                  ]
#     interpd_comparison_data = OrderedDict((variable, np.zeros((array_length, array_width), dtype=float))                # Generate the empty arrays to hold the interpolated comparison values
#                                 for variable, array_length, array_width in interpd_comparison_arrays)
#
#     def interp(row, comparison_x_locs, original_x_locs):                                                                # Function to interpolate the comparison dataset for vals @ Python mesh co-ordinates
#         interpd = interpolate.interp1d(comparison_x_locs, row)                                                          # Creating the interpolation fn. with x & y data to be interpolated
#         return interpd(original_x_locs)                                                                                 # Interpolating at the Python/FiPy mesh co-ordinates passed
#
#     for index, interpd_array in enumerate(interpd_comparison_data.values()):                                            # Loop calling the interp function for all comparison datasets
#         for row_index, row in enumerate(comsol_plot_data[index][1]):                                                    # For every row of data (timestep) in the comparison dataset
#             interpd_array[row_index] = interp(row, comsol_plot_data[index][0], OneD_aux_data[index][0])                 # Populate the interpd_comparison_arrays dict. with the interpolated data

################################## Pre-allocate Arrays for Variance Plot Data ##########################################
    # delta_array_length = len(time_dependent_metadata['simtime'])
    # delta_val_arrays = [('delta_ce', delta_array_length, len(sim_cell_axial_mesh_coordinates), interpd_comparison_data['comparison_ce_interpd'], OneD_time_dependent_data_merged['ce_merged']),
    #                     ('delta_phi_e', delta_array_length, len(sim_cell_axial_mesh_coordinates), interpd_comparison_data['comparison_phi_e_interpd'], OneD_time_dependent_data_merged['phi_e_merged']),
    #                     ('delta_phi_s_neg', delta_array_length, len(neg_axial_mesh_coordinates), interpd_comparison_data['comparison_phi_s_neg_interpd'], OneD_time_dependent_data_merged['phi_s_neg_merged']),
    #                     ('delta_phi_s_pos', delta_array_length, len(pos_axial_mesh_coordinates), interpd_comparison_data['comparison_phi_s_pos_interpd'], OneD_time_dependent_data_merged['phi_s_pos_merged']),
    #                     ('delta_flux_density_j_neg', delta_array_length, len(neg_axial_mesh_coordinates), interpd_comparison_data['comparison_flux_density_j_neg_interpd'], OneD_time_dependent_data_merged['flux_density_j_neg_merged']),
    #                     ('delta_flux_density_j_pos', delta_array_length, len(pos_axial_mesh_coordinates), interpd_comparison_data['comparison_flux_density_j_pos_interpd'], OneD_time_dependent_data_merged['flux_density_j_pos_merged']),
    #                     ('delta_exchange_flux_density_j0_neg', delta_array_length, len(neg_axial_mesh_coordinates), interpd_comparison_data['comparison_exchange_flux_density_j0_neg_interpd'], OneD_time_dependent_data_merged['exchange_flux_density_j0_neg_merged']),
    #                     ('delta_exchange_flux_density_j0_pos', delta_array_length, len(pos_axial_mesh_coordinates), interpd_comparison_data['comparison_exchange_flux_density_j0_pos_interpd'], OneD_time_dependent_data_merged['exchange_flux_density_j0_pos_merged']),
    #                     ('delta_overpotential_neg', delta_array_length, len(neg_axial_mesh_coordinates), interpd_comparison_data['comparison_overpotential_neg_interpd'], OneD_time_dependent_data_merged['overpotential_neg_merged']),
    #                     ('delta_overpotential_pos', delta_array_length, len(pos_axial_mesh_coordinates), interpd_comparison_data['comparison_overpotential_pos_interpd'], OneD_time_dependent_data_merged['overpotential_pos_merged']),
    #                     ('delta_uocp_neg', delta_array_length, len(neg_axial_mesh_coordinates), interpd_comparison_data['comparison_uocp_neg_interpd'], OneD_time_dependent_data_merged['uocp_neg_merged']),
    #                     ('delta_uocp_pos', delta_array_length, len(pos_axial_mesh_coordinates), interpd_comparison_data['comparison_uocp_pos_interpd'], OneD_time_dependent_data_merged['uocp_pos_merged']),
    #                     ('delta_terminal_voltage', delta_array_length, 1, comsol_terminal_voltage, terminal_voltage)
    #                     ]
    # delta_vals = OrderedDict((variable, np.zeros((array_length, array_width), dtype=float))
    #                     for variable, array_length, array_width, dummy_1, dummy_2 in delta_val_arrays)

###################################### Compute Deltas for Variance Plots ###############################################
    # # Assumes that both datasets have the same timestep. Datasets may have different dx, though, & code will interpolate
    # for tuple, (delta_dataset_name, delta_dataset) in enumerate(delta_vals.items()):
    #     if delta_dataset_name != 'delta_terminal_voltage':
    #         if delta_dataset_name.endswith('neg'):
    #             for time_row_index, timestep in enumerate(time_dependent_metadata['simtime']):
    #                 for x_position_index, x_position in enumerate(neg_axial_mesh_coordinates):
    #                     delta_dataset[time_row_index][x_position_index] = abs(delta_val_arrays[tuple][3][time_row_index][x_position_index] - delta_val_arrays[tuple][4][time_row_index][x_position_index])
    #         elif delta_dataset_name.endswith('pos'):
    #             for time_row_index, timestep in enumerate(time_dependent_metadata['simtime']):
    #                 for x_position_index, x_position in enumerate(pos_axial_mesh_coordinates):
    #                     delta_dataset[time_row_index][x_position_index] = abs(delta_val_arrays[tuple][3][time_row_index][x_position_index] - delta_val_arrays[tuple][4][time_row_index][x_position_index])
    #         else:
    #             for time_row_index, timestep in enumerate(time_dependent_metadata['simtime']):
    #                 for x_position_index, x_position in enumerate(sim_cell_axial_mesh_coordinates):
    #                     delta_dataset[time_row_index][x_position_index] = abs(delta_val_arrays[tuple][3][time_row_index][x_position_index] - delta_val_arrays[tuple][4][time_row_index][x_position_index])
    #     else:                                                                                                           # Get the terminal voltage delta
    #         for time_row_index, timestep in enumerate(time_dependent_metadata['simtime']):
    #             delta_dataset[time_row_index] = abs(delta_val_arrays[tuple][3][time_row_index] - delta_val_arrays[tuple][4][time_row_index])
    #
    # deltas = [(sim_cell_axial_mesh_coordinates, delta_vals['delta_ce']),
    #           (sim_cell_axial_mesh_coordinates, delta_vals['delta_phi_e']),
    #           (neg_axial_mesh_coordinates, delta_vals['delta_phi_s_neg']),
    #           (pos_axial_mesh_coordinates, delta_vals['delta_phi_s_pos']),
    #           (neg_axial_mesh_coordinates, delta_vals['delta_flux_density_j_neg']),
    #           (pos_axial_mesh_coordinates, delta_vals['delta_flux_density_j_pos']),
    #           (neg_axial_mesh_coordinates, delta_vals['delta_exchange_flux_density_j0_neg']),
    #           (pos_axial_mesh_coordinates, delta_vals['delta_exchange_flux_density_j0_pos']),
    #           (neg_axial_mesh_coordinates, delta_vals['delta_overpotential_neg']),
    #           (pos_axial_mesh_coordinates, delta_vals['delta_overpotential_pos']),
    #           (neg_axial_mesh_coordinates, delta_vals['delta_uocp_neg']),
    #           (pos_axial_mesh_coordinates, delta_vals['delta_uocp_pos']),
    #           (time_dependent_metadata['simtime'], delta_vals['delta_terminal_voltage'])
    #           ]
					
############################################## Generate Plots ##########################################################
print('Plotting datasets..')
slider_increment = settings_dict['record_period']
domain_boundaries = [(neg_elec_sep_vline, 'Neg. Electrode/Separator'), (pos_elec_sep_vline, 'Pos. Electrode/Separator')]

############################################ Terminal Voltage Plot #####################################################
def time_series_plot(x_data, y_data, ylims, scientific_y_units_bool, autoscale_bool, title, x_label, y_label, comparison_dataset_x, comparison_dataset_y):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.set_title(title, fontsize=title_fontsize)
    ax.title.set_position([.5, 1.01])
    ax.title.set_color(plot_peripherals_color)
    ax.set_ylabel(y_label, fontsize=xy_labels_fontsize)
    ax.set_xlabel(x_label, fontsize=xy_labels_fontsize)
    plt.tick_params(labelsize=ticksize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(vlinecolor)
    ax.spines['left'].set_color(vlinecolor)
    ax.xaxis.label.set_color(plot_peripherals_color)
    ax.yaxis.label.set_color(plot_peripherals_color)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', colors=plot_peripherals_color)
    ax.tick_params(axis='y', colors=plot_peripherals_color)
    ax.get_yaxis().get_major_formatter().set_useOffset(scientific_y_units_bool)
    ax.set_xlim([min(x_data), max(x_data)])
    if not ylims:
        pass
    else:
        ax.set_ylim([min(ylims), max(ylims)])

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    time_series_plot_style = dict(color=Python_line_colour, linestyle=time_series_plot_line_style,
                                  marker=time_series_plot_marker_style, markersize=marker_size, markeredgecolor=None,
                                  markerfacecolor=marker_facecolor)

    ax.plot(x_data, y_data, label='Python/FiPy', **time_series_plot_style)                                              # Will only work while sim is being started at 0 seconds
    if benchmarking:
        ax.plot(comparison_dataset_x, comparison_dataset_y, label='COMSOL', color=comparison_line_colour)

    plot_legend = plt.legend(frameon=False)
    for text in plot_legend.get_texts():
        text.set_color(plot_peripherals_color)

    fig = plt.gcf()
    plt.show()
    if save_images:
        fig.tight_layout()
        fig.set_figheight(plot_aspect_ratio[0])
        fig.set_figwidth(plot_aspect_ratio[1])
        if not path.exists(image_save_directory):
            makedirs(image_save_directory)
        fig.savefig(image_save_directory + 'Terminal_voltage.' + image_format, format=image_format, dpi=1200)

################################################# Plots of 1D Data #####################################################
class ChangingPlot(object):
    def __init__(self, x, y, ylims, scientific_y_units_bool, autoscale_bool, title, x_label, y_label, comparison_dataset_x, comparison_dataset_y):
        self.x_data = x
        self.y_data = y
        self.comparison_x_data = comparison_dataset_x
        self.comparison_y_data = comparison_dataset_y
        self.sim_start_time = float(time_dependent_metadata['simtime'][0])
        self.sim_Endtime = float(time_dependent_metadata['simtime'][-1]) + slider_max_extension
        self.timestep = settings_dict['temporal_resolution']     # Needs to change when the main_simulation_code has a calculation for the timestep & may not use the temporal_resolution value

############################################## Figure Formatting #######################################################
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('white')
        self.ax.set_title(title, fontsize=title_fontsize)
        self.ax.title.set_position([.5, 1.01])
        self.ax.title.set_color(plot_peripherals_color)
        self.ax.set_ylabel(y_label, fontsize=xy_labels_fontsize)
        self.ax.set_xlabel(x_label, fontsize=xy_labels_fontsize)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(vlinecolor)
        self.ax.spines['left'].set_color(vlinecolor)
        self.ax.xaxis.label.set_color(plot_peripherals_color)
        self.ax.yaxis.label.set_color(plot_peripherals_color)
        self.ax.yaxis.set_ticks_position('left')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.tick_params(axis='x', colors=plot_peripherals_color)
        self.ax.tick_params(axis='y', colors=plot_peripherals_color)
        self.ax.get_yaxis().get_major_formatter().set_useOffset(scientific_y_units_bool)
        self.ax.set_xlim([min(self.x_data), max(self.x_data)])

        ChangingPlot_style = dict(label='Python/FiPy', color=Python_line_colour, linestyle=ChangingPlot_line_style,
                                  marker=ChangingPlot_marker_style, markersize=marker_size, markeredgecolor=None,
                                  markerfacecolor=marker_facecolor
                                  )

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

############################################# Setting Axis Limits ######################################################
        if not ylims:
            pass
        else:
            self.ax.set_ylim([min(ylims), max(ylims)])
        if not ylims:
            ymin, ymax = self.ax.get_ylim()
        else:
            ymin = min(ylims)
            ymax = max(ylims)

########################################## Domain Boundary Plotting ####################################################
        for x_location in domain_boundaries:
            plt.vlines(x_location[0], ymin, ymax, colors=vlinecolor, linestyles='dashed', label='_nolegend_')
            # plt.vlines(x_location[0], ymin, ymax, colors=vlinecolor, linestyles='dashed')
        # if max(x) > 2:
        #     plt.text(neg_elec_sep_vline_label, ((ymax-ymin)*0.98+ymin), 'Neg. Electrode/Separator', rotation=90)
        #     plt.text(pos_elec_sep_vline_label, ((ymax-ymin)*0.98+ymin), 'Pos. Electrode/Separator', rotation=90)

############################################## Slider Setup ############################################################
        self.sliderax = self.fig.add_axes([0.2, 0.005, 0.6, 0.03], axisbg=slider_bg_colour)
        self.slider = DiscreteSlider(self.sliderax, 'Simulation Time (s)', self.sim_start_time, self.sim_Endtime,
                                     increment=slider_increment, valinit=self.sim_start_time, color=slider_fg_colour,
                                     alpha = 0.9)
        self.slider.on_changed(self.update)

        self.dataset_Python, = self.ax.plot(x, y[int(self.sim_start_time)], **ChangingPlot_style)                       # Will only work while sim is being started at 0 seconds

        if benchmarking:
            self.dataset_comparison, = self.ax.plot(comparison_dataset_x,
                                                    comparison_dataset_y[int(self.sim_start_time)],
                                                    label='COMSOL',
                                                    color=comparison_line_colour
                                                    )                                                                   # Will only work while sim is being started at 0 seconds

        handles, labels = self.ax.get_legend_handles_labels()
        plot_legend = self.ax.legend(handles, labels, frameon=False, loc='best')
        for text in plot_legend.get_texts():
            text.set_color(plot_peripherals_color)

####################################### Plot Slider Function Definition ################################################
    def update(self, slider_time_val):
        timerow_index = int(slider_time_val/self.timestep)
        self.dataset_Python.set_data([self.x_data, [self.y_data[timerow_index]]])
        if benchmarking:
            self.dataset_comparison.set_data([self.comparison_x_data, [self.comparison_y_data[timerow_index]]])
        self.fig.canvas.draw()

    def show(self):
        plt.show()

######################################## Plots of 2D Data: Electrode Cs ################################################
class TwoD_ChangingPlot(object):
    def __init__(self, TwoD_x, TwoD_y, xlims, scientific_y_units_bool, autoscale_bool, title, x_label, y_label,
                 electrode_flag):
        self.x_data = TwoD_x                                                                                            # Co-ordinates of nodes & faces in the particle radius, reading L-R is particle centre to surface
        self.y_data = TwoD_y                                                                                            # Large dataset of time & particle Cs values in y direction, particle no. in electrode in x-direction
        self.sim_start_time = float(time_dependent_metadata['simtime'][0])                                              # Extract the simulation start time (seconds)
        self.sim_Endtime = float(time_dependent_metadata['simtime'][-1]) + slider_max_extension                         # Extract the simulation end time (seconds) & add a small value to make it easier to select the last timestep in the plot slider
        self.timestep = settings_dict['temporal_resolution']                                                            # Needs to change when the main_simulation_code has a calculation for the timestep & may not use the temporal_resolution value
        if electrode_flag == 'neg':                                                                                     # Check if we're dealing with neg. or pos. electrode & set no. radial nodes accordingly (used in update method)
            self.nr = nr_neg
            self.nx = nx_neg
            self.TwoD_lege_loc = 'upper left'
        else:
            self.nr = nr_pos
            self.nx = nx_pos
            self.TwoD_lege_loc = 'upper right'

############################################## Figure Formatting #######################################################
        self.fig, self.ax = plt.subplots()                                                                              # Generate figure window
        self.fig.patch.set_facecolor('white')                                                                           # Formatting
        self.ax.set_title(title, fontsize=title_fontsize)
        self.ax.title.set_position([.5, 1.01])
        self.ax.title.set_color(plot_peripherals_color)
        self.ax.set_ylabel(y_label, fontsize=xy_labels_fontsize)
        self.ax.set_xlabel(x_label, fontsize=xy_labels_fontsize)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(vlinecolor)
        self.ax.spines['left'].set_color(vlinecolor)
        self.ax.xaxis.label.set_color(plot_peripherals_color)
        self.ax.yaxis.label.set_color(plot_peripherals_color)
        self.ax.yaxis.set_ticks_position('left')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.tick_params(axis='x', colors=plot_peripherals_color)
        self.ax.tick_params(axis='y', colors=plot_peripherals_color)
        self.ax.get_yaxis().get_major_formatter().set_useOffset(scientific_y_units_bool)                                # Toggle on/off scientific notation for the y-axis units
        self.ax.get_xaxis().get_major_formatter().set_useOffset(scientific_y_units_bool)                                # Toggle on/off scientific notation for the x-axis units (uses same passed value as for y-axis)
        self.ax.set_xlim([np.amin(self.y_data), np.amax(self.y_data)])                                                  # Plotting Cs on the x-axis, so x-axis limits are set by y_data

        figManager = plt.get_current_fig_manager()                                                                      # Grab the figure window
        figManager.window.showMaximized()                                                                               # And maximise it

        TwoD_ChangingPlot_style = dict(linestyle=ChangingPlot_line_style,
                                  marker=ChangingPlot_marker_style, markersize=marker_size, markeredgecolor=None,
                                  )

############################################# Setting Axis Limits ######################################################
        if not xlims:                                                                                                   # If xlims isn't passed, don't set any limits on Cs on the x-axis
            pass
        else:                                                                                                           # But if values are passed for xlims, then set the lower value to the min. limit & higher value to max. limit
            # print(min(xlims))
            # print(max(xlims))
            # print(TwoD_y)
            self.ax.set_xlim([min(xlims), max(xlims)])                                                                  # Set the Cs (mol/m**3) limits on the x-axis
        self.ax.set_ylim([0, 1])                                                                                        # Set the particle radius / y-axis limits

############################################## Slider Setup ############################################################
        self.sliderax = self.fig.add_axes([0.2, 0.005, 0.6, 0.03], axisbg=slider_bg_colour)                             # Define the slider size & colour
        self.slider = DiscreteSlider(self.sliderax, 'Simulation Time (s)', self.sim_start_time, self.sim_Endtime,       # Create the slider object
                                     increment=slider_increment, valinit=self.sim_start_time, color=slider_fg_colour,
                                     alpha = 0.9)
        self.slider.on_changed(self.update)                                                                             # Call the slider updating method

############################################## Plot Generation #########################################################
        self.plotted_dataset_Python = []                                                                                # Create a list to store the Cs vs. r values for every particle at this timestep. Each element in the list is Cs for a particle

        # for particle_index, particle in enumerate((TwoD_y[int(self.sim_start_time):(nr_neg + (nr_neg + 1)), :].transpose())[::-1]):     # Loop through all particles in the electrode to generate the concentration plots in each
        #     temp_dataset_Python_store, = self.ax.plot((TwoD_y[int(self.sim_start_time):(nr_neg +                                        # nr_neg to account for nodes at a timestep
        #                                                                           (nr_neg + 1)),                                        # (nr_neg+1) to account for faces at that timestep
        #                                                particle_index].transpose())[::-1],                                              # Generate the plot (will only work while sim is being started at 0 seconds)
        #                                               TwoD_x, label=('Particle ' + str(particle_index+1)),                              # Specify the co-ordinates for the y-axis & the particle numbers for legend
        #                                               **TwoD_ChangingPlot_style                                                         # Apply plot formatting, defined earlier
        #                                         )                                                                                       # transpose())[::-1] to flip column vector to row vector & reverse so reads, L-R, as particle bottom to top, matching x co-ordinates
        for particle_index in xrange(0, self.nx):
            temp_dataset_Python_store, = self.ax.plot(TwoD_y[int(self.sim_start_time):nr_neg, particle_index], TwoD_x,
                                                      label=('Particle ' + str(particle_index+1)),
                                                      **TwoD_ChangingPlot_style
                                                      )
            self.plotted_dataset_Python.append(temp_dataset_Python_store)                                               # Append a vector of Cs vs. r for a given particle at the starting time, in this electrode (storing for access via update)

        handles, labels = self.ax.get_legend_handles_labels()
        plot_legend = self.ax.legend(handles, labels, frameon=False, loc=self.TwoD_lege_loc)
        for text in plot_legend.get_texts():
            text.set_color(plot_peripherals_color)

######################################### Plot Updating with Slider ####################################################
    def update(self, slider_time_val):                                                                                  # Update the plot with the Cs values corresponding to the user-selected timestep
        timerow_index = int(slider_time_val/self.timestep)                                                              # Calculate the row value for the simulation time corresponding to the selected slider position

        # for particle in xrange(0, self.nr):                                                                             # Loop through all particles in the electrode to update the concentration plots in each
        #     self.plotted_dataset_Python[particle].set_data([[(self.y_data[timerow_index:(nr_neg + (nr_neg+1)            # nr_neg to account for nodes at a timestep, & (nr_neg+1) to account for faces at that timestep
        #                                                                + timerow_index), particle].transpose())[::-1]],
        #                                                     self.x_data])                                               # transpose())[::-1] to flip column vector to row vector & reverse so reads, L-R, as particle bottom to top, matching x co-ordinates

        for particle in xrange(0, self.nx):                                                                             # Loop through all particles in the electrode to update the concentration plots in each
            # print(timerow_index)
            # print((timerow_index + nr_neg))
            # print(particle)
            # print("\n")
            self.plotted_dataset_Python[particle].set_data([self.y_data[timerow_index:(timerow_index + nr_neg), particle],
                                                            self.x_data])

        self.fig.canvas.draw()

    def show(self):
        plt.show()

####################################### Plot Slider Function Definition ################################################
class DiscreteSlider(Slider):
    def __init__(self, *args, **kwargs):
        self.inc = kwargs.pop('increment', slider_increment)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)

########################################## Runtime & User Update #######################################################
postprocessor_clock_end_time = timegm(gmtime())
print('Post-processing complete. CPU time was ' + str(timedelta(seconds=(postprocessor_clock_end_time - postprocessor_clock_start_time))))
print('------------------- End -------------------')

############################################## Display Plots ###########################################################

################################### Python/FiPy Data AND Comparison Data ###############################################
if benchmarking:
    time_series_plot(OneD_aux_data[-1][0], terminal_voltage, OneD_aux_data[-1][1], OneD_aux_data[-1][2],
                     OneD_aux_data[-1][3], OneD_aux_data[-1][4], OneD_aux_data[-1][5], OneD_aux_data[-1][6], comsol_plot_data[-1][0], comsol_plot_data[-1][1])
    for index, variable in enumerate(OneD_time_dependent_data_merged):
        ChangingPlot(OneD_aux_data[index][0], OneD_time_dependent_data_merged.values()[index], OneD_aux_data[index][1], OneD_aux_data[index][2],
                     OneD_aux_data[index][3], OneD_aux_data[index][4], OneD_aux_data[index][5], OneD_aux_data[index][6],
                     comsol_plot_data[index][0], comsol_plot_data[index][1]).show()

############################################ Python/FiPy Data Only #####################################################
else:
    dummy_val = 0
    time_series_plot(OneD_aux_data[-1][0], terminal_voltage, OneD_aux_data[-1][1], OneD_aux_data[-1][2],
                     OneD_aux_data[-1][3], OneD_aux_data[-1][4], OneD_aux_data[-1][5], OneD_aux_data[-1][6], dummy_val,
                     dummy_val)
    for index, variable in enumerate(OneD_time_dependent_data_merged):
        ChangingPlot(OneD_aux_data[index][0], OneD_time_dependent_data_merged.values()[index], OneD_aux_data[index][1], # .values()[index] -> index captures the values associated with a given solution variable (dict. key)
                     OneD_aux_data[index][2], OneD_aux_data[index][3], OneD_aux_data[index][4], OneD_aux_data[index][5],
                     OneD_aux_data[index][6], dummy_val, dummy_val).show()

########################################## Plot Cs / Concentrations ####################################################
# for index, variable in enumerate(TwoD_time_dependent_data_merged):
#     TwoD_ChangingPlot(TwoD_aux_data[index][0], TwoD_time_dependent_data_merged.values()[index],                         # Pass all Cs values, for all particles in that electrode, at all timesteps, & index for plotting in the TwoD_ChangingPlot class
#                       TwoD_aux_data[index][1], TwoD_aux_data[index][2], TwoD_aux_data[index][3],
#                       TwoD_aux_data[index][4], TwoD_aux_data[index][5], TwoD_aux_data[index][6],
#                       TwoD_aux_data[index][7]).show()

# print(neg_radial_node_coordinates)
# print(neg_radial_node_coordinates[::1])
# print(np.fliplr([neg_radial_node_coordinates])[0])

# plt.plot(TwoD_time_dependent_data['cs_neg'][27105:27105+5,0], np.fliplr([neg_radial_node_coordinates])[0])
# plt.plot(TwoD_time_dependent_data['cs_neg'][27105:27105+5,1], np.fliplr([neg_radial_node_coordinates])[0])
# plt.plot(TwoD_time_dependent_data['cs_neg'][27105:27105+5,2], np.fliplr([neg_radial_node_coordinates])[0])
# plt.plot(TwoD_time_dependent_data['cs_neg'][27105:27105+5,3], np.fliplr([neg_radial_node_coordinates])[0])
# plt.plot(TwoD_time_dependent_data['cs_neg'][27105:27105+5,4], np.fliplr([neg_radial_node_coordinates])[0])
# plt.show()

# TwoD_ChangingPlot(TwoD_aux_data[0][0], TwoD_time_dependent_data['cs_neg'],                                              # Plotting Cs nodevalues only in the negative electrode
#                   TwoD_aux_data[0][1], TwoD_aux_data[0][2], TwoD_aux_data[0][3],
#                   TwoD_aux_data[0][4], TwoD_aux_data[0][5], TwoD_aux_data[0][6],
#                   TwoD_aux_data[0][7]).show()
#
# TwoD_ChangingPlot(TwoD_aux_data[1][0], TwoD_time_dependent_data['cs_pos'],                                              # Plotting Cs nodevalues only in the positive electrode
#                   TwoD_aux_data[1][1], TwoD_aux_data[1][2], TwoD_aux_data[1][3],
#                   TwoD_aux_data[1][4], TwoD_aux_data[1][5], TwoD_aux_data[1][6],
#                   TwoD_aux_data[1][7]).show()


