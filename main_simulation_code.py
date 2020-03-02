######################################### Python Module Imports ########################################################
from __future__ import division
from __future__ import print_function                                                                                   # Force Python 3 compatability
from load_simulation_control import *
from cell_calculations import *
from sys import exit, platform, executable, exec_prefix
from os import path, makedirs, system
from subprocess import Popen
from imp import find_module
from time import gmtime
from ctypes import CDLL
from thread import interrupt_main
from signal import signal, SIGINT
from calendar import timegm
from warnings import filterwarnings
import matplotlib
if matplotlib.get_backend() != 'Qt4Agg':
    matplotlib.use('Qt4Agg')
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cbook
import matplotlib.pyplot as plt
try:
    from pympler.tracker import SummaryTracker
except:
    print("Pympler memory debugging module cannot be imported because it's not installed. pip install pympler to solve the issue")
    exit('Simulation terminated')
from datetime import datetime, timedelta
from collections import OrderedDict
from gc import get_objects
from csv import writer
from itertools import izip
import cPickle as pickle

############################### Prevent CTRL-C Failure when Running Fortran Libraries ##################################
if platform in ('win32', 'cygwin'):
    try:
        import win32api
    except:
        print("win32api module cannot be imported because it's not installed. pip install pypiwin32 to solve the issue")
        exit('Simulation terminated')
    try:                                                                                                                # Try this path to access the dll files
        basepath = find_module('numpy')[1]
        CDLL(path.join(basepath, 'core', 'libmmd.dll'))
        CDLL(path.join(basepath, 'core', 'libifcoremd.dll'))
    except:                                                                                                             # If it fails, try this alternative path for the dll's in the 'Intel for Anaconda' distribution
        basepath = exec_prefix
        CDLL(path.join(basepath, 'Library', 'bin', 'libmmd.dll'))
        CDLL(path.join(basepath, 'pkgs', 'icc_rt-13.1.5-intel_4', 'Library', 'bin', 'libifcoremd.dll'))
    def handler(dwCtrlType, hook_sigint=interrupt_main):
        if dwCtrlType == 0:
            hook_sigint()
            return 1
        return 0
    win32api.SetConsoleCtrlHandler(handler, 1)

############################################## Simulation timing #######################################################
sim_clock_start_time = timegm(gmtime())                                                                                 # Log sim start time
print('Initialising on ' + str(datetime.fromtimestamp(sim_clock_start_time).strftime('%d/%m/%Y, %H:%M:%S'))
      + ' with the ' + str(DefaultSolver.__name__) + ' solver..')

max_permissible_timestep = 100                                                                                          # Limit the user to a maximum timestep (seconds)
timeStep = min(temporal_resolution, max_permissible_timestep)                                                           # Choose timestep, limiting it to the safest value of the options available
if simEndtime > cprofile_timestamps[-1]:                                                                                # Ensure that the current profile will run until the desired sim end time
    simEndtime = cprofile_timestamps[-1]
steps = int(simEndtime/timeStep)+1

################################################### Setup ##############################################################
numerix.set_printoptions(precision=4, threshold=numerix.inf, linewidth=numerix.inf)                                     # Prevent wrapping of printed NumPy arrays

if det_residual_live_plot == 'On':                                                                                      # Interactive plotting of PDE residuals
    plt.ion()
plt.rcParams.update({'font.size': 16})
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.minor.size'] = 6
filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
filterwarnings("ignore", category=UserWarning, module="matplotlib")

##################################### Setting up Interactive Residual Plots ############################################
def configure_residual_plots():
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    f.tight_layout()
    f.set_size_inches(18.5, 10.5)
    f.patch.set_facecolor('white')
    ax1.set_title('$\phi_{s_{pos}}$ Residual')
    ax1.set_ylabel('$\phi_{s_{pos}}$ Residual')
    ax2.set_title('$C_{e}$ Residual')
    ax2.set_ylabel('$C_{e}$ Residual')
    ax3.set_title('$\phi_{e}$ Residual')
    ax3.set_ylabel('$\phi_{e}$ Residual')
    ax4.set_title('$\phi_{s_{neg}}$ Residual')
    ax4.set_ylabel('$\phi_{s_{neg}}$ Residual')
    ax5.set_title('$C_{s_{neg}}$ Residual')
    ax5.set_ylabel('$C_{s_{neg}}$ Residual')
    ax6.set_title('$C_{s_{pos}}$ Residual')
    ax6.set_ylabel('$C_{s_{pos}}$ Residual')
    plt.minorticks_on()
    ax1.grid(True, linestyle='-', color='0.65')
    ax2.grid(True, linestyle='-', color='0.65')
    ax3.grid(True, linestyle='-', color='0.65')
    ax4.grid(True, linestyle='-', color='0.65')
    ax5.grid(True, linestyle='-', color='0.65')
    ax6.grid(True, linestyle='-', color='0.65')
    ax5.set_xlabel('Sweep Count')
    ax6.set_xlabel('Sweep Count')
    return f, ax1, ax2, ax3, ax4, ax5, ax6

##################################### Setting up Interactive Concentration Plots #######################################
def configure_Cs_plots(electrode_Cs_flag):
    num_subplot_rows = 2                                                                                                # Define the number of rows of subplots
    if electrode_Cs_flag == 'neg':
        num_subplot_columns = int(nx_neg/num_subplot_rows)                                                              # Number of columns is number of particles in electrode / number of rows of subplots
    else:
        num_subplot_columns = int(nx_pos/num_subplot_rows)
    fig, axs = plt.subplots(num_subplot_rows, num_subplot_columns, sharex=True, sharey=True)                            # Create figure and as many subplots as there are particles. Share y-axis units across all plots in a row, & x-axis for relativity
    fig.subplots_adjust(hspace=.5, wspace=.05)                                                                          # Adjust subplot spacing in figure
    axs = axs.ravel()                                                                                                   # Flatten array of axes objects, creating a 1D array

    fig.tight_layout()
    fig.set_size_inches(18.5, 10.5)
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.01, '$C_{s}$, ($mol/m^3$)', ha='center')                                                            # Adding a common x-axis label for all axes
    fig.text(0.04, 0.5, 'Particle Radius (dimensionless)', va='center', rotation='vertical')                            # Adding a common y-axis label for all axes
    if electrode_Cs_flag == 'neg':
        fig.suptitle('Neg. Electrode, Li Concentration vs. Particle Radius', size=15)                                                   # Adding a common title for all axes in figure
    else:
        fig.suptitle('Pos. Electrode, Li Concentration vs. Particle Radius', size=15)                                                   # Adding a common title for all axes in figure
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    for axes_object_index, axes_object in enumerate(axs):                                                               # For every subplot in the figure
        plt.minorticks_on()
        axes_object.grid(True, linestyle='-', color='0.65')
        axes_object.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))                                               # Limit the no. of decimal places on Cs (horizontal axis) tick values
        axes_object.set_ylim([0, 1])                                                                                    # Set the particle radius / y-axis limits, for normalised radius
        axes_object.set_title('Particle ' + str(axes_object_index+1), size=14)                                          # Add a title to each subplot, stating which particle it shows (1st particle is 1, i.e. 1-indexed)
        plt.setp(axes_object.get_xticklabels(), rotation=30, horizontalalignment='right', size=15)                      # Rotate the x-tick labels to prevent them overlapping
        plt.setp(axes_object.get_yticklabels(), size=15)                                                                # Match x-tick & y-tick label font size
    return fig, axs

############################### Instantiate Cell & Both Electrodes, inc. Properties ####################################
sim_cell = Battery(cell_temperature, params_dict['Q_Ah'], params_dict['brug'], params_dict['alpha'],                    # Instantiate battery class instance
                   params_dict['A'],params_dict['De'],params_dict['t_plus'])
sim_cell.Battery_scale_quantities(params_dict['L_neg'] + params_dict['L_sep'] + params_dict['L_pos'],
                                  params_dict['kappa_string'])

neg = Electrode(params_dict['L_neg'],params_dict['Rs_neg'],params_dict['Ds_neg'],params_dict['epsilon_s_neg'],          # Instantiate negative electrode class instance
                params_dict['sigma_neg'],params_dict['theta_min_neg'],params_dict['theta_max_neg'],
                params_dict['k_neg_norm'],params_dict['cs_max_neg'],params_dict['Uocp_neg_string'],
                params_dict['epsilon_e_neg'],params_dict['n'],cell_temperature,params_dict['Q_Ah'],
                params_dict['brug'],params_dict['alpha'],params_dict['A'],params_dict['De'],params_dict['t_plus'],
                theta_table_vals_neg, Uocp_table_vals_neg)

pos = Electrode(params_dict['L_pos'],params_dict['Rs_pos'],params_dict['Ds_pos'],params_dict['epsilon_s_pos'],          # Instantiate positive electrode class instance
                params_dict['sigma_pos'],params_dict['theta_min_pos'],params_dict['theta_max_pos'],
                params_dict['k_pos_norm'],params_dict['cs_max_pos'],params_dict['Uocp_pos_string'],
                params_dict['epsilon_e_pos'],params_dict['n'], cell_temperature,params_dict['Q_Ah'],
                params_dict['brug'],params_dict['alpha'],params_dict['A'],params_dict['De'],params_dict['t_plus'],
                theta_table_vals_pos, Uocp_table_vals_pos)

sep = Separator(params_dict['L_sep'], params_dict['epsilon_e_sep'],params_dict['n'],                                    # Instantiate separator class instance
                cell_temperature,params_dict['Q_Ah'], params_dict['brug'],params_dict['alpha'],
                params_dict['A'],params_dict['De'],params_dict['t_plus'])

############################ Create Computational Domains for the Cell & Both Electrodes ###############################
if non_uniform_axial_meshing in ('On', 'on'):                                                                           # If user has specified uniform meshing, generate uniform meshes
    neg.define_non_uniform_comp_domain_ax(nx_neg, L_neg_normalised)
    pos.define_non_uniform_comp_domain_ax(nx_pos, L_pos_normalised)
    sep.define_non_uniform_comp_domain(nx_sep, L_sep_normalised)
    dx_cell = numerix.concatenate((neg.dx, sep.dx, pos.dx))                                                             # Aggregate the node-spacing vectors from all sub-domains for the global domain
else:                                                                                                                   # User has requested non-uniform node-spacing
    neg.define_uniform_comp_domain_ax(nx_neg, L_neg_normalised)
    pos.define_uniform_comp_domain_ax(nx_pos, L_pos_normalised)
    sep.define_uniform_comp_domain(nx_sep, L_sep_normalised)
    dx_cell = numerix.zeros(nx_tot)
    dx_cell[0:nx_neg] = neg.L_normalised/nx_neg
    dx_cell[nx_neg:nx_neg+nx_sep] = sep.L_normalised/nx_sep
    dx_cell[nx_neg+nx_sep:] = pos.L_normalised/nx_pos
if non_uniform_radial_meshing in ('On', 'on'):  # Radial meshing
    neg.define_non_uniform_comp_domain_rad(nr_neg, Rs_neg_normalised)
    pos.define_non_uniform_comp_domain_rad(nr_pos, Rs_pos_normalised)
else:
    neg.define_uniform_comp_domain_rad(nr_neg, Rs_neg_normalised)
    pos.define_uniform_comp_domain_rad(nr_pos, Rs_pos_normalised)

sim_cell.define_global_mesh(dx_cell, L_tot_normalised)                                                                  # Generate the global domain mesh, passing dx_cell, which can be uniform or non-uniform

##################################### Realtime User Simulation Interaction #############################################
def termination_signal_handler(signal, frame):                                                                          # Function to deal with a CTRL-C event safely
    global sim_terminated
    sim_terminated = True
    print('Termination request received. Shutting down simulation..')
signal(SIGINT, termination_signal_handler)

############################### Compute initial values of all cell variables ###########################################
# theta_neg_initial      =   neg.theta_calc_initial(initial_soc_percent) * numerix.ones(nx_neg)                         # Initial neg. electrode stoichiometry
theta_neg_initial      =   0.63501 * numerix.ones(nx_neg)                                                               # Temporarily hard-coded to match COMSOL setup
# theta_pos_initial      =   pos.theta_calc_initial(initial_soc_percent) * numerix.ones(nx_pos)                         # Initial pos. electrode stoichiometry
theta_pos_initial      =   0.257 * numerix.ones(nx_pos)                                                                 # Temporarily hard-coded to match COMSOL setup
phi_s_neg_initial      =   numerix.zeros(nx_neg)                                                                        # Initial potential in neg. electrode (assumes equilibrium)
phi_s_pos_initial      =   pos.interpolated_Uocp_fn(theta_pos_initial) - neg.interpolated_Uocp_fn(theta_neg_initial[0]) # Initial potential in pos. electrode (assumes equilibrium)
# phi_s_pos_initial      =   interpolated_Uocp_pos_fn(theta_pos_initial) - interpolated_Uocp_neg_fn(theta_neg_initial[0])

cs_surface_neg_initial =   neg.calc_cs_surface_initial(theta_neg_initial)
cs_surface_pos_initial =   pos.calc_cs_surface_initial(theta_pos_initial)
phi_e_initial          = - neg.interpolated_Uocp_fn(theta_neg_initial[0])                                               # Negative because?
ce_initial             =   params_dict['ce_initial']

################################## Define Electrode-level FiPy Cell & Face Variables ###################################
neg.define_cellvariables(phi_s_neg_initial, cs_surface_neg_initial, phi_e_initial, theta_neg_initial,ce_initial, sim_cell)
pos.define_cellvariables(phi_s_pos_initial, cs_surface_pos_initial, phi_e_initial, theta_pos_initial,ce_initial, sim_cell)
neg.define_facemirrors(phi_s_neg_initial[0], phi_e_initial)
pos.define_facemirrors(phi_s_pos_initial[0], phi_e_initial)

##################### Define Cell-level 'FiPy CellVariables' and Initialise Operating Point ############################
j_battery_value = numerix.zeros(nx_tot)                                                                                 # Create a vector for flux density - assumes equilibrium
j_battery_value[0:nx_neg] = neg.calc_BV(neg.j0, neg.overpotential)                                                      # Calculate flux density in the neg. electrode..
j_battery_value[nx_neg + nx_sep:] = pos.calc_BV(pos.j0, pos.overpotential)                                              # ..& in the pos. electrode
sim_cell.Battery_cellvariables(neg.epsilon_e, sep.epsilon_e, pos.epsilon_e, neg.a_s, pos.a_s, nx_neg, nx_sep, nx_pos,   # Define battery-level cellvariables
                               j_battery_value, ce_initial, phi_e_initial, neg.L, sep.L, pos.L, neg.De_eff, sep.De_eff,
                               pos.De_eff)

############################### Electrolyte Concentration PDE and time-invariant BCs ###################################
Ce_source_term = sim_cell.L*sim_cell.a_s*(1.0 - sim_cell.t_plus)*sim_cell.j_battery
Ce_PDE = TransientTerm(coeff=sim_cell.epsilon_e*sim_cell.L) == (DiffusionTerm(coeff
                                                                              = (sim_cell.De_eff.harmonicFaceValue      # Define the diffusion coefficient on all faces in cell-level mesh. No interp issues here
                                                                                 / sim_cell.L.harmonicFaceValue))       # Only issue is not being able to specify two different diff. coefficients at the same face between electrode & separator. L also weighted
                                                                + Ce_source_term)
sim_cell.Ce.faceGrad.constrain(0., where=sim_cell.axial_mesh.facesLeft)                                                 # Constant, zero-flux boundary condition at LHS of battery domain
sim_cell.Ce.faceGrad.constrain(0., where=sim_cell.axial_mesh.facesRight)                                                # Constant, zero-flux boundary condition at RHS of battery domain

############################################# Electrolyte Potential PDE ################################################
kappa_eff = FaceVariable(mesh = sim_cell.axial_mesh,                                                                    # Interpolation method. Little/no difference
                         value = interpolated_kappa_fn(0.001*sim_cell.Ce.harmonicFaceValue)
                                 *sim_cell.epsilon_e_eff.harmonicFaceValue)
kappa_D_eff = FaceVariable(mesh = sim_cell.axial_mesh,
                           value = kappa_eff*(2.0*R*sim_cell.T/F)*(sim_cell.t_plus - 1.0))
phi_e_DiffusiveSource_term = -((kappa_D_eff/(sim_cell.L.harmonicFaceValue))                                             # Meld difference with above was about 10^-6 error after 1.7 sec with 1C
                               *((numerix.log(sim_cell.Ce)).faceGrad)).divergence
phi_e_ReactionSource_term  = -(sim_cell.L*sim_cell.a_s*F*sim_cell.j_battery)
phi_e_PDE = DiffusionTerm(kappa_eff/sim_cell.L.harmonicFaceValue) == (phi_e_DiffusiveSource_term
                                                                      + phi_e_ReactionSource_term)

sim_cell.phi_e.faceGrad.constrain([0.0], where = sim_cell.axial_mesh.facesRight)                                        # Constant, zero-flux boundary condition at LHS of battery domain
sim_cell.phi_e.faceGrad.constrain([0.0], where = sim_cell.axial_mesh.facesLeft)                                         # Constant, zero-flux boundary condition at RHS of battery domain

#################################### Define the Solid Potential PDEs and BCs ###########################################
phi_s_neg_PDE = DiffusionTerm(coeff=neg.sigma_eff/neg.L) == neg.L*neg.a_s*F*neg.calc_BV(neg.j0, neg.overpotential)
neg.phi_s.constrain(0., where=neg.axial_mesh.facesLeft)                                                                 # Constant, Dirichlet boundary condition at LHS of domain. Electrical ground point
neg.phi_s.faceGrad.constrain(0., where=neg.axial_mesh.facesRight)                                                       # Constant, zero-flux boundary condition at RHS of neg. electrode domain

phi_s_pos_PDE = DiffusionTerm(coeff=pos.sigma_eff/pos.L) == pos.L*pos.a_s*F*pos.calc_BV(pos.j0, pos.overpotential)
pos.phi_s.faceGrad.constrain(0., where=pos.axial_mesh.facesLeft)                                                        # Constant, zero-flux boundary condition at LHS of battery domain
solidpotential_flux_at_pos_BC = Variable()                                                                              # Create a FiPy variable for the time-varying Neumann BC at RHS of the pos. electrode
pos.phi_s.faceGrad.constrain(solidpotential_flux_at_pos_BC, where=pos.axial_mesh.facesRight)                            # Assign the value of that variable to the RHS BC

################################# Define the Solid Phase Concentration PDEs & BCs ######################################
neg.Cs_p2d_diffcoeff_function()                                                                                         # Compute the 1D diffusion coefficient for the electrode particles
Cs_neg_p2d_PDE = TransientTerm(coeff=(neg.p2d_mesh.y**2.0)*neg.Rs) == DiffusionTerm(coeff=neg.Cs_p2d_diffCoeff)
species_flux_neg_particle_surf = FaceVariable(mesh=neg.p2d_mesh)                                                        # Create a FiPy Facevariable for the time-varying particle surface flux

species_flux_neg_particle_surf.setValue(neg.surface_BC_Cs(neg.phi_s_facemirror,                                         # Compute the surface flux & set the Facevariable to that value
                                                          neg.phi_e_facemirror,
                                                          neg.Cs_p2d.faceValue,                                         # N.B. Not necessary to correct/extrap. facevalues here because we start from equilibrium with cs_surface_neg_initial everywhere
                                                          )
                                        )
neg.Cs_p2d.faceGrad.constrain(species_flux_neg_particle_surf, where=neg.p2d_mesh.facesTop)                              # Assign that Facevariable value to the FV face at the top of the particle mesh. N.B. gradient specification influences facevalue
neg.particle_centre_noflux_BC()                                                                                         # Set the no-flux BC at the particle's centre

pos.Cs_p2d_diffcoeff_function()                                                                                         # Repeat the above for the pos. electrode
Cs_pos_p2d_PDE = TransientTerm(coeff=(pos.p2d_mesh.y**2.0)*pos.Rs) == DiffusionTerm(coeff=pos.Cs_p2d_diffCoeff)
species_flux_pos_particle_surf = FaceVariable(mesh=neg.p2d_mesh)
species_flux_pos_particle_surf.setValue(pos.surface_BC_Cs(pos.phi_s_facemirror,
                                                          pos.phi_e_facemirror,
                                                          pos.Cs_p2d.faceValue                                          # N.B. Not necessary to correct/extrap. facevalues here because we start from equilibrium with cs_surface_pos_initial everywhere
                                                          )
                                        )
pos.Cs_p2d.faceGrad.constrain(species_flux_pos_particle_surf, where=pos.p2d_mesh.facesTop)
pos.particle_centre_noflux_BC()

################################################### Logging ############################################################
if datalogging_switch == 'On':
    if timeStep > record_period:
        print('Time step is larger than datalogger recording period. Data may not be logged periodically. Reduce timeStep size or increase the record_period.')
        exit('Simulation terminated')

    record_timepoint = 0.0
    array_length = int(numerix.ceil(simEndtime / record_period)) + 1
    cs_dummy_neg, cs_width_neg = (numerix.flipud(numerix.array(neg.Cs_p2d).reshape(nr_neg, nx_neg))).shape
    cs_dummy_pos, cs_width_pos = (numerix.flipud(numerix.array(pos.Cs_p2d).reshape(nr_pos, nx_pos))).shape
    array_recorder_head_general, array_recorder_head_cs_neg, array_recorder_head_cs_pos = 0, 0, 0

    time_independent_data = [['sim_cell.axial_mesh.cellCenters[0]', sim_cell.axial_mesh.cellCenters[0]],                # x-axis FV cell-centre co-ordinates in the cell-level axial mesh
                             ['sim_cell.axial_mesh.faceCenters[0]', sim_cell.axial_mesh.faceCenters[0]],                # x-axis FV face co-ordinates in the cell-level axial mesh
                             ['neg.axial_mesh.cellCenters[0]', neg.axial_mesh.cellCenters[0]],                          # x-axis FV cell-centre co-ordinates in the neg. electrode axial mesh
                             ['neg.axial_mesh.faceCenters[0]', neg.axial_mesh.faceCenters[0]],                          # x-axis FV face co-ordinates in the neg. electrode axial mesh
                             ['pos.axial_mesh.cellCenters[0]', pos.axial_mesh.cellCenters[0]],                          # x-axis FV cell-centre co-ordinates in the pos. electrode axial mesh
                             ['pos.axial_mesh.faceCenters[0]', pos.axial_mesh.faceCenters[0]],                          # x-axis FV face co-ordinates in the pos. electrode axial mesh
                             ['neg.p2d_mesh.cellCenters[1]', neg.p2d_mesh.cellCenters[1]],                              # y-axis FV cell-centre co-ordinates in the neg. electrode particle radial mesh
                             ['neg.p2d_mesh.faceCenters[1]', neg.p2d_mesh.faceCenters[1]],                              # y-axis FV face co-ordinates in the neg. electrode particle radial mesh
                             ['pos.p2d_mesh.cellCenters[1]', pos.p2d_mesh.cellCenters[1]],                              # y-axis FV cell-centre co-ordinates in the pos. electrode particle radial mesh
                             ['pos.p2d_mesh.faceCenters[1]', pos.p2d_mesh.faceCenters[1]]                               # y-axis FV face co-ordinates in the pos. electrode particle radial mesh
                             ]

    data_array_definitions = [('simtime', array_length, 1),         # Want to log a new variable? Add it here
                             ('current_applied', array_length, 1),
                             ('sweep_count', array_length, 1),
                             ('sweep_duration', array_length, 1),
                             ('res_phi_s_neg', array_length, 1),
                             ('res_phi_s_pos', array_length, 1),
                             ('res_Ce_sim_cell', array_length, 1),
                             ('res_phi_e_sim_cell', array_length, 1),
                             ('res_Cs_neg', array_length, 1),
                             ('res_Cs_pos', array_length, 1),
                             ('ce', array_length, len(sim_cell.Ce)),
                             ('ce_Facevals', array_length, len(sim_cell.Ce.faceValue)),
                             ('phi_e', array_length, len(sim_cell.phi_e)),
                             ('phi_e_Facevals', array_length, len(sim_cell.phi_e.faceValue)),
                             ('cs_neg', (nr_neg * array_length), cs_width_neg),
                             ('cs_neg_Facevals', array_length, len(neg.Cs_p2d.faceValue)),                              # N.B. Cs_p2d.faceValue is only used for dimensions here, so no need to correct/extrap. for values
                             ('cs_pos', (nr_pos * array_length), cs_width_pos),
                             ('cs_pos_Facevals', array_length, len(pos.Cs_p2d.faceValue)),                              # N.B. Cs_p2d.faceValue is only used for dimensions here, so no need to correct/extrap. for values
                             ('phi_s_neg', array_length, len(neg.phi_s)),
                             ('phi_s_neg_Facevals', array_length, len(neg.phi_s.faceValue)),
                             ('phi_s_pos', array_length, len(pos.phi_s)),
                             ('phi_s_pos_Facevals', array_length, len(pos.phi_s.faceValue)),
                             ('flux_density_j_neg', array_length, len(j_battery_value[0:nx_neg])),
                             ('flux_density_j_neg_Facevals', array_length, len(sim_cell.j_battery.faceValue[0:nx_neg+1])),
                             ('flux_density_j_pos', array_length, len(j_battery_value[nx_neg + nx_sep:])),
                             ('flux_density_j_pos_Facevals', array_length, len(sim_cell.j_battery.faceValue[nx_neg+nx_sep:])),
                             ('exchange_flux_density_j0_neg', array_length, len(neg.j0)),
                             ('exchange_flux_density_j0_neg_Facevals', array_length, len(neg.j0.faceValue)),
                             ('exchange_flux_density_j0_pos', array_length, len(pos.j0)),
                             ('exchange_flux_density_j0_pos_Facevals', array_length, len(pos.j0.faceValue)),
                             ('overpotential_neg', array_length, len(neg.overpotential)),
                             ('overpotential_neg_Facevals', array_length, len(neg.overpotential.faceValue)),
                             ('overpotential_pos', array_length, len(pos.overpotential)),
                             ('overpotential_pos_Facevals', array_length, len(pos.overpotential.faceValue)),
                             ('uocp_neg', array_length, len(neg.uocp)),
                             ('uocp_neg_Facevals', array_length, len(neg.uocp.faceValue)),
                             ('uocp_pos', array_length, len(pos.uocp)),
                             ('uocp_pos_Facevals', array_length, len(pos.uocp.faceValue)),
                             ('gc_object_count', array_length, 1)
                             ]

    time_dependent_data = OrderedDict((variable, numerix.zeros((array_length, array_width), dtype=float))
        for variable, array_length, array_width in data_array_definitions)

if det_residual_saving == 'On':
    res_phi_s_neg_tracker, res_phi_s_pos_tracker, res_Ce_sim_cell_tracker, res_phi_e_sim_cell_tracker, res_Cs_neg_tracker, res_Cs_pos_tracker = [], [], [], [], [], []
    res_trackers = [res_phi_s_neg_tracker, res_phi_s_pos_tracker, res_Ce_sim_cell_tracker, res_phi_e_sim_cell_tracker, res_Cs_neg_tracker, res_Cs_pos_tracker]

def log(*vars_logged):
    for index, allocated_array in enumerate(time_dependent_data):
        if allocated_array not in ('cs_neg', 'cs_pos'):
            time_dependent_data[allocated_array][array_recorder_head_general] = vars_logged[index]
        elif allocated_array == 'cs_neg':
            time_dependent_data[allocated_array][range(array_recorder_head_cs_neg, array_recorder_head_cs_neg + nr_neg)] = vars_logged[index]
        else:
            time_dependent_data[allocated_array][range(array_recorder_head_cs_pos, array_recorder_head_cs_pos + nr_pos)] = vars_logged[index]

def save_data(output_location, simtime):
    print('Writing data to disk..')
    output_location_timestamped = output_location + '_' + datetime.fromtimestamp(sim_clock_start_time).strftime('%d-%m-%Y_%H-%M-%S')
    if not path.exists(output_location_timestamped):
        makedirs(output_location_timestamped)

    timedelta_seconds = timedelta(seconds=(sim_clock_end_time - sim_clock_start_time)).total_seconds() % 60             # Compute CPU time for simulation in seconds
    sim_timing_data = [datetime.fromtimestamp(sim_clock_start_time).strftime('%d/%m/%Y, %H:%M:%S'),
                       datetime.fromtimestamp(sim_clock_end_time).strftime('%d/%m/%Y, %H:%M:%S'),
                       timedelta(seconds=(sim_clock_end_time - sim_clock_start_time)),
                       (timedelta_seconds/min(simEndtime, simtime))                                                     # CPU time (s)/Simulation runtime (s)
                       ]

    with open(output_location_timestamped + "\sim_settings_used.csv", 'wb') as sim_settings_CSV:
        sim_settings_writer = writer(sim_settings_CSV)
        for key, value in settings_dict.items():
            sim_settings_writer.writerow([key, value])
    with open(output_location_timestamped + "\sim_timing_data.csv", 'wb') as sim_timing_data_CSV:
        sim_timing_writer = writer(sim_timing_data_CSV)
        sim_timing_writer.writerow(sim_timing_data)
    for array in time_independent_data:
        with open(output_location_timestamped + '\\' + array[0] + '.csv', 'wb') as t_ind_data_CSV:
            t_ind_writer = writer(t_ind_data_CSV)
            t_ind_writer.writerows(izip(array[1]))
    for array in time_dependent_data:
        with open(output_location_timestamped + '\\' + array + '.csv', 'wb') as t_dep_data_CSV:
            t_dep_writer = writer(t_dep_data_CSV)
            t_dep_writer.writerows(time_dependent_data[array])

    print('Writing complete')
    return sim_timing_data, output_location_timestamped

################################# Simulation and Visualisation #########################################################
simtime = 0.0
sim_terminated = False
step = 1
if mem_leak_analysis == 'On':
    tracker = SummaryTracker()

print('Simulating..')
################################# Timestepping Loop Starts Here ########################################################
while (simtime < simEndtime) or numerix.isclose(simtime, simEndtime):
    current_applied = interpolated_current_fn(simtime)
    print('time: ' + str(simtime), '\tcurrent applied : ' + str(current_applied))
    sim_cell.Ce.updateOld()
    neg.Cs_p2d.updateOld()
    pos.Cs_p2d.updateOld()

    solidpotential_flux_at_pos_BC.setValue(-(pos.L * current_applied) / (pos.sigma_eff * sim_cell.A))

    res_phi_s_neg = res_phi_s_pos = res_Cs_neg = res_Cs_pos = res_phi_e_sim_cell = res_Ce_sim_cell = 1e6

    sweep_count = 0

    if det_residual_saving == 'On':
        for residual_list in res_trackers:
            del residual_list[:]

    sweep_start_time = timegm(gmtime())

################################# Numerical Iteration Loop Starts Here #################################################
    while ((res_phi_s_neg > tolerance_phi_s_neg_res
           or res_phi_s_pos > tolerance_phi_s_pos_res
           or res_Cs_neg > tolerance_Cs_neg_res
           or res_Cs_pos > tolerance_Cs_pos_res
           or res_phi_e_sim_cell > tolerance_phi_e_sim_cell_res
           or res_Ce_sim_cell > tolerance_Ce_sim_cell_res)
           or sweep_count < 2):

################################################# Sweep phi_s_pos ######################################################
        res_phi_s_pos = phi_s_pos_PDE.sweep(pos.phi_s, underRelaxation = under_relax_factor_phi_s_pos)

############################################ Update Pos. Overpotential #################################################
        pos.update_overpotentials()

        j_battery_value[nx_neg + nx_sep:] = pos.calc_BV(pos.j0, pos.overpotential)
        sim_cell.j_battery.value = j_battery_value

        res_Ce_sim_cell = Ce_PDE.sweep(sim_cell.Ce, dt=timeStep, underRelaxation = float(under_relax_factor_Ce))

        neg.j0.value = neg.k_norm * numerix.absolute(((neg.cs_max - numerix.array(neg.Cs_surface))/neg.cs_max)**(1-sim_cell.alpha)) * numerix.absolute((neg.Cs_surface/neg.cs_max)**sim_cell.alpha) * numerix.array(numerix.absolute((sim_cell.Ce[0:nx_neg]/ce_initial)**(1.0-sim_cell.alpha)))
        pos.j0.value = pos.k_norm * numerix.absolute(((pos.cs_max - numerix.array(pos.Cs_surface))/pos.cs_max)**(1-sim_cell.alpha)) * numerix.absolute((pos.Cs_surface/pos.cs_max)**sim_cell.alpha) * numerix.array(numerix.absolute((sim_cell.Ce[nx_neg + nx_sep:]/ce_initial)**(1.0-sim_cell.alpha)))

        j_battery_value[0:nx_neg] = neg.calc_BV(neg.j0, neg.overpotential)
        j_battery_value[nx_neg + nx_sep:] = pos.calc_BV(pos.j0, pos.overpotential)
        sim_cell.j_battery.value = j_battery_value

        kappa_eff.setValue(sim_cell.kappa(0.001*sim_cell.Ce.harmonicFaceValue) * sim_cell.epsilon_e_eff.harmonicFaceValue)
        kappa_D_eff.setValue(kappa_eff * (2.0 * R * sim_cell.T/F) * (sim_cell.t_plus - 1.0))

        res_phi_e_sim_cell = phi_e_PDE.sweep(sim_cell.phi_e, underRelaxation = float(under_relax_factor_phi_e))

        neg.phi_e_copy_from_supermesh.value = numerix.array(sim_cell.phi_e[0:nx_neg])
        pos.phi_e_copy_from_supermesh.value = numerix.array(sim_cell.phi_e[nx_neg + nx_sep:])

################################################# Update Overpotentials ################################################
        neg.update_overpotentials()
        pos.update_overpotentials()

############################################### Update Reaction Current ################################################
        j_battery_value[0:nx_neg] = neg.calc_BV(neg.j0, neg.overpotential)
        j_battery_value[nx_neg + nx_sep:] = pos.calc_BV(pos.j0, pos.overpotential)
        sim_cell.j_battery.value = j_battery_value

################################################# Sweep phi_s_neg ######################################################
        res_phi_s_neg = phi_s_neg_PDE.sweep(neg.phi_s, underRelaxation=under_relax_factor_phi_s_neg)

############################################ Update Neg. Overpotential #################################################
        neg.update_overpotentials()

######################################## Update Neg. Electrode Reaction Current ########################################
        j_battery_value[0:nx_neg] = neg.calc_BV(neg.j0, neg.overpotential)
        sim_cell.j_battery.value = j_battery_value

############################## Correct Cs Values at Particle Surfaces & Compute Li Surface Flux ########################
        surface_Cs_vals_neg, centre_Cs_vals_neg = neg.get_true_Cs_facevals()                                            # Computing the correct Cs values at particle surface & centre
        neg.insert_corrected_Cs_facevalues(neg.Cs_p2d.faceValue, surface_Cs_vals_neg, centre_Cs_vals_neg)               # Replace the incorrect top & bottom facevalues with the correct values
        species_flux_neg_particle_surf.setValue(neg.surface_BC_Cs(neg.phi_s_facemirror,                                 # Update the value of the species flux at the particle surface
                                                                  neg.phi_e_facemirror,
                                                                  neg.corrected_Cs_p2D_faceValue))                      # A revised vector of facevalues is passed, with corrected surface & centre facevalues

        surface_Cs_vals_pos, centre_Cs_vals_pos = pos.get_true_Cs_facevals()                                            # Computing the correct Cs values at particle surface & centre
        pos.insert_corrected_Cs_facevalues(pos.Cs_p2d.faceValue, surface_Cs_vals_pos, centre_Cs_vals_pos)               # Replace the incorrect top & bottom facevalues with the correct values
        species_flux_pos_particle_surf.setValue(pos.surface_BC_Cs(pos.phi_s_facemirror,
                                                                  pos.phi_e_facemirror,
                                                                  pos.corrected_Cs_p2D_faceValue))                      # A revised vector of facevalues is passed, with corrected surface & centre facevalues

############################################### Sweep for Cs ###########################################################
        res_Cs_neg = Cs_neg_p2d_PDE.sweep(neg.Cs_p2d, dt=timeStep,underRelaxation =under_relax_factor_Cs_neg)           # Generates new Cs values for neg.Cs_p2d
        res_Cs_pos = Cs_pos_p2d_PDE.sweep(pos.Cs_p2d, dt=timeStep,underRelaxation = under_relax_factor_Cs_pos)          # Generates new Cs values for pos.Cs_p2d

################################# Update the Value of Our Axial Mesh Cs_surface Variable ###############################
        surface_Cs_vals_neg, centre_Cs_vals_neg = neg.get_true_Cs_facevals()                                            # Computing the correct Cs values at particle surface. Centre vals returned here, but unused
        neg.Cs_surface.value = surface_Cs_vals_neg                                                                      # Update the axial mesh Cs value with the corrected Cs p2d mesh upper facevalues

        surface_Cs_vals_pos, centre_Cs_vals_pos = pos.get_true_Cs_facevals()                                            # Computing the correct Cs values at particle surface. Centre vals returned here, but unused
        pos.Cs_surface.value = surface_Cs_vals_pos                                                                      # Update the axial mesh Cs value with the corrected Cs p2d mesh upper facevalues

######################################## Update Value of Exchange Current Density ######################################
        neg.j0.value = neg.k_norm * numerix.absolute(((neg.cs_max - numerix.array(neg.Cs_surface))/neg.cs_max)**(1-sim_cell.alpha)) * numerix.absolute((neg.Cs_surface/neg.cs_max)**sim_cell.alpha) * numerix.array(numerix.absolute((sim_cell.Ce[0:nx_neg]/ce_initial)**(1.0-sim_cell.alpha)))
        pos.j0.value = pos.k_norm * numerix.absolute(((pos.cs_max - numerix.array(pos.Cs_surface))/pos.cs_max)**(1-sim_cell.alpha)) * numerix.absolute((pos.Cs_surface/pos.cs_max)**sim_cell.alpha) * numerix.array(numerix.absolute((sim_cell.Ce[nx_neg + nx_sep:]/ce_initial)**(1.0-sim_cell.alpha)))

################################################# Update Overpotentials ################################################
        neg.update_overpotentials()
        pos.update_overpotentials()

############################################### Update Reaction Current ################################################
        j_battery_value[0:nx_neg] = neg.calc_BV(neg.j0, neg.overpotential)
        j_battery_value[nx_neg + nx_sep:] = pos.calc_BV(pos.j0, pos.overpotential)
        sim_cell.j_battery.value = j_battery_value

        current_time = sim_clock_end_time = timegm(gmtime())
        sweep_duration = current_time - sweep_start_time

########################################## Emergency Datalogging #######################################################
        if sim_terminated:                                                                                              # If user caused CTRL-C event, exit
            if datalogging_switch == 'On':                                                                              # First check if datalogging is & data should be saved

                neg.insert_corrected_Cs_facevalues(neg.Cs_p2d.faceValue, surface_Cs_vals_neg, centre_Cs_vals_neg)       # Needs to updated again before logging, because Cs was re-calculated since this fn. was last called
                pos.insert_corrected_Cs_facevalues(pos.Cs_p2d.faceValue, surface_Cs_vals_pos, centre_Cs_vals_pos)       # Needs to updated again before logging, because Cs was re-calculated since this fn. was last called

                vars_logged = [simtime, current_applied, (sweep_count+1), sweep_duration, res_phi_s_neg, res_phi_s_pos, # If so, save it
                               res_Ce_sim_cell, res_phi_e_sim_cell, res_Cs_neg, res_Cs_pos, sim_cell.Ce,
                               sim_cell.Ce.faceValue, sim_cell.phi_e, sim_cell.phi_e.faceValue,
                               (numerix.flipud(numerix.array(neg.Cs_p2d).reshape(nr_neg, nx_neg))), neg.corrected_Cs_p2D_faceValue,
                               (numerix.flipud(numerix.array(pos.Cs_p2d).reshape(nr_pos, nx_pos))),
                               pos.corrected_Cs_p2D_faceValue, neg.phi_s, neg.phi_s.faceValue, pos.phi_s, pos.phi_s.faceValue,
                               j_battery_value[0:nx_neg], sim_cell.j_battery.faceValue[0:nx_neg + 1],
                               j_battery_value[nx_neg + nx_sep:], sim_cell.j_battery.faceValue[nx_neg + nx_sep:],
                               neg.j0, neg.j0.faceValue, pos.j0, pos.j0.faceValue, neg.overpotential,
                               neg.overpotential.faceValue, pos.overpotential, pos.overpotential.faceValue, neg.uocp,
                               neg.uocp.faceValue, pos.uocp, pos.uocp.faceValue, len(get_objects())]
                log(*vars_logged)
                sim_timing_data, output_location_timestamped = save_data(output_location, simtime)
            print('Simulation terminated. Runtime was ' + str(timedelta(seconds=(sim_clock_end_time                     # Inform user of shutdown
                                                                                 - sim_clock_start_time))))
            exit()

########################################## Realtime Residual Plots #####################################################
        if det_residual_saving == 'On':                                                                                 # If user has requested residual data be saved
            res_phi_s_pos_tracker.append(res_phi_s_pos)                                                                 # Record residual data in lists
            res_Ce_sim_cell_tracker.append(res_Ce_sim_cell)
            res_phi_e_sim_cell_tracker.append(res_phi_e_sim_cell)
            res_phi_s_neg_tracker.append(res_phi_s_neg)
            res_Cs_neg_tracker.append(res_Cs_neg)
            res_Cs_pos_tracker.append(res_Cs_pos)
        if det_residual_live_plot == 'On':                                                                              # If user has requested realtime plotting of residuals
            if sweep_count == 0:
                f1, ax1, ax2, ax3, ax4, ax5, ax6 = configure_residual_plots()                                           # Generate the figure & subplots on the first iteration
            else:
                residual_value_ax1.remove()                                                                             # On all iterations after the 1st, remove the previous value annotation to make way for the next
                residual_value_ax2.remove()
                residual_value_ax3.remove()
                residual_value_ax4.remove()
                residual_value_ax5.remove()
                residual_value_ax6.remove()
            ax1.plot(sweep_count, res_phi_s_pos, 'ro', markersize=8)                                                    # Plot the residual
            residual_value_ax1 = ax1.annotate('Residual = ' + str('%.2E' % res_phi_s_pos), xy=(1, 1),                   # Annotate the point, stating its value
                                              xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
            ax2.plot(sweep_count, res_Ce_sim_cell, 'ro', markersize=8)
            residual_value_ax2 = ax2.annotate('Residual = ' + str('%.2E' % res_Ce_sim_cell), xy=(1, 1),
                                              xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
            ax3.plot(sweep_count, res_phi_e_sim_cell, 'ro', markersize=8)
            residual_value_ax3 = ax3.annotate('Residual = ' + str('%.2E' % res_phi_e_sim_cell), xy=(1, 1),
                                              xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
            ax4.plot(sweep_count, res_phi_s_neg, 'ro', markersize=8)
            residual_value_ax4 = ax4.annotate('Residual = ' + str('%.2E' % res_phi_s_neg), xy=(1, 1),
                                              xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
            ax5.plot(sweep_count, res_Cs_neg, 'ro', markersize=8)
            residual_value_ax5 = ax5.annotate('Residual = ' + str('%.2E' % res_Cs_neg), xy=(1, 1),
                                              xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
            ax6.plot(sweep_count, res_Cs_pos, 'ro', markersize=8)
            residual_value_ax6 = ax6.annotate('Residual = ' + str('%.2E' % res_Cs_pos), xy=(1, 1),
                                              xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.pause(0.05)

        sweep_count += 1                                                                                                # Iterate the sweep counter (should be very last action in this loop)

################################### Numerical Iteration Loop Ends Here #################################################
##################################### Realtime Cs Plots, Both Electrodes ###############################################
    if det_concentrations_live_plot == 'On':                                                                            # If user has requested live-plotting of concentration gradients
        if simtime < timeStep:                                                                                          # If it's the first timestep..
            fig_Cs_live_neg, Cs_live_axs_neg = configure_Cs_plots('neg')                                                # Generate the figure & subplots for neg. electrode
            fig_Cs_live_pos, Cs_live_axs_pos = configure_Cs_plots('pos')                                                # Generate the figure & subplots for pos. electrode
            Cs_y_coordinates_reversed_neg = numerix.fliplr([numerix.unique(neg.p2d_mesh.cellCenters[1])])[0]            # Acquire the node co-ordinates to plot against & reverse them, because they're being plotted on the y-axis, neg. electrode
            top_face_coordinate_neg = numerix.amax(numerix.unique(neg.p2d_mesh.faceCenters[1]))                         # Get the y-coordinate of the particle surface
            bottom_face_coordinate_neg = numerix.amin(numerix.unique(neg.p2d_mesh.faceCenters[1]))                      # Get the y-coordinate of the particle centre
            Cs_y_coordinates_reversed_neg = numerix.concatenate([[top_face_coordinate_neg],                             # Insert surface & centre face coordinates into ends of node coordinates vector
                                                                 Cs_y_coordinates_reversed_neg,
                                                                 [bottom_face_coordinate_neg]]
                                                                )
            Cs_y_coordinates_reversed_pos = numerix.fliplr([numerix.unique(pos.p2d_mesh.cellCenters[1])])[0]            # Acquire the node co-ordinates to plot against & reverse them, because they're being plotted on the y-axis, pos. electrode
            top_face_coordinate_pos = numerix.amax(numerix.unique(pos.p2d_mesh.faceCenters[1]))                         # Get the y-coordinate of the particle surface
            bottom_face_coordinate_pos = numerix.amin(numerix.unique(pos.p2d_mesh.faceCenters[1]))                      # Get the y-coordinate of the particle centre
            Cs_y_coordinates_reversed_pos = numerix.concatenate([[top_face_coordinate_pos],                             # Insert surface & centre face coordinates into ends of node coordinates vector
                                                                 Cs_y_coordinates_reversed_pos,
                                                                 [bottom_face_coordinate_pos]]
                                                                )
            Cs_live_plot_style = dict(linestyle='-', marker='o', markersize=5, markeredgecolor=None)                    # Define the plot formatting, both plots/electrodes

        Cs_neg_nodefaces = numerix.concatenate(([surface_Cs_vals_neg], neg.formatted_Cs_nodevalue_array, [centre_Cs_vals_neg]), axis=0)      # hstack node & face values, adding surface vals vector to top of node array & centre vals vector to bottom
        Cs_pos_nodefaces = numerix.concatenate(([surface_Cs_vals_pos], pos.formatted_Cs_nodevalue_array, [centre_Cs_vals_pos]), axis=0)      # hstack node & face values, adding surface vals vector to top of node array & centre vals vector to bottom

        for particle in xrange(0, nx_neg):                                                                              # For all particles in the neg. electrode
            Cs_live_axs_neg[particle].plot((Cs_neg_nodefaces[:, particle]),                                             # Plot Cs through this particle, at this timestep (Cs on horizontal axis)
                                           Cs_y_coordinates_reversed_neg,                                               # Co-ordinates within particle radius on the plot's vertical axis
                                           **Cs_live_plot_style                                                         # Apply line & marker formatting, defined above
                                           )
        fig_Cs_live_neg.canvas.draw()                                                                                   # Redraw the figure, now that Cs at the new timestep has been plotted
        plt.pause(0.01)                                                                                                 # pause() internally calls fig.canvas.draw(), then plt.show()
        for particle in xrange(0, nx_pos):                                                                              # For all particles in the pos. electrode
            Cs_live_axs_pos[particle].plot((Cs_pos_nodefaces[:, particle]),                                             # Plot Cs through this particle, at this timestep (Cs on horizontal axis)
                                           Cs_y_coordinates_reversed_pos,                                               # Co-ordinates within particle radius on the plot's vertical axis
                                           **Cs_live_plot_style                                                         # Apply line & marker formatting, defined above
                                           )
        fig_Cs_live_pos.canvas.draw()                                                                                   # Redraw the figure, now that Cs at the new timestep has been plotted
        plt.pause(0.001)                                                                                                # pause() internally calls fig.canvas.draw(), then plt.show()


############################################# Standard Datalogging #####################################################
    if datalogging_switch == 'On' and (mpf(abs(simtime - record_timepoint)) < timeStep):
        vars_logged = [simtime, current_applied, sweep_count, sweep_duration, res_phi_s_neg, res_phi_s_pos,
                       res_Ce_sim_cell, res_phi_e_sim_cell, res_Cs_neg, res_Cs_pos, sim_cell.Ce,
                       sim_cell.Ce.faceValue, sim_cell.phi_e, sim_cell.phi_e.faceValue,
                       (numerix.flipud(numerix.array(neg.Cs_p2d).reshape(nr_neg, nx_neg))),
                       neg.corrected_Cs_p2D_faceValue, (numerix.flipud(numerix.array(pos.Cs_p2d).reshape(nr_pos, nx_pos))),
                       pos.corrected_Cs_p2D_faceValue, neg.phi_s, neg.phi_s.faceValue, pos.phi_s, pos.phi_s.faceValue,
                       j_battery_value[0:nx_neg], sim_cell.j_battery.faceValue[0:nx_neg + 1],
                       j_battery_value[nx_neg + nx_sep:], sim_cell.j_battery.faceValue[nx_neg + nx_sep:], neg.j0,
                       neg.j0.faceValue, pos.j0, pos.j0.faceValue, neg.overpotential, neg.overpotential.faceValue,
                       pos.overpotential, pos.overpotential.faceValue, neg.uocp, neg.uocp.faceValue, pos.uocp,
                       pos.uocp.faceValue, len(get_objects())]
        log(*vars_logged)

    if det_residual_live_plot == 'On':
        plt.close(f1)

####################################### Saving Realtime Residual Plots #################################################
    if det_residual_saving == 'On':
        f2, ax1, ax2, ax3, ax4, ax5, ax6 = configure_residual_plots()
        sweep_count_tracker = numerix.linspace(1, sweep_count, sweep_count)
        ax1.plot(sweep_count_tracker, res_phi_s_pos_tracker, 'ro', markersize=8)
        residual_value_ax1 = ax1.annotate('Residual = ' + str('%.2E' % res_phi_s_pos_tracker[-1]), xy=(1, 1), xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
        ax2.plot(sweep_count_tracker, res_Ce_sim_cell_tracker, 'ro', markersize=8)
        residual_value_ax2 = ax2.annotate('Residual = ' + str('%.2E' % res_Ce_sim_cell_tracker[-1]), xy=(1, 1), xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
        ax3.plot(sweep_count_tracker, res_phi_e_sim_cell_tracker, 'ro', markersize=8)
        residual_value_ax3 = ax3.annotate('Residual = ' + str('%.2E' % res_phi_e_sim_cell_tracker[-1]), xy=(1, 1), xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
        ax4.plot(sweep_count_tracker, res_phi_s_neg_tracker, 'ro', markersize=8)
        residual_value_ax4 = ax4.annotate('Residual = ' + str('%.2E' % res_phi_s_neg_tracker[-1]), xy=(1, 1), xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
        ax5.plot(sweep_count_tracker, res_Cs_neg_tracker, 'ro', markersize=8)
        residual_value_ax5 = ax5.annotate('Residual = ' + str('%.2E' % res_Cs_neg_tracker[-1]), xy=(1, 1), xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
        ax6.plot(sweep_count_tracker, res_Cs_pos_tracker, 'ro', markersize=8)
        residual_value_ax6 = ax6.annotate('Residual = ' + str('%.2E' % res_Cs_pos_tracker[-1]), xy=(1, 1), xycoords='axes fraction', fontsize=16, ha='right', va='bottom')
        if not path.exists('Residual_plots'):
            makedirs('Residual_plots')
        plt.savefig('Residual_plots/' + 'residuals_simtime_' + str(simtime) + '.png', bbox_inches='tight')
        plt.close(f2)

####################################### Realtime Memory Leak Diagnosis #################################################
    if mem_leak_analysis == 'On':
        tracker.print_diff()

########################################## End of Timestep Iteration ###################################################
    if datalogging_switch == 'On':                                                                                      # Increment datalogger variables
        record_timepoint += record_period
        array_recorder_head_general += 1
        array_recorder_head_cs_neg += nr_neg
        array_recorder_head_cs_pos += nr_pos
    simtime += timeStep                                                                                                 # Increment timestep
    step+=1

####################################### Write Data to Disk & Shutdown Sim ##############################################
print('Simulation complete. Runtime was ' + str(timedelta(seconds=(sim_clock_end_time - sim_clock_start_time))))
if datalogging_switch == 'On':
    sim_timing_data, output_location_timestamped = save_data(output_location, simtime)

    with open("sim_results_location.pkl", "w") as pickled_loc_data:
        pickle.dump(output_location_timestamped, pickled_loc_data)                                                      # Write the results location file in the local folder
    with open(output_location_timestamped + '\\' "sim_results_location.pkl", "w") as pickled_loc_data:
        pickle.dump(output_location_timestamped, pickled_loc_data)                                                      # Write a backup of the results location file in the results folder

    sim_results = [settings_dict, sim_timing_data, time_independent_data, time_dependent_data]
    with open(output_location_timestamped + '\\' "sim_results.pkl", "w") as pickled_results:
        pickle.dump(sim_results, pickled_results)                                                                       # Write the packaged simulation results to the results folder
    if auto_process == 'On':
        print('Performing post-processing..')
        Popen([executable, "post_processor.py"])

########################################### Save Any Realtime Plots ####################################################
    if det_concentrations_live_plot == 'On':                                                                            # If user has requested live-plotting of concentration gradients, & datalogging enabled, save end state of realtime plots
        plot_aspect_ratio = (12, 27)                                                                                    # (height, width)
        fig_Cs_live_neg.set_figheight(plot_aspect_ratio[0])
        fig_Cs_live_neg.set_figwidth(plot_aspect_ratio[1])
        fig_Cs_live_pos.set_figheight(plot_aspect_ratio[0])
        fig_Cs_live_pos.set_figwidth(plot_aspect_ratio[1])

        realtime_plot_save_dir = output_location_timestamped + '\\' + 'Realtime_Plots'                                  # Define a filepath to save realtime plots at their end state
        if not path.exists(realtime_plot_save_dir):                                                                     # If it doesn't already exist, create it
            makedirs(realtime_plot_save_dir)
            fig_Cs_live_neg.savefig(realtime_plot_save_dir + '\\' + 'Cs_neg.' + 'pdf', format='pdf', dpi=1200)          # Save Cs_neg image
            fig_Cs_live_pos.savefig(realtime_plot_save_dir + '\\' + 'Cs_pos.' + 'pdf', format='pdf', dpi=1200)          # Save Cs_pos image
        print('Realtime Cs plots saved..')

else:
    print('------------------- End -------------------')

