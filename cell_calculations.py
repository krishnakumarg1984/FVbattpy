######################################### Python Module Imports ########################################################
from __future__ import division
from __future__ import print_function                                                                                   # Force Python 3 compatability
from sympy import sympify, symbols, lambdify
from fipy import CellVariable, FaceVariable, numerix, Grid1D, Grid2D, DefaultSolver, TransientTerm, DiffusionTerm, Variable
from mpmath import mp,mpf
from sys import exit
from scipy import interpolate

####################################### Global Settings, Constants & Functions #########################################
mp.prec = 20                                                                                                            # mpmath package precision
Ce = symbols('Ce')                                                                                                      # Sympification/lambdification requirement
F = 96487.0                                                                                                             # Faraday constant (C/mol)
R = 8.314                                                                                                               # Universal gas constant (J/mol/K)

################################### Chebyshev Function for Non-uniform Meshing #########################################
def compute_cheb_pts(L, nodes):
    cheb_radius = 0.5*L
    cheb_circum = numerix.pi * cheb_radius
    nodes = nodes + 1
    # dx_on_curve = cheb_circum / nodes
    # semicircle_nodes = numerix.arange(0.0,cheb_circum,dx_on_curve)
    semicircle_nodes = numerix.linspace(0.0,cheb_circum,nodes)
    central_angle = semicircle_nodes / cheb_radius
    cheb_pts = cheb_radius - cheb_radius*numerix.cos(central_angle)
    return numerix.diff(cheb_pts)

######################################## Battery (Parent) Class Definition #############################################
class Battery():
    def __init__(self, cell_temperature_celcius, capacity_Ah, bruggeman_coefficient,
                 charge_transfer_coefficient, current_collector_cross_section_area, electrolyte_diffusivity, transference_number): # Initialises physical properties of cell

########################################## Compute & Assign Constants ##################################################
        self.T      = 273.0 + cell_temperature_celcius                                                                  # Could move into electrode-level later
        self.Q      = capacity_Ah * 3600.0
        self.brug   = bruggeman_coefficient                                                                             # Could move into electrode-level later when electrodes may have unique brug. values
        self.alpha  = charge_transfer_coefficient                                                                       # Could move into electrode-level later when electrodes may have unique alpha values
        self.A      = current_collector_cross_section_area
        self.De     = electrolyte_diffusivity
        self.t_plus = transference_number

    def Battery_scale_quantities(self, overall_thickness, kappa_string):
        # self.L = overall_thickness
        self.kappa = lambdify(Ce, sympify(kappa_string), "numpy")                                                       # Function to return electrolyte electrical conductivity, given a species concentration
        # NOTE: Consider changing this function's name since it's only the kappa definition now

######################################## Battery-level Mesh Generation #################################################
    def define_global_mesh(self, node_spacing, normalised_domain_thickness):                                            # Only Ce & phi_e are solved on this mesh
        self.dx = node_spacing                                                                                          # node_spacing vector passed can be either uniformly or non-uniformly spaced points
        self.L_normalised = normalised_domain_thickness
        self.axial_mesh = Grid1D(Lx=self.L_normalised, dx=self.dx)

    def Battery_cellvariables(self, neg_epsilon_e, sep_epsilon_e,pos_epsilon_e, neg_a_s, pos_a_s, nx_neg, nx_sep,
                              nx_pos, j_battery_value, ce_initial, phi_e_initial, neg_L, sep_L, pos_L, neg_De_eff,
                              sep_De_eff, pos_De_eff):
        self.Ce    = CellVariable(mesh=self.axial_mesh, value = ce_initial, hasOld=True)
        self.phi_e = CellVariable(mesh=self.axial_mesh, value = phi_e_initial)

        self.epsilon_e_value = numerix.zeros(nx_neg + nx_sep + nx_pos)
        self.epsilon_e_value[0:nx_neg], self.epsilon_e_value[nx_neg:nx_neg + nx_sep] = neg_epsilon_e, sep_epsilon_e
        self.epsilon_e_value[nx_neg + nx_sep:] = pos_epsilon_e
        self.epsilon_e = CellVariable(mesh=self.axial_mesh, value = self.epsilon_e_value)

        self.epsilon_e_eff_value = numerix.zeros(nx_neg + nx_sep + nx_pos)
        self.epsilon_e_eff_value[0:nx_neg] = neg_epsilon_e**self.brug
        self.epsilon_e_eff_value[nx_neg:nx_neg + nx_sep] = sep_epsilon_e**self.brug
        self.epsilon_e_eff_value[nx_neg + nx_sep:] = pos_epsilon_e**self.brug
        self.epsilon_e_eff = CellVariable(mesh=self.axial_mesh, value = self.epsilon_e_eff_value)

        self.a_s_value = numerix.zeros(nx_neg + nx_pos + nx_sep)
        self.a_s_value[0:nx_neg] = neg_a_s
        self.a_s_value[nx_neg:nx_neg + nx_sep] = 0.0
        self.a_s_value[nx_neg + nx_sep:] = pos_a_s
        self.a_s = CellVariable(mesh=self.axial_mesh, value = self.a_s_value)

        self.L_value = numerix.zeros(nx_neg + nx_pos + nx_sep)
        self.L_value[0:nx_neg], self.L_value[nx_neg:nx_neg + nx_sep] = neg_L, sep_L
        self.L_value[nx_neg + nx_sep:] = pos_L
        self.L = CellVariable(mesh = self.axial_mesh, value = self.L_value)

        self.De_eff_value = numerix.zeros(nx_neg + nx_pos + nx_sep)
        self.De_eff_value[0:nx_neg], self.De_eff_value[nx_neg:nx_neg + nx_sep], self.De_eff_value[nx_neg + nx_sep:] = neg_De_eff, sep_De_eff, pos_De_eff
        self.De_eff = CellVariable(mesh = self.axial_mesh, value = self.De_eff_value)

        self.j_battery = CellVariable(mesh=self.axial_mesh, value=j_battery_value)

############################# Define Eff. Diff. Coefficients & Volume Fractions ########################################
    def Battery_facevariables(self, neg_De_eff, sep_De_eff, pos_De_eff, neg_epsilon_e, sep_epsilon_e, pos_epsilon_e,
                              normalised_neg_length, normalised_sep_length):
        self.De_eff = FaceVariable(mesh=self.axial_mesh)                                                                # Define effective diffusion coefficient as a FiPy facevariable
        self.De_eff.setValue(neg_De_eff, where=(self.axial_mesh.faceCenters[0] <= normalised_neg_length))               # Set its value to that for the neg. diff. coefficient in the negative electrode domain
        self.De_eff.setValue(sep_De_eff, where=((normalised_neg_length                                                  # Set its value to the correct value in the separator
                                                 < self.axial_mesh.faceCenters[0]) &
                                                    (self.axial_mesh.faceCenters[0] <
                                                     (normalised_neg_length + normalised_sep_length))))
        self.De_eff.setValue(pos_De_eff, where=(self.axial_mesh.faceCenters[0]                                          # Set its value to the correct value in the positive electrode domain
                                                >= (normalised_neg_length + normalised_sep_length)))

        self.epsilon_e_eff_facevariable = FaceVariable(mesh=self.axial_mesh)                                            # Define electrolyte phase volume fraction as a FiPy facevariable
        self.epsilon_e_eff_facevariable.setValue(neg_epsilon_e**self.brug,                                              # Compute eff. value for neg. domain & set it in neg. electrode domain
                                                 where=(self.axial_mesh.faceCenters[0] <= normalised_neg_length))
        self.epsilon_e_eff_facevariable.setValue(sep_epsilon_e**self.brug,                                              # Compute eff. value for sep. domain & set it in sep. electrode domain
                                                 where=((normalised_neg_length < self.axial_mesh.faceCenters[0]) &
                                                        (self.axial_mesh.faceCenters[0]
                                                         < (normalised_neg_length + normalised_sep_length))))
        self.epsilon_e_eff_facevariable.setValue(pos_epsilon_e**self.brug,                                              # Compute eff. value for pos. domain & set it in pos. electrode domain
                                                 where=(self.axial_mesh.faceCenters[0]
                                                        >= (normalised_neg_length + normalised_sep_length)))

####################################### Separator (Child) Class Definition #############################################
class Separator(Battery):
    def __init__(self,domain_thickness,electrolyte_porosity_in_region,no_of_electrons_transferred, cell_temperature_celcius, capacity_Ah,
                 bruggeman_coefficient, charge_transfer_coefficient, current_collector_cross_section_area, electrolyte_diffusivity, transference_number):
        Battery.__init__(self, cell_temperature_celcius, capacity_Ah,
                         bruggeman_coefficient, charge_transfer_coefficient,
                         current_collector_cross_section_area, electrolyte_diffusivity,transference_number)
        self.L = domain_thickness
        self.epsilon_e = electrolyte_porosity_in_region                                                                 # Electrolyte phase volume fraction in separator
        self.De_eff = self.De*self.epsilon_e**self.brug                                                                 # Effective diffusion coefficient, separator

############################## Uniform & Non-uniform Separator Mesh Generation #########################################
    def define_uniform_comp_domain(self, no_of_nodes, normalised_domain_thickness):                                     # Uniform axial domain, separator
        self.nx           = int(no_of_nodes)
        self.L_normalised = normalised_domain_thickness
        self.dx           = self.L_normalised / self.nx

    def define_non_uniform_comp_domain(self, no_of_nodes, normalised_domain_thickness):                                 # Non-uniform axial domain, separator
        self.nx           = int(no_of_nodes)
        self.L_normalised = normalised_domain_thickness
        self.dx           = compute_cheb_pts(self.L_normalised, self.nx)

####################################### Electrode (Child) Class Definition #############################################
class Electrode(Battery):
    def __init__(self,domain_thickness,particle_radius,solid_diffusivity, solid_porosity,solid_conductivity,
                 min_stoichiometry, max_stoichiometry, k_norm, max_surface_conc, ocp_string,
                 electrolyte_porosity_in_region, no_of_electrons_transferred, cell_temperature_celcius,
                 capacity_Ah, bruggeman_coefficient, charge_transfer_coefficient, current_collector_cross_section_area,
                 electrolyte_diffusivity, transference_number, theta_table_vals, Uocp_table_vals):      # Initialises all the physical properties of the electrode
        Battery.__init__(self, cell_temperature_celcius, capacity_Ah,bruggeman_coefficient,
                         charge_transfer_coefficient, current_collector_cross_section_area, electrolyte_diffusivity, transference_number)

########################################## Compute & Assign Constants ##################################################
        self.L            = domain_thickness
        self.Rs           = particle_radius
        self.Ds           = solid_diffusivity
        self.epsilon_s    = solid_porosity
        self.sigma        = solid_conductivity
        self.theta_min    = min_stoichiometry
        self.theta_max    = max_stoichiometry
        self.k_norm       = k_norm                                                                                      # Reaction rate coefficient
        self.cs_max       = max_surface_conc
        self.epsilon_e    = electrolyte_porosity_in_region                                                              # Electrolyte phase volume fraction in electrodes
        self.interpolated_Uocp_fn = interpolate.interp1d(theta_table_vals, Uocp_table_vals, kind='slinear')             # Function to obtain OCP, given a stoichiometry value

        self.a_s = 3.0*self.epsilon_s/self.Rs                                                                           # Ratio of total area of all spherical particles to total volume of a particle (1/m)
        self.sigma_eff = self.sigma*self.epsilon_s**self.brug                                                           # Effective solid-phase electrical conductivity (S/m)

        self.cs_zero_percent = self.theta_min*self.cs_max                                                               # Compute solid-phase concentration corresponding to 0% SOC
        self.cs_hundred_percent = self.theta_max*self.cs_max                                                            # Compute solid-phase concentration corresponding to 100% SOC

        self.De_eff = self.De*self.epsilon_e**self.brug                                                                 # Effective diffusion coefficient, electrodes

        self.corrected_Cs_p2D_faceValue = numerix.array((), dtype=float)                                                # Create an array object, unique to the instance, which will be replaced by a new array when used
        self.formatted_Cs_nodevalue_array = numerix.array((), dtype=float)                                              # Create an array object, unique to the instance, which will be replaced by a new array when used

############################## Uniform & Non-uniform Electrode Mesh Generation #########################################
    def define_uniform_comp_domain_ax(self, no_of_nodes, normalised_domain_thickness):                                  # Uniform axial domain, electrodes
        self.nx             = int(no_of_nodes)
        self.L_normalised   = normalised_domain_thickness
        self.dx             = numerix.full((self.nx,), (self.L_normalised / self.nx))
        self.axial_mesh     = Grid1D(nx=self.nx, Lx=self.L_normalised)
        self.axial_mesh     = Grid1D(nx=self.nx, Lx=self.L_normalised)
        self.axialfaces     = self.axial_mesh.faceCenters                                                               # Create a shorthand name for accessing axial faces

    def define_non_uniform_comp_domain_ax(self, no_of_nodes, normalised_domain_thickness):                              # Non-uniform axial domain, electrodes
        self.nx             = int(no_of_nodes)
        self.L_normalised   = normalised_domain_thickness
        self.dx             = compute_cheb_pts(self.L_normalised, self.nx)
        self.axial_mesh     = Grid1D(dx=self.dx, Lx=self.L_normalised)
        self.axialfaces     = self.axial_mesh.faceCenters                                                               # Create a shorthand name for accessing axial faces

    def define_uniform_comp_domain_rad(self, shells_per_particle, normalised_particle_radius):                          # Uniform radial domain, electrode particles
        self.nr             = int(shells_per_particle)
        self.Rs_normalised  = normalised_particle_radius
        self.dr             = self.Rs_normalised/self.nr
        self.p2d_mesh       = Grid2D(nx=self.nx, ny=self.nr, Lx=self.L_normalised, Ly=self.Rs_normalised)
        self.dummy, self.radial_faces = self.p2d_mesh.faceCenters                                                       # Create a shorthand name for accessing y-axis faces in the particle mesh

    def define_non_uniform_comp_domain_rad(self, shells_per_particle, normalised_particle_radius):                      # Non-uniform radial domain, electrode particles
        self.nr             = int(shells_per_particle)
        self.Rs_normalised  = normalised_particle_radius
        self.dr             = numerix.diff(numerix.log(1
                                                       - numerix.linspace(0,1.0
                                                                          - numerix.exp(self.Rs_normalised),self.nr+1)))
        self.p2d_mesh       = Grid2D(dx=self.dx, dy=self.dr, Lx=self.L_normalised, Ly=self.Rs_normalised)
        self.dummy, self.radial_faces = self.p2d_mesh.faceCenters                                                       # Create a shorthand name for accessing y-axis faces in the particle mesh

################################# Initial Stoichiometry & Concentration Calcs ##########################################
    def theta_calc_initial(self, initial_soc_percent):
        self.initial_soc = float(initial_soc_percent)/100.0                                                             # Convert user-entered soc in percent to a normalised value
        return float(self.initial_soc * (mpf(str(self.theta_max)) - mpf(str(self.theta_min))) + mpf(str(self.theta_min)))

    def calc_cs_surface_initial(self, theta_initial):
        return theta_initial * self.cs_max

####################################### OCP from Stoichiometry Functions ###############################################
    # def calc_ocp(self, stoichiometry):                                                                                # Not currently used
    #     return self.Uocp_expr_lambdified(stoichiometry)

    def calc_ocp_interp(self, stoichiometry):
        return self.interpolated_Uocp_fn(stoichiometry)
    # NOTE: This function is redundant. The returned value can be called directly wherever used

################################# Define Variables Solved at FV Cellcentres ############################################
    def define_cellvariables(self, phi_s_initial, cs_surface_initial, phi_e_initial, theta_initial, ce_initial, sim_cell):
        self.phi_s = CellVariable(mesh=self.axial_mesh, value = phi_s_initial)
        # self.deltaPhiS = CellVariable(mesh=self.axial_mesh, value = 0.)                                               # For Newton iteration
        self.phi_e_copy_from_supermesh = CellVariable(mesh=self.axial_mesh, value = phi_e_initial)
        self.Cs_surface = CellVariable(mesh=self.axial_mesh, value = cs_surface_initial)
        self.overpotential = CellVariable(mesh=self.axial_mesh,
                                          value = (phi_s_initial - phi_e_initial - self.calc_ocp_interp(theta_initial)))
        self.Cs_p2d = CellVariable(mesh=self.p2d_mesh,   value = cs_surface_initial[0], hasOld=True)
        self.j0 = CellVariable(mesh=self.axial_mesh,
                               value = self.k_norm
                                       *numerix.absolute(((self.cs_max - cs_surface_initial)/self.cs_max)
                                                         **(1-sim_cell.alpha))
                                       *numerix.absolute((cs_surface_initial/self.cs_max)**sim_cell.alpha)
                                       *numerix.absolute((ce_initial/ce_initial)**(1.0-sim_cell.alpha)))
        self.uocp = CellVariable(mesh=self.axial_mesh, value = self.calc_ocp_interp(self.Cs_surface / self.cs_max))

############################################### Define Facemirrors #####################################################
    def define_facemirrors(self, phi_s_initial,phi_e_initial):                                                          # Facemirrors are copies of the variables solved on the axial meshes..
        self.phi_s_facemirror = FaceVariable(mesh=self.p2d_mesh, value = phi_s_initial)                                 # ..and applied to the upper face of the p2d mesh in each domain
        self.phi_e_facemirror = FaceVariable(mesh=self.p2d_mesh, value = phi_e_initial)
        self.j0_facemirror = FaceVariable(mesh=self.p2d_mesh, value = self.j0[0])

###################################### Extrapolate for Correct Facevalues ##############################################
    def get_true_Cs_facevals(self):                                                                                     # Function to calculate correct, extrapolated Cs values for top & bottom p2d faces (for when a Neumann BC has been used)
        self.formatted_Cs_nodevalue_array = numerix.flipud(numerix.array(self.Cs_p2d).reshape(self.nr, self.nx))             # Re-orientating the Cs nodevalues before extracting those values at top & bottom

        self.uppermost_nodevals = self.formatted_Cs_nodevalue_array[0, :]                                                    # Extract the Cs nodevals for all x / for all particles (L-R, smallest to largest) at the uppermost r (y) node coordinate
        self.lowermost_nodevals = self.formatted_Cs_nodevalue_array[-1, :]                                                   # Extract the Cs nodevals for all x / for all particles (L-R, smallest to largest) at the lowermost r (y) node coordinate

        self.radial_p2d_temp_between_nodes = self.radial_faces[::-1][numerix.argmax(self.radial_faces[::-1]):][::-1]    # Obtain a vector of y-locations of only the p2d mesh radial faces that are between nodes
        self.dCs_dx, self.dCs_dy = self.Cs_p2d.faceGrad                                                                 # Obtain the vector of Cs gradients through all faces in the p2d mesh. Note that dCs_dx is unused
        self.dCs_dy_isolated = self.dCs_dy[0:self.radial_p2d_temp_between_nodes.shape[0]]                               # Isolate the gradients for only those faces of interest (the radial faces between nodes)
        self.dCs_dy_isolated_upper = self.dCs_dy_isolated[-2*self.nx:-self.nx]                                          # Obtain the vector of Cs gradients through only those radial faces between the top & 2nd to top nodes
        self.dCs_dy_isolated_lower = self.dCs_dy_isolated[self.nx:2*self.nx]                                            # Obtain the vector of Cs gradients through only those radial faces between the bottom & 2nd to bottom nodes

        if isinstance(self.dr, float):                                                                                  # If the radial particle mesh is uniform, dr is a float, and is constant
            surface_Cs_vals = self.uppermost_nodevals + (self.dCs_dy_isolated_upper*self.dr/2.0)                        # Compute the vector of Cs at the top faces (distance dr/2 away) by extrapolation
            centre_Cs_vals = self.lowermost_nodevals - (self.dCs_dy_isolated_lower*self.dr/2.0)                         # Compute the vector of Cs at the bottom faces (distance dr/2 away) by extrapolation
        elif isinstance(self.dr, numerix.ndarray):                                                                      # If the radial particle mesh is non-uniform, dr is a vector of different values - hence need only end values
            surface_Cs_vals = self.uppermost_nodevals + (self.dCs_dy_isolated_upper*self.dr[-1]/2.0)                    # Compute the vector of Cs at the top faces (distance dr/2 away) by extrapolation
            centre_Cs_vals = self.lowermost_nodevals - (self.dCs_dy_isolated_lower*self.dr[0]/2.0)                      # Compute the vector of Cs at the bottom faces (distance dr/2 away) by extrapolation
        else:
            print("Warning: Check the type of self.dr - it should be a float or numerix array")

        return surface_Cs_vals, centre_Cs_vals                                                                          # Return the newly-computed particle surface & centre Li concentrations

#################################### Update FiPy Facevalues with Extrapolated Data #####################################
    def insert_corrected_Cs_facevalues(self, original_facevalues, surface_Cs_vals, centre_Cs_vals):
        self.corrected_Cs_p2D_faceValue = numerix.array(original_facevalues)                                            # Copying the facevalue array with its incorrect values to use as a starting point
        self.corrected_Cs_p2D_faceValue[numerix.array(self.p2d_mesh.facesTop)] = surface_Cs_vals                        # Inserting the corrected top-face Cs values using mask
        self.corrected_Cs_p2D_faceValue[numerix.array(self.p2d_mesh.facesBottom)] = centre_Cs_vals                      # Inserting the corrected bottom-face Cs values using mask

######################################### Define Butler-Volmer Eq. #####################################################
    def calc_BV(self, j0, overpotential):                                                                               # Coded here is the RHS of the BV equation
        # print(overpotential)                                                                                          # Code to debug the exponential overflow issue when the number of nodes in each domain is changed
        # print(numerix.exp(((1-self.alpha)*F*(overpotential))/(R*self.T)))                                             # Overpotential values are high, causing the exponents to overflow, it seems..
        # print(numerix.exp((-self.alpha*F*(overpotential))/(R*self.T)))
        return (j0 * (numerix.exp(((1-self.alpha)*F*(overpotential))/(R*self.T))                                        # Returns a value for "j", flux density on the LHS of the equation
                      - numerix.exp((-self.alpha*F*(overpotential))/(R*self.T)))
                )

#################################### Functions to Compute Particle Fluxes ##############################################
    def surface_BC_Cs(self, phi_s_facemirror, phi_e_facemirror, Cs_p2d_facevalue):                                      # Function to compute flux at the surface - constantly updated during timestepping
        self.overpotential_facemirror = (phi_s_facemirror                                                               # Compute overpotential using potentials & OCP copied from axial meshes
                                         - phi_e_facemirror
                                         - self.calc_ocp_interp(Cs_p2d_facevalue/self.cs_max))
        return -(self.Rs/self.Ds)*self.calc_BV(self.j0_facemirror, self.overpotential_facemirror)                       # Use that eta value to compute the current density via BV, & hence the surface flux

    def particle_centre_noflux_BC(self):                                                                                # Function to compute flux at the particle centre. Initialised & not updated
        self.Cs_p2d.faceGrad.constrain(0., where=self.p2d_mesh.facesBottom)

############################### Define the Electrode Particle Diffusion Coefficient ####################################
    def Cs_p2d_diffcoeff_function(self):
        self.Cs_p2d_diffCoeff = FaceVariable(mesh=self.p2d_mesh,rank=2,value=[[[0., 0.],[0., 0. ]]])                    # As always with FiPy, diffusion coefficients are defined on the faces (hence facevar)
        self.numerixed_Cs_p2d_diffCoeff = numerix.array(self.Cs_p2d_diffCoeff)                                          # Temporarily convert to a NumPy array to edit the values
        (self.numerixed_Cs_p2d_diffCoeff[1][1]) = self.Ds*(self.radial_faces**2.0)/self.Rs                              # Assign a non-zero value to the diff. coefficient in the y-axis only..
        self.Cs_p2d_diffCoeff = FaceVariable(mesh=self.p2d_mesh,rank=2,value= self.numerixed_Cs_p2d_diffCoeff)          # ..which constrains diffusion to 1D. Now recreate the facevar with the new value
        # NOTE: This previous line should update the value of the variable created on the 1st line, rather than re-creating it, which leaks memory

######################################### Update Facemirrors & Potentials ##############################################
    def update_overpotentials(self):                                                                                    # Called after every influential change in the timestepping loop
        self.phi_s_facemirror[self.radial_faces == self.Rs_normalised] = self.phi_s                                     # Copy phi_s value from the axial mesh it's solved on to the surface of the p2d mesh
        self.phi_e_facemirror[self.radial_faces == self.Rs_normalised] = self.phi_e_copy_from_supermesh                 # Copy phi_e value from the axial mesh it's solved on to the surface of the p2d mesh
        self.j0_facemirror[self.radial_faces == self.Rs_normalised] = self.j0                                           # Copy j0 value from the axial mesh it's solved on to the surface of the p2d mesh
        self.uocp.value = self.interpolated_Uocp_fn(self.Cs_surface / self.cs_max)                                      # Compute new OCP value with updated particle surface concentration
        self.overpotential.value = self.phi_s - self.phi_e_copy_from_supermesh - self.uocp                              # Use the new OCP value to compute a new value for the overpotential
        # NOTE: For speed increase, separate out the lines in this function & insert them directly in the main loop.
        # NOTE: Any repeated calls to interpolated_Uocp_fn where it's not necessary will be slow. Interpolation is slow