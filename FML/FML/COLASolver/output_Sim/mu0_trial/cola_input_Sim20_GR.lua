------------------------------------------------------------ 
-- Simulation parameter file                                 
-- Include other paramfile into this: dofile("param.lua")  
------------------------------------------------------------ 
                                                             
-- Don't allow any parameters to take optional values?       
all_parameters_must_be_in_file = true                        
------------------------------------------------------------ 
-- Simulation options                                        
------------------------------------------------------------ 
-- Label                                                     
simulation_name = "Sim20_GR"                          
-- Boxsize of simulation in Mpc/h                            
simulation_boxsize = 512.0                   
                                                             
------------------------------------------------------------ 
-- COLA                                                      
------------------------------------------------------------ 
-- Use the COLA method                                       
simulation_use_cola = true                                   
simulation_use_scaledependent_cola = false          
if simulation_use_cola then                                  
  simulation_enforce_LPT_trajectories = false                
end                                                          
------------------------------------------------------------ 
-- Choose the cosmology                                      
------------------------------------------------------------ 
-- Cosmology: LCDM, w0waCDM, DGP, JBD, ...                   
cosmology_model = "LCDM"                     
cosmology_OmegaCDM = 0.27                 
cosmology_Omegab = 0.049                     
cosmology_OmegaMNu = 0.001387                
cosmology_OmegaLambda = 0.679613              
cosmology_OmegaK = 0.0                     
cosmology_Neffective = 3.046                    
cosmology_TCMB_kelvin = 2.7255                  
cosmology_h = 0.67                                
cosmology_As = 2.1e-09                             
cosmology_ns = 0.96                             
cosmology_kpivot_mpc = 0.05                 
                                                             
-- The w0wa parametrization                                  
if cosmology_model == "w0waCDM" then                       
  cosmology_w0 = -1.0                        
  cosmology_wa = 0.0                        
end                                                          
                                                             
------------------------------------------------------------ 
-- Choose the gravity model                                  
------------------------------------------------------------ 
-- Gravity model: GR, DGP, f(R), JBD, Geff, ...              
gravity_model = "GR"                              
                                                             
if gravity_model == "Marta" then                           
 -- Parameter mu0, for GR = 0.0                              
  gravity_model_marta_mu0 = 0.03793103448275864                                
end                                                          
                                                             
------------------------------------------------------------ 
-- Particles                                                 
------------------------------------------------------------ 
-- Number of CDM+b particles per dimension                   
particle_Npart_1D = 512                      
-- Factor of how many more particles to allocate space       
particle_allocation_factor = 1.5                             
                                                             
------------------------------------------------------------ 
-- Output                                                    
------------------------------------------------------------ 
-- List of output redshifts                                  
output_redshifts = {0.0}                                     
-- Output particles?                                         
output_particles = false                                     
-- Fileformat: GADGET, FML                                   
output_fileformat = "GADGET"                               
-- Output folder                                             
output_folder = "/uio/hume/student-u24/martacor/Simulation/FML/FML/COLASolver/output/mu0_trial"          
                                                             
------------------------------------------------------------ 
-- Time-stepping                                             
------------------------------------------------------------ 
-- Number of steps between the outputs (in output_redshifts) 
timestep_nsteps = {30}                 
-- The time-stepping method: Quinn, Tassev                   
timestep_method = "Quinn"                                  
-- For Tassev: the nLPT parameter                            
timestep_cola_nLPT = -2.5                                    
-- The time-stepping algorithm: KDK                          
timestep_algorithm = "KDK"                                 
-- Spacing of the time-steps in 'a': linear, logarithmic, .. 
timestep_scalefactor_spacing = "linear"                    
                                                             
------------------------------------------------------------ 
-- Initial conditions                                        
------------------------------------------------------------ 
-- The random seed                                           
ic_random_seed = 1234567                          
-- The random generator (GSL or MT19937).                    
ic_random_generator = "GSL"                                
-- Fix amplitude when generating the gaussian random field   
ic_fix_amplitude = true                                      
-- Mirror the phases (for amplitude-fixed simulations)       
ic_reverse_phases = false                                    
ic_random_field_type = "gaussian"                          
-- The grid-size used to generate the IC                     
ic_nmesh = particle_Npart_1D                                 
-- For MG: input LCDM P(k) and use GR to scale back and      
-- ensure same IC as for LCDM                                
ic_use_gravity_model_GR = false   
-- The LPT order to use for the IC                           
ic_LPT_order = 2                                             
-- The type of input:                                        
-- powerspectrum    ([k (h/Mph) , P(k) (Mpc/h)^3)])          
-- transferfunction ([k (h/Mph) , T(k)  Mpc^2)]              
-- transferinfofile (a bunch of T(k,z) files from CAMB)      
ic_type_of_input = "transferinfofile"                      
-- When running CLASS we can just ask for outputformat CAMB  
ic_type_of_input_fileformat = "CAMB"                       
-- Path to the input                                         
ic_input_filename = "/uio/hume/student-u24/martacor/Simulation/FML/FML/COLASolver/input_Sim/mu0_trial/class_transferinfo_Sim20_GR.txt"         
-- The redshift of the P(k), T(k) we give as input           
ic_input_redshift = 0.0                                      
-- The initial redshift of the simulation                    
ic_initial_redshift = 20.0                     
-- Normalize wrt sigma8?                                     
-- If ic_use_gravity_model_GR then this is the sigma8 value  
-- in a corresponding GR universe!                           
ic_sigma8_normalization = false               
ic_sigma8_redshift = 0.0                                     
ic_sigma8 = 0.83                             
                                                             
------------------------------------------------------------ 
-- Force calculation                                         
------------------------------------------------------------ 
-- Grid to use for computing PM forces                       
force_nmesh = 512                            
-- Density assignment method: NGP, CIC, TSC, PCS, PQS        
force_density_assignment_method = "CIC"                      
-- The kernel to use for D^2 when solving the Poisson equation                    
-- Options: (fiducial = continuous, discrete_2pt, discrete_4pt)                    
force_greens_function_kernel = "fiducial"                    
-- The kernel to use for D when computing forces (with fourier)                    
-- Options: (fiducial = continuous, discrete_2pt, discrete_4pt)                    
force_gradient_kernel = "fiducial"                    
-- Include the effects of massive neutrinos when computing                    
-- the density field (density of mnu is the linear prediction)                    
-- Requires: transferinfofile above (we need all T(k,z))                    
force_linear_massive_neutrinos = true                    
-- Experimental feature: Use finite difference on the gravitational                     
-- potential to compute forces instead of using Fourier transforms.                    
force_use_finite_difference_force = false                    
force_finite_difference_stencil_order = 4                    
                                                             
------------------------------------------------------------    
-- Lightcone option    
------------------------------------------------------------    
lightcone = false    
if lightcone then    
  -- The origin of the lightcone in units of the boxsize (e.g. 0.5,0.5,0.5 is the center of the box in 3D)    
  plc_pos_observer = {0.0, 0.0, 0.0}    
  -- The boundary region we use around the shell to ensure we get all particles belonging to the lightcone    
  plc_boundary_mpch = 20.0    
  -- The redshift we turn on the lightcone    
  plc_z_init = 1.0    
  -- The redshift when we stop recording the lightcone    
  plc_z_finish = 0.0    
  -- Replicate the box to match the sky coverage we want?    
  -- If not then we need to make sure boxsize is big enough to cover the sky at z_init    
  plc_use_replicas = true    
  -- Number of dimensions where we do replicas in both + and - direction    
  -- The sky fraction is fsky = 1/2^(ndim_rep - NDIM)    
  -- For 3D: if 0 we get an octant and 3 we get the full sky    
  plc_ndim_rep = 3    
  -- Output gadget    
  plc_output_gadgetfile = false    
  -- Output ascii    
  plc_output_asciifile = false    
  -- To save memory output in batches (we only alloc as many particles as we already have to reduce memory consumption)    
  plc_output_in_batches = true    
    
  -- Make delta(z, theta) maps? This is Healpix maps in 3D where we always use the RING scheme for the maps    
  -- For 2D we use output textfiles with the binning    
  plc_make_onion_density_maps = true    
  if plc_make_onion_density_maps then    
    -- Roughly the size of the size of the bins you want in a    
    -- The exact value we use will depend on the time-steps (but not bigger than 2x this value)    
    -- At minimum we make one map per timestep    
    plc_da_maps = 0.025    
    -- Number of pixels (npix = 4*nside^2). The largest lmax we can get from    
    -- the maps is lmax ~ 2nside    
    plc_nside = 512    
    -- Use chunkpix. Only useful for very sparse maps    
    plc_use_chunkpix = false    
    if plc_use_chunkpix then    
      plc_nside_chunks = 256    
    end    
  end    
end    
    
-- On the fly analysis                                       
------------------------------------------------------------ 
                                                             
------------------------------------------------------------ 
-- Halofinding                                               
------------------------------------------------------------ 
fof = false                                                  
fof_nmin_per_halo = 20                                       
fof_linking_length = 0.2                                     
fof_nmesh_max = 0                                            
fof_buffer_length_mpch = 3.0                                 
                                                             
------------------------------------------------------------ 
-- Power-spectrum evaluation                                 
------------------------------------------------------------ 
pofk = true                                                  
pofk_nmesh = 128                                             
pofk_interlacing = true                                      
pofk_subtract_shotnoise = false                              
pofk_density_assignment_method = "PCS"                     
                                                             
------------------------------------------------------------ 
-- Power-spectrum multipole evaluation                       
------------------------------------------------------------ 
pofk_multipole = false                                       
pofk_multipole_nmesh = 128                                   
pofk_multipole_interlacing = true                            
pofk_multipole_subtract_shotnoise = false                    
pofk_multipole_ellmax = 4                                    
pofk_multipole_density_assignment_method = "PCS"           
                                                             
------------------------------------------------------------ 
-- Bispectrum evaluation                                     
------------------------------------------------------------ 
bispectrum = false                                           
bispectrum_nmesh = 128                                       
bispectrum_nbins = 10                                        
bispectrum_interlacing = true                                
bispectrum_subtract_shotnoise = false                        
bispectrum_density_assignment_method = "PCS"               
  