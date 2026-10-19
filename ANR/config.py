from sympy import true


dt = 0.05 # Sampling time
N =  30  # Prediction horizon

# The ANR network, Hamiltonian QP, and HOCBF use nominal_mass.  The deployed  plant alone uses mass, so mass != nominal_mass creates model--plant mismatch.
# The first control is the nominal acceleration command a; the plant realizes  (nominal_mass / mass) * a.  The additive disturbance is also plant-only.

nominal_mass = 1.0
mass = 1.0              #change when testing for the 1.4 mass case, otherwise use 1.0 for the nominal mass case
# Nominal acceleration-command constraints.
a_min = -1.0
a_max = 1.0
w_min = -4
w_max = 4
# Total time of simulation and time steps
T_sim = 10
total_steps_sim = int(T_sim/dt)


beta =0.99
alpha_bc = 1e-7  # Terminal costate boundary-condition loss weight

#HOCBF class-K gainsa
alpha0 = 3            
alpha1 = 3

# Deployment disturbance: d_v,k ~ Normal(0, disturbance_velocity_std**2) enters only the plant velocity equation,
# v_dot = (nominal_mass / mass) * a + d_v,k.

disturbance_enabled = False
disturbance_velocity_std = 0.3  # acceleration disturbance standard deviation [m/s^2]
disturbance_seed = 10
