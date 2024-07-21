import numpy as np
import os
import ED

# Define model parameters
parent_dir_name = "/shared/share_mala/andrew/diss"

L = 12  # try larger systems later on 14,16
sz = "1/2"
delta_start = -2
delta_end = 2
delta_interval = 4000

second_delta_start = 1
second_delta_end = 1
second_delta_interval = 1

# Driving parameter range for delta
deltas = np.linspace(delta_start, delta_end, delta_interval)  # increment of 0.005, you can adjust this

# Define driving parameter range for second_delta. If we need two deltas.
second_deltas = np.linspace(second_delta_start, second_delta_end, second_delta_interval)  # adjust range as needed

# Set directory to save to
dname = f"{parent_dir_name}/szhalf_L{L}_delta_{delta_start}_to_{delta_end}_interval_{delta_interval}_secondDelta_{second_delta_start}_to_{second_delta_end}_interval_{second_delta_interval}_data/"

# Create directory if it does not exist
os.makedirs(dname, exist_ok=True)

# Save deltas and second_deltas
np.save(os.path.join(dname, "deltas.npy"), deltas)
np.save(os.path.join(dname, "second_deltas.npy"), second_deltas)

# loop over your parameter space
# can be parallelized with joblib if your cluster has many cores
for i, delta in enumerate(deltas):
    for j, second_delta in enumerate(second_deltas):
        H = ED.H_XXZ(L=L, delta=delta, second_delta=second_delta, sz=sz)
        # ground state energy, wavefunction obtained using full diagonalization
        energy, psi = H.get_GS_full()
        # ground state energy, wavefunction obtained using sparse diagonalization
        # energy, psi = H.get_GS()

        np.save(os.path.join(dname, f"delta_{i}_second_delta_{j}.npy"), psi)
        