from multiprocessing import Pool, cpu_count
import time
from itertools import product

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
import synmorph as sm

# -------------------------
# Worker function
# -------------------------
def run_spv_sim(Dr, v0, params):
    """Run a synmorph SPV simulation with given parameters. Allows for parallel execution to perform activity parameter sweeps."""

    start = time.perf_counter()

    # ---- motility parameters ----
    v = {
            0: v0,
            1: 2*v0,
            2: 0.001,
        }
    active_params = {
        "Dr": Dr,
        "v0": v
    }

    sim = sm.simulation(tissue_params=params["tissue_params"],
                        active_params=active_params,
                        init_params=params["init_params"],
                        simulation_params=params["simulation_params"],
                        run_options=params["run_options"],
                        save_options=params["save_options"])
    
    params["save_options"]["name"] = f'Dr{round(Dr, 3)}_v{v0}'

    sim.simulate(progress_bar=False)

    sim.save(params["save_options"]["name"], id=sim.id, 
         dir_path=params["save_options"]["result_dir"], 
         compressed=params["save_options"]["compressed"])
    
    end = time.perf_counter()
    print(f"  Elapsed time of sim: {end - start:.6f} seconds")

    return (Dr, v0, sim.id)

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":

    # ---- set basic params ---- 
    tissue_radius = 4 # radius of tissue in # cells
    N_t = 10000 # final time
    dt = 0.025       # width of timestep 
    tskip = 40      # num timesteps between saved timepoints
    exp_dir = 'test2'       # make a function of key parameters to save them in title

    ctype_proportions = (0.5,0.5)       # proportions of LEP and MEP, respectively


    # ---- set relative magnitudes of tensions (Y) and interaction matrix (W)---- 
    gamma_scale = 0.12   # scaling factor for tensions
    Y = np.array([-0.1, -0.1, -0.1, -0.2, -0.3]) * gamma_scale    # 'LL', 'LM', 'LX', 'MM', 'MX' interfacial tensions
    ECM_ECM = -0.15  # -0.3       # ECM-ECM interfacial tension

    # ---- interaction matrix ----
    W = (np.array([[Y[0], Y[1], Y[2]], \
                    [Y[1], Y[3], Y[4]], \
                    [Y[2], Y[4], ECM_ECM]]))

    # ---- setup tissue parameters ----
    P0 = 3.95  # target perimeter
    tissue_params = {"L": 14,               
                    "A0": 1,
                    "P0": P0, #3.81,
                    "kappa_A": 1,
                    "kappa_P": 0.1,
                    "W": W, #np.array(((0, 0.0762), (0.0762, 0)))*0.75,
                    "a": 0.2, #0,
                    "k": 2} # 0}

    # ---- init configuration ----
    init_params = {
        "init_noise": 0.01, #0.005,
        "c_type_proportions": (0,0,0),  # vestigial
        "init_mode": "ball_by_radius",            # <— NEW
        "ball_radius": tissue_radius,                       # choose your radius in domain units
        "ball_non_ecm_types": (0, 1),     # types inside ball 0: LEP, 1: MEP
        "ball_non_ecm_proportions": ctype_proportions,  # <-- use this for 80/20 split
    }

    simulation_params = {"dt": dt,  # width of timestep
                        "tfin": N_t,  # final time
                        "tskip": tskip,  # num timesteps between saved timepoints
                        "grn_sim": None}

    save_options = {"save": "last",
                    "result_dir": "./results/Parallel_Sweep/",
                    "name": None,
                    "compressed": True} 

    run_options = {"equiangulate": True,
                "equi_nkill": 3}


    # Fixed "global" parameters
    global_params = {
        "tissue_params": tissue_params,
        "init_params": init_params,
        "simulation_params": simulation_params,
        "run_options": run_options,
        "save_options": save_options
    }


    # ---- motility param sweep values ---- 
    tau_vals = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    Dr_vals = 1/np.array(tau_vals)
    v0_vals = [0.15, 0.3, 0.6, 1.2, 2.4]

    # Generate all combinations (Cartesian product)
    param_grid = list(product(Dr_vals, v0_vals))

    # Add shared parameters to each tuple
    args = [(Dr, v0, global_params) for (Dr, v0) in param_grid]

    # Parallel computation
    with Pool(cpu_count()) as pool:
        results = pool.starmap(run_spv_sim, args)

    # Show results
    print("Results (Dr, v0, id):")
    for r in results:
        print(r)