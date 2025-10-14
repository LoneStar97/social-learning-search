# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:22:00 2025

@author: ZEL45
"""
from plotfunc import timedata, draw_video
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count
import os

# Define parameter series
rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
Rv_series = [0.01]

# Fixed simulation parameters
targets = 6000
beta = 2
agents = 50
mu = 1.1
runs = 1
file = "A50Neg_Updated.h5"
tick_speed = 1
alpha = 1e-5
save = None

NumCases = 1000
TarFixFile = '10P600_10NP60.h5'

# Worker function for timedata
def run_timedata(args):
    Rv, rho, targets, beta, agents, mu, alpha, tick_speed, runs, filename, CaseCount, target_location, Negative_targets, save = args
    try:
        print(f"Starting timedata for Rv={Rv}, rho={rho}, Case={CaseCount}")
        result = timedata(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, runs, filename, CaseCount, target_location, Negative_targets, save)
        print(f"Completed timedata for Rv={Rv}, rho={rho}, Case={CaseCount}")
        return (CaseCount, rho, Rv, True, result)  # Success with result
    except Exception as e:
        print(f"Error in timedata for Rv={Rv}, rho={rho}, Case={CaseCount}: {e}")
        return (CaseCount, rho, Rv, False, str(e))  # Failure with error message

# Parallel execution for timedata
def parallel_timedata():
    # Load target locations
    with h5py.File(TarFixFile, 'r') as hdf:
        existing_groups = list(hdf.keys())
        num_groups = len(existing_groups)
        print(f"Number of groups in {TarFixFile}: {num_groups}")

        # Precompute target locations
        target_data = []
        for i in range(min(NumCases, num_groups)):
            groupname = f'case_{i}'
            group = hdf.get(groupname)
            if group is None:
                print(f"Warning: Case {i} not found in {TarFixFile}")
                continue
            loc_x = group.get('tx')[:]
            loc_y = group.get('ty')[:]
            nloc_x = group.get('ntx')[:]
            nloc_y = group.get('nty')[:]
            target_location = np.column_stack((loc_x, loc_y))
            Negative_targets = np.column_stack((nloc_x, nloc_y))
            target_data.append((i, target_location, Negative_targets))

    # Prepare tasks
    tasks = []
    for m, Rv in enumerate(Rv_series):
        for n, rho in enumerate(rho_series):
            filename = f'Rv{Rv}rho={rho}_{file}'
            for CaseCount, target_location, Negative_targets in target_data:
                tasks.append((Rv, rho, targets, beta, agents, mu, alpha, tick_speed, runs, filename, CaseCount, target_location, Negative_targets, save))

    expected_tasks = len(Rv_series) * len(rho_series) * len(target_data)
    print(f"Expected tasks: {expected_tasks} (Rv={len(Rv_series)}, rho={len(rho_series)}, cases={len(target_data)})")

    # Run tasks in parallel
    num_processes = min(cpu_count(), len(tasks))
    print(f"Starting {len(tasks)} timedata tasks with {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_timedata, tasks)

    # Analyze results
    completed = [(case, rho, rv) for case, rho, rv, success, _ in results if success]
    failed = [(case, rho, rv, reason) for case, rho, rv, success, reason in results if not success]

    print(f"\nCompleted tasks: {len(completed)} out of {expected_tasks}")
    if completed:
        print(f"Completed cases (case, rho, Rv): {sorted(set(completed))}")
    print(f"Failed tasks: {len(failed)}")
    if failed:
        print("Failed cases (case, rho, Rv, reason):")
        for case, rho, rv, reason in failed:
            print(f"  Case {case}, rho={rho}, Rv={rv}: {reason}")

    # Verify output files
    print("\nChecking output files:")
    for rho in rho_series:
        filename = f'Rv0.01rho={rho}_{file}'
        if os.path.exists(filename):
            with h5py.File(filename, 'r') as hdf:
                cases = list(hdf.keys())
                print(f"{filename}: {len(cases)} cases - {cases}")
        else:
            print(f"{filename}: File not created")

if __name__ == '__main__':
    parallel_timedata()