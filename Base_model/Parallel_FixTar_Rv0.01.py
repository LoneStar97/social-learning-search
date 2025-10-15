# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:18:49 2024
@author: Starr
"""

from plotfunc import timedata, draw_video
import numpy as np
import h5py
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm  # Optional: for progress bar

# Parameters
rho_series = [0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
Rv_series = [0.01]
targets = 6000
beta = 2
agents = 50
mu = 1.1
runs = 1
file = "A50.h5"
tick_speed = 1
alpha = 1e-5
save = None
NumCases = 1000  # Set to 500 cases
TarFixFile = 'TarFix_50P120.h5'

# Function to process a single case and return its ID
def process_case(args):
    i, Rv, rho, targets, beta, agents, mu, alpha, tick_speed, runs, file, save, TarFixFile = args
    filename = f'Rv{Rv}_rho={rho}_{file}'  # Unique filename per case
    try:
        with h5py.File(TarFixFile, 'r') as hdf:
            available_cases = len(list(hdf.keys()))
            if available_cases == 0:
                print(f"Error: No cases found in {TarFixFile}")
                return i, False
            groupname = f'case_{i}'  # Cycle through available cases
            group = hdf.get(groupname)
            if group is None:
                print(f"Warning: Group {groupname} not found in {TarFixFile} (available: {available_cases}), skipping case {i}")
                return i, False  # Return case ID and failure status
            loc_x = group.get('tx')[:]
            loc_y = group.get('ty')[:]
            target_location = np.column_stack((loc_x, loc_y))
            timedata(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, runs, filename, i, target_location, save)
            return i, True  # Return case ID and success status
    except Exception as e:
        print(f"Error processing case {i}: {e}")
        return i, False  # Return case ID and failure status

# Prepare argument list for each case
tasks = []
# Main parallel execution
if __name__ == '__main__':
    for m, Rv in enumerate(Rv_series):
        for n, rho in enumerate(rho_series):
            # Define the existing file to append to
            filename = f'Rv{Rv}rho={rho}_{file}'            
            # Check existing cases in the file
            try:
                with h5py.File(filename, 'r') as hdf:
                    existing_cases = len(list(hdf.keys()))
                    print(f"Existing cases in {filename}: {existing_cases}")
                    start_case_id = existing_cases  # Start numbering new cases from here
            except FileNotFoundError:
                print(f"{filename} not found, creating new file")
                start_case_id = 0  # If file doesn’t exist, start from 0
            
            # Verify TarFixFile before creating tasks
            with h5py.File(TarFixFile, 'r') as hdf:
                available_cases = len(hdf.keys())
                print(f"Available target cases in {TarFixFile}: {available_cases}")
                
            for i in range(NumCases):
                tasks.append((i, Rv, rho, targets, beta, agents, mu, alpha, tick_speed, runs, file, save, TarFixFile))

    # Determine number of processes
    # num_processes = min(mp.cpu_count(), len(tasks))
    # print(f"Running {len(tasks)} tasks with {num_processes} processes")
    num_processes = 32
    
    # Run tasks in parallel and track completed case IDs
    completed_cases = []
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for real-time output as cases finish
        results = pool.imap_unordered(process_case, tasks)
        for case_id, success in tqdm(results, total=len(tasks), desc="Processing cases"):
            if success:
                print(f"Finished case ID: {case_id}")
                completed_cases.append(case_id)
            else:
                print(f"Case ID: {case_id} failed or skipped")

    # Summary of completed cases
    print(f"\nTotal cases completed: {len(completed_cases)}")
    print(f"Completed case IDs: {sorted(completed_cases)}")

    print("Parallel computation completed.")