import logging
import pandas as pd
import os

from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.benchmarks.benchmarks import get_bench_nn_bit, get_bench_nn_ant, get_bench_nn_ola, get_bench_numbers

def evaluate_network(bench_name, get_bench_fn, sim_obj, batch_size=1):
    nn = get_bench_fn(bench_name, batch_size)
    stats = get_bench_numbers(nn, sim_obj, batch_size)
    
    total_cycles = 0
    total_energy = 0
    
    for opname, s in stats.items():
        # s is a Stats object
        total_cycles += s.total_cycles
        # To get energy we need the energy_cost from simulator
        energy_cost = sim_obj.get_energy_cost()
        energy = s.get_energy(energy_cost)
        total_energy += energy
        
    return total_cycles, total_energy

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    
    config_file = 'bf_e_conf.ini'
    if not os.path.exists(config_file):
        config_file = 'conf.ini'
        
    sim_obj = Simulator(config_file, False)
    
    benchmarks = ['resnet18', 'vgg16', 'vit']
    results = []
    
    for bench in benchmarks:
        print("Evaluating {}...".format(bench))
        
        # 1. BitFusion Baseline
        cycles_bit, energy_bit = evaluate_network(bench, get_bench_nn_bit, sim_obj)
        results.append({
            'Benchmark': bench,
            'Scheme': 'BitFusion Baseline',
            'Cycles': cycles_bit,
            'Energy (uJ)': energy_bit * 1e6 # assuming get_energy returns Joules
        })
        
        # 2. ANT Quantization
        cycles_ant, energy_ant = evaluate_network(bench, get_bench_nn_ant, sim_obj)
        results.append({
            'Benchmark': bench,
            'Scheme': 'ANT Quantization',
            'Cycles': cycles_ant,
            'Energy (uJ)': energy_ant * 1e6
        })
        
        # 3. OLive Quantization
        cycles_ola, energy_ola = evaluate_network(bench, get_bench_nn_ola, sim_obj)
        results.append({
            'Benchmark': bench,
            'Scheme': 'OLive Quantization',
            'Cycles': cycles_ola,
            'Energy (uJ)': energy_ola * 1e6
        })
        
    df = pd.DataFrame(results)
    print("\n--- Simulation Results ---")
    print(df.to_string(index=False))
    
    df.to_csv('results/compare_quantization_results.csv', index=False)
    
    # Plotting Graphs
    import matplotlib
    matplotlib.use('Agg') # Headless backend
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not os.path.exists('fig'):
        os.makedirs('fig')
        
    df_pivot_cycles = df.pivot(index='Benchmark', columns='Scheme', values='Cycles')
    df_pivot_energy = df.pivot(index='Benchmark', columns='Scheme', values='Energy (uJ)')
    
    # Calculate Speedup and Energy Reduction relative to Baseline
    speedup = df_pivot_cycles['BitFusion Baseline'].values[:, None] / df_pivot_cycles.values
    energy_red = df_pivot_energy['BitFusion Baseline'].values[:, None] / df_pivot_energy.values
    
    df_speedup = pd.DataFrame(speedup, index=df_pivot_cycles.index, columns=df_pivot_cycles.columns)
    df_energy = pd.DataFrame(energy_red, index=df_pivot_energy.index, columns=df_pivot_energy.columns)
    
    # Plot Speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    df_speedup.plot(kind='bar', ax=ax, rot=0)
    ax.set_ylabel('Speedup over BitFusion Baseline')
    ax.set_title('Quantization Schemes Performance Comparison')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
    plt.tight_layout()
    plt.savefig('fig/quantization_speedup.pdf')
    
    # Plot Energy Reduction
    fig, ax = plt.subplots(figsize=(10, 6))
    df_energy.plot(kind='bar', ax=ax, rot=0)
    ax.set_ylabel('Energy Reduction over BitFusion Baseline')
    ax.set_title('Quantization Schemes Energy Comparison')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
    plt.tight_layout()
    plt.savefig('fig/quantization_energy.pdf')
    
    print("Graphs saved to fig/quantization_speedup.pdf and fig/quantization_energy.pdf")
