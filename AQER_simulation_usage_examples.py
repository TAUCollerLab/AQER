# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 09:42:28 2025

@author: anatl
"""

import matplotlib.pyplot as plt
from AQER_Simulator import Scenario, Simulator, Scenario_by_correlation
import numpy as np




############################################################################
# Appendix F1: Violating the assumption that worker responses are i.i.d
# changing the correlation between weak workers
############################################################################

# simulator = Simulator(max_iterations=1000, threshold=0.00001, simulation_repetitions=30)

# performance_vals2 = []
# performance_epoch0_vals2 = []

# cors = [0,0.3,0.5,0.6,0.7,0.8,0.9]
# for cor in cors:
#     scenario = Scenario_by_correlation(num_questions=10, answer_dim=300,cor_between_weak_workers=cor,weak_std=2)
#     scenario.add_workers(num_workers=10, standard_deviation=3.0, bias=0)
#     scenario.add_workers(num_workers=10, standard_deviation=0.5, bias=0)


#     #run simulation of this scenario
#     avg_epoch_0, std_epoch_0, avg_with_iter, std_with_iter = simulator.run_simulation(scenario)

#     performance_vals2.append(avg_with_iter)
#     performance_epoch0_vals2.append(avg_epoch_0)
#     print(f"correlation: {cor}, Performance with iterations: {avg_with_iter}")


# plt.figure(figsize=(8, 6))
# plt.plot(cors, performance_vals2, marker='o',label='AQER')
# # plt.plot(cors, performance_vals2, marker='o',label='Full AQER')
# # plt.plot(cors, performance_epoch0_vals2, marker='s', color='orange', label='Simplified AQER')

# plt.xlabel("Correlation between weak workers")
# plt.ylabel("Pearson Correlation")
# # plt.title("Performance vs. correlation between weak workers")
# plt.grid(True)
# # plt.legend()

# plt.show()




##############################################################################
# Appendix F2: Violatting the assumption that the nunber of worker is large
# Gradually decreasign the number of workers:
##############################################################################

# simulator = Simulator(max_iterations=1000, threshold=0.00001, simulation_repetitions=30)
# scenario = Scenario(num_questions=10, answer_dim=300)

# w_numbers = []
# performance_vals = []
# performance_epoch0_vals = []


# # changing the number of questions (this refreshes the correct answers and worker_answers)
# for w_num in range(1,33,3):
#     scenario = Scenario(num_questions=10, answer_dim=300)
#     scenario.add_workers(num_workers=w_num, standard_deviation=1.0, bias=0)
#     scenario.add_workers(num_workers=w_num, standard_deviation=0.5, bias=0)
#     scenario.add_workers(num_workers=w_num, standard_deviation=2, bias=0)


#     #run simulation of this scenario
#     avg_epoch_0, std_epoch_0, avg_with_iter, std_with_iter = simulator.run_simulation(scenario)

#     w_numbers.append(w_num*3,)
#     performance_vals.append(avg_with_iter)
#     performance_epoch0_vals.append(avg_epoch_0)
#     print(f"Number of workers: {w_num*3}, Performance with iterations: {avg_with_iter}")

# plt.figure(figsize=(8, 6))
# plt.plot(w_numbers, performance_vals, marker='o',label='AQER')
# # plt.plot(w_numbers, performance_vals, marker='o',label='Full AQER')
# # plt.plot(w_numbers, performance_epoch0_vals, marker='s', color='orange', label='Simplified AQER')

# plt.xlabel("Number of Workers")
# plt.ylabel("Pearson Correlation")
# # plt.title("Performance vs. Number of Workers")
# plt.grid(True)
# # plt.legend()
# # Reverse x-axis from 90 to 0
# plt.gca().invert_xaxis()

# plt.show()

###################################################################
# Appendix F3: Noisy Responses - High embeddign variance
#  Gradually increasing workers' stds
###################################################################

# simulator = Simulator(max_iterations=1000, threshold=0.00001, simulation_repetitions=30)

# stds = []
# performance_with_iter_vals = []
# performance_epoch0_vals = []

# # Vary the worker's standard deviation
# for s in np.arange(0.5, 10, 0.5):
#     scenario = Scenario(num_questions=20, answer_dim=512)
#     # Use 's' as the standard deviation while keeping bias constant (0)
#     scenario.add_workers(num_workers=10, standard_deviation=s+1.5, bias=0)
#     scenario.add_workers(num_workers=10, standard_deviation=s+0.5, bias=0)
#     scenario.add_workers(num_workers=10, standard_deviation=s+1, bias=0)
    
#     # Run simulation for this scenario
#     avg_epoch_0, std_epoch_0, avg_with_iter, std_with_iter = simulator.run_simulation(scenario)
    
#     stds.append(s+0.5)
#     performance_with_iter_vals.append(avg_with_iter)
#     performance_epoch0_vals.append(avg_epoch_0)
    
#     print(f"Standard Deviation: {s}, Performance with iterations: {avg_with_iter}, Epoch 0 Performance: {avg_epoch_0}")

# # Plot both metrics on the same diagram
# plt.figure(figsize=(8, 6))

# plt.plot(stds[0:10], performance_with_iter_vals[0:10], marker='o', label='AQER')
# # plt.plot(stds[0:10], performance_epoch0_vals[0:10], marker='s', color='orange', label='Simplified AQER')
# plt.xlabel("Mean Worker Standard Deviation")
# plt.ylabel("Pearson Correlation")
# # plt.title("Performance vs. Worker Standard Deviation")
# plt.grid(True)
# # plt.legend()
# plt.show()


##########################################################
# Appendix F4: Noisy Responses - Systematic Bias
# Gradualy increasing the percentage of biased workers:
##########################################################

simulator = Simulator(max_iterations=1000, threshold=0.00001, simulation_repetitions=30)
scenario = Scenario(num_questions=10, answer_dim=300)

biases_ratio = []
performance_with_iter_vals1 = []
performance_epoch0_vals1 = []

b=0.5
num_w = 30
# changing the number of questions (this refreshes the correct answers and worker_answers)
for biased_w in [0,3,6,9,12,15,18,21]:
    scenario = Scenario(num_questions=10, answer_dim=300)
    scenario.add_workers(num_workers=num_w-biased_w, standard_deviation=0.5, bias=0)
    scenario.add_workers(num_workers=num_w-biased_w, standard_deviation=1, bias=0)
    scenario.add_workers(num_workers=biased_w, standard_deviation=0.5, bias=b)
    scenario.add_workers(num_workers=biased_w, standard_deviation=1, bias=b)
    

    #run simulation of this scenario
    avg_epoch_0, std_epoch_0, avg_with_iter, std_with_iter = simulator.run_simulation(scenario)

    biases_ratio.append(biased_w/num_w)
    performance_with_iter_vals1.append(avg_with_iter)
    performance_epoch0_vals1.append(avg_epoch_0)

    print(f"Bias ratio: {biased_w/num_w}, Performance with iterations: {avg_with_iter}")


plt.figure(figsize=(8, 6))
plt.plot(biases_ratio[0:8], performance_with_iter_vals1[0:8], marker='o', label='AQER')
# plt.plot(biases_ratio[0:8], performance_with_iter_vals1[0:8], marker='o', label='Full AQER')
# plt.plot(biases_ratio[0:8], performance_epoch0_vals1[0:8], marker='s', color='orange', label='Simplified AQER')
plt.xlabel("Biased Worker Ratio")
plt.ylabel("Pearson Correlation")
# plt.title("Performance vs. Biased Workers Ratio")
plt.grid(True)
# plt.legend()
plt.show()




############################################################
# # Another variation of Noisy Responses - Systematic Bias
# Gradualy increasing the workers' biased:
############################################################

# simulator = Simulator(max_iterations=1000, threshold=0.00001, simulation_repetitions=100)
# scenario = Scenario(num_questions=10, answer_dim=300)

# biases = []
# performance_with_iter_vals = []
# performance_epoch0_vals = []


# # changing the number of questions (this refreshes the correct answers and worker_answers)
# for b in np.arange(0,5,0.5):
#     scenario = Scenario(num_questions=10, answer_dim=300)
#     scenario.add_workers(num_workers=10, standard_deviation=1.0, bias=b)
#     scenario.add_workers(num_workers=10, standard_deviation=0.5, bias=b)
#     scenario.add_workers(num_workers=10, standard_deviation=2, bias=b)

#     #run simulation of this scenario
#     avg_epoch_0, std_epoch_0, avg_with_iter, std_with_iter = simulator.run_simulation(scenario)

#     biases.append(b)
#     performance_with_iter_vals.append(avg_with_iter)
#     performance_epoch0_vals.append(avg_epoch_0)

#     print(f"Bias: {b}, Performance with iterations: {avg_with_iter}")


# plt.figure(figsize=(8, 6))
# plt.plot(biases, performance_with_iter_vals, marker='o', label='Performance with Iterations (Avg)')
# plt.plot(biases, performance_epoch0_vals, marker='s', color='orange', label='Epoch 0 Performance (Avg)')
# plt.xlabel("Worker Bias")
# plt.ylabel("Performance with Iterations (Avg)")
# plt.title("Performance vs. Workers' Bias ")
# plt.grid(True)
# plt.legend()
# plt.show()

