from simu import Simu
from dynammic_programming import sarsa
from visu import plot_latence_multiple_rats, plot_learning_curve_multiple_rats
import numpy as np
import random
import math
import matplotlib.patches as patches
from matplotlib import pyplot as plt


pool = Simu()
nb_rats = 50
nb_trials = 30
all_latences = []
all_norms = []
all_positions = []
all_Qs = []
eps = 0.3
eta = 0.01
gamma=0.9
exp_type = "exp"
for rats in range(nb_rats):
    print('rat number: ', rats)
    Qs,_,_,norms,_,latences,trajs = sarsa(pool,nb_trials,eta, eps, gamma)
    all_latences.append(latences)
    all_norms.append(norms)
    all_positions.append(trajs)
    all_Qs.append(Qs)
plot_latence_multiple_rats(all_latences,'plot_latences_nbTrials_' +str(nb_trials)+ '_eta_' + str(eta) +'_eps_'+str(eps)+'_gamma_'+str(gamma)+'_nb_rats_'+str(nb_rats)+exp_type)
plot_learning_curve_multiple_rats(all_norms,'plot_norms_nbTrials_'+str(nb_trials) + '_eta_' + str(eta) +'_eps_'+str(eps)+'_gamma_'+str(gamma)+'_nb_rats_'+str(nb_rats)+exp_type)

pool.plot_cells_from_Q(Qs[-1],eps,"courbe_accords_quelconque")
### TRAJS OF BEST and WORST RAT #####
all_latences = np.array(all_latences)
mean_latences = all_latences.mean(axis = 1)
print(mean_latences.shape)
idx_best_rat = np.argmin(mean_latences)
idx_worst_rat = np.argmax(mean_latences)
print(np.array(all_Qs).shape)
Q_no_learning = np.random.rand(400,8)
# idx_best_traj_of_best_rat = np.argmin(all_latences[idx_best_rat,:])
# pool.plot_trajectory(all_positions[idx_best_rat][idx_best_traj_of_best_rat],'traj_best_rat_nbTrials_'+str(nb_trials)+ '_eta_' + str(eta) +'_eps_'+str(eps)+'_gamma_'+str(gamma))
pool.plot_from_Q(all_Qs[idx_best_rat][-1],6000,eps,'traj_best_rat_nbTrials_after_'+str(nb_trials)+ '_eta_' + str(eta) +'_eps_'+str(eps)+'_gamma_'+str(gamma)+'_nb_rats_'+str(nb_rats)+exp_type)
pool.plot_from_Q(Q_no_learning,6000,eps,'traj_best_rat_nbTrials_before_'+str(nb_trials)+ '_eta_' + str(eta) +'_eps_'+str(eps)+'_gamma_'+str(gamma)+'_nb_rats_'+str(nb_rats)+exp_type)
pool.plot_from_Q(all_Qs[idx_worst_rat][-1],6000,eps,'traj_worst_rat_nbTrials_after_'+str(nb_trials)+ '_eta_' + str(eta) +'_eps_'+str(eps)+'_gamma_'+str(gamma)+'_nb_rats_'+str(nb_rats)+exp_type)
Q_no_learning = np.random.rand(400,8)
pool.plot_from_Q(Q_no_learning,6000,eps,'traj_worst_rat_nbTrials_before_'+str(nb_trials)+ '_eta_' + str(eta) +'_eps_'+str(eps)+'_gamma_'+str(gamma)+'_nb_rats_'+str(nb_rats)+exp_type)
