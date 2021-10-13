from matplotlib import pyplot as plt
import numpy as np

def plot_learning_curve_multiple_rats(all_rats_norms, file_name):
    all_rats_norms = np.array(all_rats_norms)
    mean_norm = all_rats_norms.mean(axis = 0)
    quantile_25 = np.quantile(all_rats_norms, 0.25, axis = 0)
    quantile_75 = np.quantile(all_rats_norms, 0.75, axis = 0)
    plt.plot(mean_norm, label = 'mean norm of Q matrix')
    plt.fill_between(range(len(mean_norm)), quantile_75,quantile_25, alpha = 0.3)
    plt.title('learning Curve')
    plt.ylabel('matrix norm')
    plt.xlabel('episode')
    plt.savefig(file_name + '.pdf', format = 'pdf')
    plt.clf()

def plot_latence_multiple_rats(all_rats_latences, file_name):
    all_rats_latences = np.array(all_rats_latences)
    mean_latence = all_rats_latences.mean(axis = 0)
    quantile_25 = np.quantile(all_rats_latences, 0.25, axis = 0)
    quantile_75 = np.quantile(all_rats_latences, 0.75, axis = 0)
    plt.plot(mean_latence, label = 'mean latence of rats')
    plt.legend(loc = 'best')
    plt.fill_between(range(len(mean_latence)), quantile_75,quantile_25, alpha = 0.3)
    plt.title('Latence Curve')
    plt.ylabel('latence')
    plt.xlabel('episode')
    plt.savefig(file_name +'.pdf', format = 'pdf')
    plt.clf()
