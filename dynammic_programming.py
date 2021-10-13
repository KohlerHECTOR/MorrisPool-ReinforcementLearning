import numpy as np
import random
import math

def sarsa(env,K, eta, eps, gam):
    N = 8 #nb sorties
    D = 400 #nb entrees
    # matrice de poids
    W = np.random.rand(N,D)

    # matrice etats-actions
    Q = np.random.rand(D,N)


    mem_Q  = []
    mem_norm_Q = []
    mem_delta  = []
    mem_etats = []
    mem_reward = []
    mem_latence = []
    mem_trajectories = []
    sigma = 0.05



############################### ONE RAT #######################################
    # boucle pour K epreuves
    for i in range(K):
        # l'etat initial
        env.reinit()
        total_reward = 0
        s, s_x, s_y = env.pos_to_neurones()
        # les activites de neurones d'entree encode l'état de l'animal
        # x = np.zeros(D)
        # count = 0
        # for neur_x in range(int(math.sqrt(D))):
        #     for neur_y in range(int(math.sqrt(D))):
        #         x[count] = math.exp((-(neur_x*0.05+0.025 - env.current_position[0])**2-(neur_y*0.05+0.025 - env.current_position[1])**2)/(2*sigma**2))
        #         count +=1
        neur_x = np.arange(20)
        neur_y = np.arange(20)
        x_j,y_j = np.meshgrid(neur_x,neur_y, indexing='xy')
        x = np.exp((-(x_j*0.05+0.025 - env.current_position[0])**2-(y_j*0.05+0.025 - env.current_position[1])**2)/(2*sigma**2)).flatten()

        # x[s] = math.exp((-(s_x*0.05+0.025 - env.current_position[0])**2-(s_y*0.05+0.025 - env.current_position[1])**2)/(2*sigma**2))
        # x[s] = math.exp((-(env.current_position[0] - s_x*0.05+0.025)**2-(env.current_position[1] - s_y*0.05+0.025)**2)/(2*sigma**2))

        # la fonction-valeur correspond a l'activite des neurones de sortie
        Q[s:] = np.dot( W, x)

        # choix d'une action, strategie epsilon-greedy
        if np.random.rand() < eps:
            a = np.random.randint(N)   # action aleatoire, exploration
        else:
            a = np.argmax(Q[s:])       # action optimale, exploitation

        mem_delta_epreuve = []
        mem_etats_epreuve = []

        ###################### ONE TRAJECTORY #############################
        while (not env.done):
            if env.time>=600:
                print('the rat probably died in the water')
                break
            env.step(a)
            total_reward += env.reward
            s_new,  _, _ = env.pos_to_neurones()
            # l'activite des neurones d'entree dans le nouvel etat s'
            neur_x = np.arange(20)
            neur_y = np.arange(20)
            x_j,y_j = np.meshgrid(neur_x,neur_y, indexing='xy')
            x_new = np.exp((-(x_j*0.05+0.025 - env.current_position[0])**2-(y_j*0.05+0.025 - env.current_position[1])**2)/(2*sigma**2)).flatten()
            # x_new[s_new] = math.exp((-(s_new_x*0.05+0.025 - env.current_position[0])**2-(s_new_y*0.05+0.025 - env.current_position[1])**2)/(2*sigma**2))
            # x_new[s_new] = math.exp((-(env.current_position[0] - s_new_x*0.05+0.025)**2-(env.current_position[1] - s_new_y*0.05+0.025)**2)/(2*sigma**2))
            # les valeurs d'actions dans le nouvel état
            Q[s_new] = np.dot( W, x_new)

            # choisir la nouvelle action a` dans l'etat s`
            if np.random.rand() < eps :
                a_new = np.random.randint(N)    # action aleatoire, exploration
            else:
                a_new = np.argmax(Q[s_new,:]) # action optimale, exploitation

            # mettre a jour la matrice de poids
            delta = env.reward + gam * Q[s_new, a_new] - Q[s,a]  # signal dopaminergique
            W[a, :] = W[a, :]  + eta*delta*x            # plasticite synaptique

            # initialiser la nouvelle epreuve
            a = a_new
            s = s_new
            x = x_new

            # sauvegarder les resultats de l'epreuve pour visualisation
            mem_delta_epreuve.append(delta)
            mem_etats_epreuve.append(s)
        mem_reward.append(total_reward)
        mem_latence.append(env.time)
        # print(env.time)
        mem_trajectories.append(env.all_positions)
        # sauvegarder pour visualisation
        mem_Q.append(Q.copy())
        mem_norm_Q.append(np.linalg.norm(Q))
        mem_delta.append(np.array(mem_delta_epreuve))
        mem_etats.append(np.array(mem_etats_epreuve))
    ############################################################################



    mem_Q = np.array(mem_Q)
################################################################################
    return mem_Q, mem_delta, mem_etats, mem_norm_Q, mem_reward, mem_latence, mem_trajectories
