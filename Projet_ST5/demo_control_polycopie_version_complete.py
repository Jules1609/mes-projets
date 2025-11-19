# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os

import matplotlib.pyplot as plt
import numpy as np


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import compute_alpha 
import audio
#import solutions



def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		
		return 2
	else:
		return 0
      
def compute_gradient_descent(chi, grad, domain, mu):
	"""This function makes the gradient descent.
	This function has to be used before the 'Projected' function that will project
	the new element onto the admissible space.
	:param chi: density of absorption define everywhere in the domain
	:param grad: parametric gradient associated to the problem
	:param domain: domain of definition of the equations
	:param mu: step of the descent
	:type chi: numpy.array((M,N), dtype=float64
	:type grad: numpy.array((M,N), dtype=float64)
	:type domain: numpy.array((M,N), dtype=int64)
	:type mu: float
	:return chi:
	:rtype chi: numpy.array((M,N), dtype=float64

	.. warnings also: It is important that the conditions be expressed with an "if",
			not with an "elif", as some points are neighbours to multiple points
			of the Robin frontier.
	"""

	(M, N) = numpy.shape(domain)
	# for i in range(0, M):
	# 	for j in range(0, N):
	# 		if domain_omega[i, j] != _env.NODE_ROBIN:
	# 			chi[i, j] = chi[i, j] - mu * grad[i, j]
	# # for i in range(0, M):
	# 	for j in range(0, N):
	# 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
	# 			chi[i,j] = chi[i,j] - mu*grad[i,j]
	# print(domain,'jesuisla')
	#chi[50,:] = chi[50,:] - mu*grad[50,:]
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			#print(i,j)
			#chi[i,j] = chi[i,j] - mu * grad[i,j]
			a = BelongsInteriorDomain(domain[i + 1, j])
			b = BelongsInteriorDomain(domain[i - 1, j])
			c = BelongsInteriorDomain(domain[i, j + 1])
			d = BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:
				
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi

import numpy as np

def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                Alpha, mu, chi, V_obj, mu1, V_0, skip,
                                numb_iter=100):
    """
    Version robuste de la procédure d'optimisation.
    - domain_omega: matrice indicatrice du domaine (valeurs entières ou masques)
    - chi: champ (M,N) initial avec valeurs dans [0,1]
    - V_obj: proportion cible sur les noeuds _env.NODE_ROBIN (valeur entre 0 et 1)
    - retourne: chi, energy (array shape (numb_iter+1,1)), u (champ solution), grad (dernièr grad), ene (dernière énergie)
    """
    #Validations simples pour ne pas avoir d'erreurs de dimensions
    if chi is None:
        raise ValueError("chi must be provided and have same shape as domain_omega")
    if chi.shape != domain_omega.shape:
        raise ValueError("chi and domain_omega must have same shape")
    if not (0.0 <= V_obj <= 1.0):
        raise ValueError("V_obj must be between 0 and 1")

    k = 0
    
    #On initialise les variables
    M, N = domain_omega.shape
    energy = np.zeros((numb_iter + 1, 1), dtype=np.float64)
    grad = np.zeros((M, N), dtype=np.complex128)

    #On définit une liste qui contient les alpha_rob pour chaque fréquence sur laquelle on travaille
    list_alpha_rob_current = []
    for m in range(len(Alpha)):
        alpha_rob_current = preprocessing.set2zero(Alpha[m] * chi, domain_omega)
        list_alpha_rob_current.append(alpha_rob_current)

    #On définit une liste qui contient les solutions u pour chaque fréquence sur laquelle on travaille
    list_u = []
    for n in range(len(omega)):
         u = processing.solve_helmholtz(domain_omega, spacestep, omega[n], f, f_dir[n], f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, list_alpha_rob_current[n])
         list_u.append(u)

    #On calcule la somme des énergies sur les différentes fréquences étudiées
    ene = your_compute_objective_function(domain_omega, list_u, spacestep, mu1, V_0)
    energy[k] = ene

    #Définition des paramètres de la boucle principale
    min_mu = 1e-5
    vol_tol = 1e-5    #Tolérance sur la proportion de volume
    tol_l = 1e-9    #Tolérance sur l'intervalle de dichotomie
    max_bisect_iters = 60
    max_backtrack_iters = 30

    #Si il n'y a pas de noeuds robin, on arrête la procédure
    N_robin = np.sum(domain_omega == _env.NODE_ROBIN)
    if N_robin == 0:
        raise ValueError("No ROBIN nodes found in domain_omega (_env.NODE_ROBIN).")

    #Boucle principale que l'on fait tourner seulement si skip == 0
    if skip == 0:
        while k < numb_iter and mu > min_mu:

            rhs_p_list = []
            p_list = []
            grad_sensibility = numpy.zeros((M, N), dtype=numpy.complex128)
            for n in range(len(omega)):
                #1) Résolution du problème adjoint pour chaque fréquence
                rhs_p = -2.0 * list_u[n].conjugate()

                p = processing.solve_helmholtz(domain_omega, spacestep, omega[n], rhs_p, 0 * f_dir[n], f_neu, f_rob,
                                               beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                               list_alpha_rob_current[n])
                p_list.append(p)

                #2) Calcul du Gradient paramétrique (on somme les contributions de chaque fréquence)
                grad_sensibility += -np.real(p_list[n] * list_u[n] * Alpha[n])
            
            grad = grad_sensibility.copy()

            #3) On effectue l'algorithme de descente de gradient en utilisant une projection obtenue par dichotomie
            ene_old = ene
            mu_inner = mu
            chi_temp = chi.copy()

            backtrack_iter = 0
            success_inner = False
            while backtrack_iter < max_backtrack_iters and mu_inner >= min_mu:
                #On calcule le nouveau chi temporaire
                chi_new_temp = compute_gradient_descent(chi_temp.copy(), grad.copy(), domain_omega, mu_inner)
                chi_new_temp_fixed = chi_new_temp.copy()

                #On définit une fonction pour la projection par dichotomie
                def volume_for_l(l):
                    cand = chi_new_temp_fixed + l
                    cand = np.maximum(0.0, np.minimum(1.0, cand))
                    return np.sum(cand) / float(N_robin)

                #Si le volume actuel est déjà suffisamment proche, on évite la dichotomie
                current_vol = np.sum(np.maximum(0.0, np.minimum(1.0, chi_new_temp_fixed))) / float(N_robin)
                if abs(current_vol - V_obj) <= vol_tol:
                    l_final = 0.0
                    chi_candidate = np.maximum(0.0, np.minimum(1.0, chi_new_temp_fixed + l_final))
                else:
                    #Dichotomie pour trouver le bon l
                    l_a, l_b = -200.0, 200.0
                    g_a = volume_for_l(l_a)
                    g_b = volume_for_l(l_b)

                    expand_count = 0
                    while (g_a - V_obj) * (g_b - V_obj) > 0 and expand_count < 20:
                        l_a *= 2.0
                        l_b *= 2.0
                        g_a = volume_for_l(l_a)
                        g_b = volume_for_l(l_b)
                        expand_count += 1

                    if (g_a - V_obj) * (g_b - V_obj) > 0:
                        l_grid = np.linspace(-l_b, l_b, 1001)
                        best_l = l_grid[0]
                        best_diff = abs(volume_for_l(best_l) - V_obj)
                        for ltest in l_grid:
                            diff = abs(volume_for_l(ltest) - V_obj)
                            if diff < best_diff:
                                best_diff = diff
                                best_l = ltest
                        l_final = best_l
                        chi_candidate = np.maximum(0.0, np.minimum(1.0, chi_new_temp_fixed + l_final))
                    else:
                        
                        iter_b = 0
                        l_final = 0.5 * (l_a + l_b)
                        while iter_b < max_bisect_iters and (l_b - l_a) > tol_l:
                            l_mid = 0.5 * (l_a + l_b)
                            g_mid = volume_for_l(l_mid)
                            if abs(g_mid - V_obj) <= vol_tol:
                                l_final = l_mid
                                break
                            
                            if (g_a - V_obj) * (g_mid - V_obj) <= 0:
                                l_b = l_mid
                                g_b = g_mid
                            else:
                                l_a = l_mid
                                g_a = g_mid
                            l_final = 0.5 * (l_a + l_b)
                            #On évite de stagner trop longtemps
                            if (l_b - l_a) <= tol_l:
                                break
                            iter_b += 1
                        chi_candidate = np.maximum(0.0, np.minimum(1.0, chi_new_temp_fixed + l_final))

                #On résolut le problème pour la nouvelle chi candidate
                list_alpha_rob_new = []
                list_u_new = []
                for m in range(len(Alpha)) :
                    alpha_rob_new = preprocessing.set2zero(Alpha[m] * chi_candidate, domain_omega)
                    list_alpha_rob_new.append(alpha_rob_new)
                    u_new = processing.solve_helmholtz(domain_omega, spacestep, omega[m], f, f_dir[m], f_neu, f_rob,
                                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, list_alpha_rob_new[m])
                    list_u_new.append(u_new)

                ene_new = your_compute_objective_function(domain_omega, list_u_new, spacestep, mu1, V_0)

                #On vérifie si on a une diminution de l'énergie pour accepter la nouvelle chi
                if ene_new < ene_old:
                    
                    chi = chi_candidate.copy()
                    list_u = list_u_new.copy()
                    ene = ene_new
                    list_alpha_rob_current = list_alpha_rob_new.copy()
                    mu_inner = min(mu_inner * 1.1, mu)  #On n'excède pas un certain mu
                    success_inner = True
                    break
                else:
                    
                    mu_inner = mu_inner / 2.0
                    backtrack_iter += 1

            #On met à jour les variables pour la prochaine itération
            energy[k + 1] = ene
            mu = max(mu_inner, min_mu)
            k += 1
            

            #On s'arrête si on n'a pas réussi à faire de progrès et que mu est trop petit
            if not success_inner and mu <= min_mu:
                break

    #On remplit le reste du tableau d'énergie si on a arrêté avant le nombre d'itérations maximal
    if k < numb_iter:
        energy[k + 1:] = ene

    return chi, energy, list_u, grad, ene



def your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 + mu1*(Vol(domain_omega)-V_0)

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation;
        mu1: float, it is the constant that defines the importance of the volume
        constraint;
        V_0: float, it is a reference volume.
    """

    integral_u_squared = 0.0
    for k in range(len(u)) :
        u = u[k]
        integral_u_squared += numpy.sum(numpy.abs(u)**2) * (spacestep ** 2)
    #Ne pas calculer le terme correctif sur le volume car on reste dans le volume
    '''
    num_nodes_interior = numpy.sum(domain_omega == _env.NODE_INTERIOR)
    vol_omega = num_nodes_interior * (spacestep ** 2)
    volume_penalty = mu1 * (vol_omega - V_0)
    '''
    energy = integral_u_squared 


    return energy



if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2 #Niveau de fractal sur lequel on se place
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    
    #On récupère les fréquences et amplitudes de l'audio .mp3
    freqs, amp, sr = audio.fourier_from_audio("C:\\Users\\victo\\OneDrive\\Documents\\Travail\\CentraleSupelec\\2A\\ST5\\EI\\given_4students\\audioTrain4.mp3")
    
    #On peut décommenter ce bloc pour calculer les fréquences et alphas associés à notre matériau choisi
    '''
    omegas, alphas = compute_alpha.run_compute_alpha('AE 2.93')
    '''
    #On charge les fichiers numpy déjà calculés pour gagner du temps
    omegas = np.load('omegas_2.93.npy')
    alphas = np.load('alphas_2.93.npy')

    #On initialise des listes pour stocker les résultats
    e_tot = []
    e_tot_bis = []
    e_max = 0
    
    list_f_dir = []
    list_freq_ini = []

    #On choisit les fréquences sur lesquelles on va faire l'optimisation
    n = 2  #Nombre de fréquences choisies

    #On prend la fréquence 108 et 149 (correspondant à environ 75 Hz et 81 Hz)
    j = 108
    print('Frequence 1 considere : ', freqs[j])
    f_ini = freqs[j]
    list_freq_ini.append(f_ini)
    f_dir[:, :] = 0.0
    for i in range(N):
        y_ord = i*spacestep
        f_dir[0,i] = audio.g(y_ord,j, amp)
    list_f_dir.append(f_dir)
    
    j = 149
    print('Frequence 2 considere : ', freqs[j])
    f_ini = freqs[j]
    list_freq_ini.append(f_ini)
    f_dir[:, :] = 0.0
    for i in range(N):
        y_ord = i*spacestep
        f_dir[0,i] = audio.g(y_ord,j, amp)
    list_f_dir.append(f_dir)
    
    #On calcule les omegas associés aux fréquences choisies
    list_omega_f = []
    for l in range(n) :
        omega_f = 2*np.pi * list_freq_ini[l]
        list_omega_f.append(omega_f)
    
    #On initialise chi et on l'affiche
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    plt.imshow(chi, cmap='viridis', origin='lower')
    plt.colorbar(label='Valeur')
    plt.title("Affichage d'une matrice 2D")
    plt.show()
    nbre_points_chi_egal_1 = numpy.sum(chi==1.0)
    
    #On calcule les wavenumbers et alpha_rob associés aux fréquences choisies
    list_wavenumber = []
    Alpha = []
    for m in range(len(list_omega_f)) :
        for i in range(len(omegas)):
                if omegas[i] > list_omega_f[m]:
                    Alpha_temp = alphas[i]
                    Alpha.append(Alpha_temp)
                    break
        wavenumber = list_omega_f[m]/340
        list_wavenumber.append(wavenumber)
    
    #On calcule les alpha_rob associés aux fréquences choisies
    list_alpha_rob = []
    for m in range(len(list_omega_f)) :
        alpha_rob = Alpha[m] * chi
        list_alpha_rob.append(alpha_rob)
    
    #On initialise les paramètres pour l'optimisation
    S = 0  # surface of the fractal
    for i in range(0, M):
        for l in range(0, N):
            if domain_omega[i, l] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 3  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional
    
    #On fait l'optimisation sur les deux fréquences pour calculer notre chi optimisé
    chi, energy, u, grad, enen = your_optimization_procedure(domain_omega, spacestep, list_wavenumber, f, list_f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, list_alpha_rob,
                        Alpha, mu, chi, V_obj, mu1, V_0,0)

    #On projette notre chi optimisé pour n'avoir que des 0 et des 1
    chin = chi.copy()
    robin_mask = (domain_omega == _env.NODE_ROBIN)
    chi_robin = chin[robin_mask]
    indices_sorted = numpy.argsort(chi_robin)[::-1]
    chi_proj_robin = numpy.zeros_like(chi_robin)
    chi_proj_robin[indices_sorted[:nbre_points_chi_egal_1]] = 1

    chi_proj = numpy.zeros_like(chi)
    chi_proj[robin_mask] = chi_proj_robin

    #On affiche le chi projeté obtenu après optimisation
    chin_proj = chi_proj.copy()
    plt.imshow(chin_proj, cmap='viridis', origin='lower')
    plt.colorbar(label='Valeur')
    plt.title("Affichage d'une matrice 2D")
    plt.show()

    #On fait une boucle sur toutes les fréquences pour calculer l'énergie totale obtenue avec le chi optimisé
    L = []
    mini_max = 0
    for j in range(0,len(freqs)):
        pourcent = j*100/(len(freqs))
        print(pourcent)
        
        #On définit l'expression de la source pour la fréquence considérée
        freq = freqs[j]
        f_dir[:, :] = 0.0
        for i in range(N):
            y_ord = i*spacestep
            f_dir[0,i] = audio.g(y_ord,j, amp)
        
        
        omega_f = 2*np.pi * freq
        
        #On récupère l'Alpha et wavenumber associés à la fréquence considérée
        for i in range(len(omegas)):
            if omegas[i] > omega_f:
                Alpha = alphas[i]
                break
        wavenumber = omega_f/340

        alpha_rob = Alpha * chin_proj

        #On initialise les paramètres pour résoudre le problème de Helmholtz
        S = 0  # surface of the fractal
        for i in range(0, M):
            for l in range(0, N):
                if domain_omega[i, l] == _env.NODE_ROBIN:
                    S += 1
        V_0 = 1  # initial volume of the domain
        V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
        mu = 3  # initial gradient step
        mu1 = 10**(-5)  # parameter of the volume functional

            # ----------------------------------------------------------------------
            # -- Do not modify this cell, these are the values that you will be assessed against.
            # ----------------------------------------------------------------------
            # -- compute finite difference solution
        energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
        
        #On ne refait pas l'optimisation, on met skip = 1 : on veut juste calculer l'énergie avec le chi optimisé
        sk = 1
        print(j, freq)
        chi, energy, u, grad, enen = your_optimization_procedure(domain_omega, spacestep, [wavenumber], f, [f_dir], f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, [alpha_rob],
                        [Alpha], mu, chi_proj, V_obj, mu1, V_0,sk)
        un_proj = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,alpha_rob)
    
        e_tot.append(enen.flatten())
        e_tot_bis.append(numpy.sum(numpy.abs(un_proj)**2)*(spacestep**2))
    
    #On affiche la solution controlée pour la 1ère fréquence considérée dans des fichiers .jpg
    postprocessing._plot_controled_solution(u[0], chin_proj)

    plt.figure(figsize=(10, 5))

    #Tracé du spectre énergétique
    plt.plot([freqs[j] for j in range(0, len(freqs))], e_tot, linewidth=2, label='Énergie totale', marker = 'o')

    #Mise en forme du graphique
    plt.title("Spectre énergétique", fontsize=16, fontweight='bold')
    plt.xlabel("Fréquence (Hz)", fontsize=14)
    plt.ylabel("Énergie (u.a.)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.show()

    print('End.')

    #On peut si on le souhaite décommenter ce bloc pour afficher les 30 indices des fréquences les plus énergétiques pour aider à cibler les fréquences sur lesquelles il est intéressant d'optimiser
    '''
    #Convertir en tableau numpy
    arr = np.array(e_tot_bis)

    # Trouver les indices des 30 plus grandes valeurs
    indices_top30 = np.argsort(arr)[-30:][::-1]

    print("Indices des 30 plus grandes valeurs :", indices_top30)
    print("Valeurs correspondantes :", arr[indices_top30])
    '''

