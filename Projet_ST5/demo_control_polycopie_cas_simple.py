# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os
import matplotlib.pyplot as plt


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import compute_alpha 
#import solutions

def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		print("Robin")
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
				print(i+1,j, "-----", "i+1,j")
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				print(i - 1, j, "-----", "i - 1, j")
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				print(i, j + 1, "-----", "i , j + 1")
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				print(i, j - 1, "-----", "i , j - 1")
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi

#On définit la procédure d'optimisation complète
def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj, mu1, V_0):
    
    #On initialise les variables
    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    grad = numpy.zeros((M, N), dtype=numpy.complex128)

    #On résout le problème de Helmholtz avec le chi initial
    alpha_rob_current = preprocessing.set2zero(Alpha * chi, domain_omega)
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_current)
	
    #On calcule l'énergie initiale
    ene = your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0)
    energy[k] = ene

    #On entre dans la boucle principale d'optimisation qui s'arrête soit après un nombre d'itérations fixé, soit lorsque mu devient trop petit
    while k < numb_iter and mu > 10**(-5):
        
        #1) Résolution du problème adjoint
        rhs_p = -2.0 * u.conjugate()
        
        #Résoudre l'équation de Helmholtz pour p
        p = processing.solve_helmholtz(domain_omega, spacestep, omega, rhs_p, 0*f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_current)
        
        #2) Calcul du Gradient paramétrique
        #grad_J = -Re( p * u * Alpha)
        grad_sensibility = - numpy.real(p * u * Alpha)   
        grad = grad_sensibility.copy()  


        #3) On effectue l'algorithme de descente de gradient en utilisant une projection obtenue par dichotomie
        ene_old = ene
        mu_inner = mu
        chi_temp = chi.copy() 

        while True: 
            if mu_inner < 10**(-5):
                break 
            #On calcule le nouveau chi temporaire
            chi_new_temp = compute_gradient_descent(chi_temp.copy(), grad.copy(), domain_omega, mu_inner)
            chi_new_temp_fixed = chi_new_temp.copy()
            
            #On fait la projection par dichotomie
            l = 0  #Constante à déterminer à chaque itération pour la projection sur Uadmissible
            chi_new = numpy.maximum(0.0, numpy.minimum(1.0, chi_new_temp + l))

            l_a = -100
            l_b = 100
            l_c = 0
            while abs((numpy.sum(chi_new)/numpy.sum(domain_omega == _env.NODE_ROBIN)) - V_obj) > 1e-5:
                if (numpy.sum(chi_new)/numpy.sum(domain_omega == _env.NODE_ROBIN)) > V_obj:
                    l_b = l_c
                    l_c = (l_a + l_b) / 2
                else:
                    l_a = l_c
                    l_c = (l_a + l_b) / 2
                print(l_a, l_b, l_c, "-----", "les trois lambdas")
                chi_new = numpy.maximum(0.0, numpy.minimum(1.0, chi_new_temp_fixed + l_c))

            
            #On résoud le problème de Helmholtz avec le nouveau chi
            alpha_rob_new = preprocessing.set2zero(Alpha * chi_new, domain_omega)
            u_new = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                               beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_new)
            
            #On calcule la nouvelle énergie avec le nouveau chi
            ene_new = your_compute_objective_function(domain_omega, u_new, spacestep, mu1, V_0)

            #On vérifie si on a bien une diminution de l'énergie sinon on refait tourner la boucle avec un pas plus petit
            if ene_new < ene_old:
                chi = chi_new
                u = u_new
                ene = ene_new
                alpha_rob_current = alpha_rob_new
                mu_inner = mu_inner*1.1
                break
            else:
                mu_inner = mu_inner / 2

        #On met à jour les variables pour la prochaine itération
        energy[k+1] = ene
        mu = mu_inner
        k += 1

    #On remplit le reste du tableau d'énergie si on a arrêté avant le nombre d'itérations maximal
    energy[k+1:] = ene
    print(mu, "-----", "le mu final")

    return chi, energy, u, grad


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

    integral_u_squared = numpy.sum(numpy.abs(u)**2) * (spacestep ** 2)
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
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0

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
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    #On choisit la fréquence d'optimisation
    omega_f = 3400
    wavenumber = omega_f / 340.0
    alpha_rob[:, :] = - wavenumber * 1j

    #On définit le chi initial
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    nbre_points_chi_egal_1 = numpy.sum(chi == 1.0)
    print('nbre_points_chi_egal_1 = ', nbre_points_chi_egal_1)
    # -- this is the function you have written during your project
    #On peut décommenter cette partie si on veut calculer les omegas et alphas
    '''
    omega_f = 3400
    omegas, alphas = compute_alpha.run_compute_alpha('AE 2.93')
    for i in range(len(omegas)):
        if omegas[i] > omega_f:
            Alpha = alphas[i]
            break

    numpy.save('omegas_2.93.npy', omegas)
    numpy.save('alphas_2.93.npy', alphas)
    '''
    #On charge les omegas et alphas déjà calculés pour gagner du temps
    omegas = numpy.load('omegas.npy')
    alphas = numpy.load('alphas.npy')
    
    #On définit l'Alpha et alpha_rob pour la fréquence choisie
    for i in range(len(omegas)):
        if omegas[i] > omega_f:
            Alpha = alphas[i]
            break
    alpha_rob = Alpha * chi

    #On définit les paramètres pour l'optimisation
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    print('V_obj = ', V_obj)
    
    mu = 2 # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    #On calcule la solution initiale sans optimisation
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    #On calcule la solution optimisée
    
    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                    Alpha, mu, chi, V_obj, mu1, V_0)
    print('energy=', energy)
    chin = chi.copy()
    un = u.copy()
    
    #On projette notre chi optimisé pour n'avoir que des 0 et des 1
    robin_mask = (domain_omega == _env.NODE_ROBIN)
    chi_robin = chi[robin_mask]
    indices_sorted = numpy.argsort(chi_robin)[::-1]
    chi_proj_robin = numpy.zeros_like(chi_robin)
    chi_proj_robin[indices_sorted[:nbre_points_chi_egal_1]] = 1

    chi_proj = numpy.zeros_like(chi)
    chi_proj[robin_mask] = chi_proj_robin

    chin_proj = chi_proj.copy()
    alpha_rob = Alpha * chin_proj
    un_proj = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    
    #On peut décommenter cette partie si on veut calculer l'énergie totale obtenue avec le chi optimisé projeté et on peut modifier alpha_rob pour calculer l'énergie avec le chi souhaité
    #On fait une boucle sur toutes les fréquences pour calculer l'énergie à chaque fréquence obtenue avec le chi optimisé projeté
    '''
    liste_energy = []
    i = 0
    for omega_test in omegas :
        Alpha = alphas[i]
        wavenumber = omega_test / 340.0
        alpha_rob = Alpha * chin_proj
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        
        liste_energy.append(numpy.sum(numpy.abs(u)**2) * (spacestep ** 2))
        i += 1
    
    numpy.save('liste_energy_chi_opti_proj.npy', liste_energy)
    '''

    #On charge les listes d'énergie déjà calculées pour gagner du temps
    liste_energy_chi_opti_proj = numpy.load('liste_energy_chi_opti_proj.npy')
    liste_energy_beta_1 = numpy.load('liste_energy_beta_1.npy')
    liste_energy_chi_opti = numpy.load('liste_energy_chi_opti.npy') 
    liste_energy_chi_init = numpy.load('liste_energy_chi_init.npy')
    
    #On affiche les résultats dans des fichiers .jpg
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un_proj, chin_proj)
    err = un_proj - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)
    
    #On affiche l'évolution de l'énergie en fonction de la fréquence pour les différents chi
    frequences = omegas / (2 * numpy.pi)
    plt.figure()
    plt.plot(frequences, liste_energy_chi_opti_proj, marker='^',markersize=2, label='Chi_opti projeté')
    plt.plot(frequences, liste_energy_beta_1, marker='.',markersize=2, label='Beta = 1')
    plt.plot(frequences, liste_energy_chi_opti, marker='x',markersize=2, label='Chi_opti')
    plt.plot(frequences, liste_energy_chi_init, marker='o',markersize=2, label='Chi_0')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Énergie (u.a.)')
    plt.title("Evolution de l'énergie en fonction de la fréquence")
    plt.legend()
    plt.show()
    print('End.')
