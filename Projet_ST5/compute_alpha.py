import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


def real_to_complex(z):
    return z[0] + 1j * z[1]


def complex_to_real(z):
    return np.array([np.real(z), np.imag(z)])


def compute_alpha(omega, material):
    """
    Calcule alpha(omega) pour un matériau donné.
    """
    #Propriétés de l'air
    gamma_p = 7.0 / 5.0
    rho_0 = 1.2
    c_0 = 340.0

    #Paramètres selon le matériau ou l’échantillon AE
    if material == 'mousse plastique':
        phi, sigma, alpha_h = 0.529, 151429.0, 1.37
    elif material == 'melamine':
        phi, sigma, alpha_h = 0.9921, 10.943e3, 1.009
    elif material == 'laine de verre':
        phi, sigma, alpha_h = 0.989, 11.9e3, 1.01
    elif material == 'fibre polyester':
        phi, sigma, alpha_h = 0.974, 19.7e3, 1.02
    elif material == 'materiau B':
        phi, sigma, alpha_h = 0.94, 9067, 1

    #Données ajustées des échantillons AE
    elif material == 'AE 2.93':
        phi, sigma, alpha_h = 0.45, 4720.11, 2.26
    elif material == 'AE 3.71':
        phi, sigma, alpha_h = 0.45, 5231.79, 2.14
    elif material == 'AE 5.18':
        phi, sigma, alpha_h = 0.50, 4990.16, 2.14
    else:
        raise ValueError(f"Matériau inconnu : {material}")


    # Paramètres géométriques et physiques
    L = 0.01
    resolution = 12
    mu_0 = 1.0
    ksi_0 = 1.0 / (c_0 ** 2)
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / (c_0 ** 2)
    a = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)
    A = B = 1.0

    # Définition des fonctions intermédiaires
    @Memoize
    def lambda_0(k, omega):
        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return np.sqrt(k ** 2 - (omega ** 2) * ksi_0 / mu_0)
        else:
            return 1j * np.sqrt((omega ** 2) * ksi_0 / mu_0 - k ** 2)

    @Memoize
    def lambda_1(k, omega):
        temp1 = (omega ** 2) * ksi_1 / mu_1
        temp2 = np.sqrt((k ** 2 - temp1) ** 2 + (a * omega / mu_1) ** 2)
        real = (1.0 / np.sqrt(2.0)) * np.sqrt(k ** 2 - temp1 + temp2)
        im = (-1.0 / np.sqrt(2.0)) * np.sqrt(temp1 - k ** 2 + temp2)
        return complex(real, im)

    @Memoize
    def f(x, k):
        return ((lambda_0(k, omega) * mu_0 - x) * np.exp(-lambda_0(k, omega) * L)
                + (lambda_0(k, omega) * mu_0 + x) * np.exp(lambda_0(k, omega) * L))

    def g_k(k):
        return 1.0 if k == 0 else 0.0

    @Memoize
    def chi(k, alpha):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1) / f(lambda_1(k, omega) * mu_1, k)
                          - (lambda_0(k, omega) * mu_0 - alpha) / f(alpha, k)))

    @Memoize
    def eta(k, alpha):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1) / f(lambda_1(k, omega) * mu_1, k)
                          - (lambda_0(k, omega) * mu_0 + alpha) / f(alpha, k)))

    @Memoize
    def e_k(k, alpha):
        expm = np.exp(-2.0 * lambda_0(k, omega) * L)
        expp = np.exp(+2.0 * lambda_0(k, omega) * L)
        return (A + B * (np.abs(k) ** 2)) * (np.abs(chi(k, alpha)) ** 2 + np.abs(eta(k, alpha)) ** 2)

    def sum_e_k(alpha):
        s = 0.0
        for n in range(-resolution, resolution + 1):
            k = n * np.pi / L
            s += e_k(k, alpha)
        return np.real(s)

    # Recherche du minimum
    alpha_0 = 40.0 - 40.0j  # Correction : alpha_0 est un complexe
    temp = real_to_complex(
        minimize(
            lambda z: sum_e_k(real_to_complex(z)),
            complex_to_real(alpha_0),  # x0 = [Re(alpha_0), Im(alpha_0)]
            tol=1e-3
        ).x
    )
    print(temp, "------", "je suis temp")
    return temp


def run_compute_alpha(material):
    numb_omega = 1000
    omegas = 2.0 * np.pi * np.linspace(10,1200, numb_omega)
    alphas = [compute_alpha(omega, material) for omega in omegas]
    return omegas, np.array(alphas)


if __name__ == '__main__':
    materials = [
    'mousse plastique', 'melamine', 'laine de verre', 'fibre polyester', 'materiau B',
    'AE 2.93', 'AE 3.71', 'AE 5.18'
]
    plt.figure(figsize=(8, 6))

    for mat in materials:
        print(f"Calcul de alpha pour {mat}...")
        omegas, alphas = run_compute_alpha(mat)
        ratio = np.abs(np.imag(alphas)) / np.abs(np.real(alphas))
        plt.plot(np.real(omegas), ratio, label=mat)

    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$|\Im(\alpha)| / |\Re(\alpha)|$')
    plt.title("Rapport imaginaire/réel de α(ω) pour différents matériaux")
    plt.legend()
    plt.grid(True)
    plt.savefig('fig_alpha_all_materials.jpg', dpi=300)
    plt.close()
    
    


    