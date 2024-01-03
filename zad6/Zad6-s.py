import numpy as np
import matplotlib.pyplot as plt

# Funkcja ELU
def elu(x, alpha=1.0):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)

# Gradient funkcji ELU
def elu_gradient(x, alpha=1.0):
    return np.where(x < 0, alpha * np.exp(x), 1)

# Zakres danych x
x = np.linspace(-7, 7, 200)

# Obliczamy wartości funkcji ELU i jej gradientu
elu_values = elu(x)
elu_gradient_values = elu_gradient(x)

# Tworzymy wykres
plt.figure(figsize=(8, 6))
plt.plot(x, elu_values, label='ELU')
plt.plot(x, elu_gradient_values, label='Gradient ELU')
plt.legend()
plt.xlabel('x')
plt.ylabel('Wartosc')
plt.title('Funkcja ELU i jej Gradient')
plt.grid(True)
plt.show()
