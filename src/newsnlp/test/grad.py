from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt

g = lambda x: np.tanh(x)
dgdw = grad(g)

w = np.linspace(-5, 5, 200)
g_vals = [g(x) for x in w]
dg_vals = [dgdw(x) for x in w]

fig = plt.figure()
plt.plot(w, g_vals)
plt.plot(w, dg_vals)
plt.legend(['tanh', 'derivative'])
plt.show()

