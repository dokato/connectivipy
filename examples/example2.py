import numpy as np
import matplotlib.pyplot as plt
import connectivipy as cp
from connectivipy import mvar_gen

"""
In this example we don't use Data class
"""

fs = 256.
acf = np.zeros((3, 3, 3))
# matrix shape meaning
# (p,k,k) k - number of channels,
# p - order of mvar parameters

acf[0, 0, 0] = 0.3
acf[0, 1, 0] = 0.6
acf[1, 0, 0] = 0.1
acf[1, 1, 1] = 0.2
acf[1, 2, 0] = 0.6
acf[2, 2, 2] = 0.2
acf[2, 1, 0] = 0.4

# generate 3-channel signal from matrix above
y = mvar_gen(acf, int(10e4))

# assign static class cp.Mvar to variable mv
mv = cp.Mvar

# find best model order using Vieira-Morf algorithm
best, crit = mv.order_akaike(y, 15, 'vm')
plt.plot(1+np.arange(len(crit)), crit, 'g')
plt.show()
print best
# here we know that this is 3 but in real-life cases
# we are always uncertain about it

# now let's fit parameters to the signal
av, vf = mv.fit(y, best, 'vm')

# and check whether values are correct +/- 0.01
print np.allclose(acf, av, 0.01, 0.01)

# now we can calculate Directed Transfer Function from the data
dtf = cp.conn.DTF()
dtfval = dtf.calculate(av, vf, 128)
# all possible methods are visible in that dictionary:
print cp.conn.conn_estim_dc.keys()

cp.plot_conn(dtfval, 'DTF values', fs)
