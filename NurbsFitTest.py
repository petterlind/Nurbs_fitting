#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from geomdl import BSpline
from geomdl import utilities
from geomdl import operations
from geomdl import evaluators
from geomdl import exchange
import Fitting
from geomdl.visualization import VisMPL as vis
from matplotlib import cm
import matplotlib.pyplot as plt
import pdb
import test_NURBS
            

'''
Step 1)
Create a curve to approximate, this time the curve is a special case of Branin
'''

x_samp = np.linspace(0, 1, num=20)
y_samp = test_NURBS.Branin_1d(x_samp)

Q = np.column_stack((x_samp, y_samp))  # Test points

#  Plot test function
fig = plt.figure(num=1)
plt.plot(x_samp, y_samp, 'o')  # Nicer plot
plt.plot(np.linspace(0, 1, num=1000), Branin_1d(np.linspace(0, 1, num=1000)))
# plt.show()

'''
Step 2)
Create a B-spline and fit it to the data
'''

degree = 5
n = 5

# Adapt a B-spline to the curve.
CP, U = Fitting.curve_fit2D(Q, np.ones((len(Q), 1)), None, None, None, n, degree)
# Transform to Tupled tuples


# Create a BSpline (NURBS) curve instance
test_c = BSpline.Curve()

# Set up the curve
test_c.degree = degree
test_c.ctrlpts = CP
# test_c.ctrlpts = exchange.import_txt("ex_curve02.cpt")

test_c.knotvector = U
# Auto-generate knot vector
# test_c.knotvector = utilities.generate_knot_vector(test_c.degree, len(test_c.ctrlpts))
# Set evaluation delta
test_c.delta = 0.01

# Evaluate curve in same points as the
test_c.evaluate()
# Plot the control point polygon and the evaluated curve
vis_comp = vis.VisCurve2D()
test_c.vis = vis_comp
# test_c.render()


# Evaluate derivatives at u = 0.6
ders1 = test_c.derivatives(0.6, 4)

# plot the first derivative, normed, at correct location.
fig = plt.figure(num=1)
plt.plot([ders1[0][0], ders1[0][0] + ders1[1][0]], [ders1[0][1], ders1[0][1] + ders1[1][1]], 'k')  # First derivative
plt.plot([ders1[0][0], ders1[0][0] + ders1[2][0]], [ders1[0][1], ders1[0][1] + ders1[2][1]], 'r')  # Second derivative
# plt.show()

pdb.set_trace()
# Adapt nurb from B-spline curve
# test_c
# test_c_nurb = Fitting.gauss_newton2D(, tol=1e-3, mintol=1e-3)


# good to have something here to set a breakpoint to.
pass
