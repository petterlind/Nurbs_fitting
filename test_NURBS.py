import unittest
import numpy as np
from Nurbs import Nurbs
import pdb
import Fitting as F
from geomdl import NURBS
from geomdl import exchange
from geomdl import convert
from geomdl import BSpline
from geomdl.visualization import VisPlotly
from geomdl.visualization import VisMPL as vis
import matplotlib.pyplot as plt
import copy

'''
Test functions
'''


def Branin_1d(x):
    
    # y = x
    y = 0.5
    
    X1 = 15 * x - 5
    X2 = 15 * y
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    d = 6
    e = 10
    ff = 1 / (8 * np.pi)
    return (a * (X2 - b * X1**2 + c * X1 - d)**2 + e * (1 - ff) * np.cos(X1) + e) + 5 * x


def Circle_1d(x):
    return np.sqrt(1 - x**2)
    

class Test_Nurbs(unittest.TestCase):
    '''
    Some test data
    '''
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        
        self.x_samp = np.linspace(0, 1, num=20)
        self.y_samp = Circle_1d(self.x_samp)
        self.points = np.column_stack((self.x_samp, self.y_samp))  # Test points
        #  Plot test function
        fig = plt.figure(num=1)
        plt.plot(self.x_samp, self.y_samp, 'o')  # Nicer plot
        plt.plot(np.linspace(0, 1, num=1000), Circle_1d(np.linspace(0, 1, num=1000)))
        
        self.N_cur, self.B_cur = self.set_curves()
    
    def set_curves(self):
        degree = 2
        num_cp = 2
        CP, U = F.curve_fit2D(self.points, np.ones((len(self.points), 1)), None, None, None, num_cp, degree)
        
        B_cur = BSpline.Curve()
        B_cur.degree = degree
        B_cur.ctrlpts = CP
        B_cur.knotvector = U
        B_cur.evaluate()
        B_cur.delta = 0.01
        
        N_cur = convert.bspline_to_nurbs(B_cur)
        # Set evaluation delta
        N_cur.delta = 0.01
        N_cur.evaluate()
        
        return N_cur, B_cur
    
    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for i in range(0, len(list1)):
            for a, b in zip(list1[i], list2[i]):
                self.assertAlmostEqual(a, b, tol)
    
    # def test_FindSpan(self):
    #     self.assertTrue(Nurbs.FindSpan(6, 3, 0, self.U_cube) == 3)
    #     self.assertTrue(Nurbs.FindSpan(6, 3, 8, self.U_cube) == 6)
    #     self.assertTrue(Nurbs.FindSpan(6, 3, 3, self.U_cube) == 4)
    #     self.assertTrue(Nurbs.FindSpan(6, 3, 5.5, self.U_cube) == 5)
    #     self.assertTrue(Nurbs.FindSpan(6, 3, 7.9, self.U_cube) == 6)
        
    def test_assign_ub(self):
        ub = F.assign_ub(self.points)
        # Just check if the output is correct (and that the functions actually works).
        self.assertTrue(isinstance(ub, list))
        self.assertTrue(isinstance(F.set_knots(4, 2, len(self.points), ub), np.ndarray))

    def test_fitting(self):
        CP, U = F.curve_fit2D(self.points, np.ones((len(self.points), 1)), None, None, None, 2, 2)
        self.assertTrue(isinstance(CP, tuple))
        self.assertTrue(isinstance(U, list))
    
    def test_gn_jacobi(self):
        J = F.gn_jacobi(self.N_cur, F.assign_ub(self.points), self.points, 1.0)
    
        r = len(self.points)
        n = len(self.N_cur.ctrlptsw)
        
        (rows, cols) = np.shape(np.array(J))
        self.assertTrue(rows == (2 * r + n)) 
        self.assertTrue(cols == (r + 3 * n))
        self.assertTrue(isinstance(J, list))
        
        # Write some test to test for sanity check of the contents?!
    
    def test_gn_f(self):
        alpha = F.gn_f(self.N_cur, F.assign_ub(self.points), self.points, np.nan)
        fun_val = F.gn_f(self.N_cur, F.assign_ub(self.points), self.points, alpha)
        
        r = len(self.points)
        n = len(self.N_cur.ctrlptsw)
        (rows,) = np.shape(np.array(fun_val))
        self.assertTrue(rows == (2 * r + n))
        self.assertTrue(isinstance(alpha, float))
        self.assertTrue(isinstance(fun_val, list))
    
    def test_NURBS_BSPLINE_EQUAL(self):
        # Save results
        B_res = self.B_cur.evalpts
        N_res = self.N_cur.evalpts
        
        # Plot the control point polygon and the evaluated curve
        vis_comp = vis.VisCurve2D()
        self.N_cur.vis = vis_comp
        # self.N_cur.render()
        self.assertListAlmostEqual(B_res, N_res, 3)
        
    def test_update_curve(self):
        
        
        # Initialize x
        CP_list = list()
        for CP in self.N_cur.ctrlptsw:
            for each in CP[:-1]:
                CP_list.append(each)
        W = []
        for elem in self.N_cur.ctrlptsw:
            W.append(elem[-1])
        ub = F.assign_ub(self.points)  # Should be same as for the B-spline!
        
        x = ub + CP_list + W
        r = len(self.points)
        
        New_c = F.update_curve(self.N_cur, x, r)
        
        N_res = self.N_cur.evalpts
        N_new_res = New_c.evalpts
        
        W_fix = copy.deepcopy(W)  # Change the weights
        W_fix[0] = W_fix[0] + 2
        CP_list_fix = CP_list  # Change the controlpoints
        CP_list_fix[0] = CP_list_fix[3] * 3
        
        x_w = ub + CP_list + np.log(W_fix).tolist()
        x_p = ub + CP_list_fix + np.log(W).tolist()
        
        New_w = F.update_curve(self.N_cur, x_w, r)
        New_p = F.update_curve(self.N_cur, x_p, r)
        
        self.assertListAlmostEqual(N_res, N_new_res, 3)  # nothing should be changed
        self.assertFalse(self.N_cur.ctrlptsw[0][-1] == New_w.ctrlptsw[0][-1])  # Changed
        self.assertTrue(self.N_cur.ctrlptsw[1][-1] == New_w.ctrlptsw[1][-1])  # Unchanged
        self.assertTrue(self.N_cur.ctrlptsw[2][-1] == New_w.ctrlptsw[2][-1])
        self.assertFalse(self.N_cur.ctrlptsw[0][0] == New_p.ctrlptsw[0][0])
        self.assertTrue(self.N_cur.ctrlptsw[0][-1] == New_p.ctrlptsw[0][-1])
        
    def test_gauss_newton2D(self):
        ''' Performs gauss newton search of optimum weights
        '''
    
        # Fit the NURBS weights and control points
        N_fit = F.gauss_newton2D(self.N_cur, self.points)
    
        # Plot (debug)
        # vis_fit = VisPlotly.VisCurve2D()
        # N_fit.vis = vis_fit
        # N_fit.render()
        
        Do harder geometry!
        
