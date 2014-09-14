# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:15:01 2014

@author: Eric

Houses all plot functions for county_data_analysis

Suffixes at the end of variable names:
a: numpy array
b: boolean
d: dictionary
df: pandas DataFrame
l: list
s: string
t: tuple
Underscores indicate chaining: for instance, "foo_t_t" is a tuple of tuples
"""

import matplotlib.pyplot as plt
import scipy as sp



def make_scatter_plot(x_l_t, y_l_t, color_t_t, plot_axes_at_zero_b=False,
                                               plot_regression_b=False):
    """
    Creates a scatter plot. x_l_t and y_l_t are length-n tuples containing the
    n lists to be plotted; colors of plot points are given by the length-n
    tuple color_t_t.
    """
    
    # Plot all data
    scatter_fig = plt.figure()
    ax = scatter_fig.add_subplot(1, 1, 1)
    for l_series in xrange(len(x_l_t)):
        plt.scatter(x_l_t[l_series], y_l_t[l_series],
                    c=color_t_t[l_series],
                    edgecolors='none')
                                        
    # Plot x=0 and y=0 lines
    if plot_axes_at_zero_b:
        plot_axes_at_zero(ax)
    
    # Plot regression line (one set of points only)
    if plot_regression_b and len(x_l_t) == 1:
        plot_regression(ax, x_l_t[0], y_l_t[0])
    
    return ax
    
    
    
def plot_axes_at_zero(ax):
    """
    Plots x=0 and y=0 lines on the current plot. The stretching of the lines and
    resetting of the axis limits is kind of a hack.
    """
    
    axis_limits_t = ax.axis()
    plt.plot([2*axis_limits_t[0], 2*axis_limits_t[1]], [0, 0], 'k')
    plt.plot([0, 0], [2*axis_limits_t[2], 2*axis_limits_t[3]], 'k')
    ax.set_xlim(axis_limits_t[0], axis_limits_t[1])
    ax.set_ylim(axis_limits_t[2], axis_limits_t[3])
    return ax
    
    
    
def plot_regression(ax, x_l, y_l):
    """
    Plots a regression line on the current plot. The stretching of the
    regression line and resetting of the axis limits is kind of a hack.
    """    
    
    # Find correct axis limits
    axis_limits_t = ax.axis()
    
    # Calculate and plot regression
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x_l, y_l)
    plt.plot([2*axis_limits_t[0], 2*axis_limits_t[1]],
             [slope*2*axis_limits_t[0]+intercept,
              slope*2*axis_limits_t[1]+intercept],
             'r')
    
    # Reset axis limits
    ax.set_xlim(axis_limits_t[0], axis_limits_t[1])
    ax.set_ylim(axis_limits_t[2], axis_limits_t[3])
    
    return ax