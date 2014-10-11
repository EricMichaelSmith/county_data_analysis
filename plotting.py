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

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp



def define_boolean_color(booleanSR, color_t_t):
    """
    booleanSR is the boolean-valued series to define the color from. color_t_t
    should be two tuples of length 3: one if the boolean value is
    true and one if it is false.
    """
    
    colorColumnT = ('red', 'green', 'blue')
    colorDF = pd.DataFrame(np.ndarray((len(booleanSR.index), 3)), 
                           index=booleanSR.index,
                           columns=colorColumnT)

    # Set columns one by one
    for lColumn, columnS in enumerate(colorColumnT):
        colorDF.loc[booleanSR, columnS] = color_t_t[0][lColumn]
        colorDF.loc[~booleanSR, columnS] = color_t_t[1][lColumn]
    return (colorDF, -1)
    # The second entry of the returned tuple specifies that there is no maximum
    # magnitude, like we'd have if this were define_gradient_color
    
    
    
def define_gradient_color(valueSR, color_t_t):
    """ 
    valueSR is the series to define the color from. color_t_t should be three
    tuples of length 3: one if the value is maximally negative, one if it is
    zero, and one if it is maximally positive. Intermediate values will be
    interpolated.
    """
    
    # Find the maximum-magnitude value in the column of interest: this will be
    # represented with the brightest color.
    max_magnitude = max([abs(i) for i in valueSR])

    # For each index in inputDF, interpolate between the values of color_t_t to
    # find the approprate color of the index
    gradientDF = pd.DataFrame(np.ndarray((len(valueSR.index), 3)),
                              index=valueSR.index,
                              columns=['red', 'green', 'blue'])
    for index in valueSR.index:
        gradientDF.loc[index] = \
        interpolate_gradient_color(color_t_t,
                                   valueSR[index],
                                   max_magnitude)
    return (gradientDF, max_magnitude)


                                                      
def interpolate_gradient_color(color_t_t, value, max_magnitude):
    """
    color_t_t is three tuples of length 3: one if the value is maximally negative,
    one if it is zero, and one if it is maximally positive. max_magnitude sets the
    intensity of the interpolated color. The function returns a tuple containing
    the interpolated color for the input value.
    """
    
    normalizedMagnitude = abs(value)/max_magnitude
    
    # The higher the magnitude, the closer the color to farColorT; the lower
    # the magnitude, the closter the color to nearColorT
    nearColorT = color_t_t[1]
    if value < 0:
        farColorT = color_t_t[0]
    else:
        farColorT = color_t_t[2]
    interpolatedColorA = (normalizedMagnitude * np.array(farColorT) +
                          (1-normalizedMagnitude) * np.array(nearColorT))
    return tuple(interpolatedColorA)
    
    

def make_colorbar(ax, max_magnitude, color_t_t, label_s):
    """
    Creates a colorbar with the given axis handle ax; the colors are defined
    according to color_t_t and the values are mapped from -max_magnitude to
    +max_magnitude. The colorbar is labeled with label_s.
    """

    # Create the colormap for the colorbar    
    colormap = make_colormap(color_t_t)    
    
    # Create the colorbar
    norm = mpl.colors.Normalize(vmin=-max_magnitude, vmax=max_magnitude)
    colorBarHandle = mpl.colorbar.ColorbarBase(ax, cmap=colormap,
                                               norm=norm,
                                               orientation='horizontal')
    colorBarHandle.set_label(label_s)
    
    
    
def make_colormap(color_t_t):
    """ Given colors defined in color_t_t, creates a LinearSegmentedColormap object. """
    
    # Create the colormap
    color_l = ['red', 'green', 'blue']
    color_map_entry = lambda color_t_t, i_color: \
        ((0.0, color_t_t[0][i_color], color_t_t[0][i_color]),
         (0.5, color_t_t[1][i_color], color_t_t[1][i_color]),
         (1.0, color_t_t[2][i_color], color_t_t[2][i_color]))
    color_d = {colorS: color_map_entry(color_t_t, i_color) for i_color, colorS
              in enumerate(color_l)}
    colormap = LinearSegmentedColormap('ShapePlotColorMap', color_d)
    
    return colormap



def make_scatter_plot(ax, x_l_t, y_l_t, color_t_t, plot_axes_at_zero_b=False,
                                               plot_regression_b=False):
    """
    Creates a scatter plot. x_l_t and y_l_t are length-n tuples containing the
    n lists to be plotted; colors of plot points are given by the length-n
    tuple color_t_t.
    """
    
    # Plot all data
    for l_series in xrange(len(x_l_t)):
        ax.scatter(x_l_t[l_series], y_l_t[l_series],
                    c=color_t_t[l_series],
                    edgecolors='none')
                                        
    # Plot x=0 and y=0 lines
    if plot_axes_at_zero_b:
        plot_axes_at_zero(ax)
    
    # Plot regression line (one set of points only)
    if plot_regression_b and len(x_l_t) == 1:
        plot_regression(ax, x_l_t[0], y_l_t[0])
    


def make_shape_plot(fig, valueSR, shapeIndexL, shapeL, colorTypeS, color_t_t,
                    colorBarS=None):
    """ Creates a shape plot given figure handle fig. valueSR is the Series containing the data to be plotted; shapeIndexL indexes the shapes to plot by FIPS code; shapeL contains the shapes to plot; colorTypeS defines whether the plot will be shaded according to a binary or a gradient; color_t_t defines the colors to shade with. colorBarS labels the colorbar.
    """
    
    # Set shape colors
    ax = fig.add_subplot(1, 1, 1)
    shapeBoundsAllShapesL = [float('inf'), float('inf'), float('-inf'), float('-inf')]    

    colorTypesD = {'boolean': lambda: define_boolean_color(valueSR, color_t_t),
                   'gradient': lambda: define_gradient_color(valueSR, color_t_t)}
    colorT = colorTypesD[colorTypeS]()
    colorDF = colorT[0]
    max_magnitude = colorT[1]
            
    # Add shapes to plot
    for lFIPS in valueSR.index:
        thisCountiesColorT = tuple(colorDF.loc[lFIPS])
        
        iShapeL = [i for i,j in enumerate(shapeIndexL) if j==int(lFIPS)]
        for iShape in iShapeL:       
            shapeBoundsThisShapeL = shapeL[iShape].bbox
            shapeBoundsAllShapesL[0] = \
                min(shapeBoundsThisShapeL[0], shapeBoundsAllShapesL[0])
            shapeBoundsAllShapesL[1] = \
                min(shapeBoundsThisShapeL[1], shapeBoundsAllShapesL[1])
            shapeBoundsAllShapesL[2] = \
                max(shapeBoundsThisShapeL[2], shapeBoundsAllShapesL[2])
            shapeBoundsAllShapesL[3] = \
                max(shapeBoundsThisShapeL[3], shapeBoundsAllShapesL[3])
            
            thisShapesPatches = []
            pointsA = np.array(shapeL[iShape].points)
            shapeFileParts = shapeL[iShape].parts
            allPartsL = list(shapeFileParts) + [pointsA.shape[0]]
            for lPart in xrange(len(shapeFileParts)):
                thisShapesPatches.append(mpl.patches.Polygon(
                    pointsA[allPartsL[lPart]:allPartsL[lPart+1]]))
            ax.add_collection(mpl.collections.PatchCollection(thisShapesPatches,
                                              color=thisCountiesColorT))
    ax.set_xlim(-127, -65)
    ax.set_ylim(20, 50)
    ax.set_axis_off()
    
    # Add colorbar
    if colorBarS and max_magnitude != -1:
        color_ax = fig.add_axes([0.25, 0.10, 0.50, 0.05])
        make_colorbar(color_ax, max_magnitude, color_t_t, colorBarS)
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

	
	
def plot_line_score_of_features(ax, feature_s_l, score_value_l,
                                extremum_func=None,
                                is_backward_selection_b=False,
                                ylabel_s=None):
    """ Given a list of features feature_s_l and a corresponding list of scores score_value_l, creates a line plot with axes ax. If extremum_func is either max() or min(), the maximum or minimum value, respectively, will be circled.  """
	
    # Indicate addition or subtraction of features
    if not is_backward_selection_b:
        feature_s_l[1:] = ['+ ' + feature_s for feature_s in feature_s_l[1:]]
    else:
        feature_s_l[:-1] = ['- ' + feature_s for feature_s in feature_s_l[:-1]]
        feature_s_l[-1] = '(last remaining feature: ' + feature_s_l[-1] + ')'
    
    for l_odd in range(len(score_value_l))[1::2]:
        ax.axvspan(l_odd-0.5, l_odd+0.5, alpha=0.15,
                   edgecolor='none', facecolor=[0, 0, 0])
    ax.plot(range(len(score_value_l)), score_value_l)
    ax.set_xlim(-0.5, len(score_value_l)-0.5)
    ax.set_xticks(range(len(score_value_l)))
    ax.set_xticklabels(feature_s_l, rotation=90)
    ax.set_ylabel(ylabel_s)
    
    # Circle extreme value
    if extremum_func:
        extremum_value = \
            extremum_func(value for value in score_value_l if value is not None)
        i_extremum = score_value_l.index(extremum_value)
        ax.scatter(i_extremum, extremum_value)
    
    
    
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