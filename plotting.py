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



def define_boolean_color(booleanSR, colorT_T):
    """
    booleanSR is the boolean-valued series to define the color from. colorT_T
    should be two tuples of length 3: one if the boolean value is
    true and one if it is false.
    """
    
    colorColumnT = ('red', 'green', 'blue')
    colorDF = pd.DataFrame(np.ndarray((len(booleanSR.index), 3)), 
                           index=booleanSR.index,
                           columns=colorColumnT)

    # Set columns one by one
    for lColumn, columnS in enumerate(colorColumnT):
        colorDF.loc[booleanSR, columnS] = colorT_T[0][lColumn]
        colorDF.loc[~booleanSR, columnS] = colorT_T[1][lColumn]
    return (colorDF, -1)
    # The second entry of the returned tuple specifies that there is no maximum
    # magnitude, like we'd have if this were define_gradient_color
    
    
    
def define_gradient_color(valueSR, colorT_T):
    """ 
    valueSR is the series to define the color from. colorT_T should be three
    tuples of length 3: one if the value is maximally negative, one if it is
    zero, and one if it is maximally positive. Intermediate values will be
    interpolated.
    """
    
    # Find the maximum-magnitude value in the column of interest: this will be
    # represented with the brightest color.
    maxMagnitude = max([abs(i) for i in valueSR])

    # For each index in inputDF, interpolate between the values of colorT_T to
    # find the approprate color of the index
    gradientDF = pd.DataFrame(np.ndarray((len(valueSR.index), 3)),
                              index=valueSR.index,
                              columns=['red', 'green', 'blue'])
    for index in valueSR.index:
        gradientDF.loc[index] = \
        interpolate_gradient_color(colorT_T,
                                   valueSR[index],
                                   maxMagnitude)
    return (gradientDF, maxMagnitude)


                                                      
def interpolate_gradient_color(colorT_T, value, maxMagnitude):
    """
    colorT_T is three tuples of length 3: one if the value is maximally negative,
    one if it is zero, and one if it is maximally positive. maxMagnitude sets the
    intensity of the interpolated color. The function returns a tuple containing
    the interpolated color for the input value.
    """
    
    normalizedMagnitude = abs(value)/maxMagnitude
    
    # The higher the magnitude, the closer the color to farColorT; the lower
    # the magnitude, the closter the color to nearColorT
    nearColorT = colorT_T[1]
    if value < 0:
        farColorT = colorT_T[0]
    else:
        farColorT = colorT_T[2]
    interpolatedColorA = (normalizedMagnitude * np.array(farColorT) +
                          (1-normalizedMagnitude) * np.array(nearColorT))
    return tuple(interpolatedColorA)
    
    

def make_colorbar(fig, maxMagnitude, colorT_T, labelS):
    """
    Creates a colorbar with the given figure handle fig; the colors are defined
    according to colorT_T and the values are mapped from -maxMagnitude to
    +maxMagnitude. The colorbar is labeled with labelS.
    """

    # Create the colormap for the colorbar
    colorL = ['red', 'green', 'blue']
    colorMapEntry = lambda colorT_T, iColor: \
        ((0.0, colorT_T[0][iColor], colorT_T[0][iColor]),
         (0.5, colorT_T[1][iColor], colorT_T[1][iColor]),
         (1.0, colorT_T[2][iColor], colorT_T[2][iColor]))
    colorD = {colorS: colorMapEntry(colorT_T, iColor) for iColor, colorS
              in enumerate(colorL)}
    colorBarMap = LinearSegmentedColormap('ShapePlotColorMap', colorD)
    
    # Create the colormap
    colorAx = fig.add_axes([0.25, 0.10, 0.50, 0.05])
    norm = mpl.colors.Normalize(vmin=-maxMagnitude, vmax=maxMagnitude)
    colorBarHandle = mpl.colorbar.ColorbarBase(colorAx, cmap=colorBarMap,
                                               norm=norm,
                                               orientation='horizontal')
    colorBarHandle.set_label(labelS)



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



def make_shape_plot(valueSR, shapeIndexL, shapeL, colorTypeS, colorT_T,
                    colorBarS=None):
    """
    Creates a shape plot. valueSR is the Series containing the data to be
    plotted; shapeIndexL indexes the shapes to plot by FIPS code; shapeL contains
    the shapes to plot; colorTypeS defines whether the plot will be shaded
    according to a binary or a gradient; colorT_T defines the colors to shade
    with. colorBarS labels the colorbar.
    """
    
    # Set shape colors
    shapeFig = plt.figure(figsize=(11,6))
    ax = shapeFig.add_subplot(1, 1, 1)
    shapeBoundsAllShapesL = [float('inf'), float('inf'), float('-inf'), float('-inf')]    

    colorTypesD = {'boolean': lambda: define_boolean_color(valueSR, colorT_T),
                   'gradient': lambda: define_gradient_color(valueSR, colorT_T)}
    colorT = colorTypesD[colorTypeS]()
    colorDF = colorT[0]
    maxMagnitude = colorT[1]
        
    # Add shapes to plot
    for lFIPS in valueSR.index:
        thisCountiesColorT = tuple(colorDF.loc[lFIPS])        
        
        iShapeL = [i for i,j in enumerate(shapeIndexL) if j==lFIPS]
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
    if maxMagnitude != -1:
        make_colorbar(shapeFig, maxMagnitude, colorT_T, colorBarS)
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