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



def define_boolean_color(boolean_sr, color_t_t):
    """
    boolean_sr is the boolean-valued series to define the color from. color_t_t
    should be two tuples of length 3: one if the boolean value is
    true and one if it is false.
    """
    
    color_column_t = ('red', 'green', 'blue')
    color_df = pd.DataFrame(np.ndarray((len(boolean_sr.index), 3)), 
                           index=boolean_sr.index,
                           columns=color_column_t)

    # Set columns one by one
    for l_column, column_s in enumerate(color_column_t):
        color_df.loc[boolean_sr, column_s] = color_t_t[0][l_column]
        color_df.loc[~boolean_sr, column_s] = color_t_t[1][l_column]
    return (color_df, None)
    # The second entry of the returned tuple specifies that there is no maximum or minimum magnitude, like we'd have if this were define_gradient_color
    
    
    
def define_balanced_gradient_color(value_sr, color_t_t):
    """ 
    This will return a gradient spanning from minus the highest-magnitude value to plus the highest-magnitude value. value_sr is the series to define the color from. color_t_t should be three tuples of length 3: one if the value is maximally negative, one if it is zero, and one if it is maximally positive. Intermediate values will be interpolated.
    """
    
    # Find the maximum-magnitude value in the column of interest: this will be
    # represented with the brightest color.
    max_magnitude = max([abs(i) for i in value_sr])

    # For each value, interpolate between the values of color_t_t to find the approprate color
    gradient_df = pd.DataFrame(np.ndarray((len(value_sr.index), 3)),
                              index=value_sr.index,
                              columns=['red', 'green', 'blue'])
    for index in value_sr.index:
        gradient_df.loc[index] = \
        interpolate_balanced_gradient_color(color_t_t,
                                            value_sr[index],
                                            max_magnitude)
    return (gradient_df, (-max_magnitude, 0, max_magnitude))



def define_unbalanced_gradient_color(value_sr, color_t_t, color_value_t):
    """ 
    This will return a gradient spanning from lowest value to the highest value. value_sr is the series to define the color from. color_t_t should be three tuples of length 3: one for the lowest value, one for the midrange value, and one for the highest value (unless this is overridden by manual color value specification in color_value_t). Intermediate values will be interpolated.
    """
    
    # Find the max, min, and mean values
    if color_value_t:
        min_value, mid_value, max_value = color_value_t
    else:
        min_value = np.min(value_sr)
        max_value = np.max(value_sr)
        mid_value = float(min_value + max_value)/2.0

    # For each value, interpolate between the values of color_t_t to find the approprate color
    gradient_df = pd.DataFrame(np.ndarray((len(value_sr.index), 3)),
                              index=value_sr.index,
                              columns=['red', 'green', 'blue'])
    for index in value_sr.index:
        gradient_df.loc[index] = \
        interpolate_unbalanced_gradient_color(color_t_t,
                                              value_sr[index],
                                              min_value,
                                              mid_value,
                                              max_value)
    return (gradient_df, (min_value, mid_value, max_value))
    

                                                      
def interpolate_balanced_gradient_color(color_t_t, value, max_magnitude):
    """
    color_t_t is three tuples of length 3: one if the value is maximally negative, one if it is zero, and one if it is maximally positive. max_magnitude sets the intensity of the interpolated color. The function returns a tuple containing the interpolated color for the input value.
    """
    
    normalized_magnitude = abs(value)/max_magnitude
    
    # The higher the magnitude, the closer the color to far_color_t; the lower
    # the magnitude, the closter the color to near_color_t
    near_color_t = color_t_t[1]
    if value < 0:
        far_color_t = color_t_t[0]
    else:
        far_color_t = color_t_t[2]
    interpolated_color_a = (normalized_magnitude * np.array(far_color_t) +
                          (1-normalized_magnitude) * np.array(near_color_t))
    return tuple(interpolated_color_a)



def interpolate_unbalanced_gradient_color(color_t_t, value, min_value,
                                          mid_value, max_value):
    """
    color_t_t is three tuples of length 3: one for min_value, one for mid_value, and one for max_value. The function returns a tuple containing the interpolated color for the input value.
    """
    
    if value < mid_value:
        low_value = min_value
        low_color_t = color_t_t[0]
        high_value = mid_value
        high_color_t = color_t_t[1]
    else:
        low_value = mid_value
        low_color_t = color_t_t[1]
        high_value = max_value
        high_color_t = color_t_t[2]
    interval = high_value - low_value
    interpolated_color_a = (value - low_value)/interval * np.array(high_color_t) + \
        (high_value - value)/interval * np.array(low_color_t)
        
    return tuple(interpolated_color_a)
    
    

def make_colorbar(ax, color_t_t, color_value_t, label_s):
    """ Creates a colorbar with the given axis handle ax; the colors are defined according to color_t_t and the values are mapped according to color_value_t. color_t_t and color_value_t must currently both be of length 3. The colorbar is labeled with label_s. """

    # Create the colormap for the colorbar    
    colormap = make_colormap(color_t_t, color_value_t)    
    
    # Create the colorbar
    norm = mpl.colors.Normalize(vmin=color_value_t[0], vmax=color_value_t[2])
    color_bar_handle = mpl.colorbar.ColorbarBase(ax, cmap=colormap,
                                               norm=norm,
                                               orientation='horizontal')
    color_bar_handle.set_label(label_s)
    
    
    
def make_colormap(color_t_t, color_value_t):
    """ Given colors defined in color_t_t and values defined in color_value_t, creates a LinearSegmentedColormap object. Works with only three colors and corresponding values for now. """
        
    # Find how far the second color is from the first and third
    second_value_fraction = float(color_value_t[1] - color_value_t[0]) / \
        float(color_value_t[2] - color_value_t[0])
    
    # Create the colormap
    color_s_l = ['red', 'green', 'blue']
    color_map_entry = lambda color_t_t, i_color: \
        ((0.0, color_t_t[0][i_color], color_t_t[0][i_color]),
         (second_value_fraction, color_t_t[1][i_color], color_t_t[1][i_color]),
         (1.0, color_t_t[2][i_color], color_t_t[2][i_color]))
    color_d = {color_s: color_map_entry(color_t_t, i_color) for i_color, color_s
              in enumerate(color_s_l)}
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
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
    
    # Plot regression line (one set of points only)
    if plot_regression_b and len(x_l_t) == 1:
        plot_regression(ax, x_l_t[0], y_l_t[0])
    


def make_shape_plot(fig, value_sr, shape_index_l, shape_l, color_type_s, color_t_t,
                    ax=None, colorbar_s=None, colorbar_ax=None, color_value_t=None):
    """ Creates a shape plot given figure handle fig. value_sr is the Series containing the data to be plotted; shape_index_l indexes the shapes to plot by FIPS code; shape_l contains the shapes to plot; color_type_s defines whether the plot will be shaded according to a binary or a gradient; color_t_t defines the colors to shade with. colorbar_s labels the colorbar. ax and colorbar_ax are optional pre-defined axes. If color_value_t is defined, it sets the values that will be shown by the respective colors in color_t_t if 'unbalanced_gradient' is specified for color_type_s.
    """
    
    # Set shape colors
    if not ax:
        ax = fig.add_subplot(1, 1, 1)
    shape_bounds_all_shapes_l = [float('inf'), float('inf'), float('-inf'), float('-inf')]    

    color_types_d = {'boolean': lambda: define_boolean_color(value_sr, color_t_t),
                     'balanced_gradient': \
                         lambda: define_balanced_gradient_color(value_sr, color_t_t),
                     'unbalanced_gradient': \
                         lambda: define_unbalanced_gradient_color(value_sr, color_t_t,
                                                                  color_value_t)}
    color_df, color_value_t = color_types_d[color_type_s]()
    color_df = np.around(color_df, decimals=5)
    # To prevent rounding errors leading to values outside the [0, 1] interval
                
    # Add shapes to plot
    for l_fips in value_sr.index:
        this_counties_color_t = tuple(color_df.loc[l_fips])
        
        i_shape_l = [i for i,j in enumerate(shape_index_l) if j==int(l_fips)]
        for i_shape in i_shape_l:       
            shape_bounds_this_shape_l = shape_l[i_shape].bbox
            shape_bounds_all_shapes_l[0] = \
                min(shape_bounds_this_shape_l[0], shape_bounds_all_shapes_l[0])
            shape_bounds_all_shapes_l[1] = \
                min(shape_bounds_this_shape_l[1], shape_bounds_all_shapes_l[1])
            shape_bounds_all_shapes_l[2] = \
                max(shape_bounds_this_shape_l[2], shape_bounds_all_shapes_l[2])
            shape_bounds_all_shapes_l[3] = \
                max(shape_bounds_this_shape_l[3], shape_bounds_all_shapes_l[3])
            
            this_shapes_patches = []
            points_a = np.array(shape_l[i_shape].points)
            shape_file_parts = shape_l[i_shape].parts
            all_parts_l = list(shape_file_parts) + [points_a.shape[0]]
            for l_part in xrange(len(shape_file_parts)):
                this_shapes_patches.append(mpl.patches.Polygon(
                    points_a[all_parts_l[l_part]:all_parts_l[l_part+1]]))
            ax.add_collection(mpl.collections.PatchCollection(this_shapes_patches,
                                              color=this_counties_color_t))
    ax.set_xlim(-127, -65)
    ax.set_ylim(20, 50)
    ax.set_axis_off()
    
    # Add colorbar
    if colorbar_s and color_value_t:
        if not colorbar_ax:
            colorbar_ax = fig.add_axes([0.25, 0.10, 0.50, 0.05])
        make_colorbar(colorbar_ax, color_t_t, color_value_t, colorbar_s)
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