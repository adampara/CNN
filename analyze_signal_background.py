#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:17:34 2023

@author: para
"""

import numpy                            as np

from fetch_object_from_ROOT_file       import fetch_object_from_ROOT_file
from plot_histogram                    import plot_histogram


import ROOT

import matplotlib.pyplot as plt

def plot_values(val, err, first_component, third_component, title, 
                c0=0, c1=1 , c2=2, xl='x', yl='y', plot=False, fig=False, lab='', lnsty=''):
    """
    
    val end err are vlues and errors. Plot the values and the errors as a function of the key component c1
    when the component c0 and c2 have values 'first cmponent' and 'third component' accordingly
    
    """
    x    = []  # The second component of the tuple
    y    = []  # Values from the 'val' dictionary
    yerr = []  # Errors from the 'err' dictionary

    for key, value in val.items():
        if key[c0] == first_component and key[c2] == third_component:
            x.append(key[c1])
            y.append(value)
            yerr.append(err[key])

    if fig: 
        plt.figure()
        
    plt.errorbar(x, y, yerr=yerr, fmt='o', markersize=5, capsize=3, label=lab, linestyle=lnsty)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title( title )
    
    if plot: plt.show()


def plot_overlayed_histograms(hist_list, hist_leg):
    
    # Create a canvas to draw the histograms on
    c1 = ROOT.TCanvas("c1", "Overlayed Histograms", 800, 600)
    c1.SetGrid()
    
    # Set up the legend
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    
    # Define a list of colors for the histograms
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kOrange]
        

        
    # Loop over the histograms in hist_list
    for i, hist in enumerate(hist_list):

        # Set the color for the copied histogram using the colors list
        hist.SetLineColor(colors[i % len(colors)])  # Loop over the colors if there are more histograms than colors    
        
        draw_option = "HIST SAME" if i > 0 else "HIST"
        hist.Draw(draw_option)
        
        # Add the corresponding descriptor to the legend
        legend.AddEntry(hist, hist_leg[i], "l")
    
    # Draw the legend
    legend.Draw()
    
    # Show the canvas
    c1.Draw()
    
    c1.Update()
    
    #nto exit doubleclick on Canvas
    loop = True
    while loop:
        if c1.WaitPrimitive() == None:
            loop = False
            c1.Close()
        

    # # Wait for the user to press Enter
    # input("Press Enter to close...")
    
    # # Close the canvas
    # c1.Close()

def get_hist_mean_err(f_root, title):
    """
    fetch histogram, get the mean vakue and error

    """
    
    aux = fetch_object_from_ROOT_file(f_root, title)        
    
    return aux, aux.GetMean(), aux.GetMeanError()
proc  = 'sgn'
proc  = 'bkg'


bars = [ 1, 5, 10, 15]

blob_radii = np.array([100., 200., 300., 400., 500.])    # blob radii at 1 bar
bl_rad     =  blob_radii.tolist()    

#   ranges  along the trajectory ( at 1 bar)   
bl_rad.insert(0, 0.)
rad_l_u = [(bl_rad[i], bl_rad[i+1]) for i in range(len(bl_rad) - 1)] 

#   histograms
blob_beg    = {}
loc_en_loss = {}
blob_beg_2    = {}
loc_en_loss_2 = {}

blob_end    = {}
bragg_prof  = {}
blob_end_2    = {}
bragg_prof_2   = {}

#    mean values
blob_beg_mean    = {}
loc_en_loss_mean = {}
blob_beg_2_mean    = {}
loc_en_loss_2_mean = {}

blob_end_mean    = {}
bragg_prof_mean  = {}
blob_end_2_mean    = {}
bragg_prof_2_mean   = {}

#   errors

blob_beg_mean_err    = {}
loc_en_loss_mean_err = {}
blob_beg_2_mean_err   = {}
loc_en_loss_2_mean_err = {}

blob_end_mean_err    = {}
bragg_prof_mean_err  = {}
blob_end_2_mean_err    = {}
bragg_prof_2_mean_err   = {}

for proc in ['sgn', 'bkg']:
    for bar in bars:
        
        bar_s = str(bar)
        
        tit= proc + '_' + bar_s + 'bar_'
            
        f_root = 'hist/adampara.' + proc + '.' + bar_s + 'bar.nexus.hist_v1'
        
        sc = 1./bar
        blob_size_bar = sc * blob_radii 
        
        rad_l_u_bar = [(x * sc, y * sc) for x, y in rad_l_u]
        
        hist_list = []
        hist_leg  = []
    
        for i, br in enumerate(blob_size_bar):
    
            title = tit + 'blob_beg_1_rad_' + f"{br:.3f}"  
            blob_beg[proc,bar,i], blob_beg_mean[proc,bar,i], blob_beg_mean_err[proc,bar,i] = get_hist_mean_err(f_root, title)
                    
            title = tit + 'blob_beg_2_rad_' + f"{br:.3f}"  
            blob_beg_2[proc,bar,i], blob_beg_2_mean[proc,bar,i], blob_beg_2_mean_err[proc,bar,i] = get_hist_mean_err(f_root, title)
            
            title = tit + 'blob_end1_rad_' + f"{br:.3f}"     # end1 indeed, needs to be fixed in the creating program
            blob_end[proc,bar,i], blob_end_mean[proc,bar,i], blob_end_mean_err[proc,bar,i] = get_hist_mean_err(f_root, title)
            
            title = tit + 'blob_end_2_rad_' + f"{br:.3f}"  
            blob_end_2[proc,bar,i], blob_end_2_mean[proc,bar,i], blob_end_2_mean_err[proc,bar,i] = get_hist_mean_err(f_root, title)
            
            # aux = fetch_object_from_ROOT_file(f_root, title)        
            # blob_beg         [proc,bar,i]       = aux
            # blob_beg_mean    [proc,bar,i]       = aux.GetMean()
            # blob_beg_mean_err[proc,bar,i]       = aux.GetMeanError()
            
        for i, p in enumerate(rad_l_u_bar):    
            
            title = tit + 'blob_beg_1_rad_' + f"{p[0]:.3f}" + '_to_' +   f"{p[1]:.3f}"
            loc_en_loss[proc,bar,i], loc_en_loss_mean[proc,bar,i], loc_en_loss_mean_err[proc,bar,i] = get_hist_mean_err(f_root, title)
    
            title = tit + 'blob_beg_2_rad_' + f"{p[0]:.3f}" + '_to_' +   f"{p[1]:.3f}"
            loc_en_loss_2[proc,bar,i], loc_en_loss_2_mean[proc,bar,i], loc_en_loss_2_mean_err[proc,bar,i] = get_hist_mean_err(f_root, title)
    
            title = tit + 'blob_end1_rad_' + f"{p[0]:.3f}" + '_to_' +   f"{p[1]:.3f}"    #  fix end1 to end_1 in the originating program
            bragg_prof[proc,bar,i], bragg_prof_mean[proc,bar,i], bragg_prof_mean_err[proc,bar,i] = get_hist_mean_err(f_root, title)
    
            title = tit + 'blob_end_2_rad_' + f"{p[0]:.3f}" + '_to_' +   f"{p[1]:.3f}"
            bragg_prof_2[proc,bar,i], bragg_prof_2_mean[proc,bar,i], bragg_prof_2_mean_err[proc,bar,i] = get_hist_mean_err(f_root, title)
            # aux = fetch_object_from_ROOT_file(f_root, title) 
            # loc_en_loss         [proc,bar,i]       = aux    
            # loc_en_loss_mean    [proc,bar,i]       = aux.GetMean()
            # loc_en_loss_mean_err[proc,bar,i]       = aux.GetMeanError()
    

for proc in ['sgn', 'bkg']:
#     track beginning, piecewise    
    plt.figure()
    
    title = proc + ' dE/dx for scaled path segments'
    
    for i in range(len(rad_l_u)):
        lab = str(rad_l_u[i][0]) + ' to '+ str(rad_l_u[i][1])         
    
        plot_values(loc_en_loss_mean, loc_en_loss_mean_err, proc, i, title, 
                    c0=0, c1=1 , c2=2, xl='pressure', yl='energy loss',lab=lab, lnsty='-.') 
    
    plt.grid()
    plt.legend()
    
    #   track beginning, blob
    
    plt.figure()
    title = proc + ' dE/dx for different pressures '
    xlab = 'track segment, ' + str(blob_radii[1]-blob_radii[0]) + 'mm/bar'    
    for i in range(len(bars)):
            
        lab = str(bars[i]) + ' bar'
        plot_values(loc_en_loss_mean, loc_en_loss_mean_err, proc, bars[i], title, 
                    c0=0, c1=2 , c2=1, xl=xlab, yl='energy loss', lab=lab, lnsty='-.') 
    plt.grid()
    plt.legend()  
    
    
    
    #  track end, blob
    
    plt.figure()
    title = proc + ' Brag blob '
    xlab = 'pressure'    
    for i in range(len(blob_radii)):
            
        lab = 'blob R = ' + str(blob_radii[i]) + '/bars'
        plot_values(blob_end_mean, blob_end_mean_err, proc, i , title, 
                    c0=0, c1=1 , c2=2, xl=xlab, yl='blob energy', lab=lab, lnsty='-.') 
    plt.grid()
    plt.legend()  
  
plt.show()

# exit()
#     for irad in range(len(blob_radii)):
#         rad = blob_radii[irad]
#         title = tit + 'blob_beg_1_rad_' + f"{rad:.3f}"
        
#         aux = fetch_object_from_ROOT_file(f_root, title)
#         hist_list.append(aux.Clone()) 
#         hist_leg.append(f"{rad:.3f}")

hist_list = []
hist_leg  = []

for bl in loc_en_loss:
    if bl[1] == 1: 
        hist_leg.append(str(bl[2]))        
        hist_list.append( loc_en_loss[bl])

plot_overlayed_histograms(hist_list, hist_leg)
    
hist_list = []
hist_leg  = []

for bl in blob_beg:
    if bl[2] == 0: 
        hist_leg.append(str(bl[1]))        
        hist_list.append( blob_beg[bl])

plot_overlayed_histograms(hist_list, hist_leg)
# plot_histogram(hist_list[0])