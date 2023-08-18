#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 11:58:19 2023

@author: para
"""

import matplotlib.pyplot         as plt
  
def plot_scatter_plot(tit, x , y, val, scal=1000.0, col='red', plot='False'):
          
    plt.figure()
    plt.title(tit)
    if len(val) == len(x): 
        plt.scatter(x, y , s=scal*val, color = col) 
    else:
        plt.scatter(x, y , color = col)   
        
    plt.grid()
    
    if plot:
        plt.show()