#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:22:53 2023

@author: para
"""


import matplotlib.pyplot as plt
import math
import numpy             as np
import random
import seaborn           as sns
import subprocess 
import sys

def NEXT_data(f, nev=1, n_points=4, dist_cut=1.0, debug=False, ext='.GNN_inp'):
    
    """
    read a single file.. it contains multiple events
    load nev events, n_points is the maximal number of nodes (for debugging/efficiency)
    dis_cut defines the adjancent limit
    """
    
    np.set_printoptions(linewidth=9999999)
    
    of = f.split('/')[-1].replace('.txt',ext)
    fo = open(of,'w')   
    
    if f.find('1eroi') > 0:
        targ = 0                 # background tag
    if f.find('0nubb') > 0:
        targ = 1                 # signal tag

    rd_events = read_events(f, debug=False) 
    
    if debug:
        #    examine specific events
        iev = 46
        print ('  event ', iev,'  length',len(rd_events[iev])  )
        print (rd_events[iev][0][0:2])
        print (rd_events[iev][3][0:2])

    debug = True
    
    #   loop over all events 
    
    for table in rd_events:
        
        n_nodes, vtrk_x, vtrk_y, vtrk_z, vtrk_E, nnbrMat, adjMat = get_nodes(table, n_req=n_points, dist_cut=dist_cut, ecut=4.e-4, plot=False )
    
        if debug:
            print ('  NEXT_data from file ', f )
            for nn in range(n_nodes):
                print ('x,y,z,E ',vtrk_x[nn],vtrk_y[nn],vtrk_z[nn],vtrk_E[nn])
            print ('AdjMat\n',adjMat)
            print ('number of neighbours ', nnbrMat)
    
        dlim = 10.
        
        feats = []   #  node features 
    
        edge_ind = []
        edge_val = []
    
        for nn in range(n_nodes):
            n_neigh = nnbrMat[nn]
    #             print ('  node ', nn, n_neigh, ' neighbours')
            av_en   = 0.
            en_dens = 0.
            av_len  = 0.
            
            for mm in range( n_nodes):
                if adjMat[nn,mm] > dlim: continue
                #print ('  link ', nn,mm, vtrk_E[mm], adjMat[nn,mm])
                av_en   += vtrk_E[mm]
                en_dens += vtrk_E[mm]/adjMat[nn,mm] 
                av_len  += adjMat[nn,mm] 
                
                edge_ind.append((nn,mm))
                edge_val.append(adjMat[nn,mm] )
                
            if n_neigh>0:
                
                av_en   = av_en/n_neigh
                av_len  = av_len/n_neigh
                en_dens = en_dens/n_neigh
                
            
            feat = (vtrk_E[nn], n_neigh, av_en, en_dens, av_len)
            feats.append(feat)
            
    #             print ('  node ', nn,'   mean energy', av_en, '  en_dens',en_dens, n_neigh)
            pos = np.array([vtrk_x.tolist(), vtrk_y.tolist(), vtrk_z.tolist()])
    
            ev = (targ, pos, feats, edge_ind, edge_val, n_nodes)
            
        #  write the event information
        sp = ' '
        eol = '\n'
        ev_head = str(targ) + sp + str(n_nodes) + eol
        fo.write(ev_head)
        for mm in range( n_nodes):    
            feat_line = str(round(feats[mm][0],5)) + sp + str(feats[mm][1]) + sp + str(round(feats[mm][2],5)) + sp +  str(round(feats[mm][3],6)) + sp + str(round(feats[mm][4],3)) + eol
            fo.write(feat_line)
        
        np.savetxt(fo, vtrk_x, fmt='%6.1f', newline = ' ')
        np.savetxt(fo, vtrk_y, fmt='%6.1f', newline = ' ')
        np.savetxt(fo, vtrk_z, fmt='%6.1f', newline = ' ')
        
        np.savetxt(fo, adjMat, fmt='%6.3e', newline = ' ')
        np.savetxt(fo, nnbrMat, fmt='%i'  , newline = ' ')

        
        
           
    return 



def get_nodes(table, n_req=5, dist_cut=10.0 , ecut=4.e-4, debug=False, plot=False):
    
    """
    get the node data: list of nodes and their atributes, adjacency matrix
    """
    if debug:
        print ('---->   get nodes, the input table check ')
        print (type(table))
        print ('  nodes ', table.shape)
    tableT = table.T
    
    x = tableT[0]
    y = tableT[1]
    z = tableT[2]
    E = tableT[3]
 
    if plot: 
        #   analyze the spectrum of nodes to optimize the energy cutoff
        sns.histplot(data=E , color="red", bins=100, cumulative=True)  
        plt.show()
        
    vtrk_x, vtrk_y, vtrk_z, vtrk_E = [], [], [], []

    for xx,yy,zz,ee in zip(x,y,z,E):
        
         if ee < ecut: continue        #  eliminate low signal nodes
    
         vtrk_x.append(xx)
         vtrk_y.append(yy)
         vtrk_z.append(zz)
         vtrk_E.append(ee)
 
    length = len(vtrk_x)
    n_nodes = length          
    vtrk_ID = np.arange(n_nodes)

    
    #   convert to the n array of he requested length
    
    vtrk_x = np.array(vtrk_x[:n_nodes])
    vtrk_y = np.array(vtrk_y[:n_nodes])
    vtrk_z = np.array(vtrk_z[:n_nodes])
    vtrk_E = np.array(vtrk_E[:n_nodes])


    vtrk_ID = np.arange(n_nodes)
    
    #    for some reason it was necessary to make sure that the x values are never identical
    for ID in vtrk_ID:
        vtrk_x[ID] = vtrk_x[ID] + (ID * 10e-5)
            
    
    # ---------------------------------------------------------------------------
    # Create the adjacency matrix
    # -1 --> self
    # 0 --> not a neighbor
    # (distance) --> nodes are neighbors
    # ---------------------------------------------------------------------------
     
    # Iterate through all nodes, and for each one find the neighboring nodes.
    adjMat = []; nnbrMat = []
    
    for vID1,vx1,vy1,vz1,vE1 in zip(vtrk_ID,vtrk_x,vtrk_y,vtrk_z,vtrk_E):
        nbr_list = [];
        nnbrs = 0;
        for vID2,vx2,vy2,vz2,vE2 in zip(vtrk_ID,vtrk_x,vtrk_y,vtrk_z,vtrk_E):  
            if(vx1 == vx2 and vy1 == vy2 and vz1 == vz2):
                nbr_list.append(1.e6);
            else:
                # proximity cut: accept liks shorter than dist_cut
                dist = math.sqrt((vx2-vx1)**2 + (vy2-vy1)**2 + (vz2-vz1)**2);
                if (dist < dist_cut):
                    #print ('\n nodes ',vID1,vID2, vx1, vy1, vz1, vE1, vx2,vy2,vz2, vE2, vE2/dist)
                    nbr_list.append(dist);
                    nnbrs += 1;
                else:
                    nbr_list.append(1.e6);
        nnbrMat.append(nnbrs);
        adjMat.append(np.array(nbr_list));

    return n_nodes, vtrk_x, vtrk_y, vtrk_z, vtrk_E, np.array(nnbrMat), np.array(adjMat)

def read_events(dfile, debug=False):
    """
    read events from the data file
    """
    
    events = []
    f = open(dfile)
    linea = f.readlines()  
    f.close()

    table = np.loadtxt(dfile,comments='E')

    if debug:
        print ('  file ',dfile, '\n', '  number of lines ',len(linea))
    #   identify the event header
    
    ev_head = []

    lin_no = -1
    
    for lin in linea:
        lin_no += 1
        if lin.find('Ev') > -1:
            ev_head.append(lin_no)

    n_events = len(ev_head) 
    

    if debug:     
        print (' number of events  ',n_events)
        for beg in ev_head:
            print (' event header', linea[beg])

    ev_head.append(len(linea))  # end of the last event
    
    for iev in range(n_events):
        
        len_ev = ev_head[iev+1] - ev_head[iev] -1
        
        jbeg =  ev_head[iev] - iev
        jend =  ev_head[iev] + len_ev - iev - 1    #   poinsta numberng starts at 'local 0'

        event = table[jbeg:jend+1]
        events.append(event)        
        
        if debug:
            print ( 'jbeg, jend ',jbeg, jend)
            print ('  event no', iev, '  length ', len_ev )
            print ('   first point ',table[jbeg])
            print ('   last point ',table[jend])
        
            for linn in range(jend-3, jend+5):
                print ('  line ', linn, '  ', linea[linn].rstrip())
    
            for linn in range(jend-3, jend+1):
    
               print ('  table[linn-iev] ', linn-iev, '   ', table[linn-iev])        

    
    return events

inp_files = sys.argv[1]
#inp_files  = '/Users/para/GNN/NEXT100/0nubb/beersheba/beersh*h5'
file_list = subprocess.getoutput(' echo "`ls ' + inp_files + '`"').split('\n')
#print (file_list)

for f in file_list:
    print ('  process file  ',f)
    NEXT_data(f, nev=1, n_points=4, dist_cut=5.0, debug=True)
    
 