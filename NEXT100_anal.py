#!/usr/bin/env python
# coding: utf-8

# Analyzde events in NEXT100 geometry


import tables                   as tb
import math 
#import seaborn                  as sns
import matplotlib.pyplot        as plt
import numpy                    as np
import subprocess
import sys

from mpl_toolkits.mplot3d import Axes3D

# In[3]:


def load_MC_events(filename):
    """
    read Monte Carlo events
    
    """


    with tb.open_file(filename, 'r') as h5f:

        hits  =  h5f.root.MC.hits.read()

    return hits



def get_events(data_set,data_type):
    """
    convert the dataset into x,y,z,E collections
    """
    ev = {}
    ev_x = []
    ev_y = []
    ev_z = []
    ev_e = []
    ev_no = -1
    line_no = -1

    for  eline in data_set:
        line_no += 1

        if data_type == 'DECO':
            (event,npeak,X,Y,Z,E) = eline 
        if data_type == 'MC':
            (event,part_id, hit_id,X,Y,Z,time, E,act) = eline    
        #print (  'line read ',event,npeak,X,Y,Z,E)


        if event not in ev:

            #print ('  start a new event', ev_no)

            if ev_no > -1:    #  new event, complete the previous one
                #print ('  line number ', line_no,' new event found')
                ev[ev_no] = (ev_x,ev_y, ev_z, ev_e)
                #print  ('   completed an event number ', ev_no, '\n  number of points ', len(ev_x))


            #   new event 
            ev_x = []
            ev_y = []
            ev_z = []
            ev_e = []
            ev_no = event

            ev[event] = []   # placeholder for the new event

        ev_x.append(X)
        ev_y.append(Y)
        ev_z.append(Z)
        ev_e.append(E)
        #print ('  event number', event)


    #   assemble last eventev[event] = (ev_x,ev_y, ev_z, ev_e) def get_events(data_set):
    ev[event] = (ev_x,ev_y, ev_z, ev_e) 
    
    return ev



#     ============================================

inp_files = sys.argv[1]
#inp_files  = '/Users/para/GNN/NEXT100/0nubb/beersheba/beersh*h5'
file_list = subprocess.getoutput(' echo "`ls ' + inp_files + '`"').split('\n')
print (file_list)

ev_energy = []
radius    = []
z_val     = []
n_ev = 0

for fn in file_list:
    
    print ("  convert file ", fn)
    
    hits = load_MC_events(fn)
    
    print ("  loaded file ", fn)     
    mc = get_events(hits,'MC')
    

    
    for iev in mc:
        
        iev_bsb = 2*iev
        #print (len(ev[iev]))
        n_ev += 1
    
        n_hit = 3
        for ih in range(n_hit):
            print ('  hit no ', ih,
                   ' x ',mc[iev][0][ih],
                   ' y ',mc[iev][1][ih],
                   ' z ',mc[iev][2][ih],
                   ' E ',mc[iev][3][ih]
                   )
    
        x = np.array(mc[iev][0])
        y = np.array(mc[iev][1])
        z = np.array(mc[iev][2])
    
        rad = np.sqrt(x**2 + y**2)
    
        radius += rad.tolist()
        z_val  += z.tolist()
    
        sig_siz = np.array(mc[iev][3])
        tot_en = sum(mc[iev][3])
    
        ev_energy.append(tot_en)
    
        #input("Press Enter to continue...")
        print ('   event ', iev, '   E = ', tot_en, ' max z ', max(z), 'maxR' , max(rad))
    

    
print ('total number of events ', len(ev_energy))

plt.hist(ev_energy, bins=100)
# sns.histplot(data=ev_energy , color="red")
plt.title('  energy')
plt.show() 

n, bins, patches = plt.hist(ev_energy, bins=10, range=(2.4,2.5))
print (n)
print (bins)
print (patches)
# sns.histplot(data=ev_energy , color="red")
plt.title('  energy')
plt.show() 


#sns.histplot(data=radius , color="red")
plt.hist(radius, bins=100)
plt.title('  radius')
plt.show() 

#sns.histplot(data=z_val , color="red")
plt.hist(z_val, bins=100)
plt.title('  Z')
plt.show() 




