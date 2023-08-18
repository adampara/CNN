#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        Events   =  h5f.root.DECO.Events.read()
    return hits, Events



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





def granulate(data_set, xmin=-500., dx=3., ymin=-500., dy=3. , zmin=0., dz=3.):

    """
    rebun the x,,y,z to some  requested granularity
    """

    ev = {}

    
    for iev in data_set:

        if len(data_set[iev])==0: continue
        x = np.array(data_set[iev][0])
        y = np.array(data_set[iev][1])
        z = np.array(data_set[iev][2])
        e = np.array(data_set[iev][3])
        
        xn = {}
        yn = {}
        zn = {}
        en = {}

        for xo,yo,zo,eo in zip (x,y,z,e):
        
            ix = int((xo-xmin)/dx)
            iy = int((yo-ymin)/dy)
            iz = int((zo-zmin)/dz)

            if (ix,iy,iz) not in xn: xn[(ix,iy,iz)] = 0.
            if (ix,iy,iz) not in yn: yn[(ix,iy,iz)] = 0.
            if (ix,iy,iz) not in zn: zn[(ix,iy,iz)] = 0.
            if (ix,iy,iz) not in en: en[(ix,iy,iz)] = 0.
            
            if eo <= 0 :
                print (  ix, iy,iz, eo)

            xn[(ix,iy,iz)] += xo*eo
            yn[(ix,iy,iz)] += yo*eo
            zn[(ix,iy,iz)] += zo*eo
            en[(ix,iy,iz)] += eo

        
        ev_x = []
        ev_y = []
        ev_z = []
        ev_e = []

        #print ('   dx =  ',dx)        
        for hits in xn:
            if en[hits] == 0:   
                print ('   -    ', hits, ' ', xn[hits], '  ', yn[hits],  ' ', zn[hits], '  ', en[hits] )
            ev_x.append(xn[hits]/en[hits])
            ev_y.append(yn[hits]/en[hits])
            ev_z.append(zn[hits]/en[hits])
            ev_e.append(en[hits])


        #   assemble last eventev[event] = (ev_x,ev_y, ev_z, ev_e) def get_events(data_set):
        ev[iev] = (ev_x,ev_y, ev_z, ev_e) 
    
    return ev
def write_converted_events(fn, ext, dset):
    

    of = fn.split('/')[-1].replace('.h5',ext)
    f = open(of,'w')

    for iev in dset:
      
        ev_head = 'Ev  ' + str(iev) + '\n' 
        f.write(ev_head)
        x = np.array(dset[iev][0])
        y = np.array(dset[iev][1])
        z = np.array(dset[iev][2])
        e = np.array(dset[iev][3])
        
        for xx,yy,zz,ee in zip(x,y,z,e):
            ev = str(round(xx,1)) + ' ' + str(round(yy,1)) + ' ' + str(round(zz,1)) + ' ' + str(round(ee,6)) + '\n'
                                                                    
            f.write(ev)




#     ============================================

inp_files = sys.argv[1]
#inp_files  = '/Users/para/GNN/NEXT100/0nubb/beersheba/beersh*h5'
file_list = subprocess.getoutput(' echo "`ls ' + inp_files + '`"').split('\n')
print (file_list)

for fn in file_list:
    
    print ("  convert file ", fn)
    
    hits, Events = load_MC_events(fn)
    
    print ("  loaded file ", fn)     
    mc = get_events(hits,'MC')
    write_converted_events(fn, '_mc.txt', mc)   
    
    bshb = get_events(Events,'DECO')
    write_converted_events(fn, '_bshb_1.txt', bshb)   
    
    bshb_2 = granulate(bshb,dx=2.,dy=2.,dz=2.)
    write_converted_events(fn, '_bshb_2.txt', bshb_2)    

    bshb_5 = granulate(bshb,dx=5.,dy=5.,dz=5.)
    write_converted_events(fn, '_bshb_5.txt', bshb_5)    
    
    bshb_10 = granulate(bshb,dx=10.,dy=10.,dz=10.)
    write_converted_events(fn, '_bshb_10.txt', bshb_10)    

    print (len(mc), len(bshb), len(bshb_2), len(bshb_5), len(bshb_10) )       
    plot = True
    
    coarse = granulate(bshb,dx=5.,dy=5.,dz=5.)
    if plot:
        col = 0
        color = { 1: 'red',
                  2: 'blue'}
        
        
        
        
        col += 1
        
        #print (ev[2])
        
        ev_energy = []
        radius    = []
        z_val     = []
        n_ev = 0
        
        for iev in mc:
            iev_bsb = 2*iev
            #print (len(ev[iev]))
            n_ev += 1
        
        
            x = np.array(mc[iev][0])
            y = np.array(mc[iev][1])
            z = np.array(mc[iev][2])
        
            rad = np.sqrt(x**2 + y**2)
        
            radius += rad.tolist()
            z_val += z.tolist()
        
            sig_siz = np.array(mc[iev][3])
            tot_en = sum(mc[iev][3])
        
            ev_energy.append(tot_en)
        
            #input("Press Enter to continue...")
            print ('   event ', iev, '   E = ', tot_en, ' max z ', max(z), 'maxR' , max(rad))
        
            fig = plt.figure()
            ax = Axes3D(fig)
        
            # creating the plot
            plot_geeks = ax.scatter(x, y, z, s=1000.0*sig_siz, color = 'red')
        
            xbsb = np.array(bshb[iev_bsb][0])
            ybsb = np.array(bshb[iev_bsb][1])
            zbsb = np.array(bshb[iev_bsb][2])
            sig_siz_bsb = np.array(bshb[iev_bsb][3])
        
            #plot_geeks = ax.scatter(xbsb, ybsb, zbsb, s=1000.0*sig_siz_bsb, color = 'blue')
        
            xn = np.array(coarse[iev_bsb][0])
            yn = np.array(coarse[iev_bsb][1])
            zn = np.array(coarse[iev_bsb][2])
            sig_siz_n= np.array(coarse[iev_bsb][3])
             
            plot_geeks = ax.scatter(xn, yn, zn, s=1000.0*sig_siz_n, color = 'green')
                
            # setting title and labels
            ax.set_title("3D plot")
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')
            ax.set_zlabel('z-axis')
        
            plt.show()
        
        sns.histplot(data=ev_energy , color="red")
        plt.title('  energy')
        plt.show() 
        
        sns.histplot(data=radius , color="red")
        plt.title('  radius')
        plt.show() 
        
        sns.histplot(data=z_val , color="red")
        plt.title('  Z')
        plt.show() 




