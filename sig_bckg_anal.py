#!/usr/bin/env python
# coding: utf-8

"""
 read nexus events
 display, and analyze
 python sig_bckg_anal.py 99999999 "events/*"
 histograms v2 : only tracks above cutoff energy (1 MeV) included in histograms
""" 

import                           glob
import                           math

import tables                    as tb
#import math 
#import seaborn                  as sns
import matplotlib.pyplot         as plt
import numpy                     as np
import subprocess
import sys

from get_avg_y_in_range         import get_avg_y_in_range
from mpl_toolkits.mplot3d       import Axes3D
from plot_scatter_plot          import plot_scatter_plot
from plot_weighted_histogram    import plot_weighted_histogram
from Save_Hist                  import Save_Hist
from sort_y_based_on_x          import sort_y_based_on_x

# In[3]:


def load_MC_events(filename,nrec=1000):
    """
    read hits collection produced by nexus
    """


    with tb.open_file(filename, 'r') as h5f:

        hits       =  h5f.root.MC.hits.read(start=0, stop=nrec)     # limit the  length of hits only
        particles  =  h5f.root.MC.particles.read(start=0, stop=nrec)

    return hits, particles

def get_particles(data_set):
    """
    fetch primary particles and secondary gammas
    """
    
    deb = False
    
    ev = {}

    
    p_name = []    # primary particle name
    p_ekin = []    # primary particle kinetic energy
    p_len  = []    # primary particle length
    
    p_x0   = []    # primary, starting x
    p_y0   = []    # primary, starting y
    p_z0   = []    # primary, starting z
    
    p_x1   = []    # primary, final x
    p_y1   = []    # primary, final y
    p_z1   = []    # primary, final z
    
    g_en   = []    # gamma energy, brehmsstrahlung only     
    g_len  = []    # gamma length

    ev_out = 0

    for  eline in data_set:


        if deb: print ('line ', eline)
        if deb: print ('  current event number  ', ev_out)

        (event, pid, pn, prim, mid, x, y, z, t, 
         fx, fy, fz, ft, inv, fv, ipx, ipy, ipz, fpx, fpy, fpz, 
         ek, tlen, cr_pr, fin_proc  ) = eline    
        
        pn = pn.decode('utf-8')
        if deb: print ('event number ',event,'  current event', ev_out)

        if event not in ev:

            if deb: print ('  event', event)

            if event > ev_out:    #  new event, complete the previous one
                #print ('  line number ', line_no,' new event found')
                nprim = len(p_name)
                nsec_g = len(g_en)
                ev[ev_out] = (nprim, nsec_g, p_name, p_ekin, p_len, p_x0, p_y0, p_z0,
                              p_x1, p_y1, p_z1, g_en, g_len)
                if deb: print  ('   completed an event number ', ev_out)
                if deb: print (ev[ev_out])


                #   new event 
                
                p_name = []    # primary particle name
                p_ekin = []    # primary particle kinetic energy
                p_len  = []    # primary particle length
                
                p_x0   = []    # primary, starting x
                p_y0   = []    # primary, starting y
                p_z0   = []    # primary, starting z
                
                p_x1   = []    # primary, final x
                p_y1   = []    # primary, final y
                p_z1   = []    # primary, final z  
                
                g_en   = []    # gamma energy, brehmsstrahlung only     
                g_len  = []    # gamma length
                ev_out = event


        #ev[event] = []   # placeholder for the new event

        if prim == 1:
            p_name.append(pn)
            p_ekin.append(ek)
            p_len .append(tlen)
            
            p_x0  .append(x)
            p_y0  .append(y)
            p_z0  .append(z)
            
            p_x1  .append(fx)
            p_y1  .append(fy)
            p_z1  .append(fz)
            
        if deb: print ('  primeary ekin ',p_ekin)
            
        if prim == 0:
            if cr_pr == b'eBrem':
                g_en.append(ek)
                g_len.append(tlen)

        #print ('  event number', event)


    #   assemble last eventev[event] = (ev_x,ev_y, ev_z, ev_e) def get_events(data_set):
    nprim = len(p_name)
    nsec_g = len(g_en)
    ev[event] = (nprim, nsec_g, p_name, p_ekin, p_len, p_x0, p_y0, p_z0,
                  p_x1, p_y1, p_z1, g_en, g_len)

    # ndeb = 3
    # for i in range(ndeb):
    #     print ('  event ',i)
    #     print (ev[i])

    return ev

def get_hits(data_set,data_type):
    """
    convert the dataset into x,y,z,E   collections (DECO  for deconvoluted daa sets, add t=0)
    convert the dataset into x,y,z,t,E collections (MC    for mc hits daa sets)
    """
    ev = {}
    
    #   this declaration is to avoid 'undefined' errors in line 152 
    ev_x = []
    ev_y = []
    ev_z = []
    ev_e = []
    ev_t = []
    ev_no = -1
    line_no = -1

    for  eline in data_set:
        line_no += 1

        if data_type == 'DECO':
            time = 0.
            (event,npeak,X,Y,Z,time, E) = eline 
        if data_type == 'MC':
            # print ('line ', eline)
            (event, X, Y, Z, time, E, reg, act, flag ) = eline    
            # print (  'line read ',event,X,Y,Z,E)
            # exit()


        if event not in ev:

            #print ('  start a new event', ev_no)

            if ev_no > -1:    #  new event, complete the previous one
                #print ('  line number ', line_no,' new event found')
                ev[ev_no] = (ev_x,ev_y, ev_z, ev_e, ev_t )
                #print  ('   completed an event number ', ev_no, '\n  number of points ', len(ev_x))


            #   new event 
            ev_x = []
            ev_y = []
            ev_z = []
            ev_e = []
            ev_t = []
            ev_no = event

            ev[event] = []   # placeholder for the new event

        ev_x.append(X)
        ev_y.append(Y)
        ev_z.append(Z)
        ev_t.append(time)
        ev_e.append(E)
        #print ('  event number', event)


    #   assemble last eventev[event] = (ev_x,ev_y, ev_z, ev_e) def get_events(data_set):
    ev[event] = (ev_x,ev_y, ev_z, ev_e, ev_t) 
    
    return ev

#     ============================================

from ROOT                       import TH1F
def Book(sigbckg, pres, blob_radii, pairs):

    """
    Book histograms
    """

    tit= sigbckg + '_' + str(pres) + '_'
    
    nbin = 100

    title = tit + 'Etot'
    Etot = TH1F(title, title, nbin,0.5,3.0)     #  total hit energy
  
    title = tit + 'track - hits '
    tr_hits = TH1F(title, title, nbin,-0.01, 0.01)     #  track -  hits energy   
    
    title = tit + 'N_hits'
    Nhit = TH1F(title, title, 1000,0.,10000.)  
        
    title = tit + 'N_prim'
    N_prim = TH1F(title, title, 5,0.5,5.5)  
        
    title = tit + 'N_gam'
    N_gam = TH1F(title, title, 10,-0.5,9.5)  
    
    title = tit + 'gam_en'
    gam_en = TH1F(title, title, nbin ,0. ,1.)  
    
    title = tit + 'event_volume'
    ev_rad_max = 5000.
    ev_vol = TH1F(title, title, 1000 ,0., ev_rad_max) 
    
    #    end point energies for different blob radii
    
    max_blob_e = 2.0
    
    bl_beg_1 = []
    bl_beg_2 = []
    bl_end_1 = []
    bl_end_2 = []
    
    nbin_r = 1000
    for rad in blob_radii:
        
        title = tit + 'blob_beg_1_rad_' + f"{rad:.3f}"
        aux = TH1F(title, title, nbin_r ,0. , max_blob_e)   
        bl_beg_1.append(aux)
        
        title = tit + 'blob_beg_2_rad_' + f"{rad:.3f}"
        aux = TH1F(title, title, nbin_r ,0. , max_blob_e)   
        bl_beg_2.append(aux)
        
        title = tit + 'blob_end1_rad_' + f"{rad:.3f}"
        aux = TH1F(title, title, nbin_r ,0. , max_blob_e)   
        bl_end_1.append(aux)
        
        title = tit + 'blob_end_2_rad_' + f"{rad:.3f}"
        aux = TH1F(title, title, nbin_r ,0. , max_blob_e)   
        bl_end_2.append(aux)
        
    
    bl_beg_dif_1 = []
    bl_beg_dif_2 = []
    bl_end_dif_1 = []
    bl_end_dif_2 = []
    
    for p in pairs:
        
        title = tit + 'blob_beg_1_rad_' + f"{p[0]:.3f}" + '_to_' +   f"{p[1]:.3f}"
        aux = TH1F(title, title, nbin_r ,0. , max_blob_e)   
        bl_beg_dif_1.append(aux)
        
        title = tit + 'blob_beg_2_rad_' + f"{p[0]:.3f}" + '_to_' +   f"{p[1]:.3f}"
        aux = TH1F(title, title, nbin_r ,0. , max_blob_e)   
        bl_beg_dif_2.append(aux)
        
        title = tit + 'blob_end1_rad_' + f"{p[0]:.3f}" + '_to_' +   f"{p[1]:.3f}"
        aux = TH1F(title, title, nbin_r ,0. , max_blob_e)   
        bl_end_dif_1.append(aux)
        
        title = tit + 'blob_end_2_rad_' + f"{p[0]:.3f}" + '_to_' +   f"{p[1]:.3f}"
        aux = TH1F(title, title, nbin_r ,0. , max_blob_e)   
        bl_end_dif_2.append(aux)
        
        
    
    hists = (Etot, tr_hits, Nhit, N_prim, N_gam, gam_en, ev_vol,
             bl_beg_1, bl_beg_2, bl_end_1, bl_end_2,
             bl_beg_dif_1, bl_beg_dif_2, bl_end_dif_1, bl_end_dif_2,
             )

    return hists


#     ============================================

def parse_fname(fn):
    """
    extract info from the file name
    """
    tok = fn.split('.')

    return tok[1], tok[2]


#    =============================================

def dist(t,v):
    """
    t is time,
    v is velocity
    calculate the distance travelled

    """
    
    npoint = len(t)
    
    dist = np.zeros(npoint)
    
    for i in range(1,npoint):
        dist[i] = dist[i-1] + 0.5 * (t[i] - t[i-1]) * (v[i] + v[i-1])
        
    return dist
    
print ('  number of command line arguments  ', len(sys.argv))
file_list = ['events/adampara.bkg.15bar.nexus.h5']   # default file list

nrec = 10000
if len(sys.argv) > 1:
    nrec = int(sys.argv[1])   
if len(sys.argv) > 2:
    inp_files = sys.argv[2]
    inp_files  = '/Users/para/GNN/' + inp_files
    file_list = subprocess.getoutput(' echo "`ls ' + inp_files + '`"').split('\n')
    glob.glob(inp_files)

#   
print ('  list of files to analyze  \n',file_list)

plots   = False     #  show 'per event plots
deb_ev  = False     #  debugging prints per event
  
m_e  = 0.511      # electron mass



ev_energy = []
radius    = []
z_val     = []
n_ev = 0

for fn in file_list:

    hist_file  = fn.replace('events', 'hist').replace('h5', 'hist_v2')     # histograms
    event_file = fn.replace('events', 'evfiles').replace('h5', 'txt')   # GNN info
    print ('  histogram file ', hist_file)
    print ('  GNN event file ', event_file)    
    print ("  Analyze file ", fn)
    
    #   signal/background, pressure
    sigbckg,pres = parse_fname(fn)
    bar = float(pres.replace('bar',''))
    
    blob_radii = np.array([100., 200., 300., 400., 500.]) * 1.0/bar
    bl_r =  blob_radii.tolist()
    bl_r.insert(0, 0.)
    pairs = [(bl_r[i], bl_r[i+1]) for i in range(len(bl_r) - 1)] 
    # print (len(bl_r))
    # print (bl_r)
    # print (len(pairs))
    # print (pairs)
    
    max_len = 20                # event length at 1 bar, in nsec
    ev_len = max_len/bar        #expected event length at this pressure    
    
    hits, particles = load_MC_events(fn, nrec=nrec)   
    mc = get_hits(hits,'MC')
    if len(particles) > 0:
        prtcls = get_particles(particles)       # this is for the mc true files only
 

    #  book histograms
    hists = Book(sigbckg, pres, blob_radii, pairs)
    (Etot, tr_hits, Nhit, N_prim, N_gam, gam_en, ev_vol,
             bl_beg_1, bl_beg_2, bl_end_1, bl_end_2,
             bl_beg_dif_1, bl_beg_dif_2, bl_end_dif_1, bl_end_dif_2,
             ) = hists

    #   hits and particles are dictionaries indexed by the event number    
    for iev in mc:
        
        if deb_ev: 
            print ('  particles, event no ',iev, prtcls[iev])
            # print ('  hits, event no ', iev, mc[iev])
            # exit()
            #print (len(ev[iev]))
            
            # n_hit = 3
            # for ih in range(n_hit):
            #            ' x ',mc[iev][0][ih],
            #            ' y ',mc[iev][1][ih],
            #            ' z ',mc[iev][2][ih],
            #            ' E ',mc[iev][3][ih]
                       # )
        
        n_ev += 1     # numnber of analyzed events
    

        x = np.array(mc[iev][0])            # x hit
        y = np.array(mc[iev][1])            # y hit
        z = np.array(mc[iev][2])            # z hit
        sig_siz = np.array(mc[iev][3])      # hit energy deposited
        t = np.array(mc[iev][4])            # t hit
        
        tot_en = sum(sig_siz)
       
        rad = np.sqrt(x**2 + y**2 + z**2)
        
        # tr_len = [0., 100., 200., 300., 400., 500.]
        
        # for itr in range(1,len(tr_len)):
        #     av_en, e0,e1 = get_avg_y_in_range(distance,track_energy,tr_len[itr-1], tr_len[itr])

        #     dedx = 0.1* (e1-e0)/(tr_len[itr] - tr_len[itr-1])   # dEdx per 1 cm
        #     print(tr_len[itr-1], tr_len[itr], e0,e1,dedx)
        

        (nprim, nsec_g, p_name, p_ekin, p_len, p_x0, p_y0, p_z0,
                      p_x1, p_y1, p_z1, g_en, g_len) = prtcls[iev]
        

        #if nprim == 2: continue
        tot_prim = sum(p_ekin)
        

        
        #   fill histograms
        
        Etot.     Fill(tot_en)
        tr_hits.  Fill(tot_en - tot_prim)
        Nhit.     Fill(float(len(x)))
        N_prim.   Fill(float(nprim))
        N_gam.    Fill(float(len(g_en)))
        gam_en.   Fill(sum(g_en))
        
        #   hit-based histograms  
        
        for rr,ss in zip(rad,sig_siz):
            ev_vol.    Fill(rr,ss)
            
        bl_1_b     = np.zeros(len(blob_radii))
        bl_1_e     = np.zeros(len(blob_radii))
        bl_2_b     = np.zeros(len(blob_radii))
        bl_2_e     = np.zeros(len(blob_radii))
        
        bl_1_d_b     = np.zeros(len(blob_radii))
        bl_1_d_e     = np.zeros(len(blob_radii))
        bl_2_d_b     = np.zeros(len(blob_radii))
        bl_2_d_e     = np.zeros(len(blob_radii))
        
        for xx,yy,zz,ss in zip(x,y,z,sig_siz):
            for rad_num in range(len(blob_radii)):
                
                if math.sqrt( (xx-p_x0[0])**2 + (yy-p_y0[0])**2 + (zz-p_z0[0])**2 ) < blob_radii[rad_num]:
                    bl_1_b[rad_num] += ss
                if math.sqrt( (xx-p_x1[0])**2 + (yy-p_y1[0])**2 + (zz-p_z1[0])**2) < blob_radii[rad_num]:
                    bl_1_e[rad_num] += ss

                if math.sqrt( (xx-p_x0[0])**2 + (yy-p_y0[0])**2 + (zz-p_z0[0])**2) > pairs[rad_num][0]:
                    if math.sqrt( (xx-p_x0[0])**2 + (yy-p_y0[0])**2 + (zz-p_z0[0])**2) < pairs[rad_num][1]:
                        bl_1_d_b[rad_num] += ss
                if math.sqrt( (xx-p_x1[0])**2 + (yy-p_y1[0])**2 + (zz-p_z1[0])**2) > pairs[rad_num][0]:
                    if math.sqrt( (xx-p_x1[0])**2 + (yy-p_y1[0])**2 + (zz-p_z1[0])**2) < pairs[rad_num][1]:
                        bl_1_d_e[rad_num] += ss
                        
                if nprim > 1: 
                
                    if math.sqrt( (xx-p_x0[1])**2 + (yy-p_y0[1])**2 + (zz-p_z0[1])**2) < blob_radii[rad_num]:
                        bl_2_b[rad_num] += ss
                    if math.sqrt( (xx-p_x1[1])**2 + (yy-p_y1[1])**2 + (zz-p_z1[1])**2) < blob_radii[rad_num]:
                        bl_2_e[rad_num] += ss
    
                    if math.sqrt( (xx-p_x0[1])**2 + (yy-p_y0[1])**2 + (zz-p_z0[1])**2) > pairs[rad_num][0]:
                        if math.sqrt( (xx-p_x0[1])**2 + (yy-p_y0[1])**2 + (zz-p_z0[1])**2) < pairs[rad_num][1]:
                            bl_2_d_b[rad_num] += ss
                    if math.sqrt( (xx-p_x1[1])**2 + (yy-p_y1[1])**2 + (zz-p_z1[1])**2) > pairs[rad_num][0]:
                        if math.sqrt( (xx-p_x1[1])**2 + (yy-p_y1[1])**2 + (zz-p_z1[1])**2) < pairs[rad_num][1]:
                            bl_2_d_e[rad_num] += ss

        e_track_mkin = 1.      #   cut on track energy to be included
            
        for rad_num in range(len(blob_radii)):
            
            if p_ekin[0] > e_track_mkin:
                    
                bl_beg_1[rad_num].    Fill(bl_1_b[rad_num])
                bl_end_1[rad_num].    Fill(bl_1_e[rad_num])
                
                bl_beg_dif_1[rad_num].    Fill(bl_1_d_b[rad_num])
                bl_end_dif_1[rad_num].    Fill(bl_1_d_e[rad_num])

            if nprim > 1: 
                if p_ekin[1] > e_track_mkin:   
                    
                    bl_beg_2[rad_num].    Fill(bl_2_b[rad_num]) 
                    bl_end_2[rad_num].    Fill(bl_2_e[rad_num])
                    
                    bl_beg_dif_2[rad_num].    Fill(bl_2_d_b[rad_num])
                    bl_end_dif_2[rad_num].    Fill(bl_2_d_e[rad_num])
                    

        if deb_ev:
            print ('   event ', iev, '   E = ', tot_en, ' max z ', '  primary energy ', tot_prim)
            print ('     p_ekin', p_ekin, ' \n       p_len ',p_len)
        
        if plots:    
            #input("Press Enter to continue...")
            
            # order all hits by the time
            x_sort = sort_y_based_on_x(t,x)
            y_sort = sort_y_based_on_x(t,y)        
            z_sort = sort_y_based_on_x(t,z)
            
            sig_siz_sort = np.array(sort_y_based_on_x(t, sig_siz))        
            t_sort = sort_y_based_on_x(t,np.copy(t))          

            #  track energy, time sequenced
            integ_track_en = np.cumsum(sig_siz_sort) 
            track_energy = integ_track_en[-1] - integ_track_en 
            
            #  velocity
            v_over_c = np.sqrt(1. - (m_e/(track_energy + m_e))**2)
            #  path length
            cdist = 300.
            distance = cdist*dist(t_sort, v_over_c )
            
            
            tit = sigbckg + ' ' + pres + '  event no ' + str(iev)
            print (type(tit))
            print (tit)
            nbins = 100
            xlow = 0.
            xup = ev_len        
            plot_weighted_histogram(t, sig_siz, nbins, xlow, xup, xl='time', tit='E deposited vs time')
            plt.title(tit + 'energy deposition vs time')

            plot_scatter_plot(tit + ' X vs time',            t_sort,    x_sort,       sig_siz_sort, plot=False)     
            plot_scatter_plot(tit + ' track energy vs path', distance,  track_energy, sig_siz_sort, plot=False)     
            plot_scatter_plot(tit + ' Y vs time',            t_sort,    y_sort,       sig_siz_sort, plot=False)     
            plot_scatter_plot(tit + ' Z vs time',            t_sort,    z_sort,       sig_siz_sort, plot=False)     
            plot_scatter_plot(tit + ' Track energy vs time', t_sort,    track_energy, sig_siz_sort, plot=False)     
            plot_scatter_plot(tit + ' velocity vs time',     t_sort,    v_over_c ,    sig_siz_sort, plot=False)     
            plot_scatter_plot(tit + ' Path length vs time',  t_sort,    distance,     sig_siz_sort, plot=False)     
           
            fig = plt.figure()
            ax = Axes3D(fig)
            plot_geeks = ax.scatter(x, y, z, s=1000.0*sig_siz, color = 'red')
            #    add star/end points of primary particles
            plot_geeks = ax.scatter( p_x0, p_y0, p_z0, s=50, color = 'blue',  marker='x')
            plot_geeks = ax.scatter( p_x1, p_y1, p_z1, s=50, color = 'green', marker='o')
            plt.title(tit + ' Trajectory')
            plt.show()

    Save_Hist([], hist_file, delete=True)
    
    print ('total number of events ', len(ev_energy))



