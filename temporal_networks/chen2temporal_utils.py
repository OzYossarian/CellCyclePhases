import numpy as np
import matplotlib.pyplot as plt

import seaborn as sb
sb.set_context('paper')

from ODEs.xppcall import xpprun


# =========================================================

def normed_series(x) :
    return x / np.max(x)
       
#================= run chen model
npa, variables = xpprun('../example_data/bychen04_xpp.ode', clean_after=True)

i_st = 100
i_end = 300

times = npa[i_st:i_end,0]
npa = npa[i_st:i_end,:]

series = lambda name : npa[:, 1+variables.index(name)]
variables = [var.upper() for var in variables]
data = {var : series(var) for var in variables}
#================== find important events
mass = data['MASS']
bud = data['BUD']
spn = data['SPN']
ori = data['ORI']

idx_mass = abs(np.diff(mass)) > 1 # abrupt change when cell divides
times_mass = times[:-1][idx_mass]

# bud emergence
idx_bud = (bud[1:] >= 1) * (bud[:-1] <= 1) # where bud increases past 1
times_bud = times[:-1][idx_bud]

# chromosome alignment on spindle completed
idx_spn = (spn[1:] >= 1) * (spn[:-1] <= 1) 
times_spn = times[:-1][idx_spn]

#
idx_ori = (ori[1:] >= 1) * (ori[:-1] <= 1)
times_ori = times[:-1][idx_ori]

#=======================

def plot_concentrations(var, ax=None, norm=False) :

    """
    Plot concentration over time of each variable in var
    """
    
    if ax==None :
        ax = plt.gca()
        
    for i in var :
        if norm:
            ax.plot(times, normed_series(data[i]), label=i)
        else :
            ax.plot(times, series(i), label=i)

#     ax.legend()
    ax.set_xlabel('Time (min)')
    if norm :
        ax.set_ylabel('Concentration (normed)')
    else :
        ax.set_ylabel('Concentration')

    sb.despine()
    
    return ax    
    
#====================== plot timings


def plot_timings(ax=None, labels=True, y_pos=None, tstart=None, tend=None) :
    
    if ax==None :
        ax = plt.gca()
    if tstart==None: 
        tstart = times_mass[0]
    if tend==None :
        tend = 300
        
    tshift = - times_mass[0] + tstart
        
    # plot vertical lines
    for i in range(len(times_mass)) :
    
        if times_ori[i] + tshift <= tend : # control number of periods
            ax.axvline(times_mass[i] + tshift, ls='-', color='k', lw=3, zorder=-2)
            ax.axvline(times_bud[i] + tshift, ls='-', color='k', lw=1, zorder=-2)
            ax.axvline(times_spn[i] + tshift, ls='-', color='k', lw=1, zorder=-2)
            ax.axvline(times_ori[i] + tshift, ls='-', color='k', lw=1, zorder=-2)


    # plot text labels
    events_model = ['bud', 'ori', 'spn']
    events_model_times = [times_bud[0], times_ori[0], times_spn[0]]
    if y_pos==None:
        y_pos = ax.get_ylim()[1] * 1.05

    if labels:
        for i, event in enumerate(events_model):
            ax.text(events_model_times[i] + tshift, y_pos, events_model[i],
            rotation=90, fontsize='small', ha="center", va="bottom")

    
#events = ['START', 'E3']
#events_times = [105, 170]
#for i, event in enumerate(events):
#    ax1.text(events_times[i], 2.6, events[i], rotation=90, c='grey', fontsize='small', ha="center", va="bottom")
    
    
def plot_phases(ax=None, tstart=0, ypos=1, hwidth=0.2, ypos_txt=1) :

    if ax==None :
        ax = plt.gca()
        
    tshift = tstart
        
    phases = ['G1', 'S', 'G2', 'M']
    phases_times = np.array([0, 35, 70, 78, 101]) + tshift
    
    for i, phase in enumerate(phases) :
    
        # plot backgorund fill
        ax.axvspan(phases_times[i], phases_times[i+1], 
            ymin=ypos-hwidth/2, ymax=ypos+hwidth/2, color='k', alpha=0.1*i)
        
        # plot text
        mid_points = (phases_times[1:] + phases_times[:-1]) / 2
        ax.text(mid_points[i], ypos_txt, phases[i], 
        ha='center', fontsize='small')  
        


def curve2binary(data, p=1, method='average') :
    
    """Return true for all points above threshold"""
    
    avg = np.mean(data)
    ymax = np.max(data)
    
    if method=="average" :
        threshold = p * avg
    elif method=="percentage" :
        threshold = p * ymax
    
    binary = np.zeros_like(data)
    mask = (data >= threshold)
    binary[mask] = 1
    
    return binary #, threshold

