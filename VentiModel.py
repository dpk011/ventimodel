#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:21:36 2020

@author: dgupta
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

class Lung():

    def __init__(self): ##, Vtp, BPM, IE, Ptrig):

        ## Input from user
        self.Vtp = 75        # Tidal Volume [0-100] Percentage(%) of bag compression  
        self.BPM = 20        # Breaths per minutes [8-30] 
        self.IE = 2          # 1 by I/E ratio [1-4]; Typcial I/E ratio 1:1 to 1:4
        self.Ptrig = 10      # Trigger Pressure Threshold on [in cm_H2O] for patient-triggered inhale cycle

        ## Advanced Parameters
        self.P_peep = 10      # PEEP; cm_H2O
        self.Thp =  10       # Inspiratory hold or pause (in % on Inhale time) at the end of inhale during the inspiratory phase for plateau pressure
        self.VCon = 0.5/75   ## Volume conversion [Lites/%] ## MUST BE REPLACED WITH CORRECT VALUE
        
        ## Lung Model Specs
        self.C = 20e-3           # Compliance; L/cm_H2O
        self.R = 20              # Airway Resistance; cm_H2O / (L/s)
        self.Rp = 100*self.R     # Leak Resistance; cm_H2O / (L/s)

        ## Calculating timing and defining state
        self.T, self.Tin, self.Tex = self.RespTime()
        self.Tho = self.Tin * self.Thp/100          # Inspiratory hold or pause (in sec)

        self.dt = 0.01      # in sec ;time resolution
        self.state = pd.DataFrame(index=np.arange(0, self.T + self.dt, self.dt), columns = ['Faw', 'Paw', 'Plung', 'Vt'])

    def RespTime(self):
        T = 60/self.BPM                  # Breath duration in sec
        Tinh = (T/(1+self.IE))           # InhaleHold time of the inspiratory phase (in sec)
        Tin = Tinh*(100- self.Thp)/100   # Inhale time of the inspiratory phase (in sec)
        Tex = T - Tinh                   # expiratory phase duration (in sec)
        return(T, Tin, Tex)

    def Flowin_constant(self): #, flow=0.55): # flow in Liters/sec
        flow = self.VCon * self.Vtp/(self.Tin)  # Or divide by (self.Tin - self.Tho)
        self.state.loc[0:self.Tin, 'Faw'] = flow

    def Inspiration(self):
        stt =  self.state.loc[0:self.Tin]
        t = stt.index - stt.index[0]
        temp = stt.Faw*self.Rp*(1 - np.exp(-t/(self.Rp*self.C)))
        self.state.loc[0:self.Tin, 'Plung'] = temp + self.P_peep
        self.state.loc[0:self.Tin, 'Paw'] = stt.Faw*self.R + self.state.loc[0:self.Tin, 'Plung']
        self.state.loc[0:self.Tin, 'Vt'] = self.C*(self.state.loc[0:self.Tin, 'Plung'] - self.P_peep)

    def Hold(self):
        t_start, t_end = (self.Tin + self.dt, self.Tin + self.Tho)
        stt =  self.state.loc[t_start:t_end]
        t = stt.index - stt.index[0]
        plo = self.state.loc[self.Tin, 'Plung']
        self.state.loc[t_start:t_end, 'Plung'] = (plo - self.P_peep)*np.exp(-t/(self.Rp*self.C)) + self.P_peep
        self.state.loc[t_start:t_end, 'Paw'] = self.state.loc[t_start:t_end, 'Plung']
        self.state.loc[t_start:t_end, 'Vt'] = self.C*(self.state.loc[t_start:t_end, 'Plung'] - self.P_peep)
        self.state.loc[t_start:t_end, 'Faw'] = 0.0

    def Expiration(self):
        t_start, t_end = (self.Tin + self.Tho + self.dt, self.T)
        stt =  self.state.loc[t_start:t_end]
        t = stt.index - stt.index[0]
        Rpp = self.Rp*self.R/(self.Rp + self.R)
        plo = self.state.loc[self.Tin + self.Tho, 'Plung']
        self.state.loc[t_start:t_end, 'Plung'] = (plo - self.P_peep)*np.exp(-t/(Rpp*self.C)) + self.P_peep
        self.state.loc[t_start:t_end, 'Paw'] = self.state.loc[t_start:t_end, 'Plung']
        self.state.loc[t_start:t_end, 'Vt'] = self.C*(self.state.loc[t_start:t_end, 'Plung'] - self.P_peep)
        self.state.loc[t_start:t_end, 'Faw'] = -(self.state.loc[t_start:t_end, 'Plung'] - self.P_peep)/self.R

    def Breath(self, n=1):
        self.Flowin_constant()
        self.Inspiration()
        self.Hold()
        self.Expiration()
        df = pd.concat(n*[self.state], ignore_index=True)
        df.index = df.index*self.dt
        return df

if __name__ == "__main__":
    print('==> Direct Script Execution...')

lung = Lung()

# lung.Flowin_constant()
# lung.Inspiration()
# lung.Hold()
df = lung.Breath(4)

fig, ax = plt.subplots(3,1, figsize=(6,6), sharex=True)
df['Paw'].plot(ax=ax[2])
df['Vt'].plot(ax=ax[1])
df['Faw'].plot(ax=ax[0])
ax[0].grid();ax[1].grid();ax[2].grid()

ax[0].set_title('Flow (L/s)')
ax[1].set_title('Delivered Tidal Volume (L)')
ax[2].set_title('Airway Pressure (cm_H2O)')
ax[2].set_xlabel('Time (s)')
plt.show()
