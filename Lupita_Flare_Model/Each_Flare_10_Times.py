import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import astropy
import lightkurve as lk
from scipy.optimize import minimize
from lightkurve import search_lightcurvefile
from lightkurve import search_lightcurve

from astropy.table import Table, join, MaskedColumn, vstack, Column

import sys  
sys.path.append('/Users/Tobin/Dropbox/Stellar_Flares_Project/Lupita_Flare_Model/Llamaradas-Estelares/') #Edit this to your own file path
from Flare_model import flare_model

import astropy.units as u

sys.path.append('/Users/Tobin/Dropbox/Stellar_Flares_Project/Getting_Started/')
import stella
from tqdm import tqdm_notebook
import os, sys
from stella.download_nn_set import DownloadSets

lupita_tab=Table.read('lupita_tab_3.mrt', format='mrt')

labeling_flare=np.zeros(len(lupita_tab))

for i in range(len(lupita_tab)):
    labeling_flare[i]=int(i)
    
lupita_tab.add_column(Column(labeling_flare.astype(int)), name='ID', index=0)

import pickle

flare_info=pickle.load(open("/Users/Tobin/Dropbox/Stellar_Flares_Project/Getting_Started/Energies_and_rates.pkl", 'rb'))

def get_lc(tic, sector_ind):

    search = lk.search_lightcurve(target=tic, mission='TESS', author='SPOC')

    search.table["dataURL"] = search.table["dataURI"]
    lc = search[sector_ind].download()
    lc=lc[~np.isnan(lc.flux.value)]    
    
    return lc

def syn_flare_insertion(lc, flare_amp, flare_fwhm, inserted_time_step):
    
    model_flare_flux=flare_model(lc['time'].value, lc['time'].value[inserted_time_step], 
                             flare_fwhm, flare_amp*np.median(lc.flux.value))
    
    
    
    fixed_mask=np.ma.filled(model_flare_flux, fill_value=0)
    
    fixed_mask2=np.nan_to_num(fixed_mask, nan=0)
    
    lc_with_inserted_flare=fixed_mask2+lc.flux.value
    
    return lc_with_inserted_flare


def recovery(lc, inserted_flare_flux, inserted_time_step):
    "Initialize Stella"
    OUT_DIR='/Users/Tobin/Dropbox/Stellar_Flares_Project/Getting Started/Results/'

    cnn = stella.ConvNN(output_dir=OUT_DIR)

    ds = DownloadSets()
    ds.download_models()

    MODELS=ds.models
    
    #Test recovery:
    preds = np.zeros((len(MODELS),len(lc['time'].value)))

    for i, model in enumerate(MODELS):
        cnn.predict(modelname=model,
                times=lc.time.value,
                fluxes=inserted_flare_flux,
                errs=lc['flux_err'].value)
        preds[i] = cnn.predictions[0]

    avg_pred = np.nanmedian(preds, axis=0)
    
    pred_at_inserterd_timestep = avg_pred[inserted_time_step]
    
    one_before=avg_pred[inserted_time_step-1]
    one_after=avg_pred[inserted_time_step+1]
    
    if pred_at_inserterd_timestep > 0.3 and one_before > 0.3 and one_after > 0.3:
        return True, avg_pred
    else:
        return False, None
    
def Injecting_and_recovery(tic, flare, sector_ind):
    
    lc = get_lc(tic, sector_ind)
    
    sector_bool=flare_info['Flare_Bool'][sector_ind]
    
    
    all_inds = np.arange(len(lc))
    not_flare_inds = all_inds[sector_bool & (all_inds > 5) & (all_inds < len(lc) - 5)]
    rand_insertion_point = np.random.choice(not_flare_inds)
    
#     while True:
#         rand_insertion_point=np.random.randint(2, len(lc)-3, size=1)
        
#         print(rand_insertion_point, ~flare_info['Flare_Bool'][0][rand_insertion_point])
        
#         if ~flare_info['Flare_Bool'][0][rand_insertion_point]:
#             break

    inserted_flare=syn_flare_insertion(lc, lupita_tab['Amp'][flare], lupita_tab['FWHM'][flare], rand_insertion_point)

    recovered, pred = recovery(lc, inserted_flare, rand_insertion_point)

    return recovered

list_of_rows=[]

for i in range(len(lupita_tab)):
    row = Table(lupita_tab[i])
    
    row.add_column(Column([0]), name='Sector_Insertion_Trial', index=1)
    
    row.add_column(Column([0]), name='Sector', index=1)
    
    list_of_10_rows=[]
    
    for j in range(10):
        list_of_10_rows.append(row)
        
    table_of_length_10=vstack(list_of_10_rows)
    
    for j in range(len(table_of_length_10)):
        table_of_length_10['Sector_Insertion_Trial'][j]=j+1
    
    list_of_50_rows=[]
    
    for k in range(5):
        list_of_50_rows.append(table_of_length_10)
    
    table_of_length_50=vstack(list_of_50_rows)
    
    for j in range(len(table_of_length_50)):
        if j <10:
            table_of_length_50['Sector'][j]=0
        if j >10 and j < 20:
            table_of_length_50['Sector'][j]=1
        if j >20 and j < 30:
            table_of_length_50['Sector'][j]=2
        if j >30 and j < 40:
            table_of_length_50['Sector'][j]=3
        if j > 40:
            table_of_length_50['Sector'][j]=4
            
    
    list_of_rows.append(table_of_length_50)
        
trial_table=vstack(list_of_rows)

Recovery=np.zeros(len(trial_table))

trial_table.add_column(Column(Recovery), name='Recovered', index=3)

trial_table['Recovered'] = trial_table['Recovered'].astype('bool')

trial_table

for i in range(len(trial_table)):
    trial_table['Recovered'][i]=Injecting_and_recovery('tic272272592', trial_table[i]['ID'], trial_table[i]['Sector'])
    print("Finished Flare:", i)
    
    
    
print(trial_table)    
    
    
trial_table.write('Injection_Recovery_Table_Results.fits')