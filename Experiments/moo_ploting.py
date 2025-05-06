#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:17:22 2024

@author: sorourifar.1
"""




import os
import time
import random
import numpy as np
import pandas as pd
import torch
import sys
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


import pandas as pd
import seaborn as sns
from scipy import stats



import matplotlib



import torch
import matplotlib.pyplot as plt



matplotlib.rcParams.update({# Use mathtext, not LaTeX
                            'text.usetex': False,
                            # Use the Computer modern font
                            'font.family': 'serif',
                            'font.serif': 'cmr10',
                            'mathtext.fontset': 'cm',
                            # Use ASCII minus
                            'axes.unicode_minus': False,
                            'figure.dpi' : 300,

                            })

font = {'weight' : 'bold',
        'size'   : 44}
plt.rc('font', **font)





name = 'dG'; data_file = 'MORDRED_SMILES_dft_Gsolv.csv'
df_dg = pd.read_csv('../prop_data/'+data_file)#.drop(columns=['Unnamed: 0'])


name = 'E'; data_file = 'MORDRED_SMILES_dft_E.csv'
df_e0 = pd.read_csv('../prop_data/'+data_file)#.drop(columns=['Unnamed: 0'])



from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

def compute_molecular_weights(smiles_list):
    molecular_weights = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            mw = Descriptors.MolWt(molecule)
            molecular_weights.append(mw)
        else:
            molecular_weights.append(np.nan)  # Use NaN for invalid SMILES
    return np.array(molecular_weights)

smiles_list = df_dg['SMILES'].to_list()
molecular_weights = compute_molecular_weights(smiles_list)


# Step 1: load data
smiles_list = df_dg['SMILES'].to_list()
values1 = -1*torch.tensor(df_dg['Gsolv'].values, dtype=torch.float32).reshape(-1,1)
F = 96485
values2 = torch.tensor(((df_e0['E'].values+0.76)*2*F)/ (3.600*molecular_weights), dtype=torch.float32).reshape(-1,1)



values = torch.cat([values1,values2], dim=1)



# Step 1: load data
smiles_list = df_dg['SMILES'].to_list()
values1 = -1*torch.tensor(df_dg['Gsolv'].values, dtype=torch.float32).reshape(-1,1)
F = 96485
values2 = torch.tensor(((df_e0['E'].values+0.76)*2*F)/ (3.600*molecular_weights), dtype=torch.float32).reshape(-1,1)



values = torch.cat([values1,values2], dim=1)
hv = values1* values2




moldais_moo = []
sgp_moo  = []
rnd_moo  = []
tmfp_moo = []



for seed in range(10 ):
    moldais_moo.append( torch.load(f'./mic_res/results/MOO_E_sparsity-CMI10_model-GP_acq-EI_iter_90_seed-{seed}.pkl')['best_values'])
moldais = np.array(moldais_moo)
moldais_m = moldais.mean(axis=0)
moldais_s = moldais.std(axis=0)



#for seed in range(10):
#    sgp_moo.append(torch.load(f'./mic_res/results/MOO_E_sparsity-MI10000_model-GP_acq-EI_iter_40_seed-{seed}.pkl')['best_values'])
#sgp = np.array(sgp_moo)
#sgp_m = sgp.mean(axis=0)
#sgp_s = sgp.std(axis=0)



for seed in range(10 ):
    Yi = torch.load(f'./mic_res/results/MOO_E_sparsity-MI10_model-GP_acq-Random_iter_0_seed-{seed}.pkl')['y']
    Y = Yi.prod(1)
    ybest = [Y[:k].max().item() for k in range(10,100)]
    rnd_moo.append(ybest)
rnd = np.array(rnd_moo)
rnd_m = rnd.mean(axis=0)
rnd_s = rnd.std(axis=0)



num_list = []
df_g = pd.read_csv('./New_gauche_MOO_(0_9).csv')['y1']
for i in range(10 ):
    str_list = df_g[i].replace('\n','').replace('tensor([','').replace('Tensor([','').replace('       ','').replace('],dtype=torch.float64)','').replace('], dtype=torch.float64)','').replace(')','').split(', ')
    num_list.append( np.array([float(x.replace(']', '')) for x in str_list]) )
gauche = np.array(num_list)[:, 11:]
gauche_m = gauche.mean(axis=0)
gauche_s = gauche.std(axis=0)










plt.figure(figsize=(12,9))


plt.plot([0,91], [np.log(hv.max()),np.log(hv.max())], 'k:'  )

plt.plot(np.arange(1,91), np.log(moldais_m), label='MolDAIS-CMI', color='tab:cyan')
plt.fill_between(np.arange(1,91),np.log(moldais_m+1.95*moldais_s/10**0.5),np.log(moldais_m-1.95*moldais_s/10**0.5), alpha=.3, color='tab:cyan')

#plt.plot(np.arange(1,41), sgp_m, label='SGP')
#plt.fill_between(np.arange(1,41),sgp_m+1.95*sgp_s/10**0.5,sgp_m-1.95*sgp_s/10**0.5, alpha=.3)


plt.plot(np.arange(1,91),np.log(gauche_m), label='TM-FP', color='tab:purple')
plt.fill_between(np.arange(1,91),np.log(gauche_m+1.95*gauche_s/10**0.5),np.log(gauche_m-1.95*gauche_s/10**0.5), alpha=.3, color='tab:purple')

plt.plot(np.arange(1,91), np.log(rnd_m), label='Random', color='tab:red')
plt.fill_between(np.arange(1,91),np.log(rnd_m+1.95*rnd_s/10**0.5),np.log(rnd_m-1.95*rnd_s/10**0.5), alpha=.3, color='tab:red')



#plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

#plt.yscale('log')
#plt.ylim(0)
plt.xlim(1,90)
plt.xlabel('Iteration')
plt.legend(fontsize=24, ncols = 3, loc=1)
plt.ylabel(r"$\ln(-\Delta G_{solv} \cdot E^o)$")
plt.tight_layout()
plt.savefig('./figs/moo_convergence.png', dpi=300)












def CQ(results, y):
    y_sorted = np.sort(y)
    quantiles = np.array([stats.percentileofscore(y_sorted, x) / 100 for x in results])
    return quantiles



seeds = [0,1,2,3,4,]#5,6,7,8,9]

df2 = pd.DataFrame()

col_name = r'Quantile scores of $m^* $'

end = -1
hv = hv.flatten()

d =  pd.DataFrame({'Method':['MolDAIS']*10, col_name:CQ(moldais[:,end],hv)})
df2 = pd.concat([df2, d])

#d =  pd.DataFrame({'Method':['SGP']*len(seeds), col_name:CQ(sgp[:,end],hv)})
#df2 = pd.concat([df2, d])


d =  pd.DataFrame({'Method':['Random']*(10), col_name:CQ(rnd[:,end],hv)})
df2 = pd.concat([df2, d])



d =  pd.DataFrame({'Method':['TM-FP']*(10), col_name:CQ(gauche[:,end],hv)})
df2 = pd.concat([df2, d])


C = [  'tab:cyan',
 'tab:orange',
 'tab:green',
 'tab:red',
 'tab:purple',

         ]

plt.figure(figsize=(13,9))
for i,ls in enumerate(range(4)):
  plt.axvline(i, color=C[i], linestyle='-', zorder=0, linewidth=184, alpha=.15)
#plt.axhline(1,color='k', linestyle=':',zorder=0)
#plt.axhline(.99,color='k', linestyle=':',zorder=0)
#plt.axhline(.98,color='k', linestyle=':',zorder=0)
plt.axhline(1,color='k', linestyle=':',zorder=0)

sns.violinplot(data=df2, x='Method', y=col_name, cut=0, inner="box", scale='width', palette=C, linewidth=3,linecolor='k')
#plt.title('Lipophilicity')
plt.xlabel(None)
#plt.ylim(.93,1.001)
#density_norm{“area”, “count”, “width”}
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('./figs/MOO_finalVals.png')









def find_non_dominated_points(values):
    # Sort values by the first objective (ascending order)
    sorted_indices = torch.argsort(values[:, 0], dim=0)
    sorted_values = values[sorted_indices]

    non_dominated = []
    current_best = float('inf')

    # Iterate through sorted points to find non-dominated points
    for point in sorted_values:
        if point[1].item() < current_best:
            non_dominated.append(point)
            current_best = point[1].item()

    return torch.stack(non_dominated)



non_dominated_points = find_non_dominated_points(values*-1)


for i in range(1):
    md_out =  torch.load(f'./mic_res/results/MOO_E_sparsity-CMI10_model-GP_acq-EI_iter_90_seed-{1}.pkl')
    tm_out =  pd.read_csv('./New_gauche_MOO_(1)_y.csv')
    rd_out =  torch.load(f'./mic_res/results/MOO_E_sparsity-MI10_model-GP_acq-Random_iter_0_seed-{1}.pkl')


    Y = md_out['y']
    Y3 = rd_out['y']
    Y2 = torch.tensor(tm_out[['y1', 'y2']].to_numpy())


    nd_md = find_non_dominated_points(Y*-1)
    nd_tm = find_non_dominated_points(Y2*-1)
    nd_rd = find_non_dominated_points(Y3*-1)



    plt.figure(figsize=(12,9))
    # Plot all points
    plt.step(nd_md[:, 0]*-1, nd_md[:, 1]*-1, color='tab:cyan', marker='s',label='MolDAIS Pareto Set', alpha=.7,zorder=100, linewidth=3, linestyle='-',where='post')
    plt.step(nd_tm[:, 0]*-1, nd_tm[:, 1]*-1, color='tab:purple', marker='o',label='TM-FP Pareto Set', alpha=.9,zorder=100, linewidth=3, linestyle='-',where='post')
    plt.step(nd_rd[:, 0]*-1, nd_rd[:, 1]*-1, color='tab:red', marker='o',label='Random Pareto Set', alpha=.9,zorder=100, linewidth=3, linestyle='-',where='post')
    plt.step(non_dominated_points[:, 0]*-1, non_dominated_points[:, 1]*-1, color='k', marker='o',label='Pareto Set', alpha=.7,zorder=0, linewidth=3, linestyle='-',where='post')

    plt.scatter(values[:, 0], values[:, 1], color='k', marker='x',label='All Molecules', alpha=.15, zorder=0)
    plt.scatter(Y[:, 0], Y[:, 1], 100, color='tab:cyan',marker='o', alpha=.24 ,label='MolDAIS queries', zorder=10)
    plt.scatter(Y2[:, 0], Y2[:, 1],100, color='tab:purple',marker='x', alpha=.24 ,label='TM-FP queries', zorder=10)
    plt.scatter(Y3[:, 0], Y3[:, 1], 100, color='tab:red',marker='+', alpha=.24 ,label='Random queries', zorder=10)

    #plt.scatter(non_dominated_points[:, 0]*-1, non_dominated_points[:, 1]*-1, s=300,c=(non_dominated_points.prod(1)*-1).tolist(), marker='o',alpha=.5,zorder=0, )

    # Annotate and show the plot
    plt.xlabel(r'$-\Delta G_{solv}$')
    plt.ylabel('$E^o$')
    plt.legend(fontsize=24, ncols=2)
    plt.grid(False)
    #plt.colorbar()
    plt.tight_layout()
    plt.xlim(0, 2100)
    plt.ylim(0, )
    plt.tight_layout()
    plt.savefig('./figs/moo_pareto.png', dpi=300)
    plt.show()


