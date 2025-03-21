# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 03:49:16 2021

@author: xinxi
"""

import numpy as np
# import pandas as pd
import sys

# from .similarity import get_Jaccard_Similarity as getSim
# from utils.similarity import get_Gauss_Similarity as getSim
# from sklearn.metrics.pairwise import cosine_similarity as getSim
def readData(dataName, usedDataPath):
    def read_ZhangDDA(dataPrefix):
        names = {'A': 'drug',
                 'b': 'disease'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'association.txt',dtype=float,delimiter=' ',encoding='utf-8-sig')
        # drug
        dr_enzyme_sim=np.loadtxt(dataPrefix+'drug_1d_sim.txt',dtype=float,delimiter=' ',encoding='utf-8-sig')
        dr_target_sim=np.loadtxt(dataPrefix+'drug_3d_sim.txt',dtype=float,delimiter=' ',encoding='utf-8-sig')
        dr_struct_sim=np.loadtxt(dataPrefix+'drug_finger_sim.txt',dtype=float,delimiter=' ',encoding='utf-8-sig')
       #dr_pathwy_sim=np.loadtxt(dataPrefix+'pathway_sim.txt',dtype=float,delimiter=' ')
       #dr_intera_sim=np.loadtxt(dataPrefix+'drug_interaction_sim.txt',dtype=float,delimiter=' ')
        AAr=np.array([dr_enzyme_sim,dr_target_sim,dr_struct_sim])           
        ANet = {}
        # disease
        dis_sim=np.loadtxt(dataPrefix+'dis_sim.txt',dtype=float,delimiter=' ',encoding='utf-8-sig')
        bAr=np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    
    if dataName.startswith('ZhangDDA'):
        dataY, AAr, bAr, ANet, bNet, names = read_ZhangDDA(usedDataPath)
    else:
        print('error: no data named '+dataName)
        sys.exit()
    return dataY, AAr, bAr, ANet, bNet, names

