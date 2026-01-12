from netpyne import specs, sim
import json
import os
from neuron import h
import numpy as np
from collections import defaultdict

'''
Sigmoidal decay of weight from w_max to w_min.
    - x_pos  : scalar or array (distance: 0, 1, 2...)
    - w_max  : maximum weight
    - w_min  : minimum weight
    - smooth : controls slope ("smoothness") of decay
    - x_half : distance where weight ≈ (w_max + w_min) / 2
'''
def sigmoid_weight(x_pos, w_max, w_min, smooth, x_half):
    s = 1.0 / (1.0 + np.exp(smooth * (x_pos - x_half)))
    return w_min + (w_max - w_min) * s

#----------- Basic conns -----------
# Initial parameters of synaptic contacts in the Dorsal Column + PSDC layers
def getPSDCOffStruct(numNeurnons, ItoE_weight, PANtoE_weight, PANtoI_weight):
    connParams = {}
    for i in range(1, numNeurnons+1):
        connParams[f'STIM->ECell{i}'] = {
                'preConds': {'pop': 'stim'}, 
                'postConds': {'pop': f'ECell{i}'},  
                'probability': 1,         
                'weight': PANtoE_weight,          
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell',           
            }
        connParams[f'STIM->ICell{i}'] = {
                'preConds': {'pop': 'stim'}, 
                'postConds': {'pop': f'ICell{i}'},  
                'probability': 1,         
                'weight': PANtoI_weight,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
            }

    #EXC
    for i in range(1, numNeurnons+1):
        #INH
        for j in range(1, numNeurnons+1):
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': ItoE_weight,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell',
            }        

    return  connParams  

def getNormalStruct(numNeurnons, persentPSDC, ItoE_weight, InterPSDC_weight, PANtoE_weight, PANtoI_weight):
    numPSDC = int(numNeurnons * persentPSDC)
    connParams = {}

    for i in range(1, numNeurnons+1):
        connParams[f'STIM->ECell{i}'] = {
                'preConds': {'pop': 'stim'}, 
                'postConds': {'pop': f'ECell{i}'},  
                'probability': 1,         
                'weight': PANtoE_weight,          
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell',           
            }
        connParams[f'STIM->ICell{i}'] = {
                'preConds': {'pop': 'stim'}, 
                'postConds': {'pop': f'ICell{i}'},  
                'probability': 1,         
                'weight': PANtoI_weight,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
            }

    for i in range(1, numPSDC+1):
        connParams[f'STIM->PSDCCell{i}'] = {
                'preConds': {'pop': 'stim'}, 
                'postConds': {'pop': f'PSDCCell{i}'},  
                'probability': 1,         
                'weight': PANtoE_weight,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
            } 

    for i in range(1, numPSDC+1):
        for j in range(1, numNeurnons+1):
            connParams[f'PSDCCell{i}->ECell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ECell{j}'},  
                    'probability': 1,         
                    'weight': InterPSDC_weight,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToECell',            
                }
            
            connParams[f'PSDCCell{i}->ICell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ICell{j}'},  
                    'probability': 1,         
                    'weight': InterPSDC_weight,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToICell',            
                }

    #EXC
    for i in range(1, numNeurnons+1):
        #INH
        for j in range(1, numNeurnons+1):
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': ItoE_weight,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell',
            }   

    return  connParams  
#----------- Basic conns -----------

#----------- The strength of synaptic connections depends on the distance -----------
# To slightly diversify synaptic strength weights based on the distance to the target neuron. 
# At first glance, this seems like a weak factor, but its purpose is to prevent hypersynchronization in the model, 
# which can strongly distort results. 
# The easiest way is to systematically distribute synaptic strength, and this can be done depending on the 
# axon length to the target neuron. 
# We use a sigmoidal dependence to make it smoother, since the actual distribution of synaptic strengths in the 
# somatosensory cuneate nucleus is currently unknown. It's only clear that it is probably very minor or soft. 
# P.S. The model represents a slice of the somatosensory nucleus, a plate. So this should work fine.
def getPSDCOffStruct_3(numNeurnons, ItoE_weight, ItoE_w_min, ItoE_smooth, ItoE_x_half, 
                       PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half, 
                       PANtoI_weight, PANtoI_w_min, PANtoI_smooth, PANtoI_x_half):
    connParams = {}
    for i in range(1, numNeurnons+1):
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half*numNeurnons)
            connParams[f'STIM{i}->ECell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,         
                #'weight': PANtoE_weight,      
                'weight': w,        
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell',           
                }
            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoI_weight,PANtoI_w_min, PANtoI_smooth, PANtoI_x_half*numNeurnons) 
            connParams[f'STIM{i}->ICell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,         
                #'weight': PANtoI_weight,  
                'weight': w,            
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
                } 
    #EXC
    for i in range(1, numNeurnons+1):
        #INH
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, ItoE_weight, ItoE_w_min, ItoE_smooth, ItoE_x_half*numNeurnons) 
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': w,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell',
            }        

    return  connParams  

def getNormalStruct_3(numNeurnons, persentPSDC, 
                      ItoE_weight, ItoE_w_min, ItoE_smooth, ItoE_x_half, 
                      InterPSDC_weight, InterPSDC_w_min, InterPSDC_smooth, InterPSDC_x_half, 
                      PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half, 
                      PANtoI_weight, PANtoI_w_min, PANtoI_smooth, PANtoI_x_half
                      ):
    numPSDC = int(numNeurnons * persentPSDC)
    connParams = {}

    for i in range(1, numNeurnons+1):
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half*numNeurnons)
            connParams[f'STIM{i}->ECell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,         
                'weight': w,          
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell',           
                }
            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoI_weight,PANtoI_w_min, PANtoI_smooth, PANtoI_x_half*numNeurnons) 
            connParams[f'STIM{i}->ICell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,         
                'weight': w,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
                } 

    for i in range(1, numNeurnons+1):
        for j in range(1, numPSDC+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half*numPSDC) 
            connParams[f'STIM{i}->PSDCCell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'PSDCCell{j}'},  
                'probability': 1,         
                'weight': w,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
            }    

    for i in range(1, numPSDC+1):
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, InterPSDC_weight, InterPSDC_w_min, InterPSDC_smooth, InterPSDC_x_half*numNeurnons) 
            connParams[f'PSDCCell{i}->ECell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ECell{j}'},  
                    'probability': 1,         
                    'weight': w,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToECell',            
                }
            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, InterPSDC_weight, InterPSDC_w_min, InterPSDC_smooth, InterPSDC_x_half*numNeurnons) 
            connParams[f'PSDCCell{i}->ICell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ICell{j}'},  
                    'probability': 1,         
                    'weight': w,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToICell',            
                }

    #EXC
    for i in range(1, numNeurnons+1):
        #INH
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, ItoE_weight, ItoE_w_min, ItoE_smooth, ItoE_x_half*numNeurnons) 
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': w,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell',
            }     

    return  connParams  
#----------- The strength of synaptic connections depends on the distance -----------

#----------- The strength of synaptic connections depends on the distance + Proj to Inhs loop -----------
#P.S. not used in present
def getPSDCOffStruct_4(numNeurnons, ItoE_weight, ItoE_w_min, ItoE_smooth, ItoE_x_half, 
                       PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half, 
                       PANtoI_weight, PANtoI_w_min, PANtoI_smooth, PANtoI_x_half,
                       EtoI_weight, EtoI_w_min, EtoI_smooth, EtoI_x_half, delayEtoI
                       ):
    connParams = {}
    for i in range(1, numNeurnons+1):
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half*numNeurnons)
            connParams[f'STIM{i}->ECell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,         
                'weight': w,          
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell',           
                }
            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoI_weight,PANtoI_w_min, PANtoI_smooth, PANtoI_x_half*numNeurnons) 
            connParams[f'STIM{i}->ICell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,         
                'weight': w,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
                } 
    #EXC
    for i in range(1, numNeurnons+1):
        #INH
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, ItoE_weight, ItoE_w_min, ItoE_smooth, ItoE_x_half*numNeurnons) 
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': w,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell',
            }   

            if (i != j):
                x_pos = abs(i - j)
                w = sigmoid_weight(x_pos, EtoI_weight, EtoI_w_min, EtoI_smooth, EtoI_x_half*numNeurnons)  
                connParams[f'ECell{j}->ICell{i}'] = {
                    'preConds': {'pop': f'ECell{j}'}, 
                    'postConds': {'pop': f'ICell{i}'}, 
                    'probability': 1,         
                    'weight': w,   
                    'delay': delayEtoI,                
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'ECellToICell',
                }        

    return  connParams  

def getNormalStruct_4(numNeurnons, persentPSDC, 
                      ItoE_weight, ItoE_w_min, ItoE_smooth, ItoE_x_half, 
                      InterPSDC_weight, InterPSDC_w_min, InterPSDC_smooth, InterPSDC_x_half, 
                      PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half, 
                      PANtoI_weight, PANtoI_w_min, PANtoI_smooth, PANtoI_x_half,
                      EtoI_weight, EtoI_w_min, EtoI_smooth, EtoI_x_half, delayEtoI
                      ):
    numPSDC = int(numNeurnons * persentPSDC)
    connParams = {}

    for i in range(1, numNeurnons+1):
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half*numNeurnons)
            connParams[f'STIM{i}->ECell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,         
                'weight':  w,          
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell',           
                }
            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoI_weight,PANtoI_w_min, PANtoI_smooth, PANtoI_x_half*numNeurnons) 
            connParams[f'STIM{i}->ICell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,         
                'weight':  w,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
                } 

    for i in range(1, numNeurnons+1):
        for j in range(1, numPSDC+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight, PANtoE_w_min, PANtoE_smooth, PANtoE_x_half*numPSDC) 
            connParams[f'STIM{i}->PSDCCell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'PSDCCell{j}'},  
                'probability': 1,         
                'weight': w,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell',           
            }    

    for i in range(1, numPSDC+1):
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, InterPSDC_weight, InterPSDC_w_min, InterPSDC_smooth, InterPSDC_x_half*numNeurnons) 
            connParams[f'PSDCCell{i}->ECell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ECell{j}'},  
                    'probability': 1,         
                    'weight': w,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToECell',            
                }
            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, InterPSDC_weight, InterPSDC_w_min, InterPSDC_smooth, InterPSDC_x_half*numNeurnons) 
            connParams[f'PSDCCell{i}->ICell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ICell{j}'},  
                    'probability': 1,         
                    'weight': w,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToICell',            
                }

    #EXC
    for i in range(1, numNeurnons+1):
        #INH
        for j in range(1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, ItoE_weight, ItoE_w_min, ItoE_smooth, ItoE_x_half*numNeurnons) 
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': w,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell',
            }   

            if (i != j):
                x_pos = abs(i - j)
                w = sigmoid_weight(x_pos, EtoI_weight, EtoI_w_min, EtoI_smooth, EtoI_x_half*numNeurnons)  
                connParams[f'ECell{j}->ICell{i}'] = {
                    'preConds': {'pop': f'ECell{j}'}, 
                    'postConds': {'pop': f'ICell{i}'}, 
                    'probability': 1,         
                    'weight': w,   
                    'delay': delayEtoI,                
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'ECellToICell',
                }   

    return  connParams  
#----------- The strength of synaptic connections depends on the distance + Proj to Inhs loop -----------

#----------- With the somatosensory nucleus divided into a sector that only receives responses from RA and SA mechanoreceptors -----------
#It is recommended to read the article https://doi.org/10.1038/s41593-024-01821-1 
def getPSDCOffStruct_5_2(numNeurnons, numRA, numCommons, 
                         ItoE_weight_RA, ItoE_w_min_RA, ItoE_smooth_RA, ItoE_x_half_RA, 
                         PANtoE_weight_RA, PANtoE_w_min_RA, PANtoE_smooth_RA, PANtoE_x_half_RA, 
                         PANtoI_weight_RA, PANtoI_w_min_RA, PANtoI_smooth_RA, PANtoI_x_half_RA,

                         ItoE_weight_SA, ItoE_w_min_SA, ItoE_smooth_SA, ItoE_x_half_SA, 
                         PANtoE_weight_SA, PANtoE_w_min_SA, PANtoE_smooth_SA, PANtoE_x_half_SA, 
                         PANtoI_weight_SA, PANtoI_w_min_SA, PANtoI_smooth_SA, PANtoI_x_half_SA):
    
    #Inputs RA
    RAzone = numRA+numCommons
    connParams = {}
    for i in range(1, numRA+1):
        #RA climb into the SA zone
        for j in range(1, numRA+numCommons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight_RA, PANtoE_w_min_RA, PANtoE_smooth_RA, PANtoE_x_half_RA*RAzone)
            connParams[f'STIM{i}->ECell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,          
                'weight': w,        
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell_RA',           
                }
            
        #RA Ihs are only stimulated by the RA part
        for j in range(1, numRA+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoI_weight_RA, PANtoI_w_min_RA, PANtoI_smooth_RA, PANtoI_x_half_RA*RAzone) 
            connParams[f'STIM{i}->ICell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,         
                'weight': w,            
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell_RA',           
                } 
              
    #Inputs SA
    for i in range(numRA+1, numNeurnons+1):
        #SA here are the incentives that stimulate the SA region
        for j in range(numRA+1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight_SA, PANtoE_w_min_SA, PANtoE_smooth_SA, PANtoE_x_half_SA*numNeurnons)
            connParams[f'STIM{i}->ECell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,             
                'weight': w,        
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell_SA',           
                }
            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoI_weight_SA, PANtoI_w_min_SA, PANtoI_smooth_SA, PANtoI_x_half_SA*numNeurnons) 
            connParams[f'STIM{i}->ICell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,          
                'weight': w,            
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell_SA',           
                }

    #Inhs RA
    #EXC
    for i in range(1, numRA+numCommons+1):
    #for i in range(1, numRA+1):
        #INH
        for j in range(1, numRA+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, ItoE_weight_RA, ItoE_w_min_RA, ItoE_smooth_RA, ItoE_x_half_RA*RAzone) 
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': w,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell_RA',
            } 

    #Inhs SA
    #EXC
    for i in range(numRA+1, numNeurnons+1):
        #INH
        for j in range(numRA+1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, ItoE_weight_SA, ItoE_w_min_SA, ItoE_smooth_SA, ItoE_x_half_SA*numNeurnons) 
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': w,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell_SA',
            }        
       

    return  connParams  

def getNormalStruct_5_2(numNeurnons, numRA, numCommons, persentPSDC, RApersentPSDC,
                      ItoE_weight_RA, ItoE_w_min_RA, ItoE_smooth_RA, ItoE_x_half_RA, 
                      InterPSDC_weight_RA, InterPSDC_w_min_RA, InterPSDC_smooth_RA, InterPSDC_x_half_RA, 
                      PANtoE_weight_RA, PANtoE_w_min_RA, PANtoE_smooth_RA, PANtoE_x_half_RA, 
                      PANtoI_weight_RA, PANtoI_w_min_RA, PANtoI_smooth_RA, PANtoI_x_half_RA,

                      ItoE_weight_SA, ItoE_w_min_SA, ItoE_smooth_SA, ItoE_x_half_SA, 
                      InterPSDC_weight_SA, InterPSDC_w_min_SA, InterPSDC_smooth_SA, InterPSDC_x_half_SA, 
                      PANtoE_weight_SA, PANtoE_w_min_SA, PANtoE_smooth_SA, PANtoE_x_half_SA, 
                      PANtoI_weight_SA, PANtoI_w_min_SA, PANtoI_smooth_SA, PANtoI_x_half_SA
                      ):
    numPSDC = int(numNeurnons * persentPSDC)
    numRA_PSDC =  int(numPSDC * RApersentPSDC)
    numRA_PSDC = 1 if numRA_PSDC < 1 else numRA_PSDC
    connParams = {}
    RAzone = numRA+numCommons

    #Inputs RA
    RAzone = numRA+numCommons
    connParams = {}
    for i in range(1, numRA+1):
        for j in range(1, numRA+numCommons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight_RA, PANtoE_w_min_RA, PANtoE_smooth_RA, PANtoE_x_half_RA*RAzone)
            connParams[f'STIM{i}->ECell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,          
                'weight': w,        
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell_RA',           
                }
            
        for j in range(1, numRA+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoI_weight_RA, PANtoI_w_min_RA, PANtoI_smooth_RA, PANtoI_x_half_RA*RAzone) 
            connParams[f'STIM{i}->ICell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,         
                'weight': w,            
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell_RA',           
                } 
              
    #Inputs SA
    for i in range(numRA+1, numNeurnons+1):
        for j in range(numRA+1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight_SA, PANtoE_w_min_SA, PANtoE_smooth_SA, PANtoE_x_half_SA*numNeurnons)
            connParams[f'STIM{i}->ECell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,             
                'weight': w,        
                'delay': 0,                
                'sec': 'dend',             
                'loc': 1.0,                 
                'synMech': 'InputToECell_SA',           
                }
            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoI_weight_SA, PANtoI_w_min_SA, PANtoI_smooth_SA, PANtoI_x_half_SA*numNeurnons) 
            connParams[f'STIM{i}->ICell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,          
                'weight': w,            
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell_SA',           
                }
            
    #Srim PSDC RA
    for i in range(1, numRA+1):
        for j in range(1, numRA_PSDC+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight_RA, PANtoE_w_min_RA, PANtoE_smooth_RA, PANtoE_x_half_RA*numPSDC) 
            connParams[f'STIM{i}->PSDCCell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'PSDCCell{j}'},  
                'probability': 1,         
                'weight': w,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell_RA',           
            }   

    #Srim PSDC SA
    for i in range(numRA+1, numNeurnons+1):
        for j in range(numRA_PSDC+1, numPSDC+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, PANtoE_weight_SA, PANtoE_w_min_SA, PANtoE_smooth_SA, PANtoE_x_half_SA*numPSDC) 
            connParams[f'STIM{i}->PSDCCell{j}'] = {
                'preConds': {'pop': f'stim{i}'}, 
                'postConds': {'pop': f'PSDCCell{j}'},  
                'probability': 1,         
                'weight': w,         
                'delay': 0,                
                'sec': 'dend',            
                'loc': 1.0,                
                'synMech': 'InputToICell_SA',           
            }     

    #PSDC -> Proj, Inh RA
    for i in range(1, numRA_PSDC+1):
        for j in range(1, numRA+numCommons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, InterPSDC_weight_RA, InterPSDC_w_min_RA, InterPSDC_smooth_RA, InterPSDC_x_half_RA*RAzone) 
            connParams[f'PSDCCell{i}->ECell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ECell{j}'},  
                    'probability': 1,         
                    'weight': w,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToECell_RA',            
                }
            
        for j in range(1, numRA+1):            
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, InterPSDC_weight_RA, InterPSDC_w_min_RA, InterPSDC_smooth_RA, InterPSDC_x_half_RA*RAzone) 
            connParams[f'PSDCCell{i}->ICell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ICell{j}'},  
                    'probability': 1,         
                    'weight': w,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToICell_RA',            
                }
            
    #PSDC -> Proj, Inh SA
    for i in range(numRA_PSDC+1, numPSDC+1):
        for j in range(numRA+1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, InterPSDC_weight_SA, InterPSDC_w_min_SA, InterPSDC_smooth_SA, InterPSDC_x_half_SA*numNeurnons) 
            connParams[f'PSDCCell{i}->ECell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ECell{j}'},  
                    'probability': 1,         
                    'weight': w,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToECell_SA',            
                }
       
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, InterPSDC_weight_SA, InterPSDC_w_min_SA, InterPSDC_smooth_SA, InterPSDC_x_half_SA*numNeurnons) 
            connParams[f'PSDCCell{i}->ICell{j}'] = {
                    'preConds': {'pop': f'PSDCCell{i}'}, 
                    'postConds': {'pop': f'ICell{j}'},  
                    'probability': 1,         
                    'weight': w,         
                    'delay': 0,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'PSDCCellToICell_SA',            
                }

    #Inhs RA
    #EXC
    for i in range(1, numRA+numCommons+1):
    #for i in range(1, numRA+1):
        #INH
        for j in range(1, numRA+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, ItoE_weight_RA, ItoE_w_min_RA, ItoE_smooth_RA, ItoE_x_half_RA*RAzone) 
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': w,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell_RA',
            } 

    #Inhs SA
    #EXC
    for i in range(numRA+1, numNeurnons+1):
        #INH
        for j in range(numRA+1, numNeurnons+1):
            x_pos = abs(i - j)
            w = sigmoid_weight(x_pos, ItoE_weight_SA, ItoE_w_min_SA, ItoE_smooth_SA, ItoE_x_half_SA*numNeurnons) 
            connParams[f'ICell{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': w,   
                'delay': 2,                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': 'ICellToECell_SA',
            }                

    return  connParams  