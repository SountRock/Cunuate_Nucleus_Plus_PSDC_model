from netpyne import specs, sim
import json
import numpy as np
import statistics
from getStruct_5 import getPSDCOffStruct, getNormalStruct, getPSDCOffStruct_5_2, getNormalStruct_5_2
import matplotlib.pyplot as plt
import csv
import math
import os
from plotsPatterns import get_spike_times, plot_raster_dict, plot_raster_dict2
import glob
from plotsPatterns import psi_analysis_top, globalize_and_wct_plot_from_peaks, load_spike_data_from_checkpoint, transformDictToMatrix, filterAndRenumberDict, align_by_first_spike, save_raster_data_to_json, load_pop_spikes_from_json, plot_ISI_analysis, plot_phase_circles_groups, plot_phase_histogram_by_group, save_phase_histogram_data_to_json, plot_phase_histogram_from_json, save_phase_difference_data_to_json, plot_phase_difference_histogram_from_json, plot_kde_from_spike_groups, plot_spike_spectrogram, plot_spike_spectrogram2

def megaTargetFuncOnlySyn(savePath, 
                    ItoE_weight_SA, ItoE_w_min_SA, ItoE_smooth_SA, ItoE_x_half_SA,
                    InterPSDC_weight_SA, InterPSDC_w_min_SA, InterPSDC_smooth_SA, InterPSDC_x_half_SA,
                    PANtoE_weight_SA, PANtoE_w_min_SA, PANtoE_smooth_SA, PANtoE_x_half_SA,
                    PANtoI_weight_SA, PANtoI_w_min_SA, PANtoI_smooth_SA, PANtoI_x_half_SA,
                    tauExcProj_SA, tauExcInh_SA, tauExcPSDC_SA, tauPSDCtoProj_SA, tauPSDCtoInh_SA, tauInh_SA,
                    tau_recExcProj_SA, tau_recExcInh_SA, tau_recExcPSDC_SA, tau_recPSDCtoProj_SA, tau_recPSDCtoInh_SA, tau_recInh_SA,
                    UExcProj_SA, UExcInh_SA, UExcPSDC_SA, UPSDCtoProj_SA, UPSDCtoInh_SA, UInh_SA,
                    tau_facilExcProj_SA, tau_facilExcInh_SA, tau_facilExcPSDC_SA, tau_facilPSDCtoProj_SA, tau_facilPSDCtoInh_SA, tau_facilInh_SA,

                    ItoE_weight_RA, ItoE_w_min_RA, ItoE_smooth_RA, ItoE_x_half_RA,
                    InterPSDC_weight_RA, InterPSDC_w_min_RA, InterPSDC_smooth_RA, InterPSDC_x_half_RA,
                    PANtoE_weight_RA, PANtoE_w_min_RA, PANtoE_smooth_RA, PANtoE_x_half_RA,
                    PANtoI_weight_RA, PANtoI_w_min_RA, PANtoI_smooth_RA, PANtoI_x_half_RA,
                    tauExcProj_RA, tauExcInh_RA, tauExcPSDC_RA, tauPSDCtoProj_RA, tauPSDCtoInh_RA, tauInh_RA,
                    tau_recExcProj_RA, tau_recExcInh_RA, tau_recExcPSDC_RA, tau_recPSDCtoProj_RA, tau_recPSDCtoInh_RA, tau_recInh_RA,
                    UExcProj_RA, UExcInh_RA, UExcPSDC_RA, UPSDCtoProj_RA, UPSDCtoInh_RA, UInh_RA,
                    tau_facilExcProj_RA, tau_facilExcInh_RA, tau_facilExcPSDC_RA, tau_facilPSDCtoProj_RA, tau_facilPSDCtoInh_RA, tau_facilInh_RA                   
                    ):
    netParams = specs.NetParams() 
    netParams.ItoE_weight_SA = abs(ItoE_weight_SA) * 1 / 10
    netParams.InterPSDC_weight_SA = abs(InterPSDC_weight_SA) * 1 / 10
    netParams.PANtoE_weight_SA = abs(PANtoE_weight_SA) * 1 / 10
    netParams.PANtoI_weight_SA = abs(PANtoI_weight_SA) * 1 / 10

    netParams.ItoE_weight_RA = abs(ItoE_weight_RA) * 1 / 10
    netParams.InterPSDC_weight_RA = abs(InterPSDC_weight_RA) * 1 / 10
    netParams.PANtoE_weight_RA = abs(PANtoE_weight_RA) * 1 / 10
    netParams.PANtoI_weight_RA = abs(PANtoI_weight_RA) * 1 / 10

    ItoE_weight_SA = abs(ItoE_weight_SA) * 1 / 10
    InterPSDC_weight_SA = abs(InterPSDC_weight_SA) * 1 / 10
    PANtoE_weight_SA = abs(PANtoE_weight_SA) * 1 / 10
    PANtoI_weight_SA = abs(PANtoI_weight_SA) * 1 / 10

    ItoE_weight_RA = abs(ItoE_weight_RA) * 1 / 10
    InterPSDC_weight_RA = abs(InterPSDC_weight_RA) * 1 / 10
    PANtoE_weight_RA = abs(PANtoE_weight_RA) * 1 / 10
    PANtoI_weight_RA = abs(PANtoI_weight_RA) * 1 / 10

    ItoE_w_min_SA = abs(ItoE_w_min_SA) * 1 / 10
    InterPSDC_w_min_SA = abs(InterPSDC_w_min_SA) * 1 / 10
    PANtoE_w_min_SA = abs(PANtoE_w_min_SA) * 1 / 10
    PANtoI_w_min_SA= abs(PANtoI_w_min_SA) * 1 / 10

    ItoE_w_min_RA = abs(ItoE_w_min_RA) * 1 / 10
    InterPSDC_w_min_RA = abs(InterPSDC_w_min_RA) * 1 / 10
    PANtoE_w_min_RA = abs(PANtoE_w_min_RA) * 1 / 10
    PANtoI_w_min_RA= abs(PANtoI_w_min_RA) * 1 / 10

    ## Cell types -------------------------------------------------------------
    IzhiPSDC = {'secs': {}}
    IzhiPSDC['secs']['soma'] = {'geom': {}, 'pointps': {}}                       
    IzhiPSDC['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiPSDC['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': 5,
        'k': 1.5,
        'vr': -60,
        'vt': -45,
        'vpeak': 25,
        'a': 0.1,
        'b': 0.26,
        'c': -65,
        'd': 2,
        'celltype': 2
    } 

    IzhiExc = {'secs': {}}
    IzhiExc['secs']['soma'] = {'geom': {}, 'pointps': {}}                        
    IzhiExc['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiExc['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': 2,
        'k': 2.5,
        'vr': -60,
        'vt': -45,
        'vpeak': 25,
        'a': 0.1,
        'b': 0.26,
        'c': -65,
        'd': 2,
        'celltype': 2
    }

    IzhiInh = {'secs': {}}
    IzhiInh['secs']['soma'] = {'geom': {}, 'pointps': {}}                       
    IzhiInh ['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiInh['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': 5,
        'k': 1.5,
        'vr': -60,
        'vt': -45,
        'vpeak': 25,
        'a': 0.1,
        'b': 0.26,
        'c': -65,
        'd': 2,
        'celltype': 2
    }  

    netParams.cellParams['IzhiExc'] = IzhiExc
    netParams.cellParams['IzhiInh'] = IzhiInh
    netParams.cellParams['IzhiPSDC'] = IzhiPSDC 
    ## Cell types -------------------------------------------------------------           
    numNeurnons = 100
    persentPSDC = 0.4
    numPSDC = int(numNeurnons * persentPSDC)

    for i in range(1, numNeurnons + 1):
        netParams.popParams[f'ECell{i}'] = {'cellType': 'IzhiExc', 'numCells': 1}
        netParams.popParams[f'ICell{i}'] = {'cellType': 'IzhiInh', 'numCells': 1}

    #Создание PSDC слоев
    for i in range(1, numPSDC + 1):
            netParams.popParams[f'PSDCCell{i}'] = {'cellType': 'IzhiPSDC', 'numCells': 1}

    ## Tsodyks-Markram Pressets RA::::::::::::::::::::::::::::::::::::::::::::::::::::::
    netParams.synMechParams['InputToECell_RA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcProj_RA),           
        'tau_rec': abs(tau_recExcProj_RA),    
        'tau_facil': abs(tau_facilExcProj_RA),    
        'U': abs(UExcProj_RA),          
        'u0': 0.0,          
        'e': 0.0            
    }
    netParams.synMechParams['InputToICell_RA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcInh_RA),           
        'tau_rec': abs(tau_recExcInh_RA),    
        'tau_facil': abs(tau_facilExcInh_RA),    
        'U': abs(UExcInh_RA),          
        'u0': 0.0,          
        'e': 0.0            
    }
    netParams.synMechParams['InpuToPSDCCell_RA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcPSDC_RA),           
        'tau_rec': abs(tau_recExcPSDC_RA),    
        'tau_facil': abs(tau_facilExcPSDC_RA),     
        'U': abs(UExcPSDC_RA),         
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['PSDCCellToECell_RA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauPSDCtoProj_RA),           
        'tau_rec': abs(tau_recPSDCtoProj_RA),    
        'tau_facil': abs(tau_facilPSDCtoProj_RA),    
        'U': abs(UPSDCtoProj_RA),          
        'u0': 0.0,          
        'e': 0.0            
    }
    netParams.synMechParams['PSDCCellToICell_RA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauPSDCtoInh_RA),           
        'tau_rec': abs(tau_recPSDCtoInh_RA),    
        'tau_facil': abs(tau_facilPSDCtoInh_RA),     
        'U': abs(UPSDCtoInh_RA),          
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['ICellToECell_RA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauInh_RA),          
        'tau_rec': abs(tau_recInh_RA),    
        'tau_facil': abs(tau_facilInh_RA),    
        'U': abs(UInh_RA),        
        'u0': 0.0,         
        'e': -70             
    }

    ## Tsodyks-Markram Pressets SA::::::::::::::::::::::::::::::::::::::::::::::::::::::
    netParams.synMechParams['InputToECell_SA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcProj_SA),           
        'tau_rec': abs(tau_recExcProj_SA),    
        'tau_facil': abs(tau_facilExcProj_SA),    
        'U': abs(UExcProj_SA),          
        'u0': 0.0,          
        'e': 0.0            
    }
    netParams.synMechParams['InputToICell_SA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcInh_SA),           
        'tau_rec': abs(tau_recExcInh_SA),    
        'tau_facil': abs(tau_facilExcInh_SA),    
        'U': abs(UExcInh_SA),          
        'u0': 0.0,          
        'e': 0.0            
    }
    netParams.synMechParams['InpuToPSDCCell_SA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcPSDC_SA),           
        'tau_rec': abs(tau_recExcPSDC_SA),    
        'tau_facil': abs(tau_facilExcPSDC_SA),     
        'U': abs(UExcPSDC_SA),         
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['PSDCCellToECell_SA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauPSDCtoProj_SA),           
        'tau_rec': abs(tau_recPSDCtoProj_SA),    
        'tau_facil': abs(tau_facilPSDCtoProj_SA),    
        'U': abs(UPSDCtoProj_SA),          
        'u0': 0.0,          
        'e': 0.0            
    }
    netParams.synMechParams['PSDCCellToICell_SA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauPSDCtoInh_SA),           
        'tau_rec': abs(tau_recPSDCtoInh_SA),    
        'tau_facil': abs(tau_facilPSDCtoInh_SA),     
        'U': abs(UPSDCtoInh_SA),          
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['ICellToECell_SA'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauInh_SA),          
        'tau_rec': abs(tau_recInh_SA),    
        'tau_facil': abs(tau_facilInh_SA),    
        'U': abs(UInh_SA),        
        'u0': 0.0,         
        'e': -70             
    }
    ## Tsodyks-Markram Pressets ::::::::::::::::::::::::::::::::::::::::::::::::::::::
    connParamsAllPathActive = getNormalStruct_5_2(numNeurnons=numNeurnons, persentPSDC=persentPSDC, numRA=20, numCommons=40, RApersentPSDC=0.25,
                                              ItoE_weight_RA=ItoE_weight_RA, ItoE_w_min_RA=ItoE_w_min_RA, ItoE_smooth_RA=ItoE_smooth_RA, ItoE_x_half_RA=ItoE_x_half_RA,
                                              InterPSDC_weight_RA=InterPSDC_weight_RA, InterPSDC_w_min_RA=InterPSDC_w_min_RA, InterPSDC_smooth_RA=InterPSDC_smooth_RA, InterPSDC_x_half_RA=InterPSDC_x_half_RA, 
                                              PANtoE_weight_RA=PANtoE_weight_RA, PANtoE_w_min_RA=PANtoE_w_min_RA, PANtoE_smooth_RA=PANtoE_smooth_RA, PANtoE_x_half_RA=PANtoE_x_half_RA,  
                                              PANtoI_weight_RA=PANtoI_weight_RA, PANtoI_w_min_RA=PANtoI_w_min_RA, PANtoI_smooth_RA=PANtoI_smooth_RA, PANtoI_x_half_RA=PANtoI_x_half_RA,
                                              
                                              ItoE_weight_SA=ItoE_weight_SA, ItoE_w_min_SA=ItoE_w_min_SA, ItoE_smooth_SA=ItoE_smooth_SA, ItoE_x_half_SA=ItoE_x_half_SA,
                                              InterPSDC_weight_SA=InterPSDC_weight_SA, InterPSDC_w_min_SA=InterPSDC_w_min_SA, InterPSDC_smooth_SA=InterPSDC_smooth_SA, InterPSDC_x_half_SA=InterPSDC_x_half_SA, 
                                              PANtoE_weight_SA=PANtoE_weight_SA, PANtoE_w_min_SA=PANtoE_w_min_SA, PANtoE_smooth_SA=PANtoE_smooth_SA, PANtoE_x_half_SA=PANtoE_x_half_SA,  
                                              PANtoI_weight_SA=PANtoI_weight_SA, PANtoI_w_min_SA=PANtoI_w_min_SA, PANtoI_smooth_SA=PANtoI_smooth_SA, PANtoI_x_half_SA=PANtoI_x_half_SA,
                                              )
    
    #1KHz async test
    inputsFiles = glob.glob(os.path.join("inputs_hard_pattern", '*.pkl')) 
    for inputsFile in inputsFiles:
        (total_time, 
        regular_range, 
        irregular_range, 
        semiregular_range, 
        spike_times_start, 
        spike_times_middle, 
        spike_times_end) = load_spike_data_from_checkpoint(inputsFile)
        dictTimes = {
            "start" : transformDictToMatrix(spike_times_start),
            #"middle" : transformDictToMatrix(spike_times_middle),
            "end" : transformDictToMatrix(spike_times_end)
            }
    
        for key, value in dictTimes.items():
            temp = inputsFile.replace("\\", "").replace(".pkl", "")
            path = f'{savePath}_{temp}_{key}' 
            os.makedirs(path, exist_ok=True)     

            for neu in range(1, len(value) + 1): 
                netParams.popParams[f'stim{neu}'] = {'cellModel': 'VecStim', 'numCells': 1, 'spkTimes': value[neu-1]} 

            netParams.connParams = connParamsAllPathActive

            simConfig = specs.SimConfig()      
            simConfig.duration = total_time + 500             
            simConfig.dt = 0.025 
            simConfig.printPopAvgRates = [ 0, simConfig.duration ]  

            sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

            allPopRates = sim.allSimData['popRates']
            save_raster_data_to_json(
                sim,
                pop_groups=['ECell', 'ICell'],
                time_range=(0, total_time),
                json_filename=f'{path}/raster_data.json',
                all_pop_rates=allPopRates
            )

            popSpikes = load_pop_spikes_from_json(f'{path}/raster_data.json')

            ESpikes = filterAndRenumberDict(popSpikes, 'ECell')
            plot_ISI_analysis(f'{path}', ESpikes, total_time, 'ECellISI_analysis')
            ISpikes = filterAndRenumberDict(popSpikes, 'ICell')
            plot_ISI_analysis(f'{path}', ISpikes, total_time, 'ICellISI_analysis')

            ESpikes = transformDictToMatrix(ESpikes)
            ISpikes = transformDictToMatrix(ISpikes)

            ESpikes_align = align_by_first_spike(ESpikes, t_phase=0.0)
            ISpikes_align = align_by_first_spike(ESpikes, t_phase=0.0)
            value_align = align_by_first_spike(value, t_phase=0.0)

            save_phase_histogram_data_to_json(
                neuron_groups=[ESpikes_align, ISpikes_align],
                period=100,
                bins=64,
                group_names=['Proj', 'Input'],
                group_colors=['tab:green', 'tab:grey'],
                t_min_slice=400,
                output_json_path=f'{path}/hist_data.json'
            )
            plot_phase_histogram_from_json(f'{path}/hist_data.json', save_path=f'{path}/phase_hist_groups_Input_Output.png', save_individual=True)

            save_phase_difference_data_to_json(
                target_spike_trains=ESpikes_align,
                reference_spike_trains=value_align,
                period=100,
                bins=64,
                t_min_slice=400,
                output_json_path=f'{path}/hist_data_diff.json'
            )

            plot_phase_difference_histogram_from_json(f'{path}/hist_data_diff.json', f'{path}/phase_hist_diff_groups.png', vmax=140)
        
            plot_kde_from_spike_groups(
                spike_time_groups=[ESpikes_align, ISpikes_align],
                t_min=400,
                t_max=total_time,
                period=100,
                group_names=['Proj', 'Inh'],
                group_colors=['tab:green', 'tab:orange'],
                save_path=f'{path}/circular_kde.png'
            )
                
            plot_kde_from_spike_groups(
                spike_time_groups=[ESpikes_align, value_align],
                t_min=400,
                t_max=total_time,
                period=100,
                group_names=['Proj', 'Input'],
                group_colors=['tab:green', 'tab:grey'],
                save_path=f'{path}/circular_kdeInput_Output.png'
            )

            custom_colors = ['blue', 'green', 'red', 'orange', 'yellow']
            plot_spike_spectrogram2(
                ESpikes,
                bin_size=1.0,        
                window_size=100.0,   
                overlap=50.0,        
                fs=3000.0,           
                save_path=f'{path}/spectrogramProjCell.png',   
                z_clip=3.0,
                max_freq=600,
                on_filter=True,
                sigma=(0.5,0.5),
                cmap_colors=custom_colors      
            )

            plot_spike_spectrogram2(
                value,
                bin_size=1.0,        
                window_size=100.0,   
                overlap=50.0,        
                fs=3000.0,           
                save_path=f'{path}/spectrogramInputCell.png',   
                z_clip=3.0,  
                max_freq=600,
                on_filter=True,
                sigma=(0.5,0.5),   
                cmap_colors=custom_colors
            )  

            sigA, sigB, tt = globalize_and_wct_plot_from_peaks(
                A_peaks_ms=value,
                B_peaks_ms=ESpikes,
                dt_ms=0.025,
                freq_range=(3, 200),
                base=0.1,
                dj=1/48,
                save_path=f'{path}/WCT.png'
            )

            psi_static, psi_time, times, top_idx = psi_analysis_top(
                 A_spikes_ms=ESpikes, 
                 B_spikes_ms=value, 
                 A_spikes_label="Proj", B_spikes_label= "Input", 
                 dt_ms=0.025, 
                 fmin=3, fmax=200.0,
                 window_ms=500, step_ms=100.0,
                 top_n=10,
                 save_path=f'{path}/PSI')

with open("paramsRA_x.json", "r", encoding="utf-8") as f:
    init_params_RA = json.load(f)

with open("paramsSA_x.json", "r", encoding="utf-8") as f:
    init_params_SA = json.load(f)

path = f'111100_hard_pattern_vibro'
os.makedirs(path, exist_ok=True)
params = {**init_params_RA, **init_params_SA}

y_pred = megaTargetFuncOnlySyn(savePath=path, 
                    ItoE_weight_SA=params['ItoE_weight_SA'], ItoE_w_min_SA=params['ItoE_w_min_SA'], ItoE_smooth_SA=params['ItoE_smooth_SA'], ItoE_x_half_SA=params['ItoE_x_half_SA'], 
                    InterPSDC_weight_SA=params['InterPSDC_weight_SA'], InterPSDC_w_min_SA=params['InterPSDC_w_min_SA'], InterPSDC_smooth_SA=params['InterPSDC_smooth_SA'], InterPSDC_x_half_SA=params['InterPSDC_x_half_SA'], 
                    PANtoE_weight_SA=params['PANtoE_weight_SA'], PANtoE_w_min_SA=params['PANtoE_w_min_SA'], PANtoE_smooth_SA=params['PANtoE_smooth_SA'], PANtoE_x_half_SA=params['PANtoE_x_half_SA'],
                    PANtoI_weight_SA=params['PANtoI_weight_SA'], PANtoI_w_min_SA=params['PANtoI_w_min_SA'], PANtoI_smooth_SA=params['PANtoI_smooth_SA'], PANtoI_x_half_SA=params['PANtoI_x_half_SA'],
                    tauExcProj_SA=params['tauExcProj_SA'], tauExcInh_SA=params['tauExcInh_SA'], tauExcPSDC_SA=params['tauExcPSDC_SA'], tauPSDCtoProj_SA=params['tauPSDCtoProj_SA'], tauPSDCtoInh_SA=params['tauPSDCtoInh_SA'], tauInh_SA=params['tauInh_SA'],
                    tau_recExcProj_SA=params['tau_recExcProj_SA'], tau_recExcInh_SA=params['tau_recExcInh_SA'], tau_recExcPSDC_SA=params['tau_recExcPSDC_SA'], tau_recPSDCtoProj_SA=params['tau_recPSDCtoProj_SA'], tau_recPSDCtoInh_SA=params['tau_recPSDCtoInh_SA'], tau_recInh_SA=params['tau_recInh_SA'],
                    UExcProj_SA=params['UExcProj_SA'], UExcInh_SA=params['UExcInh_SA'], UExcPSDC_SA=params['UExcPSDC_SA'], UPSDCtoProj_SA=params['UPSDCtoProj_SA'], UPSDCtoInh_SA=params['UPSDCtoInh_SA'], UInh_SA=params['UInh_SA'],
                    tau_facilExcProj_SA=params['tau_facilExcProj_SA'], tau_facilExcInh_SA=params['tau_facilExcInh_SA'], tau_facilExcPSDC_SA=params['tau_facilExcPSDC_SA'], tau_facilPSDCtoProj_SA=params['tau_facilPSDCtoProj_SA'], tau_facilPSDCtoInh_SA=params['tau_facilPSDCtoInh_SA'], tau_facilInh_SA=params['tau_facilInh_SA'],
                    
                    ItoE_weight_RA=params['ItoE_weight_RA'], ItoE_w_min_RA=params['ItoE_w_min_RA'], ItoE_smooth_RA=params['ItoE_smooth_RA'], ItoE_x_half_RA=params['ItoE_x_half_RA'], 
                    InterPSDC_weight_RA=params['InterPSDC_weight_RA'], InterPSDC_w_min_RA=params['InterPSDC_w_min_RA'], InterPSDC_smooth_RA=params['InterPSDC_smooth_RA'], InterPSDC_x_half_RA=params['InterPSDC_x_half_RA'], 
                    PANtoE_weight_RA=params['PANtoE_weight_RA'], PANtoE_w_min_RA=params['PANtoE_w_min_RA'], PANtoE_smooth_RA=params['PANtoE_smooth_RA'], PANtoE_x_half_RA=params['PANtoE_x_half_RA'],
                    PANtoI_weight_RA=params['PANtoI_weight_RA'], PANtoI_w_min_RA=params['PANtoI_w_min_RA'], PANtoI_smooth_RA=params['PANtoI_smooth_RA'], PANtoI_x_half_RA=params['PANtoI_x_half_RA'],
                    tauExcProj_RA=params['tauExcProj_RA'], tauExcInh_RA=params['tauExcInh_RA'], tauExcPSDC_RA=params['tauExcPSDC_RA'], tauPSDCtoProj_RA=params['tauPSDCtoProj_RA'], tauPSDCtoInh_RA=params['tauPSDCtoInh_RA'], tauInh_RA=params['tauInh_RA'],
                    tau_recExcProj_RA=params['tau_recExcProj_RA'], tau_recExcInh_RA=params['tau_recExcInh_RA'], tau_recExcPSDC_RA=params['tau_recExcPSDC_RA'], tau_recPSDCtoProj_RA=params['tau_recPSDCtoProj_RA'], tau_recPSDCtoInh_RA=params['tau_recPSDCtoInh_RA'], tau_recInh_RA=params['tau_recInh_RA'],
                    UExcProj_RA=params['UExcProj_RA'], UExcInh_RA=params['UExcInh_RA'], UExcPSDC_RA=params['UExcPSDC_RA'], UPSDCtoProj_RA=params['UPSDCtoProj_RA'], UPSDCtoInh_RA=params['UPSDCtoInh_RA'], UInh_RA=params['UInh_RA'],
                    tau_facilExcProj_RA=params['tau_facilExcProj_RA'], tau_facilExcInh_RA=params['tau_facilExcInh_RA'], tau_facilExcPSDC_RA=params['tau_facilExcPSDC_RA'], tau_facilPSDCtoProj_RA=params['tau_facilPSDCtoProj_RA'], tau_facilPSDCtoInh_RA=params['tau_facilPSDCtoInh_RA'], tau_facilInh_RA=params['tau_facilInh_RA'],
                    )
    

