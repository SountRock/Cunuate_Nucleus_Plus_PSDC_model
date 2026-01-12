'''
Sample of optimization script
'''

from optim_prot import run_nsga3_optimization_threads, corr_loss
from netpyne import specs, sim
import json
import numpy as np
from getStruct import getPSDCOffStruct_5_2, getNormalStruct_5_2
from analyzers.plotsPatterns import get_spike_times

with open('inputs/others_patterns/SCS_sync_input_50Hz.json', 'rb') as sync_spkt: spikes_50_sync = json.load(sync_spkt) 
with open('inputs/others_patterns/SCS_async_input_1kHz_center.json', 'rb') as sync_spkt: spikes_1kHz_async_center = json.load(sync_spkt) 
with open('inputs/others_patterns/SCS_async_input_1kHz_low.json', 'rb') as sync_spkt: spikes_1kHz_async_low = json.load(sync_spkt) 
with open('inputs/others_patterns/SCS_async_input_1kHz_hight.json', 'rb') as sync_spkt: spikes_1kHz_async_hight = json.load(sync_spkt)

def Vm_dyn_score(Vm, baseline_ref=55.0, Vm_min=65.0):
    Vm = np.asarray(Vm, dtype=float)

    baseline = np.mean(Vm)
    sign_penalty = (0 if baseline < 0 else 500)
    abs_baseline = abs(baseline)
    nan_penalty = (abs_baseline if np.isfinite(baseline) else 800)
    baseline_penalty = abs(nan_penalty + sign_penalty - baseline_ref)

    total_score = 1 * baseline_penalty + 0.5 * abs(abs(np.min(Vm)) - Vm_min) 
    return total_score

'''
The func generating feature that we will optimize
'''
def generatorFunc(t_matrix, 
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
    netParams.ItoE_weight_SA = abs(ItoE_weight_SA)
    netParams.InterPSDC_weight_SA = abs(InterPSDC_weight_SA)
    netParams.PANtoE_weight_SA = abs(PANtoE_weight_SA)
    netParams.PANtoI_weight_SA = abs(PANtoI_weight_SA)

    netParams.ItoE_weight_RA = abs(ItoE_weight_RA)
    netParams.InterPSDC_weight_RA = abs(InterPSDC_weight_RA)
    netParams.PANtoE_weight_RA = abs(PANtoE_weight_RA)
    netParams.PANtoI_weight_RA = abs(PANtoI_weight_RA)

    ## Cell types -------------------------------------------------------------
    # Post Syncaptic Dorsal Column neurons, Tonic type 
    IzhiPSDC = {'secs': {}}
    IzhiPSDC['secs']['soma'] = {'geom': {}, 'pointps': {}}                       
    IzhiPSDC['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiPSDC['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007',
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
    
    # Projection neurons, Burst type 
    IzhiExc = {'secs': {}}
    IzhiExc['secs']['soma'] = {'geom': {}, 'pointps': {}}                        
    IzhiExc['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiExc['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007',
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

    # Inhs neurons, Tonic type 
    IzhiInh = {'secs': {}}
    IzhiInh['secs']['soma'] = {'geom': {}, 'pointps': {}}                       
    IzhiInh ['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiInh['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007',
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
    persentPSDC = 0.4 #40% of A-alpha afferents project to the PSDC layer
    numPSDC = int(numNeurnons * persentPSDC)

    for i in range(1, numNeurnons + 1):
        netParams.popParams[f'ECell{i}'] = {'cellType': 'IzhiExc', 'numCells': 1}
        netParams.popParams[f'ICell{i}'] = {'cellType': 'IzhiInh', 'numCells': 1}

    # Creating Inh and Proj layers
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
    # Template with PSDC
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
    # Template without PSDC
    connParamsPSDCoff = getPSDCOffStruct_5_2(numNeurnons=numNeurnons, numRA=20, numCommons=40, 
                                              ItoE_weight_RA=ItoE_weight_RA, ItoE_w_min_RA=ItoE_w_min_RA, ItoE_smooth_RA=ItoE_smooth_RA, ItoE_x_half_RA=ItoE_x_half_RA,
                                              PANtoE_weight_RA=PANtoE_weight_RA, PANtoE_w_min_RA=PANtoE_w_min_RA, PANtoE_smooth_RA=PANtoE_smooth_RA, PANtoE_x_half_RA=PANtoE_x_half_RA,  
                                              PANtoI_weight_RA=PANtoI_weight_RA, PANtoI_w_min_RA=PANtoI_w_min_RA, PANtoI_smooth_RA=PANtoI_smooth_RA, PANtoI_x_half_RA=PANtoI_x_half_RA,

                                              ItoE_weight_SA=ItoE_weight_SA, ItoE_w_min_SA=ItoE_w_min_SA, ItoE_smooth_SA=ItoE_smooth_SA, ItoE_x_half_SA=ItoE_x_half_SA,
                                              PANtoE_weight_SA=PANtoE_weight_SA, PANtoE_w_min_SA=PANtoE_w_min_SA, PANtoE_smooth_SA=PANtoE_smooth_SA, PANtoE_x_half_SA=PANtoE_x_half_SA,  
                                              PANtoI_weight_SA=PANtoI_weight_SA, PANtoI_w_min_SA=PANtoI_w_min_SA, PANtoI_smooth_SA=PANtoI_smooth_SA, PANtoI_x_half_SA=PANtoI_x_half_SA
                                              )

    # ------------------- Simulation with PSDC layers -------------------
    netParams.connParams = connParamsAllPathActive
    
    # They are only needed for calculating and monitoring the basic dynamics, in this case
    scoreAllPathActive = []
    for s, f in enumerate(t_matrix[1]):
        if (s == 1 or s > 3):
            continue
        fRA = f['RA']
        fSA = f['SA']
        for neu in range(1, 3): 
            netParams.popParams[f'stim{neu}'] = {'cellModel': 'VecStim', 'numCells': 1, 'spkTimes': fRA[neu-1]} 
        for neu in range(3, numNeurnons): 
            netParams.popParams[f'stim{neu}'] = {'cellModel': 'VecStim', 'numCells': 1, 'spkTimes': fSA[neu-1]} 

        simConfig = specs.SimConfig()      
        simConfig.duration = 600         
        simConfig.dt = 0.025 
        simConfig.printPopAvgRates = [ 0, simConfig.duration ]  

        bool_temp = s == 2 or s == 3
        if bool_temp:
            simConfig.recordTraces = {
                'V_soma':{'sec':'soma','loc':0.5,'var':'v'}
                }
            simConfig.recordStep = 0.025 
            simConfig.analysis['plotTraces'] = {'include': ['ECell5']} 

        sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

        if bool_temp:
            score_temp = getBaseDynamic(sim, simConfig.duration)
            scoreAllPathActive.append(score_temp)
            simConfig.recordTraces = None
            simConfig.recordStep = None
            simConfig.analysis['plotTraces'] = None

    # Pressure control
    maxISIdata = []
    commons_control = []
    ys_pressure = []
    for s, f in enumerate(t_matrix[1]):
        fRA = f['RA']
        fSA = f['SA']
        for neu in range(1, 3): 
            netParams.popParams[f'stim{neu}'] = {'cellModel': 'VecStim', 'numCells': 1, 'spkTimes': fRA[neu-1]} 
        for neu in range(3, numNeurnons): 
            netParams.popParams[f'stim{neu}'] = {'cellModel': 'VecStim', 'numCells': 1, 'spkTimes': fSA[neu-1]} 

        #SIM_1============================================================
        netParams.connParams = connParamsPSDCoff
          
        simConfig = specs.SimConfig()      
        simConfig.duration = 400          
        simConfig.dt = 0.025 
        simConfig.printPopAvgRates = [ 0, 350 ]  

        sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

        # Control of preserving temporal dynamics in the common zone.
        # Exact numerical matching is not important;
        # what matters is that in the initial phase there are more spikes than in the final phase.
        spikes_times_left_commons = get_spike_times(sim, 'ECell4', time_range=(350, 400))
        spikes_times_center_commons = get_spike_times(sim, 'ECell4', time_range=(200, 300))
        try: 
            spikes_left_commons = list(spikes_times_left_commons.values())[0]
            spikes_left_commons = sorted([x for x in spikes_left_commons if x < 400 and x > 350])

            spikes_center_commons = list(spikes_times_center_commons.values())[0]
            spikes_center_commons = sorted([x for x in spikes_center_commons if x < 400 and x > 350])

            control = len(spikes_left_commons) - len(spikes_center_commons)
            commons_control.append(control)
        except Exception as e:
            commons_control.append(100)

        # Controlling the relief of an empty spot in raster dynamics
        try: 
            spikes_times = get_spike_times(sim, 'ECell4', time_range=(50, 350))
            spikes_times = list(spikes_times.values())
            spikes_times = spikes_times[0]
            isi_spikes = np.diff(spikes_times)
            maxISIdata.append(np.max(isi_spikes)) 
        except Exception as e:
            penalty = 1000
            maxISIdata.append(penalty) 

        allPopRates_off = sim.allSimData['popRates']

        center = numNeurnons / 2
        low = int(center - 1)
        hight = int(center + 1)
        freqs_off = []
        for l in range(low, hight+1):
            freqs_off.append(allPopRates_off[f'ECell{l}'])

        output_freq_psdc_off = np.max(freqs_off)

        #SIM_2============================================================
        netParams.connParams = connParamsAllPathActive
          
        simConfig = specs.SimConfig()      
        simConfig.duration = 400          
        simConfig.dt = 0.025 
        simConfig.printPopAvgRates = [ 0, 350 ]  

        sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

        center = numNeurnons / 2
        low = int(center - 1)
        hight = int(center + 1)
        allPopRates_on = sim.allSimData['popRates']
        freqs_on = []
        for l in range(low, hight+1):
            freqs_on.append(allPopRates_on[f'ECell{l}'])

        output_freq_psdc_on = np.max(freqs_on)

        #Diifusion Pressure. We control how much the frequency of spikes is when the PSDC 
        # is disabled compared to the frequency when all paths are active
        y_pressure = output_freq_psdc_off / output_freq_psdc_on
        if not np.isfinite(y_pressure):
            y_pressure = 10
        else: 
            y_pressure = y_pressure
        ys_pressure.append(y_pressure)

    y_final = [
        scoreAllPathActive, 
        maxISIdata, 
        ys_pressure,
        commons_control
    ]

    return  y_final

'''
Function for controlling the basic dynamics of membrane potential. 
'''
def getBaseDynamic(sim, duration):
    data = sim.simData['V_soma']  

    Vm = np.array(data['cell_8'])
    t = np.linspace(0, duration, len(Vm))

    mask = (t >= 340) & (t <= 400)
    t_interval = t[mask]
    Vm_interval = Vm[mask]

    n = min(t_interval.shape[0], Vm_interval.shape[0])
    t = t_interval[:n]
    Vm = Vm_interval[:n]

    return Vm_dyn_score(Vm, baseline_ref=55.0, Vm_min=65.0)

#--------------------------------------------
# To change the parameters within the appropriate limits
param_bounds = {
    'ItoE_weight_RA': (0.005, 0.1), 
    'ItoE_w_min_RA': (0.001, 0.05), 
    'ItoE_smooth_RA': (0.5, 3.0),
    'ItoE_x_half_RA': (0.1, 0.9),

    'InterPSDC_weight_RA': (0.005, 0.1), 
    'InterPSDC_w_min_RA': (0.001, 0.05), 
    'InterPSDC_smooth_RA': (0.5, 3.0),
    'InterPSDC_x_half_RA': (0.1, 0.9),

    'PANtoE_weight_RA': (0.005, 0.1),
    'PANtoE_w_min_RA': (0.001, 0.05), 
    'PANtoE_smooth_RA': (0.5, 3.0),
    'PANtoE_x_half_RA': (0.1, 0.9),

    'PANtoI_weight_RA': (0.005, 0.1),
    'PANtoI_w_min_RA': (0.001, 0.05), 
    'PANtoI_smooth_RA': (0.5, 3.0),
    'PANtoI_x_half_RA': (0.1, 0.9),

    'tauExcProj_RA': (2.0, 1000.0),
    'tauExcInh_RA': (2.0, 1000.0),
    'tauExcPSDC_RA': (2.0, 1000.0),
    'tauPSDCtoProj_RA': (2.0, 1000.0),
    'tauPSDCtoInh_RA': (2.0, 1000.0),
    'tauInh_RA': (2.0, 1000.0),

    'tau_recExcProj_RA': (2.0, 1000.0),
    'tau_recExcInh_RA': (2.0, 1000.0),
    'tau_recExcPSDC_RA': (2, 1000.0),
    'tau_recPSDCtoProj_RA': (2, 1000.0),
    'tau_recPSDCtoInh_RA': (2, 1000.0),
    'tau_recInh_RA': (2, 1000.0),

    'UExcProj_RA': (0.05, 1.0),
    'UExcInh_RA': (0.05, 1.0),
    'UExcPSDC_RA': (0.05, 1.0),
    'UPSDCtoProj_RA': (0.05, 1.0),
    'UPSDCtoInh_RA': (0.05, 1.0),
    'UInh_RA': (0.05, 1.0),

    'tau_facilExcProj_RA': (2.0, 1000.0),
    'tau_facilExcInh_RA': (2.0, 1000.0),
    'tau_facilExcPSDC_RA': (2, 1000.0),
    'tau_facilPSDCtoProj_RA': (2, 1000.0),
    'tau_facilPSDCtoInh_RA': (2, 1000.0),
    'tau_facilInh_RA': (2, 1000.0),

    #=========================================
    'ItoE_weight_SA': (0.003, 0.01), 
    'ItoE_w_min_SA': (0.001, 0.005), 
    'ItoE_smooth_SA': (0.5, 3.0),
    'ItoE_x_half_SA': (0.1, 0.9),

    'InterPSDC_weight_SA': (0.003, 0.01), 
    'InterPSDC_w_min_SA': (0.001, 0.005), 
    'InterPSDC_smooth_SA': (0.5, 3.0),
    'InterPSDC_x_half_SA': (0.1, 0.9),

    'PANtoE_weight_SA': (0.003, 0.01),
    'PANtoE_w_min_SA': (0.001, 0.005), 
    'PANtoE_smooth_SA': (0.5, 3.0),
    'PANtoE_x_half_SA': (0.1, 0.9),

    'PANtoI_weight_SA': (0.003, 0.01),
    'PANtoI_w_min_SA': (0.001, 0.005), 
    'PANtoI_smooth_SA': (0.5, 3.0),
    'PANtoI_x_half_SA': (0.1, 0.9),

    'tauExcProj_SA': (2.0, 500.0),
    'tauExcInh_SA': (2.0, 500.0),
    'tauExcPSDC_SA': (2.0, 500.0),
    'tauPSDCtoProj_SA': (2.0, 500.0),
    'tauPSDCtoInh_SA': (2.0, 500.0),
    'tauInh_SA': (2.0, 500.0),

    'tau_recExcProj_SA': (2.0, 500.0),
    'tau_recExcInh_SA': (2.0, 500.0),
    'tau_recExcPSDC_SA': (2, 500.0),
    'tau_recPSDCtoProj_SA': (2, 500.0),
    'tau_recPSDCtoInh_SA': (2, 500.0),
    'tau_recInh_SA': (2, 500.0),

    'UExcProj_SA': (0.05, 0.8),
    'UExcInh_SA': (0.05, 0.8),
    'UExcPSDC_SA': (0.05, 0.8),
    'UPSDCtoProj_SA': (0.05, 0.8),
    'UPSDCtoInh_SA': (0.05, 0.8),
    'UInh_SA': (0.05, 0.8),

    'tau_facilExcProj_SA': (2.0, 500.0),
    'tau_facilExcInh_SA': (2.0, 500.0),
    'tau_facilExcPSDC_SA': (2, 500.0),
    'tau_facilPSDCtoProj_SA': (2, 500.0),
    'tau_facilPSDCtoInh_SA': (2, 500.0),
    'tau_facilInh_SA': (2, 500.0),
} 

y_true_matrix = []
t_matrix = []

#base Vm dyn score
y_true_matrix.append([3.32, 3.32]) #Optimal constants from 50Hz sync
t_matrix.append([0.0, 0.0])

#New control
with open('characteristics/masISI_pressurePSDCoff.json', 'r') as json_file:
    y_true = json.load(json_file)
    y_values = list(y_true['y'])
    x_tags = list(y_true['x'])
    x_values = []
    for tag in x_tags:
        with open(f'new_pressure_stim_8to2_RA_SA_1000ms/{tag}NRA.json', 'r') as json_file:
            spikesRA = json.load(json_file)
        with open(f'new_pressure_stim_8to2_RA_SA_1000ms/{tag}NSA.json', 'r') as json_file:
            spikesSA = json.load(json_file)
        spikes = {'RA': spikesRA, 'SA': spikesSA}

        x_values.append(spikes)
    y_true_matrix.append(y_values)
    t_matrix.append(x_values)
y_true_matrix.append(np.full(len(y_values), 0.2))
t_matrix.append(x_values)

y_true_matrix.append(np.full(len(y_values), 25.0))
t_matrix.append(np.full(len(y_values), 25.0))

# Load init params
with open("optimisation/all_init_params/params_pressure_split51.json", "r", encoding="utf-8") as f:
    init_params = json.load(f)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    
    param_names, results, pareto = run_nsga3_optimization_threads(
        mega_func=generatorFunc,
        t_matrix=t_matrix,
        y_true_matrix=y_true_matrix,
        param_bounds=param_bounds,
        n_gen=300,
        pop_size=300,
        log_path="optimisation_log_deap_test.csv",
        checkpoint_path='save_state_optim_deap_test.json',
        checkpoint_every=5,
        priorities=[1, 3, 2, 2],
        init_params=init_params,
        delta_init_params=0.01,
        loss_func=corr_loss,
        n_workers=64
    )
    
    