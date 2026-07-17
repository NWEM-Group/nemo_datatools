import ioos_qc
from ioos_qc import qartod
from ioos_qc.config import Config
from ioos_qc.streams import PandasStream
from ioos_qc.stores import PandasStore

import pandas as pd
import datetime
import numpy as np

##################################
## QARTOD Sensor Configurations ##
##################################

def load_sensor_qartod_config(instrument_model, sensor, target_depth, sample_rate):
    
    config = {}
    
    if sensor == 'sea_water_pressure':
        config = {
            "sea_water_pressure": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [np.max([0,target_depth-5]), target_depth+5],
                        "fail_span": [np.max([-1,target_depth-10]), target_depth+10]},
                    "spike_test": {
                        "suspect_threshold": 0.15,
                        "fail_threshold": 0.3},
                    "rate_of_change_test": {
                        "threshold": 0.5/(60*60)},
                    "flat_line_test": {
                        "tolerance": 0.001,
                        "suspect_threshold": int(sample_rate*4),
                        "fail_threshold": int(sample_rate*12)}
                }
            }
        }
        if sample_rate > 60:
            config['sea_water_pressure']['qartod']['spike_test']['suspect_threshold'] = 2*config['sea_water_pressure']['qartod']['spike_test']['suspect_threshold']
            config['sea_water_pressure']['qartod']['spike_test']['fail_threshold'] = 2*config['sea_water_pressure']['qartod']['spike_test']['fail_threshold']
            config['sea_water_pressure']['qartod']['rate_of_change_test']['threshold'] = 4*config['sea_water_pressure']['qartod']['rate_of_change_test']['threshold']
            
    elif sensor == 'sea_water_temperature':
        config = {
            "sea_water_temperature": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [0,35],
                        "fail_span": [-5,45]},
                    "spike_test": {
                        "suspect_threshold": 0.5,
                        "fail_threshold": 1.0},
                    "rate_of_change_test": {
                        "threshold": 2/(30*60)},
                    "flat_line_test": {
                        "tolerance": 0.0005,
                        "suspect_threshold": int(sample_rate*4),
                        "fail_threshold": int(sample_rate*12)}
                }
            }
        }
        if sample_rate > 60:
            config['sea_water_temperature']['qartod']['spike_test']['suspect_threshold'] = 2*config['sea_water_temperature']['qartod']['spike_test']['suspect_threshold']
            config['sea_water_temperature']['qartod']['spike_test']['fail_threshold'] = 2*config['sea_water_temperature']['qartod']['spike_test']['fail_threshold']
            config['sea_water_temperature']['qartod']['rate_of_change_test']['threshold'] = 4*config['sea_water_temperature']['qartod']['rate_of_change_test']['threshold']

    elif sensor == 'sea_water_electrical_conductivity':
        config = {
            "sea_water_electrical_conductivity": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [2.5,3.8],
                        "fail_span": [1.5,4.5]},
                    "spike_test": {
                        "suspect_threshold": 0.5,
                        "fail_threshold": 1.0},
                    "rate_of_change_test": {
                        "threshold": 1.0/(30*60)},
                    "flat_line_test": {
                        "tolerance": 0.001,
                        "suspect_threshold": int(sample_rate*4),
                        "fail_threshold": int(sample_rate*12)}
                }
            }
        }
    elif sensor == 'sea_water_practical_salinity':
        config = {
            "sea_water_practical_salinity": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [2,35],
                        "fail_span": [0,40]},
                    "spike_test": {
                        "suspect_threshold": 0.25,
                        "fail_threshold": 0.5},
                    "rate_of_change_test": {
                        "threshold": 5.0/(30*60)},
                    "flat_line_test": {
                        "tolerance": 0.001,
                        "suspect_threshold": int(sample_rate*4),
                        "fail_threshold": int(sample_rate*12)}
                }
            }
        }
    elif sensor == 'sea_water_sigma_theta':
        config = {
            "sea_water_sigma_theta": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [10, 35],
                        "fail_span": [0, 55]},
                    "spike_test": {
                        "suspect_threshold": 0.25,
                        "fail_threshold": 0.5},
                    "rate_of_change_test": {
                        "threshold": 5.0/(30*60)},
                    "flat_line_test": {
                        "tolerance": 0.001,
                        "suspect_threshold": int(sample_rate*4),
                        "fail_threshold": int(sample_rate*12)}
                }
            }
        }
    elif sensor == 'mass_concentration_of_oxygen_in_sea_water':
        config = {
            "mass_concentration_of_oxygen_in_sea_water": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [0.01,20],
                        "fail_span": [0,30]},
                    "spike_test": {
                        "suspect_threshold": 0.25,
                        "fail_threshold": 0.5},
                    "rate_of_change_test": {
                        "threshold": 1.0/(30*60)},
                    "flat_line_test": {
                        "tolerance": 0.001,
                        "suspect_threshold": int(sample_rate*4),
                        "fail_threshold": int(sample_rate*12)}
                }
            }
        }
    elif sensor == 'mass_concentration_of_chlorophyll_a_in_sea_water':
        config = {
            "mass_concentration_of_chlorophyll_a_in_sea_water": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [0.0001, 50.0],
                        "fail_span": [0.0, 100.0]},
                    "spike_test": {
                        "suspect_threshold": 0.1,
                        "fail_threshold": 0.2},
                    "rate_of_change_test": {
                        "threshold": 1.0/(60*60)},
                    "flat_line_test": {
                        "tolerance": 0.001,
                        "suspect_threshold": int(sample_rate*4),
                        "fail_threshold": int(sample_rate*12)}
                }
            }
        }
    elif sensor == 'sea_water_turbidity':
        config = {
            "sea_water_turbidity": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [0.05, 10.0],
                        "fail_span": [0.0, 20.0]},
                    "spike_test": {
                        "suspect_threshold": 0.05,
                        "fail_threshold": 0.1},
                    "rate_of_change_test": {
                        "threshold": 1.0/(60*60)},
                    "flat_line_test": {
                        "tolerance": 0.005,
                        "suspect_threshold": int(sample_rate*4),
                        "fail_threshold": int(sample_rate*12)}
                }
            }
        }
    elif sensor == 'sea_water_ph_reported_on_total_scale':
        config = {
            "sea_water_ph_reported_on_total_scale": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [7, 9],
                        "fail_span": [6.5, 9.5]},
                    "spike_test": {
                        "suspect_threshold": 0.5,
                        "fail_threshold": 1.0},
                    "rate_of_change_test": {
                        "threshold": 1.0/(60*60)},
                    "flat_line_test": {
                        "tolerance": 0.01,
                        "suspect_threshold": 6*(60*60),
                        "fail_threshold": 12*(60*60)}
                }
            }
        }
        
        
    return config

def run_qartod_tests(var_df, instrument_model, sensor, target_depth, sample_rate):
    
    
    # Ensure that the variable is a pandas dataframe
    if not(isinstance(var_df,pd.DataFrame)):
        var_df = pd.DataFrame(data=var_df,columns=['time',sensor])
    
    config = load_sensor_qartod_config(instrument_model, sensor, target_depth, sample_rate)
        
    c = Config(config)

    # Setup the stream
    ps = PandasStream(var_df.loc[:,['time',sensor]], time='time')
    
    # Pass the run method the config to use
    results = ps.run(c)
    
    # Store the results in another DataFrame
    store = PandasStore(
        results
    )

    # Write only the test results to the store
    results_store = store.save(write_data=False, write_axes=False)

    #
    aggr_flags = qartod.qartod_compare(results_store.to_numpy().T)
    results_store[sensor + '_qartod_rollup_qc'] = aggr_flags
    
    return results_store

def concat_test_results_into_string(temp_qartod_df):
    
    # Initialize an empty list to store the concatenated qartod results
    qartod_results = np.zeros(len(temp_qartod_df)).astype(int)
    
    # Extract out the relevant column names
    qar_cols = temp_qartod_df.columns
    # Only concatenate the result if it is not the rollup result
    qar_cols = [ii for ii in qar_cols if 'qartod_rollup_qc' not in ii]
    
    # Loop through each column, and append on the digit of the current
    # flag
    # Example, if there are 3 flags (flag 1 = 1, flag 2 = 4, flag 3 = 1),
    # then the final result should be 141.
    # To get there, we use 100*flag1 + 10*flag2 + 1*flag3
    # which is the same as (10^2)*flag1 + (10^1)*flag2 + (10^0)*flag3
    # The maximum exponent factor is equal to (# of flags - 1)
    fact = len(qar_cols) - 1
    for column in qar_cols:
        qartod_results = qartod_results + (temp_qartod_df.loc[:,column].astype(float) * (10**fact)).astype(int)
        fact = fact - 1
     
    # Change the result from an array into a list
    qartod_results = [ii for ii in qartod_results]

    return qartod_results


##################################
## QARTOD Wrapper Functions     ##
##################################


def process_qartod_tests(data_df, instrument_model, target_depth, sample_rate, qartod_valid_sensors=None):
    
    qartod_df = []
    NT = data_df.shape[0]
    
    # Calculate qartod tests
    sensor_names = [col for col in data_df.columns if 'time' not in col] 
    run_sensors = []
    for sensor in sensor_names:
        if (qartod_valid_sensors is None) or (sensor in qartod_valid_sensors):
                run_sensors.append(sensor)


    for sensor in run_sensors:
        if not((all(data_df[sensor].isna())) or (all(data_df[sensor] == -555))):
            
            temp_df = pd.DataFrame(data = list(zip(np.ndarray.flatten(data_df[sensor].to_numpy()),
                                                    np.ndarray.flatten(data_df['time'].to_numpy()))),
                                    columns=[sensor,'time'])

            # Perform the qartod tests for the specified sensor
            temp_qartod_df = run_qartod_tests(temp_df, instrument_model, sensor, 
                                              target_depth, sample_rate)
            temp_qartod_df[sensor+'_qc_manually_flagged_test'] = 9*np.ones(NT).astype(int)
            temp_results = concat_test_results_into_string(temp_qartod_df)
            temp_qartod_df[sensor+'_qc_aggregate'] = temp_qartod_df[sensor+'_qartod_rollup_qc'].astype(int)
            temp_qartod_df[sensor+'_qc_tests'] = temp_results
            

        else:
            # If all the data is bad, apply "fail" flags everywhere, and assign 9999 for the tests, indicating
            # that no tests were performed

            sensor_config = load_sensor_qartod_config(instrument_model, sensor, target_depth, sample_rate)[sensor]['qartod']
            ntests = len(sensor_config.keys())
            flag_val = int('9'*(ntests+1))
            temp_qartod_df = pd.DataFrame(data=list(zip(4*np.ones(NT).astype(int),
                                                            flag_val*np.ones(NT).astype(int))),
                                                columns=[sensor+'_qc_aggregate',
                                                        sensor+'_qc_tests'])
            for test_name in sensor_config.keys():
                temp_qartod_df[sensor+'_qartod_'+test_name] = flag_val*np.ones(NT).astype(int)
            temp_qartod_df[sensor+'_qartod_rollup_qc'] = 9*np.ones(NT).astype(int)
            temp_qartod_df[sensor+'_qc_manually_flagged_test'] = 9*np.ones(NT).astype(int)
            
    
        
        # Append on the newly created qartod flags to the qartod dataframe
        if isinstance(qartod_df,list):
            qartod_df = temp_qartod_df
        else:
            qartod_df = pd.concat([qartod_df, temp_qartod_df],axis=1)
                
                
                
    return qartod_df

