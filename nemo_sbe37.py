#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr
import datetime

import netCDF4
from netCDF4 import Dataset

import os

import json
import shutil
import sys
import getopt
import gc
import traceback
from importlib import reload

import gsw

import seabirdscientific.instrument_data as sb_id
import seabirdscientific.conversion as sb_conv

import matplotlib
from matplotlib import pyplot as plt

import struct
import codecs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import nemo_general_functions as nemo_funcs
import nemo_deployments
import nemo_qartod as nemo_qc

# # SBE37 Functions

def find_sbe37_headerend(sbe37_lines):

    for ii, line in enumerate(sbe37_lines):
        if line.startswith('*END*'):
            return ii
    raise ValueError('SBE37 file missing #END header line.')

def find_sbe37_instrument_start_time(sbe37_lines):

    for ii, line in enumerate(sbe37_lines):
        if line.startswith('# start_time = '):
            start_time_str = line.split('=')[1].strip()
            start_time = datetime.datetime.strptime(start_time_str, '%b %d %Y %H:%M:%S')
            return start_time
    raise ValueError('SBE37 file missing data start line.')

def find_sbe37_sample_rate(sbe37_lines):

    for ii, line in enumerate(sbe37_lines):
        if line.startswith('# interval = seconds:'):
            interval_str = line.split(':')[1].strip()
            sample_rate = float(interval_str)
            return sample_rate
    raise ValueError('SBE37 file missing sample rate line.')

def read_sbe37_nemo_rawdataframe(sbe37_file):
    
    # Read in the raw file
    sbe37_interim = pd.read_csv(sbe37_file, header=1, skiprows=[2,3])
    
    # Extract out the CTD response string
    sbe_ctdstring = [ii for ii in sbe37_interim['CTDresponseString']]
    sbe_temp = []
    sbe_cond = []
    sbe_pres = []
    sbe_oxy = []
    sbe_sal = []
    sbe_date = []
    sbe_time = []

    # Parse each line of the CTD response string    
    def parse_line(line):
        splitstring = line.split(',') 
        
        try: 
            # Check if the split of the string has a date
            # for the 5th entry
            testdate = datetime.datetime.strptime(splitstring[4], ' %d %b %Y')
            split_dateflag = True
        except:
            split_dateflag = False
            
        salt_flag = True
        if ((len(splitstring) == 6) and split_dateflag):
            # If there is a date in the 5th entry of the split string,
            # and there are only 6 entries in the split string,
            # indicate that salinity value is not provided
            salt_flag = False
        
        tempval = splitstring[0]
        condval = splitstring[1]
        presval = splitstring[2]
        oxyval = splitstring[3]
        if salt_flag:
            # If salinity values are provided,
            # pull out that data
            salval = splitstring[4]
            dateval = splitstring[5]
            timeval = splitstring[6]
        else:
            # Otherwise, provide a "NAN" for salinity
            salval = 'NAN'
            dateval = splitstring[4]
            timeval = splitstring[5]
        return tempval, condval, presval, oxyval, salval, dateval, timeval
            
    
    for ii in range(0,len(sbe_ctdstring)):
        
        # If the string is not parseable (i.e., has a NAN, a line prompt,
        #                                 or is not enough variables),
        # do not parse, and fill with "NAN"
        splitstring = sbe_ctdstring[ii].split(',') 
        if ((ii != 'NAN') or (ii != ' S>')) and (len(splitstring) >= 6):
            [tempval, condval, presval, 
             oxyval, salval, dateval, timeval] = parse_line(sbe_ctdstring[ii])
            sbe_temp.append(tempval)
            sbe_cond.append(condval)
            sbe_pres.append(presval)
            sbe_oxy.append(oxyval)
            sbe_sal.append(salval)
            sbe_date.append(dateval)
            sbe_time.append(timeval)
        else:
            sbe_temp.append('NAN')
            sbe_cond.append('NAN')
            sbe_pres.append('NAN')
            sbe_oxy.append('NAN')
            sbe_sal.append('NAN')
            sbe_date.append('NAN')
            sbe_time.append('NAN')

    # Build a new data frame to match the format of the normal "_Data.dat" file
    sbe_interim2 = pd.DataFrame(data = list(zip(sbe37_interim['TIMESTAMP'], sbe37_interim['RECORD'], 
                                               sbe_temp, sbe_cond, sbe_pres, sbe_oxy,
                                               sbe_sal, sbe_date, sbe_time)),
                               columns = ['TIMESTAMP', 'RECORD', 
                                          'SBE37Temp_C','SBE37Cond_S_m',
                                          'SBE37Pres_dbar', 'SBE37DO_mGperL',
                                          'SBE37Sal_PSU', 'SBE37Date', 'SBE37Time'])

    
    
    return sbe_interim2


# In[ ]:


def read_sbe37_nemo_dataframe(sbe37_file):
    
    if '_Raw.dat' in sbe37_file:
        sbe37_interim = read_sbe37_nemo_rawdataframe(sbe37_file)
    else:
        sbe37_interim = pd.read_csv(sbe37_file, header=1, skiprows=[2,3])

    # Replace all "NAN" or '"NAN"' values with "NAN"
    sbe37_interim = sbe37_interim.replace(['NAN', '"NAN"'], 'NAN')

    
    sbe_inst_time = [pd.NaT if ((sbe37_interim['SBE37Date'][ii] == 'NAN')
                                or (sbe37_interim['SBE37Date'][ii] == '"NAN"')
                                or (sbe37_interim['SBE37Time'][ii] == 'NAN')
                                or (sbe37_interim['SBE37Time'][ii] == '"NAN"'))
                     else datetime.datetime.strptime(sbe37_interim['SBE37Date'][ii] + 
                                                     sbe37_interim['SBE37Time'][ii],
                                                     ' %d %b %Y %H:%M:%S.%f')  
                     if len(sbe37_interim['SBE37Time'][ii]) > 9 
                     else datetime.datetime.strptime(sbe37_interim['SBE37Date'][ii] + 
                                                     sbe37_interim['SBE37Time'][ii],
                                                     ' %d %b %Y %H:%M:%S') 
                     for ii in range(0,len(sbe37_interim))]
    sbe_time = [pd.NaT if ((ii == 'NAN') or (ii == '"NAN"')) 
                else datetime.datetime.strptime(ii, '%Y-%m-%d %H:%M:%S.%f') if len(ii) > 19 
                else datetime.datetime.strptime(ii, '%Y-%m-%d %H:%M:%S') 
                for ii in sbe37_interim['TIMESTAMP']]
    
    if 'SBE37DO_mLperL' in sbe37_interim.columns:
        sbe_oxy = [np.nan if ii == 'NAN' else float(ii) 
                   for ii in sbe37_interim['SBE37DO_mLperL']]
        sbe_oxy = 1.4276 * np.array(sbe_oxy)
    elif 'SBE37DO_mGperL' in sbe37_interim.columns:
        sbe_oxy = [np.nan if ii == 'NAN' else float(ii) 
                   for ii in sbe37_interim['SBE37DO_mGperL']]
        #sbe_oxy = [ii for ii in sbe_oxy]
    
    vars_to_check = ['RECORD', 'SBE37Temp_C', 'SBE37Cond_S_m',
                     'SBE37Pres_dbar', 'SBE37Sal_PSU',
                     'SBE37DO_mGperL', ]
    for var in vars_to_check:
        if var in sbe37_interim.columns:
            sbe37_interim[var] = [np.nan if ii == 'NAN' else float(ii)
                                  for ii in sbe37_interim[var]]
    
    sbe37_rawdf = pd.DataFrame(data = list(zip(sbe_time, sbe37_interim['RECORD'], 
                                               sbe37_interim['SBE37Temp_C'], 
                                               sbe37_interim['SBE37Cond_S_m'], 
                                               sbe37_interim['SBE37Pres_dbar'], 
                                               sbe_oxy,
                                               sbe37_interim['SBE37Sal_PSU'], 
                                               sbe_inst_time)),
                           columns = ['Timestamp', 'RecordNumber', 
                                      'Temperature_degC','Conductivity_S_m',
                                      'Pressure_dbar', 'DO_mg_L',
                                      'Salinity_PSU', 'Instrument_Timestamp'])
    
    
    return sbe37_rawdf


# In[ ]:

def read_sbe37_seabirdcnv(sbe37_file):
    # Use the seabird function to read in the file
    if '.cnv' not in sbe37_file:
        print('File is not a .cnv file. Cannot read with seabird function.')
        return None
    inst_data = sb_id.cnv_to_instrument_data(filepath=sbe37_file)
    inst_vars = inst_data.measurements.keys()
    inst_df = inst_data._to_dataframe()
    sample_rate = inst_data.interval_s
    ref_start = datetime.datetime(inst_data.start_time.timetuple().tm_year,1,1,0,0,0)
    ref_date = datetime.datetime(1970,1,1,0,0,0)

    # Find the time variable and convert to timestamps
    time_var = None
    time_units = None
    for var in inst_vars:
        if 'time' in inst_data.measurements[var].description.lower():
            time_var = var
            time_units = inst_data.measurements[var].units
            break
    if 'julian days' in time_units.lower():
        inst_df['date'] = [ref_start + datetime.timedelta(days=(ii-1)) for ii in inst_df['timeJV2']]
        inst_df.drop(columns=['timeJV2'], inplace=True)
    inst_df['timestamp'] = [np.round((ii-ref_date).total_seconds(),6) for ii in inst_df['date']]

    # Find the temperature variable and check its scale.
    # SeaBird CNV variable names embed the scale: '90' → ITS-90, '68' → IPTS-68.
    temp_var = None
    temp_units = None
    for var in inst_vars:
        if inst_data.measurements[var].description.lower() == 'temperature' :
            temp_var = var
            temp_units = inst_data.measurements[var].units
            break
    if temp_units is not None and 'ITS-90' not in temp_units:
        inst_df[temp_var] = gsw.t90_from_t68(inst_df[temp_var].to_numpy())

    # Find the oxygen variable and check its units.
    oxy_var = None
    oxy_units = None
    for var in inst_vars:
        if 'oxygen' in inst_data.measurements[var].description.lower():
            oxy_var = var
            oxy_units = inst_data.measurements[var].units
            break
    if oxy_units is not None and oxy_units == 'ml/l':
        inst_df[oxy_var] = sb_conv.convert_oxygen_to_mg_per_l(inst_df[oxy_var].to_numpy())

    newname_dict = {
        time_var: 'timestamp',
        temp_var: 'sea_water_temperature',
        "cond0S/m": 'sea_water_electrical_conductivity',
        'prdM': 'sea_water_pressure',
    }
    if oxy_var is not None:
        newname_dict[oxy_var] = 'mass_concentration_of_oxygen_in_sea_water'

    # Rename the columns to match the standard names we use for processing
    sbe_interim = inst_df.rename(columns=newname_dict)
    if 'mass_concentration_of_oxygen_in_sea_water' not in sbe_interim.columns:
        sbe_interim['mass_concentration_of_oxygen_in_sea_water'] = np.nan

    # Process the derived variables

    # Salinity and Absolute Salinity
    sbe_interim['sea_water_practical_salinity'] = gsw.SP_from_C(
        np.array([(1000*ii)/100 for ii in sbe_interim['sea_water_electrical_conductivity']]),
        sbe_interim['sea_water_temperature'].to_numpy(),
        sbe_interim['sea_water_pressure'].to_numpy()
    )
    inst_SA = gsw.SA_from_SP(sbe_interim['sea_water_practical_salinity'].to_numpy(),
                             sbe_interim['sea_water_pressure'].to_numpy(),
                             -124,48)
    
    # Conservative Temperature
    inst_CT = gsw.CT_from_t(inst_SA,
                            sbe_interim['sea_water_temperature'].to_numpy(),
                            sbe_interim['sea_water_pressure'].to_numpy()
    )

    # Dissolved Oxygen Saturation Concentration
    inst_DOsat = gsw.O2sol(inst_SA, inst_CT, sbe_interim['sea_water_pressure'].to_numpy(),-124,48)

    # Potential Density referenced to 0 dbar (sigma-0)#
    sbe_interim['sea_water_sigma_theta'] = gsw.sigma0(inst_SA, inst_CT)
    # Speed of sound in sea water
    sbe_interim['speed_of_sound_in_sea_water'] = gsw.sound_speed(inst_SA, inst_CT, 
                                                                 sbe_interim['sea_water_pressure'].to_numpy())
    # Fractional saturation of oxygen in sea water
    if oxy_var is None:
        sbe_interim['fractional_saturation_of_oxygen_in_sea_water'] = np.nan
    else:
        sbe_interim['fractional_saturation_of_oxygen_in_sea_water'] = 100 * sbe_interim['mass_concentration_of_oxygen_in_sea_water'] / inst_DOsat

    # Drop unnecessary variables and round the data to 6 decimal places for consistency with the line-parsing method
    for var in sbe_interim.columns:
        if var not in ['timestamp', 'sea_water_temperature', 'sea_water_electrical_conductivity',
                       'sea_water_pressure', 'mass_concentration_of_oxygen_in_sea_water',
                       'sea_water_practical_salinity', 'sea_water_sigma_theta',
                       'speed_of_sound_in_sea_water', 'fractional_saturation_of_oxygen_in_sea_water']:
            sbe_interim.drop(columns=[var], inplace=True)
    for var in sbe_interim.columns:
        if var not in ['timestamp', 'date']:
            sbe_interim[var] = [np.round(ii,6) for ii in sbe_interim[var].to_numpy()]

    return sbe_interim, sample_rate


def read_sbe37_nemo_lineparsing(sbe37_file):
    
    #
    fillVal = -555

    # Read in the file
    with open(sbe37_file, encoding = 'utf-8', errors='replace') as f:
        lines = f.readlines()

    hdr_end_ind = find_sbe37_headerend(lines)
    startind = hdr_end_ind + 1
    instrument_start_time = find_sbe37_instrument_start_time(lines)
    instrument_start_year = instrument_start_time.year
    ref_start = datetime.datetime(instrument_start_year, 1, 1, 0, 0, 0) - datetime.timedelta(days=1)
    sample_rate = find_sbe37_sample_rate(lines)

    hdr_line = lines[0]
    hdr_linesplit = hdr_line.split(',')
    lander = hdr_linesplit[1].strip('"')

    timestamp = []
    record = []
    temper_degC = []
    cond_S_m = []
    pres_db = []
    do_mL_per_L = []
    sal_psu = []
    sbe_date = []
    sbe_timestamp = []

    for ii in range(startind,len(lines)):

        line = lines[ii]
        linesplit = line.split(',')
        
        if (linesplit[0] == 'NAN') or (linesplit[0] == '"NAN"'):
            timestamp.append(pd.NaT)
        else:
            timestamp.append(datetime.datetime.strptime(linesplit[0],'"%Y-%m-%d %H:%M:%S"'))
            
        if (linesplit[1] == 'NAN') or (linesplit[1] == '"NAN"'):
            record.append(np.nan)
        else:
            record.append(int(linesplit[1]))
            
        if (linesplit[2] == 'NAN') or (linesplit[2] == '"NAN"'):
            temper_degC.append(np.nan)
        else:
            temper_degC.append(float(linesplit[2]))
        
        if (linesplit[3] == 'NAN') or (linesplit[3] == '"NAN"'):
            cond_S_m.append(np.nan)
        else:
            cond_S_m.append(float(linesplit[3]))
        
        if (linesplit[4] == 'NAN') or (linesplit[4] == '"NAN"'):
            pres_db.append(np.nan)
        else:
            pres_db.append(float(linesplit[4]))
        
        if (linesplit[5] == 'NAN') or (linesplit[5] == '"NAN"'):
            do_mL_per_L.append(np.nan)
        else:
            do_mL_per_L.append(float(linesplit[5]))
        
        if (linesplit[6] == 'NAN') or (linesplit[6] == '"NAN"'):
            sal_psu.append(np.nan)
        else:
            sal_psu.append(float(linesplit[6]))
        
        if (linesplit[7] == 'NAN') or (linesplit[8] == 'NAN') or (linesplit[7] == '"NAN"') or (linesplit[8] == '"NAN"'):
            sbe_timestamp.append(pd.NaT)
        else:
            try:
                sbe_timestamp.append(datetime.datetime.strptime(linesplit[7] + linesplit[8],
                                                                '" %d %b %Y"" %H:%M:%S"\n'))
            except:
                sbe_timestamp.append(datetime.datetime.strptime(linesplit[7] + linesplit[8],
                                                                '" %d %b %Y"" %H:%M:%S.%f"\n'))
    
    
    # Package all the data into a data frame
    sbe37_rawdf = pd.DataFrame(data = list(zip(timestamp, record, 
                                               temper_degC, cond_S_m, pres_db,
                                               do_mL_per_L, sal_psu, sbe_timestamp)),
                               columns = ['Timestamp', 'RecordNumber', 
                                          'Temperature_degC','Conductivity_S_m',
                                          'Pressure_dbar', 'DO_mL_L',
                                          'Salinity_PSU', 'Instrument_Timestamp'])
    
    return sbe37_rawdf, sample_rate


# In[ ]:


def _parse_deployment_time(time_str):
    
    if time_str is None:
        return None
    
    parsed_time = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    if parsed_time.tzinfo is not None:
        parsed_time = parsed_time.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    
    return parsed_time



def _load_sbe37_deployment_window(nemo_name, deployment_name):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{nemo_name}_deployments.json')
    
    with open(json_path, 'r') as f:
        deployment_info = json.load(f)
    
    deployment_data = deployment_info.get('deployments', {}).get(deployment_name)
    if deployment_data is None:
        raise KeyError(f"Deployment '{deployment_name}' not found for {nemo_name}.")
    
    lat = deployment_data.get('latitude')
    lon = deployment_data.get('longitude')
    depth = deployment_data.get('depth')
    start_deployment = _parse_deployment_time(deployment_data.get('start'))
    end_deployment = _parse_deployment_time(deployment_data.get('end'))
    status = deployment_data.get('status', 'unknown')
    
    return lat, lon, depth, start_deployment, end_deployment, status



def _find_sbe37_files(datadir, nemo_name, deployment_name, serial_num=None, status=None):
    """Discover SBE37 data files in datadir matching deployment and optional serial."""
    if not os.path.isdir(datadir):
        print(datadir + ' is not a valid directory. Cannot search for SBE37 files.')
        return []

    if deployment_name is None or str(deployment_name).strip() == '':
        raise ValueError('deployment_name is required to search for SBE37 files.')

    required_suffix = '.cnv'
    required_tokens = [str(deployment_name).strip().lower()]
    if serial_num is not None and str(serial_num).strip() != '':
        required_tokens.append(str(serial_num).strip().lower())

    files = []
    for fname in os.listdir(datadir):
        if not os.path.isfile(os.path.join(datadir, fname)):
            continue
        flower = fname.lower()
        if not flower.endswith(required_suffix):
            continue
        if not ('sbe37' in flower or 'ctd' in flower):
            continue
        if all(token in flower for token in required_tokens):
            files.append(fname)
    return sorted(files)


def _estimate_sample_rate_seconds(sample_time_values, default_seconds=60.0):
    """Estimate sample interval in seconds from timestamp values."""
    if sample_time_values is None:
        return float(default_seconds)

    sample_times = pd.to_datetime(sample_time_values, errors='coerce')
    sample_times = sample_times.dropna()
    if len(sample_times) < 2:
        return float(default_seconds)

    sample_times = sample_times.sort_values()
    deltas = sample_times.diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return float(default_seconds)

    return float(deltas.median())


def nemo_sbe37_wrapper(nemo_name, deployment_name, instrument_name='sbe37', serial_num=None):
    
    lat, lon, depth, start_deployment, end_deployment, status = _load_sbe37_deployment_window(
        nemo_name,
        deployment_name
    )
    if not((status == 'realtime') or (status == 'recovered')):
        print(f"Deployment status is '{status}'. Skipping processing for {nemo_name} - {deployment_name}.")
        return
    datadir, savedir = nemo_funcs.get_datalocations(nemo_name, deployment_name, 'ctd', status=status)

    # Load shorthand for file token matching
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{nemo_name}_deployments.json')
    with open(json_path, 'r') as f:
        deployment_info = json.load(f)

    if serial_num is None:
        deployment_data = deployment_info.get('deployments', {}).get(deployment_name, {})
        instruments = deployment_data.get('instruments', {})
        sbe37_serials = []

        if isinstance(instruments, dict):
            for _, instrument_record in instruments.items():
                if not isinstance(instrument_record, dict):
                    continue

                model_name = str(instrument_record.get('model', '')).strip().lower()
                serial_value = instrument_record.get('sn')
                if serial_value is None:
                    continue
                if 'sbe37' not in model_name:
                    continue

                serial_text = str(serial_value).strip()
                if serial_text == '':
                    continue
                if serial_text not in sbe37_serials:
                    sbe37_serials.append(serial_text)

        if len(sbe37_serials) == 0:
            print(
                f"No SBE37 serial numbers were found in deployment info for "
                f"{nemo_name} - {deployment_name}."
            )
            return

        print(
            f"No serial number provided. Found {len(sbe37_serials)} SBE37 serial(s): "
            f"{', '.join(sbe37_serials)}"
        )
        for current_serial in sbe37_serials:
            print(f"\nProcessing SBE37 serial {current_serial}...")
            nemo_sbe37_wrapper(nemo_name, deployment_name, instrument_name, current_serial)
        return

    # Discover matching data files
    sbe37_files = _find_sbe37_files(datadir, nemo_name, deployment_name, serial_num, status)
    if len(sbe37_files) == 0:
        print(f'   No SBE37 files found in {datadir}')
        return
    print(f'   Found {len(sbe37_files)} SBE37 file(s): {sbe37_files}')

    # Read and concatenate all found files.
    # recovered: use seabird cnv parser; realtime: try dataframe parser, then line parser fallback.
    frames = []
    for fname in sbe37_files:
        fpath = os.path.join(datadir, fname)
        if status == 'recovered':
            df, sample_rate = read_sbe37_seabirdcnv(fpath)
            frames.append(df)
        else:
            try:
                df, sample_rate = read_sbe37_nemo_dataframe(fpath)
                frames.append(df)
            except Exception:
                print(f'   Failed dataframe parsing for {fname}; trying line parsing.')
                df, sample_rate = read_sbe37_nemo_lineparsing(fpath)
                frames.append(df)

    if len(frames) == 1:
        sbe37_rawdf = frames[0].copy()
    else:
        sbe37_rawdf = pd.concat(frames, ignore_index=True)

    # Standardize to seabird-CNV naming conventions for processing.
    # Expected standardized columns:
    # sample_time, instrument_timestamp, record_number,
    # sea_water_temperature, sea_water_electrical_conductivity,
    # sea_water_pressure, sea_water_practical_salinity,
    # sea_water_sigma_theta, mass_concentration_of_oxygen_in_sea_water,
    # fractional_saturation_of_oxygen_in_sea_water
    sbe37_std = pd.DataFrame()
    instrument_info = {}


    if status == 'recovered':
        # Find the matching CTD instrument for this serial number when possible.
        if serial_num is not None:
            instrument_info = nemo_deployments.find_matching_deployment_instrument(
                nemo_name,
                deployment_name,
                'ctd',
                serial_num
            )
            if instrument_info is None:
                print(
                    f"   No matching deployment instrument found for CTD serial {serial_num}. "
                    'Proceeding without deployment-specific time drift metadata.'
                )
                instrument_info = {}
            else:
                print(instrument_info)
        else:
            print('   Serial number not provided. Skipping deployment instrument lookup.')

        # read_sbe37_seabirdcnv provides timestamp seconds and derived seawater variables.
        if 'timestamp' in sbe37_rawdf.columns:
            sample_time = pd.to_datetime(sbe37_rawdf['timestamp'], unit='s', origin='unix', utc=True).dt.tz_localize(None)
        elif 'date' in sbe37_rawdf.columns:
            sample_time = pd.to_datetime(sbe37_rawdf['date'])
            if getattr(sample_time.dt, 'tz', None) is not None:
                sample_time = sample_time.dt.tz_localize(None)
        else:
            raise KeyError('Recovered SBE37 dataframe missing timestamp/date columns.')
        
        # Apply time drift correction if necessary
        if instrument_info.get('time_drift_secs', False):
            drift_seconds = instrument_info.get('time_drift_secs', 0)
            inst_start_time = datetime.datetime.strptime(instrument_info.get('start_time', None), '%Y-%m-%dT%H:%M:%SZ') if instrument_info.get('start_time', None) else None
            inst_end_time = datetime.datetime.strptime(instrument_info.get('end_time', None), '%Y-%m-%dT%H:%M:%SZ') if instrument_info.get('end_time', None) else None
            timerange = (inst_end_time - inst_start_time).total_seconds() if inst_start_time and inst_end_time else 0
            drift_rate = drift_seconds / timerange if timerange > 0 else 0
            print(f'   Applying time drift correction of {drift_seconds} seconds over {timerange} seconds ({drift_rate} seconds per second).')
            # Apply a linear drift correction across the time range of the data
            sample_time = sample_time - pd.to_timedelta(drift_rate * (sample_time - inst_start_time).dt.total_seconds(), unit='s')

        sbe37_std['sample_time'] = sample_time
        sbe37_std['instrument_timestamp'] = sample_time
        sbe37_std['record_number'] = np.arange(len(sbe37_rawdf))
        sbe37_std['sea_water_temperature'] = sbe37_rawdf['sea_water_temperature'].to_numpy()
        sbe37_std['sea_water_electrical_conductivity'] = sbe37_rawdf['sea_water_electrical_conductivity'].to_numpy()
        sbe37_std['sea_water_pressure'] = sbe37_rawdf['sea_water_pressure'].to_numpy()
        sbe37_std['sea_water_practical_salinity'] = sbe37_rawdf['sea_water_practical_salinity'].to_numpy()
        sbe37_std['sea_water_sigma_theta'] = sbe37_rawdf['sea_water_sigma_theta'].to_numpy()
        sbe37_std['speed_of_sound_in_sea_water'] = sbe37_rawdf['speed_of_sound_in_sea_water'].to_numpy()
        sbe37_std['mass_concentration_of_oxygen_in_sea_water'] = sbe37_rawdf['mass_concentration_of_oxygen_in_sea_water'].to_numpy()
        sbe37_std['fractional_saturation_of_oxygen_in_sea_water'] = sbe37_rawdf['fractional_saturation_of_oxygen_in_sea_water'].to_numpy()
    else:
        # Realtime parsing output is converted to seabird-CNV naming conventions.
        sbe37_std['sample_time'] = pd.to_datetime(sbe37_rawdf['Timestamp'])
        sbe37_std['instrument_timestamp'] = pd.to_datetime(sbe37_rawdf['Instrument_Timestamp'])
        sbe37_std['record_number'] = sbe37_rawdf['RecordNumber'].to_numpy()
        sbe37_std['sea_water_temperature'] = sbe37_rawdf['Temperature_degC'].to_numpy()
        sbe37_std['sea_water_electrical_conductivity'] = sbe37_rawdf['Conductivity_S_m'].to_numpy()
        sbe37_std['sea_water_pressure'] = sbe37_rawdf['Pressure_dbar'].to_numpy()



        sal_input = sbe37_rawdf['sea_water_practical_salinity'].to_numpy()
        sal_calc = gsw.SP_from_C(
            np.array([10 * ii for ii in sbe37_std['sea_water_electrical_conductivity']]),
            sbe37_std['sea_water_temperature'].to_numpy(),
            sbe37_std['sea_water_pressure'].to_numpy()
        )
        sbe37_std['sea_water_practical_salinity'] = np.array([
            sal_calc[ii] if pd.isna(sal_input[ii]) else sal_input[ii]
            for ii in range(0, len(sal_input))
        ])

        sal_SA = gsw.SA_from_SP(
            sbe37_std['sea_water_practical_salinity'].to_numpy(),
            sbe37_std['sea_water_pressure'].to_numpy(),
            lon,
            lat
        )
        temp_CT = gsw.CT_from_t(
            sal_SA,
            sbe37_std['sea_water_temperature'].to_numpy(),
            sbe37_std['sea_water_pressure'].to_numpy()
        )
        sbe37_std['sea_water_sigma_theta'] = gsw.sigma0(sal_SA, temp_CT)
        sbe37_std['speed_of_sound_in_sea_water'] = gsw.sound_speed(sal_SA, temp_CT, sbe37_std['sea_water_pressure'].to_numpy())

        O2_sol_mg_L = (
            gsw.O2sol(
                sal_SA,
                temp_CT,
                sbe37_std['sea_water_pressure'].to_numpy(),
                lon,
                lat
            )
            * (32 / 1000) * (1000 + sbe37_std['sea_water_sigma_theta'].to_numpy()) * (1 / 1000)
        )
        sbe37_std['fractional_saturation_of_oxygen_in_sea_water'] = (
            100 * sbe37_std['mass_concentration_of_oxygen_in_sea_water'].to_numpy() / O2_sol_mg_L
        )

    # Keep data only within the deployment window.
    valid_time_mask = ~sbe37_std['sample_time'].isna()
    if start_deployment is not None:
        valid_time_mask = valid_time_mask & (sbe37_std['sample_time'] >= start_deployment)
    if end_deployment is not None:
        valid_time_mask = valid_time_mask & (sbe37_std['sample_time'] <= end_deployment)
    sbe37_std = sbe37_std[valid_time_mask]

    if len(sbe37_std) == 0:
        print('   No new SBE37 data. Continue on.')
        return

    sample_rate = _estimate_sample_rate_seconds(sbe37_std['sample_time'])

    instrument_depth = instrument_info.get('instrument_depth', depth)
    sbe37_std['depth'] = [instrument_depth for _ in range(len(sbe37_std))]
    
    

    # Repackage standardized data into legacy netCDF-writing schema.
    sbe37_std = sbe37_std.reset_index(drop=True)
    n_samples = len(sbe37_std)
    sbe37_df = pd.DataFrame({
        'buoyname': [nemo_name for _ in range(0, n_samples)],
        'time': sbe37_std['sample_time'].to_numpy().squeeze(),
        'instrument_timestamp': sbe37_std['instrument_timestamp'].to_numpy().squeeze(),
        'record_number': sbe37_std['record_number'].to_numpy().squeeze(),
        'sea_water_pressure': sbe37_std['sea_water_pressure'].to_numpy().squeeze(),
        'depth': sbe37_std['depth'].to_numpy().squeeze(),
        'sea_water_temperature': sbe37_std['sea_water_temperature'].to_numpy().squeeze(),
        'sea_water_electrical_conductivity': sbe37_std['sea_water_electrical_conductivity'].to_numpy().squeeze(),
        'sea_water_practical_salinity': sbe37_std['sea_water_practical_salinity'].to_numpy().squeeze(),
        'sea_water_sigma_theta': sbe37_std['sea_water_sigma_theta'].to_numpy().squeeze(),
        'mass_concentration_of_oxygen_in_sea_water': sbe37_std['mass_concentration_of_oxygen_in_sea_water'].to_numpy().squeeze(),
        'fractional_saturation_of_oxygen_in_sea_water': sbe37_std['fractional_saturation_of_oxygen_in_sea_water'].to_numpy().squeeze(),
        'speed_of_sound_in_sea_water': sbe37_std['speed_of_sound_in_sea_water'].to_numpy().squeeze()
    })

    # Add empty variable columns for future use
    sbe37_df['sea_water_ph_reported_on_total_scale'] = [np.nan for _ in range(0, n_samples)]
    sbe37_df['mass_concentration_of_chlorophyll_a_in_sea_water'] = [np.nan for _ in range(0, n_samples)]
    sbe37_df['sea_water_turbidity'] = [np.nan for _ in range(0, n_samples)]

    # Run QARTOD tests on the SBE37 data.
    qartod_valid_sensors = [
        'sea_water_pressure', 'sea_water_temperature', 'sea_water_electrical_conductivity',
        'sea_water_practical_salinity', 'sea_water_sigma_theta',
        'mass_concentration_of_oxygen_in_sea_water', 'sea_water_ph_reported_on_total_scale',
        'mass_concentration_of_chlorophyll_a_in_sea_water', 'sea_water_turbidity'
    ]
    sbe37_qcdf = nemo_qc.process_qartod_tests(
        sbe37_df,
        instrument_info.get('model', ''),
        instrument_depth,
        sample_rate,
        qartod_valid_sensors=qartod_valid_sensors
    )

    # Package a dictionary of the relavant information for the netCDF
    nemo_info = {'BuoyName': nemo_name,
                 "BuoyTitle": deployment_info.get('buoy_title', deployment_info.get('nemo_title', '')),
                 'info_url': deployment_info.get('info_url', ''),
                 'DeploymentName': deployment_name,
                 'Latitude': lat,
                 'Longitude': lon,
                 'Depth': depth,
                 "institution_info": deployment_info.get('institution_info', {}),
                 'InstrumentType': 'ctd',
                 "InstrumentInfo": instrument_info
    }

    # Loop through all the physical variables, and round to the nearest 6 decimal places for consistency with the line-parsing method.
    for var in ['sea_water_pressure', 'sea_water_temperature', 
                'sea_water_electrical_conductivity', 'sea_water_practical_salinity', 
                'sea_water_sigma_theta', 'speed_of_sound_in_sea_water', 
                'mass_concentration_of_oxygen_in_sea_water', 'fractional_saturation_of_oxygen_in_sea_water']:
        sbe37_df[var] = [np.round(ii,6) if not pd.isna(ii) else np.nan for ii in sbe37_df[var].to_numpy()]

    # Replace any NaN values with -555 for netCDF compatibility
    sbe37_df.fillna(-555, inplace=True)
    
    ########################################
    # Make the lander netCDF
    nemo_funcs.make_ctd_netCDF(nemo_info, savedir, sbe37_df, sbe37_qcdf)
    
    return


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process SBE37 data for a buoy deployment')
    parser.add_argument('--buoy', type=str,
                        help='Buoy name to process')
    parser.add_argument('--deployment', type=str,
                        help='Deployment name to process')
    parser.add_argument('--instrument', type=str,
                        help='Instrument name to process')
    parser.add_argument('--serial', type=str,
                        help='Instrument serial number to process')
    args = parser.parse_args()
    nemo_sbe37_wrapper(args.buoy, args.deployment, args.instrument, args.serial)


if __name__ == '__main__':
    main()

