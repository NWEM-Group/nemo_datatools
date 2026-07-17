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

#################################
# SBE56 Functions
#################################

def find_sbe56_headerend(sbe56_lines):

    for ii, line in enumerate(sbe56_lines):
        if line.startswith('*END*'):
            return ii
    raise ValueError('SBE56 file missing #END header line.')

def find_sbe56_instrument_start_time(sbe56_lines):

    for ii, line in enumerate(sbe56_lines):
        if line.startswith('# start_time = '):
            start_time_str = line.split('=')[1].strip()
            start_time = datetime.datetime.strptime(start_time_str, '%b %d %Y %H:%M:%S')
            return start_time
    raise ValueError('SBE56 file missing data start line.')

def find_sbe56_sample_rate(sbe56_lines):

    for ii, line in enumerate(sbe56_lines):
        if line.startswith('# interval = seconds:'):
            interval_str = line.split(':')[1].strip()
            sample_rate = float(interval_str)
            return sample_rate
    raise ValueError('SBE56 file missing sample rate line.')

def read_sbe56_nemo_dataframe(sbe56_file):
    
    # Read in the file
    with open(sbe56_file, encoding = 'utf-8', errors='replace') as f:
        lines = f.readlines()

    hdr_end_ind = find_sbe56_headerend(lines)
    startind = hdr_end_ind + 1
    instrument_start_time = find_sbe56_instrument_start_time(lines)
    instrument_start_year = instrument_start_time.year
    ref_start = datetime.datetime(instrument_start_year, 1, 1, 0, 0, 0) - datetime.timedelta(days=1)
    sample_rate = find_sbe56_sample_rate(lines)

    sbe56_rawdf = pd.read_csv(sbe56_file, names=['timestamp', 'sea_water_temperature', 'data_flag'], 
                              skiprows=startind, delim_whitespace=True, comment='#', engine='python')
    
    sbe56_rawdf['timestamp'] = pd.to_datetime(ref_start + pd.to_timedelta(sbe56_rawdf['timestamp'], unit='D'))

    return sbe56_rawdf, sample_rate



def read_sbe56_nemo_lineparsing(sbe56_file):
    
    # Read in the file
    with open(sbe56_file, encoding = 'utf-8', errors='replace') as f:
        lines = f.readlines()

    print(sbe56_file)

    hdr_end_ind = find_sbe56_headerend(lines)
    startind = hdr_end_ind + 1
    instrument_start_time = find_sbe56_instrument_start_time(lines)
    instrument_start_year = instrument_start_time.year
    ref_start = datetime.datetime(instrument_start_year, 1, 1, 0, 0, 0)
    sample_rate = find_sbe56_sample_rate(sbe56_lines)

    timestamp = []
    temper_degC = []
    dataflag = []

    for ii in range(startind,len(lines)):

        line = lines[ii].strip()
        if line == '':
            continue

        normalized_line = line.strip('"').strip()
        if normalized_line == '':
            continue

        if '\t' in normalized_line:
            linesplit = [entry.strip().strip('"') for entry in normalized_line.split('\t') if entry.strip() != '']
        elif ',' in normalized_line:
            linesplit = [entry.strip().strip('"') for entry in normalized_line.split(',') if entry.strip() != '']
        else:
            linesplit = normalized_line.split()

        if len(linesplit) < 2:
            raise ValueError(
                f"Could not parse SBE56 data line {ii + 1}: expected at least 2 fields, got {len(linesplit)}. "
                f"Line content: {line}"
            )
        
        if (linesplit[0] == 'NAN') or (linesplit[0] == '"NAN"'):
            timestamp.append(pd.NaT)
        else:
            timestamp.append(ref_start + datetime.timedelta(days=float(linesplit[0])))
            
        if (linesplit[1] == 'NAN') or (linesplit[1] == '"NAN"'):
            temper_degC.append(np.nan)
        else:
            temper_degC.append(float(linesplit[1]))
            
        if len(linesplit) < 3 or (linesplit[2] == 'NAN') or (linesplit[2] == '"NAN"'):
            dataflag.append(np.nan)
        else:
            dataflag.append(int(linesplit[2]))

    # Package the fallback output to match the wrapper's realtime expectations.
    sbe56_rawdf = pd.DataFrame(data = list(zip(timestamp,
                                               timestamp,
                                               np.arange(len(timestamp)),
                                               temper_degC,
                                               dataflag)),
                               columns = ['Timestamp', 'Instrument_Timestamp',
                                          'RecordNumber', 'Temperature_degC',
                                          'DataFlag'])
    
    return sbe56_rawdf, sample_rate

# In[ ]:


def _parse_deployment_time(time_str):
    
    if time_str is None:
        return None
    
    parsed_time = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    if parsed_time.tzinfo is not None:
        parsed_time = parsed_time.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    
    return parsed_time



def _load_sbe56_deployment_window(nemo_name, deployment_name):
    
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



def _find_sbe56_files(datadir, nemo_name, deployment_name, serial_num=None, status=None):
    """Discover SBE56 data files in datadir matching deployment and optional serial."""
    if not os.path.isdir(datadir):
        print(datadir + ' is not a valid directory. Cannot search for SBE56 files.')
        return []

    if deployment_name is None or str(deployment_name).strip() == '':
        raise ValueError('deployment_name is required to search for SBE56 files.')

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
        if not ('sbe56' in flower or 'ctd' in flower):
            continue
        if all(token in flower for token in required_tokens):
            files.append(fname)
    return sorted(files)


def _read_sbe56_temperature_only_cnv(sbe56_file):
    """Read recovered SBE56 CNV data when only time and temperature are available."""
    inst_data = sb_id.cnv_to_instrument_data(filepath=sbe56_file)
    inst_vars = inst_data.measurements.keys()
    inst_df = inst_data._to_dataframe()
    ref_start = datetime.datetime(inst_data.start_time.timetuple().tm_year, 1, 1, 0, 0, 0)
    ref_date = datetime.datetime(1970, 1, 1, 0, 0, 0)

    time_var = None
    time_units = None
    for var in inst_vars:
        if 'time' in inst_data.measurements[var].description.lower():
            time_var = var
            time_units = inst_data.measurements[var].units
            break

    if time_var is None:
        raise KeyError('Recovered SBE56 CNV file missing a time variable.')

    if time_units is not None and 'julian days' in time_units.lower():
        inst_df['date'] = [ref_start + datetime.timedelta(days=ii) for ii in inst_df[time_var]]
    elif 'date' not in inst_df.columns:
        raise KeyError('Recovered SBE56 CNV file missing a usable date/time column.')

    inst_df['timestamp'] = [np.round((ii - ref_date).total_seconds(), 6) for ii in inst_df['date']]

    temp_var = None
    temp_units = None
    for var in inst_vars:
        if inst_data.measurements[var].description.lower() == 'temperature':
            temp_var = var
            temp_units = inst_data.measurements[var].units
            break

    if temp_var is None:
        raise KeyError('Recovered SBE56 CNV file missing a temperature variable.')

    if temp_units is not None and 'ITS-90' not in temp_units:
        inst_df[temp_var] = gsw.t90_from_t68(inst_df[temp_var].to_numpy())

    sbe56_interim = inst_df.rename(columns={
        time_var: 'timestamp',
        temp_var: 'sea_water_temperature'
    })

    for var in sbe56_interim.columns:
        if var not in ['timestamp', 'date', 'sea_water_temperature']:
            sbe56_interim.drop(columns=[var], inplace=True)
    for var in sbe56_interim.columns:
        if var not in ['timestamp', 'date']:
            sbe56_interim[var] = [np.round(ii, 6) for ii in sbe56_interim[var].to_numpy()]

    return sbe56_interim

def get_deployment_serials(deployment_info, deployment_name):

    deployment_data = deployment_info.get('deployments', {}).get(deployment_name, {})
    instruments = deployment_data.get('instruments', {})
    
    sbe56_serials = []
    if isinstance(instruments, dict):
        for _, instrument_record in instruments.items():
            if not isinstance(instrument_record, dict):
                continue

            model_name = str(instrument_record.get('model', '')).strip().lower()
            serial_value = instrument_record.get('sn')
            if serial_value is None:
                continue
            if 'sbe56' not in model_name:
                continue

            serial_text = str(serial_value).strip()
            if serial_text == '':
                continue
            if serial_text not in sbe56_serials:
                sbe56_serials.append(serial_text)

    return sbe56_serials

def nemo_sbe56_wrapper(nemo_name, deployment_name, instrument_name='sbe56', serial_num=None):
    
    lat, lon, depth, start_deployment, end_deployment, status = _load_sbe56_deployment_window(
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
        sbe56_serials = get_deployment_serials(deployment_info, deployment_name)
        if len(sbe56_serials) == 0:
            print(
                f"No SBE56 serial numbers were found in deployment info for "
                f"{nemo_name} - {deployment_name}."
            )
            return

        print(
            f"No serial number provided. Found {len(sbe56_serials)} SBE56 serial(s): "
            f"{', '.join(sbe56_serials)}"
        )
        for current_serial in sbe56_serials:
            print(f"\nProcessing SBE56 serial {current_serial}...")
            nemo_sbe56_wrapper(nemo_name, deployment_name, instrument_name, current_serial)
        return

    # Discover matching data files
    sbe56_files = _find_sbe56_files(datadir, nemo_name, deployment_name, serial_num, status)
    if len(sbe56_files) == 0:
        print(f'   No SBE56 files found in {datadir}')
        return
    print(f'   Found {len(sbe56_files)} SBE56 file(s): {sbe56_files}')

    instrument_info = {}

    # Read and concatenate all found files.
    # recovered: use seabird cnv parser; realtime: try dataframe parser, then line parser fallback.
    frames = []
    for fname in sbe56_files:
        fpath = os.path.join(datadir, fname)
        try:
            
            df, sample_rate = read_sbe56_nemo_dataframe(fpath)
            frames.append(df)
        except Exception:
            print(f'   Failed standard CNV dataframe read for {fname}; try manual line parsing.')
            try:
                df, sample_rate = read_sbe56_nemo_lineparsing(fpath)
                frames.append(df)
            except Exception:
                print(f'   Failed manual line parsing for {fname}; unable to proceed.')
                return

    if len(frames) == 1:
        sbe56_rawdf = frames[0].copy()
    else:
        sbe56_rawdf = pd.concat(frames, ignore_index=True)

    # Standardize to seabird-CNV naming conventions for processing.
    sbe56_std = pd.DataFrame()


    if status == 'recovered':
        # Find the matching SBE56 instrument for this serial number when possible.
        if serial_num is not None:
            instrument_info = nemo_deployments.find_matching_deployment_instrument(
                nemo_name,
                deployment_name,
                'sbe56',
                serial_num
            )
            if instrument_info is None:
                print(
                    f"   No matching deployment instrument found for SBE56 serial {serial_num}. "
                    'Proceeding without deployment-specific time drift metadata.'
                )
                instrument_info = {}
            else:
                print(instrument_info)
        else:
            print('   Serial number not provided. Skipping deployment instrument lookup.')

        # read_sbe56_seabirdcnv provides timestamp seconds; SBE56 uses temperature only.
        if 'timestamp' in sbe56_rawdf.columns:
            sample_time = pd.to_datetime(sbe56_rawdf['timestamp'], unit='s', origin='unix', utc=True).dt.tz_localize(None)
        elif 'date' in sbe56_rawdf.columns:
            sample_time = pd.to_datetime(sbe56_rawdf['date'])
            if getattr(sample_time.dt, 'tz', None) is not None:
                sample_time = sample_time.dt.tz_localize(None)
        else:
            raise KeyError('Recovered SBE56 dataframe missing timestamp/date columns.')
        
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

        sbe56_std['sample_time'] = sample_time
        sbe56_std['instrument_timestamp'] = sample_time
        sbe56_std['record_number'] = np.arange(len(sbe56_rawdf))
        sbe56_std['sea_water_temperature'] = sbe56_rawdf['sea_water_temperature'].to_numpy()
    else:
        sbe56_std['sample_time'] = pd.to_datetime(sbe56_rawdf['Timestamp'])
        sbe56_std['instrument_timestamp'] = pd.to_datetime(sbe56_rawdf['Instrument_Timestamp'])
        sbe56_std['record_number'] = sbe56_rawdf['RecordNumber'].to_numpy()
        sbe56_std['sea_water_temperature'] = sbe56_rawdf['Temperature_degC'].to_numpy()


    # Keep data only within the deployment window.
    valid_time_mask = ~sbe56_std['sample_time'].isna()
    print(start_deployment, end_deployment)
    if start_deployment is not None:
        valid_time_mask = valid_time_mask & (sbe56_std['sample_time'] >= start_deployment)
    if end_deployment is not None:
        valid_time_mask = valid_time_mask & (sbe56_std['sample_time'] <= end_deployment)
    sbe56_std = sbe56_std[valid_time_mask]

    if len(sbe56_std) == 0:
        print('   No new SBE56 data. Continue on.')
        return
    
    # Add a "depth" variable, which corresponds to the deployment depth for SBE56 instruments.
    instrument_depth = instrument_info.get('instrument_depth', None)
    sbe56_std['depth'] = [instrument_depth for _ in range(len(sbe56_std))]

    # Repackage standardized data into the same schema used by the SBE37 workflow.
    sbe56_std = sbe56_std.reset_index(drop=True)
    n_samples = len(sbe56_std)
    sbe56_df = pd.DataFrame({
        'buoyname': [nemo_name for _ in range(0, n_samples)],
        'time': sbe56_std['sample_time'].to_numpy().squeeze(),
        'instrument_timestamp': sbe56_std['instrument_timestamp'].to_numpy().squeeze(),
        'record_number': sbe56_std['record_number'].to_numpy().squeeze(),
        'depth': sbe56_std['depth'].to_numpy().squeeze(),
        'sea_water_temperature': sbe56_std['sea_water_temperature'].to_numpy().squeeze()
    })


    # Run QARTOD tests on the SBE56 data.
    qartod_valid_sensors=['sea_water_temperature']
    sbe56_qcdf = nemo_qc.process_qartod_tests(sbe56_df, instrument_info.get('model',''), 
                                              instrument_info.get('instrument_depth',''), sample_rate,
                                              qartod_valid_sensors=qartod_valid_sensors)




    # Package a dictionary of the relevant information for the netCDF.
    nemo_info = {'BuoyName': nemo_name,
                 'BuoyTitle': deployment_info.get('buoy_title', deployment_info.get('nemo_title', '')),
                 'info_url': deployment_info.get('info_url', ''),
                 'DeploymentName': deployment_name,
                 'Latitude': lat,
                 'Longitude': lon,
                 'Depth': depth,
                 'institution_info': deployment_info.get('institution_info', {}),
                 'InstrumentType': 'sbe56',
                 'InstrumentInfo': instrument_info}

    # Round the physical variables for consistency with the SBE37 workflow.
    for var in ['sea_water_temperature']:
        sbe56_df[var] = [np.round(ii,6) if not pd.isna(ii) else np.nan for ii in sbe56_df[var].to_numpy()]

    # Replace any NaN values with -555 for netCDF compatibility.
    sbe56_df.fillna(-555, inplace=True)

    ########################################
    # Make the lander netCDF
    nemo_funcs.make_thermistor_netCDF(nemo_info, savedir, sbe56_df, sbe56_qcdf)
    
    return


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process SBE56 data for a buoy deployment')
    parser.add_argument('--buoy', type=str,
                        help='Buoy name to process')
    parser.add_argument('--deployment', type=str,
                        help='Deployment name to process')
    parser.add_argument('--instrument', type=str,
                        help='Instrument name to process')
    parser.add_argument('--serial', type=str,
                        help='Instrument serial number to process')
    args = parser.parse_args()
    nemo_sbe56_wrapper(args.buoy, args.deployment, args.instrument, args.serial)


if __name__ == '__main__':
    main()

