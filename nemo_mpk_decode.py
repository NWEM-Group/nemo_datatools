#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import getopt
import gc

import json

import shutil

from copy import deepcopy

import numpy as np
import pandas as pd

import datetime

import msgpack

curdir = os.getcwd()
if '/home/nwem/' in curdir:
    orcamain = '/home/nemo/nemo_datatools/'
else:
    orcamain = 'C:/Users/APLUser/NWEM_DataProcessing'
sys.path.insert(0, orcamain)
import nemo_general_functions as nemo


# In[ ]:


"""
Data Encoding for ORCA 2.0
https://bitbucket.org/uwaploe/sampler/src/master/doc/dataencoding.md

Message Pack python info:
https://msgpack-python.readthedocs.io/en/latest/api.html
"""


# # MessagePack file data extraction functions

# In[ ]:


def read_mpk_testfile(mpk_file):
    with open(mpk_file, 'rb') as mpk_data:
        # raw=False to decode strings as strings instead of bytes
        unpacker = msgpack.Unpacker(mpk_data, raw=False, timestamp=1) 
        msg_data = []
        for unpacked_data in unpacker:
            msg_data.append(unpacked_data)
            
    return msg_data


# In[ ]:



def check_ctd_vars(entrydata, sensortype):
    
    ctdvars = [ii for ii in entrydata.keys()]    
    new_voltage_vars = False
    if ('v0' not in ctdvars) and ('v4' in ctdvars):
        new_voltage_vars = True
        voltage_update = {'v1': 'v0', 'v2': 'v1', 'v3': 'v2', 'v4': 'v3'}
        entrydata = {voltage_update.get(k, k): v for k, v in entrydata.items()}
        ctdvars = [ii for ii in entrydata.keys()]

    reqvars = ['v0', 'v1', 'v2', 'v3', 'v4',
               'temperature', 'conductivity', 'pressure']
    
    for reqvar in reqvars:
        if reqvar not in ctdvars:
            entrydata[reqvar] = np.nan
    entrydata['sensortype'] = sensortype
            
    return entrydata

# In[ ]:

def build_engdict(entrydata):

    engdict = {}
    for key in entrydata.keys():
        entryval = entrydata[key]
        if key == 'alarms':
            req_alarms = ['ext_v12_fault', 'ext_v24_fault',
                          'hotel_12v_fault', 'leak_detect']
            if len(entryval) == 0:
                for alarm in req_alarms:
                    engdict[alarm] = False
            else:
                found_alarms = [ii.strip() for ii in entryval.split(',')]
                for alarm in req_alarms:
                    if alarm in found_alarms:
                        engdict[alarm] = True
                    else:
                        engdict[alarm] = False
        else:
            for key2 in entryval.keys():
                engdict[key2] = entryval[key2]

    return engdict


# In[ ]:


def build_datadicts(sensor_names, mpk_data):
    
    # Build an empty dictionary for all the relevant sensors
    sensor_data = {sensor_name:{'timestamp':[], 'time': [], 'data':[]}
                   for sensor_name in sensor_names}    
    sensor_datakeys = sensor_data.keys()

    reftime = datetime.datetime(1970,1,1)
    for entry in mpk_data:
        try:
            entry_data = entry['sensor']
        except:
            continue
            
        if entry['sensor'] not in sensor_datakeys:
            print(entry['sensor'], "not in sensor_datakeys")
            continue
        sensor_data[entry['sensor']]['timestamp'].append(entry['t'])
        sensor_data[entry['sensor']]['time'].append(reftime+
                                                    datetime.timedelta(microseconds=(entry['t']/1e3)))
        if (('ctd/1' == entry['sensor']) or
            ('sbe19/1' == entry['sensor']) or
            ('sbe19p/1' == entry['sensor']) or
            ('sbe37/1' == entry['sensor'])):
            sensor_data[entry['sensor']]['data'].append(check_ctd_vars(entry['data'], entry['sensor']))
        elif entry['sensor'] == 'eng/1':
            sensor_data[entry['sensor']]['data'].append(build_engdict(entry['data']))
        else:
            sensor_data[entry['sensor']]['data'].append(entry['data'])
        
    return sensor_data


# # Sensor dataframe processing functions

# In[ ]:


def build_sensor_df(sensor_data, sensor_name):
    
    data_df = pd.DataFrame(index=sensor_data[sensor_name]['time'],
                           data=sensor_data[sensor_name]['data'])
    data_df.index.names= ['time']
    data_df = data_df.reset_index()
    
    # Get the time stamps year and month
    data_df['time.year'] = [ii.year for ii in data_df.loc[:,'time']]
    data_df['time.month'] = [ii.month for ii in data_df.loc[:,'time']]
    data_df['time.day'] = [ii.day for ii in data_df.loc[:,'time']]
    data_df['timestamp'] = sensor_data[sensor_name]['timestamp']
    
    return data_df


def ensure_csv_column_order(data_df, sensor_name):

    sensor_column_order = {
        'bmon/1': ['timestamp', 'Adc_vb_f', 'Adc_va_f', 'Adc_vl_f', 'Adc_ic_f',
                   'Adc_il_f', 'T_hs', 'T_batt', 'T_amb', 'Array_fault',
                   'Vb_f', 'Vb_ref', 'Load_fault', 'Hmeter', 'Alarm'],
        'adcp/1': ['timestamp', 'fmt', 'pd0'],
        'ctd/1': ['timestamp', 'v0', 'v1', 'v2', 'v3', 'v4', 'temperature',
                  'conductivity', 'pressure', 'salinity', 'sensortype'],
        'met/1': ['timestamp', 'temperature', 'dewpoint', 'pressure',
                  'humidity', 'status', 'voltage'],
        'sbe19/1': ['timestamp', 'v0', 'v1', 'v2', 'v3', 'v4', 'temperature',
                    'conductivity', 'pressure', 'salinity', 'sensortype'],
        'sbe19p/1': ['timestamp', 'v0', 'v1', 'v2', 'v3', 'v4', 'temperature',
                     'conductivity', 'pressure', 'salinity', 'sensortype'],
        'sbe37/1': ['timestamp', 'v0', 'v1', 'v2', 'v3', 'v4', 'temperature',
                    'conductivity', 'pressure', 'salinity', 'sensortype'],
        's9/ica/1': ['timestamp', 'psi', 'inc', 'mag', 'theta', 'phi', 'tilt',
                     'temp', 'rate', 'vs'],
        's9/icaraw/1': ['timestamp', 'ax', 'ay', 'az', 'mx', 'my', 'mz',
                        'temp', 'vs'],
        's9/wind/1': ['timestamp', 'ws', 'wx', 'wy', 'wd', 'gu', 'temp',
                      'va', 'fb', 'fs', 'fv', 'vi'],
        'eng/1': ['timestamp', 'ext_v12_fault', 'ext_v24_fault',
                  'hotel_12v_fault', 'leak_detect'],
        'rad/1': ['timestamp','sensor1', 'sensor1_stddev', 'sensor2', 'sensor2_stddev',
                  'body_temp', 'op_mode', 'status_flags', 'dm_version', 'device_type'],
        'rad/2': ['timestamp','sensor1', 'sensor1_stddev', 'sensor2', 'sensor2_stddev',
                  'body_temp', 'op_mode', 'status_flags', 'dm_version', 'device_type'],
    }

    preferred_columns = sensor_column_order.get(sensor_name, ['timestamp'])
    ordered_columns = [column for column in preferred_columns if column in data_df.columns]
    remaining_columns = sorted([column for column in data_df.columns if column not in ordered_columns])

    return data_df.loc[:, ordered_columns + remaining_columns]


# In[ ]:


def load_latest_df(sensor_dir, sensor_type, filedate, loglevel=1, log=None):
    sensor_files = [file for file in os.listdir(sensor_dir) 
                    if (('.csv' in file) and (sensor_type in file))]
    if len(sensor_files) <= 0:
        return None

    sensor_dates = [datetime.datetime.strptime(file[(file.find(sensor_type)
                                                     +len(sensor_type)+1):file.find('.csv')],
                                               '%Y%m%d')
                    for file in sensor_files]

    closest_dateind = np.argmin([abs((filedate - sensor_date).total_seconds()) for sensor_date in sensor_dates])

    with open(os.path.join(sensor_dir, sensor_files[closest_dateind]), "r") as f:
        latest_df = pd.read_csv(f)
    
    # Convert time from strings to pandas timestamp objects
    ref_time = datetime.datetime(1970,1,1)
    df_times = [pd.Timestamp(ref_time + datetime.timedelta(microseconds=float(ii)/1e3)) 
                for ii in latest_df.loc[:,'timestamp']]
    latest_df.loc[:,'time'] = df_times
    
    # Get the time stamps year and month
    latest_df['time.year'] = [ii.year for ii in latest_df.loc[:,'time']]
    latest_df['time.month'] = [ii.month for ii in latest_df.loc[:,'time']]
    latest_df['time.day'] = [ii.day for ii in latest_df.loc[:,'time']]

    if loglevel >= 4:
        print('      Loaded latest data file: ' + sensor_files[closest_dateind] + ': ' 
            + latest_df['time'].min().strftime('%Y-%b-%d %H:%M:%S') +
            ' to ' + latest_df['time'].max().strftime('%Y-%b-%d %H:%M:%S'))

    return latest_df


# # General MessagePack Processing functions

# In[ ]:

def get_deployment_info(buoy_designation, basedir):
    # Load in the buoy deployment info
    with open(os.path.join(basedir,'info_jsons','deployments',
                            buoy_designation.lower()+'_deployments.json'), 'r') as file:
        depinfo = json.load(file)

    return depinfo


# In[ ]:


def load_valid_sensors():
    
    sensor_names = ['bmon/1', 'adcp/1', 'ctd/1', 'met/1',
                    'sbe19/1','sbe19p/1', 'sbe37/1',
                    'sbe37odo/1', 'sbeflbb/1',
                    'imctdo/1','imctdo/2','imctdo/3',
                    'imseafet/1', 'imseaphox/1',
                    's9/ica/1', 's9/icaraw/1', 's9/wind/1',
                    'rad/1','rad/2','eng/1', 'gps/1']
    sensor_types = ['sunsaver', 'adcp', 'surfctd', 'met',
                    'surfctd', 'surfctd', 'surfctd',
                    'surfctd', 'fluorometer',
                    'imm-ctd1','imm-ctd2','imm-ctd3',
                    'seafet', 'seaphox',
                    'compass', 'compass_raw', 'wind',
                    'pyranometer_shortwave','pyrgeometer_longwave',
                    'engineering', 'gps']
    
    return sensor_names, sensor_types


# In[ ]:


def process_mpkfile(buoy_designation, mpk_filepath, loglevel=1, log=None, rebuildFlag=False):
    
    try:
        basedir, datadir, savedir = nemo.get_mpkpaths(buoy_designation)

        mpk_filedate = datetime.datetime.strptime(mpk_filepath[mpk_filepath.find('orcadata_')+len('orcadata_'):mpk_filepath.find('.mpk')], 
                                                  '%Y%m%d-%H').replace(hour=0, minute=0, second=0, microsecond=0)
        mpk_data = read_mpk_testfile(mpk_filepath)
        
                
        if loglevel >= 4:
            print('      ' + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') 
                  + ' - Data loaded', file=log)

        sensor_names, sensor_types = load_valid_sensors()
        sensor_data = build_datadicts(sensor_names, mpk_data)


        for sensor_ind in range(0,len(sensor_names)):
            if loglevel >= 2:
                print('   Begin processing data for sensor: ' + sensor_names[sensor_ind])
            temp_df = build_sensor_df(sensor_data, sensor_names[sensor_ind])
            if (len(temp_df) == 0) or (temp_df is None):
                if loglevel >= 2:
                    print('   No data for: ' + sensor_names[sensor_ind], file=log)
                continue
            else:
                sensor_dir = os.path.join(savedir, sensor_types[sensor_ind])
                if not(os.path.exists(sensor_dir)):
                    os.mkdir(sensor_dir)

            # Try to load in the most recent dataframe, if it exists
            latest_df = load_latest_df(sensor_dir, sensor_types[sensor_ind], mpk_filedate, loglevel, log)
            if loglevel >= 4:
                print('      ' + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') 
                  + ' - Existing data loaded: ' + sensor_names[sensor_ind], file=log)
            if latest_df is not None:
                if rebuildFlag:
                    temp_df = pd.concat([temp_df, latest_df]).drop_duplicates(subset=['time'])
                else:
                    temp_df = pd.concat([latest_df, temp_df]).drop_duplicates(subset=['time'])

            # Sort the data by time
            temp_df = temp_df.sort_values(by=['time']).reset_index(drop=True)
                

            base_filename = (buoy_designation + '_' + sensor_types[sensor_ind] + '_')
            
            # Subset data by data year (likely only one year)
            df_years = np.unique(temp_df['time.year'])            
            for year in df_years:
                # Extract out the data for that specific year
                temp_yeardf = temp_df[temp_df['time.year'] == year]
                
                # Subset data the data month (likely only one month)
                df_months = np.unique(temp_yeardf['time.month'])
                for month in df_months:
                    # Extract out the data for that specific month
                    temp_monthdf = temp_yeardf[temp_yeardf['time.month']==month].reset_index(drop=True)
                    
                    # Subset the data by data day (will perhaps have multiple days)
                    df_days = np.unique(temp_monthdf['time.day'])
                    for day in df_days:
                        # Extract out data for that specifc day
                        temp_daydf = temp_monthdf[temp_monthdf['time.day']==day].reset_index(drop=True)
                        
                        # Build a sensor filename, and create the file path
                        if month < 10:
                            sensor_filename = base_filename + str(year) + '0' + str(month)
                        else:
                            sensor_filename = base_filename + str(year) + str(month)
                        if day < 10:
                            sensor_filename = sensor_filename + '0' + str(day)
                        else:
                            sensor_filename = sensor_filename + str(day)
                        sensor_filepath = os.path.join(sensor_dir, sensor_filename+'.csv')
                    
                    
                        # Write out the data file
                        if loglevel >= 4:
                            print('      ' + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') 
                                  + ' - Writing out data file: ' + sensor_filename, file=log)
                        ordered_daydf = ensure_csv_column_order(
                            temp_daydf.drop(columns=['time',
                                                     'time.year',
                                                     'time.month',
                                                     'time.day']),
                            sensor_names[sensor_ind]
                        )
                        with open(sensor_filepath, "w") as f:
                            ordered_daydf.to_csv(f, index=False,
                                                 na_rep='None',
                                                 lineterminator='\n')
                        if loglevel >= 4:
                            print('      ' + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') 
                                  + ' - Data file written: ' + sensor_filename, file=log)
    except Exception as e:
        if loglevel >= 2:
            print('An error occurred while processing the file.', file=log)
            print('Error message: ', e)
        return False
                
    return True


# # Data log updating functions

# In[ ]:


def check_processlog(buoy_designation):
    
    basedir, datadir, savedir = nemo.get_mpkpaths(buoy_designation)
    depinfo = get_deployment_info(buoy_designation, basedir)
    
    if os.path.exists(os.path.join(savedir, buoy_designation+'_mpk_filelog.csv')):
        filelog = pd.read_csv(os.path.join(savedir, buoy_designation+'_mpk_filelog.csv'))
    else:
        filelog = pd.DataFrame(data=list(zip([],[],[])),
                               columns=['filename','controller_id','processed_flag','processed_date'])

    if "<controller_id>" in datadir:
        print('Checking for controller deployments in the deployment info...')
        controller_deps = depinfo['deployments']['orca20_1']['controller_deployments']
        controller_ids = [controller_dep['controller_id'] for controller_dep in controller_deps]
        datadirs = [datadir.replace("<controller_id>", controller_id)
                    for controller_id in controller_ids]
        datastarts = [datetime.datetime.strptime(controller_dep['start_date'], "%Y-%m-%dT%H:%M:%SZ") 
                      for controller_dep in controller_deps]
        dataends = [datetime.datetime.strptime(controller_dep['end_date'], "%Y-%m-%dT%H:%M:%SZ")
                    if controller_dep['end_date'] is not None else None
                    for controller_dep in controller_deps ]
    else:
        controller_ids = [None]
        datadirs = [datadir]
        datastarts = [None]
        dataends = [None]

    total_new = 0
    for controller_id, datadir, datastart, dataend in zip(controller_ids, datadirs, datastarts, dataends):
        if not(os.path.exists(datadir)):
            print('Data directory does not exist: ' + datadir)
            continue
        
        mpk_files = sorted([file for file in os.listdir(datadir) if '.mpk' in file])
        mpk_filedatesstr = [file[file.find('orcadata_')+9:file.find('.mpk')] for file in mpk_files]
        mpk_filedates = [datetime.datetime.strptime(file, '%Y%m%d-%H') for file in mpk_filedatesstr]

        
        if datastart is not None:
            mpk_files = [mpk_files[ii] for ii in range(0,len(mpk_files)) 
                          if mpk_filedates[ii] >= datastart]
        if dataend is not None:
            mpk_files = [mpk_files[ii] for ii in range(0,len(mpk_files)) 
                          if mpk_filedates[ii] <= dataend]


        logged_files = filelog['filename'].to_list()
        newfiles = sorted(list(set(mpk_files)-set(logged_files)))
        
        if len(newfiles) == 0:
            continue
        else:
            print(datetime.datetime.strftime(datetime.datetime.now(),
                                            '%Y-%b-%d %H:%M:%S') + ': Updating the file log...')
            for newfile in newfiles:
                newrow = {'filename': newfile, 'controller_id': controller_id, 
                          'processed_flag': False, 'processed_date': None}
                filelog.loc[len(filelog),:] = newrow
            total_new += len(newfiles)

    if total_new > 0:
        with open(os.path.join(savedir, buoy_designation+'_mpk_filelog.csv'), "w") as f:
            filelog.to_csv(f, index=False, na_rep='None', lineterminator='\n')

    return 


# In[ ]:


def get_filelist_to_process(buoy_designation, rebuildDict=None):
    basedir, datadir, savedir = nemo.get_mpkpaths(buoy_designation)

    if os.path.exists(os.path.join(savedir, buoy_designation+'_mpk_filelog.csv')):
        filelog = pd.read_csv(os.path.join(savedir, buoy_designation+'_mpk_filelog.csv'),
                              keep_default_na=False, na_values='None')
    else:
        return None, None, None

    if rebuildDict is not None:
        startdate = rebuildDict['startdate']
        enddate = rebuildDict['enddate']
        for fileind in range(0,len(filelog)):
            filedate_str = filelog.loc[fileind,'filename'][filelog.loc[fileind,'filename'].find('orcadata_')+len('orcadata_'):filelog.loc[fileind,'filename'].find('.mpk')]
            filedate = datetime.datetime.strptime(filedate_str, '%Y%m%d-%H').replace(hour=0, minute=0, second=0, microsecond=0)
            if (filedate >= startdate) and (filedate <= enddate):
                filelog.loc[fileind,'processed_flag'] = False
                filelog.loc[fileind,'processed_date'] = None
    filelist = sorted(filelog[filelog['processed_flag']==False]['filename'].to_list())
    controller_ids = sorted(filelog[filelog['processed_flag']==False]['controller_id'].to_list())
    
    return filelog, filelist, controller_ids


# # Message Pack Decoder Wrapper

# In[ ]:


def decode_buoy_msgpack_files_wrapper(buoy_designation, rebuildDict=None, loglevel=1, log=None):

    basedir, datadir, savedir = nemo.get_mpkpaths(buoy_designation)
    check_processlog(buoy_designation)
    filelog, filelist, controller_ids = get_filelist_to_process(buoy_designation, rebuildDict)

    for ind, mpk_file in enumerate(filelist):
        controller_id = controller_ids[ind]
        if loglevel >= 2:
            print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') + 
                  ': Processing ' + buoy_designation + ' mpk file: ' + mpk_file, file=log)
        if controller_id is None:
            mpk_filepath = os.path.join(datadir.replace("<controller_id>", controller_id), mpk_file)
        else:
            mpk_filepath = os.path.join(datadir, mpk_file)

        successFlag = process_mpkfile(buoy_designation, mpk_filepath, loglevel=loglevel, 
                                      log=log, rebuildFlag=rebuildDict is not None)
        if successFlag:
            if loglevel >= 2:
                print('   ...Processing successful! - ' + 
                      datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S'), file=log)
            update_row = {'filename': mpk_file,
                          'processed_flag': True,
                          'processed_date': datetime.datetime.strftime(datetime.datetime.now(),
                                                                       '%Y-%m-%d %H:%M:%S')}
            logindex = filelog[filelog['filename']==mpk_file].index[0]
            filelog.loc[logindex,:] = update_row
        else:
            if loglevel >= 2:
                print('   *** AN ERROR OCCURRED IN FILE PROCESSING ***', file=log)

    with open(os.path.join(savedir, buoy_designation+'_mpk_filelog.csv'), "w") as f:
        filelog.to_csv(f, index=False, na_rep='None', lineterminator='\n')
    
    return


# In[ ]:


def load_sensor_datatypes(sensor_name):
    
    datatypes = {
        "sensor": {
            'sbe37/1':
            {
                "temperature": "float64",
                "conductivity": "float64",
                "pressure": "float64",
                "salinity": "float64",
            },
            'sbe19/1':
            {
                "temperature": "float64",
                "conductivity": "float64",
                "pressure": "float64"
            },
            'sbe19p/1':
            {
                "v1": "float64",
                "v2": "float64",
                "v3": "float64",
                "v4": "float64",
                "temperature": "float64",
                "conductivity": "float64",
                "pressure": "float64"
            },
            'sbe37odo/1':
            {
                "temperature": "float64",
                "conductivity": "float64",
                "pressure": "float64",
                "doxy": "float64",
                "salinity": "float64",
            },
            'sbeflbb/1':
            {
                "chl-temp": "float64",
                "chl-raw": "float64",
                "chl": "float64",
                "bs-temp": "float64",
                "bs-raw": "float64",
                "bs-eng": "float64",
                "bs": "float64",
                "v": "float64",
            },
            'adcp/1':
            {
                "fmt": "string",
                "pd0": "bytearray"
            },
            'met/1':
            {
                "temperature": "float32",
                "dewpoint": "float32",
                "pressure": "float32",
                "humidity": "float32",
                "status": "int8",
                "voltage": "float32"
            },
            's9/wind/1':
            {
                "ws": "float32",
                "wx": "float32",
                "wy": "float32",
                "wd": "float32",
                "gu": "float32",
                "temp": "float32",
                "va": "int32",
                "fb": "int32",
                "fs": "int32",
                "fv": "int32",
                "vi": "float32"
            },
            's9/ica/1':
            {
                "psi": "float32",
                "inc": "float32",
                "mag": "float32",
                "theta": "float32",
                "phi": "float32",
                "tilt": "float32",
                "temp": "float32",
                "rate": "int32",
                "vs": "int32"
            },
            's9/icaraw/1':
            {
                "ax": "float32",
                "ay": "float32",
                "az": "float32",
                "mx": "float32",
                "my": "float32",
                "mz": "float32",
                "temp": "float32",
                "vs": "int32"
            },
            'bmon/1':
            {
                "Adc_vb_f": "float32",
                "Adc_va_f": "float32",
                "Adc_vl_f": "float32",
                "Adc_ic_f": "float32",
                "Adc_il_f": "float32",
                "T_hs": "int",
                "T_batt": "int",
                "T_amb": "int",
                "Array_fault": "uint",
                "Vb_f": "float32",
                "Vb_ref": "float32",
                "Load_fault": "uint",
                "Hmeter": "uint",
                "Alarm": "uint"
            },
            'gps/1':
            {
                "lat": "float64",
                "lon": "float64",
                "hdop": "float64"
            }
        }
    }
    
    if not(any([sensor_name == sensor for sensor in datatypes['sensor'].keys()])):
        print('Sensor type does not have any data types. Check if these need to be added')
        return None
    
    
    return datatypes['sensor'][sensor_name]


# # Main Processing Code:

# In[ ]:

def help_func():
    helptext = '''
    ORCA MessagePack Decoder
    -------------------------
    Options:
    -h, --help              : Show this help message
    -b, --buoy              : Buoy designation (e.g., ORCA1, ORCA2, ORCA3, ORCA4, NPBY1, NPBY2, PISCES1, PISCES2, PISCES3)
    -r, --rebuild           : Rebuild flag (True/False). Default is False. 
                              If True, additional arguments required: startdate=YYYY-MM-DD enddate=YYYY-MM-DD
    -l, --loglevel          : Log level (1-4). Default is 1.

    Example Usage:
    
    Basic run for ORCA1 buoy:
    python ORCA_MessagePack_Decode.py -b ORCA1
    
    Rebuild data for ORCA1 buoy from Jan 1, 2023 to Jan 31, 2023 with log level 2:
    python ORCA_MessagePack_Decode.py -b ORCA1 -r True startdate=2023-01-01 enddate=2023-01-31 -l 2
    '''
    print(helptext)
    return


def main():
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hu:b:l:r:", ["help", "buoy=", "rebuild=", "loglevel="])
    except Exception as inst:
        # print help information and exit:
        print('Error in getting options: ' + str(inst)) # will print something like "option -a not recognized"
        sys.exit(2)
        
    
    # step through command-line options and their arguments
    buoyName = False
    logFlag = False
    rebuildFlag = False
    
    # For all the command-line arguments, update a flag to indicate if the option was given,
    # and extract the value of the command-line argument
    for o, a in opts:
        if o == "-v":
            bVerbose = True
            # but currently verbose run is not implemented!
        elif o in ("-h", "--help"):
            help_func()
            sys.exit(0)
        elif o in ("-b", "--buoy"):
            buoyName = True
            buoy_designation = a
        elif o in ("-r", "--rebuild"):
            rebuildFlag = True
            rebuild = a
        elif o in ("-l", "--loglevel"):
            logFlag = True
            loglevel = a
        else:
            assert False, "unhandled option"
    
    ###############################
    # General processing function #
    # No logging required         #
    ###############################
    log = None
    print('\n')
        
    # If the buoy designation was given, ensure that the buoy designation is in uppercase,
    # and that it matches one of the valid buoy designations
    valid_buoys = ['chaba','nemoss']
    if buoyName:
        buoy_designation = buoy_designation.lower()
        if buoy_designation in valid_buoys:
            print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') 
                  + ': Selected buoy - ' + buoy_designation, file=log)
        else:
            help_func()
            print('Invalid buoy designation. Check name, and retry.', file=log)
            sys.exit(2)
    else:
        help_func()
        print('No buoy designation given. Try again with options: "-b *buoy_designation*"')
        sys.exit(2)
        
    # If a log level was given, ensure that it is valid;
    # otherwise, use a default log level of 1
    if logFlag:
        loglevel = int(loglevel)
        if loglevel > 4:
            loglevel = 4
        elif loglevel < 1:
            loglevel = 1
    else:
        loglevel=1

    # If a rebuild flag was given, ensure that it is valid;
    # otherwise, use a default rebuild flag of False
    if rebuildFlag:
        if rebuild.lower() == 'true':
            rebuildFlag = True
        else:
            rebuildFlag = False
    else:
        rebuildFlag = False
    
    # If the rebuild flag is True, get additional arguments
    # to get a start and end date for rebuilding data
    rebuildDict = None
    if rebuildFlag:
        rebuildDict = {}
        startdate=datetime.datetime(2025,7,1)
        enddate=datetime.datetime.now()+datetime.timedelta(hours=7)
        for arg in args:
            if 'startdate=' in arg.lower():
                startdate_str = arg[arg.find('=')+1:]
                startdate = datetime.datetime.strptime(startdate_str, '%Y-%m-%d')
            elif 'enddate=' in arg.lower():
                enddate_str = arg[arg.find('=')+1:]
                enddate = datetime.datetime.strptime(enddate_str, '%Y-%m-%d')
        rebuildDict = {'startdate': startdate, 'enddate': enddate}
        print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') + 
              ': Rebuild flag set to True for dates: ' + 
              startdate.strftime('%Y-%b-%d') + ' to ' + enddate.strftime('%Y-%b-%d'), file=log)
        
    
    ##################################################################
    # Begin processing new data for a given buoy
    print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') + 
          ': Begin decoding MessagePack files for: ' + buoy_designation,file=log)
    try:
        decode_buoy_msgpack_files_wrapper(buoy_designation, rebuildDict, loglevel, log)
        print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') + ': File decoding completed.',file=log)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('Error occurred in new data processing: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'),file=log)
        print(e,file=log)
        print('Error occurs on line ' + str(exc_tb.tb_lineno), file=log)
        
    
    if log is not None:
        log.close()    
        # Move the log to the appropriate folder
        curdir = os.getcwd()
        logdir = nwem.get_info_json()['logs']
        os.replace(os.path.join(curdir,logfile), 
                   os.path.join(logdir, buoy_designation.lower(), logfile))
        
    print('\n')
    sys.exit(0)


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




