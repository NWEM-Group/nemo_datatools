#!/usr/bin/env python
# coding: utf-8
# %%

import numpy as np
import pandas as pd
import xarray as xr
import scipy

import datetime
import time as pytime

import netCDF4
from netCDF4 import Dataset

import json

import os
import sys
import getopt
import gc
import shutil

import requests
from urllib import request

import matplotlib
from matplotlib import pyplot as plt

import gsw

curdir = os.getcwd()
if '/home/nwem/' in curdir:
    orcamain = '/home/nemo/nemo_datatools/'
else:
    orcamain = 'C:/Users/APLUser/NWEM_DataProcessing'
sys.path.insert(0, orcamain)
import nemo_general_functions as nemo
import nemo_qartod as nemo_qc

import ioos_qc
from ioos_qc import qartod
from ioos_qc.config import Config
from ioos_qc.streams import PandasStream
from ioos_qc.stores import PandasStore

# # General Processing Functions 

# %%



def get_procdirs(buoy_designation):

    
    procdir, basedir, datadir = nemo.get_mpkpaths(buoy_designation)
    savedir = nemo.get_info_json()['data_savedir']
    savedir = os.path.join(savedir, 'realtime', buoy_designation, 'ctd')
    depdir = os.path.join(procdir, 'info_jsons', 'deployments')
    
    return procdir, basedir, datadir, savedir, depdir


def drop_nondeploy_inds(data_df, dep_starttime, dep_endtime):

    if data_df is None:
        return None

    if any(np.logical_not(np.logical_and(data_df['time'] >= dep_starttime,
                                         data_df['time'] <= dep_endtime))):

        if all(np.logical_not(np.logical_and(data_df['time'] >= dep_starttime,
                                             data_df['time'] <= dep_endtime))):
            print('All data is outside a deployment time. Do not create netCDF.')
            return None

        dropinds = np.where(np.logical_not(np.logical_and(data_df['time'] >= dep_starttime,
                                                          data_df['time'] <= dep_endtime)))[0]
        data_df = data_df.drop(index=dropinds).reset_index(drop=True)

    return data_df


# %%


def load_latest_file(buoy_designation, savedir, startdate = None, ctd_pos='surfctd'):

    def _load_ctd_nc_as_time_df(nc_path):
            ds = xr.load_dataset(nc_path)
            if 'location_id' in ds.dims:
                # This workflow expects a single platform row; collapse to time-only.
                ds = ds.isel(location_id=0, drop=True)
            if 'location_id' in ds.coords:
                ds = ds.drop_vars('location_id')
    
            df = ds.to_dataframe()
            # Some files already expose 'time' as a dataframe column; avoid duplicate insertion.
            if 'time' in df.columns:
                df = df.reset_index(drop=True)
            else:
                df = df.reset_index()
    
            if 'time' not in df.columns and 'time' in ds.variables:
                df.loc[:, 'time'] = pd.to_datetime(ds['time'].values)
    
            return ds, df

    latest_date, latest_path, prior_path = find_latest_datafile(buoy_designation, savedir, startdate, ctd_pos)
    if latest_date is None:
        return None, None

    ds_latest, ctd_latest = _load_ctd_nc_as_time_df(latest_path)
    ctd_latest = (ctd_latest.drop_duplicates(subset='time')
                  .sort_values(by='time')
                  .reset_index(drop=True))
    ctd_latest = round_vars(ctd_latest)
    
    prior_path = None
    
    if prior_path is not None:
        ctd_prior = xr.load_dataset(prior_path).to_pandas().reset_index()
        ctd_prior = (ctd_prior.drop_duplicates(subset='time')
                      .sort_values(by='time')
                      .reset_index(drop=True))
        ctd_prior = round_vars(ctd_prior)
        
        ctd_latest = (pd.concat([ctd_prior, ctd_latest]).drop_duplicates(subset='time')
                      .sort_values(by='time')
                      .reset_index(drop=True))
        
    # Drop out data that is within the last 3 hours
    checktime = (datetime.datetime.now() + datetime.timedelta(hours=7)
                 - datetime.timedelta(hours=3))
    ctd_latest = ctd_latest.loc[ctd_latest.loc[:,'time']<=checktime,:].reset_index(drop=True)
        
    # Do a double check to remove duplicate points
    ctd_latest = ctd_latest.drop_duplicates()
    ctd_latest = ctd_latest.reset_index(drop=True)
        
    required_vars = ['sea_water_temperature',
       'sea_water_electrical_conductivity',
       'sea_water_pressure',
       'mass_concentration_of_oxygen_in_sea_water',
       'sea_water_ph_reported_on_total_scale',
       'sea_water_turbidity',
       'mass_concentration_of_chlorophyll_a_in_sea_water',
       'sea_water_turbidity_std',
       'mass_concentration_of_chlorophyll_a_in_sea_water_std',
       'fractional_saturation_of_oxygen_in_sea_water',
       'sea_water_practical_salinity',
       'sea_water_sigma_theta']
    
    req_qcvar = ['qc_gross_range_test', 'qc_spike_test',
                 'qc_rate_of_change_test', 'qc_flat_line_test',
                 'qc_aggregate']
    ctd_latest_cols = ctd_latest.columns
    blankqc = 9*np.ones(len(ctd_latest))
    for var in required_vars:
        for qc_var in req_qcvar:
            if (var+'_'+qc_var) not in ctd_latest_cols:
                ctd_latest.loc[:,var+'_'+qc_var] = blankqc
                
    
    return ctd_latest, ctd_latest.loc[len(ctd_latest)-1,'time'].to_pydatetime()

## Variable processing functions

def oxy_umolkg_to_mgL(oxy_umolkg, sigmaTheta):

    # Convert oxygen concentration from units of
    # micromoles oxygen/kilogram water to
    # milligrams oxygen/Liter water

    # Per Sea-Bird Application Note 64 (Rev Jun 2013)
    # Data Conversions
    # Sea-Bird uses the following equations to convert oxygen to various engineering units:
    # [mg/L] = [ml/L] * 1.42903
    # [µmole/Kg] = [ml/L] * 44660 / (sigma_theta(P=0,Theta,S) + 1000
    mL_to_mg_fac = 1.42903

    oxy_mgL = mL_to_mg_fac * ((oxy_umolkg / 44660) * (sigmaTheta + 1000))

    return oxy_mgL


# # Instrument processing functions
def process_sbe37data(sbe37_df, depinfo, instinfo, timePeriod='recent'):

    if sbe37_df is None or sbe37_df.shape[0] == 0:
        print('No data in file. Return without processing.')
        return None

    # Parse the data timestamps and convert into python datetimes
    ref_date = datetime.datetime(1970,1,1)
    sbe37_time = [ref_date + datetime.timedelta(microseconds=(ii/1e3)) for ii in sbe37_df['timestamp']]
    sbe37_df['time'] = sbe37_time

    # Drop out any "all nan" rows
    sbe37_df[sbe37_df==-1] = np.nan
    sbe37_df = sbe37_df.dropna(axis=0, how='all', 
                               subset=sbe37_df.columns[1:-1],
                               ignore_index=True).reset_index(drop=True)

    # Drop any duplicates in the "HCAT Time" field
    sbe37_df = sbe37_df.drop_duplicates(subset='time', keep='first').reset_index(drop=True)

    # Check the oxygen column name:
    convert_mll_to_mgl = False
    if ('dissolved_oxy (mg/l)' in sbe37_df.columns):
        oxycol = 'dissolved_oxy (mg/l)'
    elif ('dissolved_oxy (ml/l)' in sbe37_df.columns):
        oxycol = 'dissolved_oxy (ml/l)'
        convert_mll_to_mgl = True
    elif ('dissolved_oxy' in sbe37_df.columns):
        oxycol = 'dissolved_oxy'
    elif ('dissovled_oxy' in sbe37_df.columns):
        oxycol = 'dissovled_oxy'
    elif 'doxy' in sbe37_df.columns:
        oxycol = 'doxy'
    else:
        oxycol = None
    if oxycol is None:
        blankoxy = [np.nan for ii in range(0,len(sbe37_df))]
        oxycol = 'do_temp'
        sbe37_df[oxycol] = blankoxy
    if instinfo['o2_unit'] == 'ml/L':
        convert_mll_to_mgl = True
    elif instinfo['o2_unit'] == 'mg * L^-1':
        convert_mll_to_mgl = False

    # Drop out any "all nan" rows
    sbe37_df[sbe37_df==-1] = np.nan

    subset_cols = ['temperature', 'conductivity', 'pressure', oxycol]
    sbe37_df = sbe37_df.dropna(axis=0, how='all', 
                               subset=subset_cols,
                               ignore_index=True).reset_index(drop=True)
    # *** Note, the oxygen data is likely actually in ml/L, not mg/L ***

    if sbe37_df is None or sbe37_df.shape[0] == 0:
        print('No data in file. Return without processing.')
        return None

    sbe37_df['depth'] = [instinfo['instrument_depth'] for ii in range(0,len(sbe37_df))]

    # Ensure that salinity is present in the dataframe, even if it needs to be filled with NaNs
    if 'salinity' not in sbe37_df.columns:
        sbe37_df['salinity'] = gsw.SP_from_C([(ii*1000/100) for ii in sbe37_df['conductivity']],
                                             sbe37_df['temperature'], sbe37_df['pressure'])

    sbe37_df = sbe37_df.rename({'conductivity': 'sea_water_electrical_conductivity', 
                                'temperature': 'sea_water_temperature', 
                                'salinity': 'sea_water_practical_salinity', 
                                'pressure': 'sea_water_pressure',
                                oxycol: 'mass_concentration_of_oxygen_in_sea_water'},
                               axis=1)


    # Per Sea-Bird Application Note 64 (Rev Jun 2013)
    # Data Conversions
    # Sea-Bird uses the following equations to convert oxygen to various engineering units:
    # [mg/L] = [ml/L] * 1.42903
    if convert_mll_to_mgl:
        sbe37_df['mass_concentration_of_oxygen_in_sea_water'] = 1.42903 * sbe37_df['mass_concentration_of_oxygen_in_sea_water']

    SP = gsw.SP_from_C([(ii*1000/100) for ii in sbe37_df['sea_water_electrical_conductivity']],
                       sbe37_df['sea_water_temperature'], sbe37_df['sea_water_pressure'])
    SA = gsw.SA_from_SP(SP, sbe37_df['sea_water_pressure'], 
                        depinfo['longitude'], depinfo['latitude'])
    CT = gsw.CT_from_t(SA, sbe37_df['sea_water_temperature'], 
                       sbe37_df['sea_water_pressure'])
    sigma0 = gsw.sigma0(SA, CT)

    dosat_umolkg = gsw.O2sol(SA, CT, sbe37_df['sea_water_pressure'],
                             depinfo['longitude'], depinfo['latitude'])
    dosat_mgL = oxy_umolkg_to_mgL(dosat_umolkg, sigma0)
    dosat_per = 100*sbe37_df['mass_concentration_of_oxygen_in_sea_water']/dosat_mgL

    sbe37_df['sea_water_practical_salinity'] = SP
    sbe37_df['sea_water_sigma_theta'] = sigma0
    sbe37_df['fractional_saturation_of_oxygen_in_sea_water'] = dosat_per

    dep_starttime = datetime.datetime.strptime(depinfo['start'],'%Y-%m-%dT%H:%M:%SZ')
    dep_endtime = (datetime.datetime.strptime(depinfo['end'],'%Y-%m-%dT%H:%M:%SZ') if depinfo['end'] else 
                   datetime.datetime.now() + datetime.timedelta(hours=12))
    sbe37_df = drop_nondeploy_inds(sbe37_df, dep_starttime, dep_endtime)

    return sbe37_df


# In[ ]:


def process_ecov2data(ecov2_df, depinfo, instinfo, timePeriod='recent'):

    # ecov2_df is now passed directly to the function, no need to read from CSV

    if ecov2_df is None or ecov2_df.shape[0] == 0:
        print('No data in file. Return without processing.')
        return None

    # Parse the data timestamps and convert into python datetimes
    ref_date = datetime.datetime(1970,1,1)
    ecov2_time = [ref_date + datetime.timedelta(microseconds=(ii/1e3)) for ii in ecov2_df['timestamp']]
    ecov2_df['time'] = ecov2_time

    # Drop out any "all nan" rows
    ecov2_df[ecov2_df==-1] = np.nan
    ecov2_df = ecov2_df.dropna(axis=0, how='all', 
                               subset=ecov2_df.columns[1:-1],
                               ignore_index=True).reset_index(drop=True)

    # Drop any duplicates in the "HCAT Time" field
    ecov2_df = ecov2_df.drop_duplicates(subset='time', keep='first').reset_index(drop=True)

    # Drop all the engineering based variables
    ecov2_df = ecov2_df.drop(columns=['chl-temp','chl-raw',
                                      'bs-temp', 'bs-raw', 'bs-eng', 'v'])

    if ecov2_df is None or ecov2_df.shape[0] == 0:
        print('No data in file. Return without processing.')
        return None

    ecov2_df['depth'] = [instinfo['instrument_depth']
                         for ii in range(0,len(ecov2_df))]

    # Rename the hydrocat variable names
    ecov2_df = ecov2_df.rename({'bs': 'sea_water_turbidity', 
                                'chl': 'mass_concentration_of_chlorophyll_a_in_sea_water'},
                               axis=1)

    dep_starttime = datetime.datetime.strptime(depinfo['start'],'%Y-%m-%dT%H:%M:%SZ')
    dep_endtime = (datetime.datetime.strptime(depinfo['end'],'%Y-%m-%dT%H:%M:%SZ') if depinfo['end'] else 
                    datetime.datetime.now() + datetime.timedelta(hours=12))
    ecov2_df = drop_nondeploy_inds(ecov2_df, dep_starttime, dep_endtime)

    return ecov2_df

def process_seaphoxdata(seaphox_df, depinfo, instinfo, timePeriod='recent'):

    # seaphox_df is now passed directly to the function, no need to read from CSV

    if seaphox_df is None or seaphox_df.shape[0] == 0:
        print('No data in file. Return without processing.')
        return None

    # Parse the data timestamps and convert into python datetimes
    ref_date = datetime.datetime(1970,1,1)
    seaphoxtime = [ref_date + datetime.timedelta(microseconds=(ii/1e3)) for ii in seaphox_df['timestamp']]
    seaphox_df['time'] = seaphoxtime


    # Drop out any "all nan" rows
    seaphox_df[seaphox_df==-1] = np.nan
    seaphox_df = seaphox_df.dropna(axis=0, how='all', 
                               subset=seaphox_df.columns[1:-1],
                               ignore_index=True).reset_index(drop=True)

    # Drop any duplicates in the "HCAT Time" field
    seaphox_df = seaphox_df.drop_duplicates(subset='time', keep='first').reset_index(drop=True)


    dep_starttime = datetime.datetime.strptime(depinfo['start'],'%Y-%m-%dT%H:%M:%SZ')
    dep_endtime = (datetime.datetime.strptime(depinfo['end'],'%Y-%m-%dT%H:%M:%SZ') if depinfo['end'] else 
                   datetime.datetime.now() + datetime.timedelta(hours=12))
    seaphox_df = drop_nondeploy_inds(seaphox_df, dep_starttime, dep_endtime)

    if seaphox_df is None or seaphox_df.shape[0] == 0:
        print('No data in file. Return without processing.')
        return None

    seaphox_df['depth'] = [instinfo['instrument_depth']
                           for ii in range(0,len(seaphox_df))]

    # Check the oxygen column name:
    convert_mll_to_mgl = False
    if ('dissolved_oxy (mg/l)' in seaphox_df.columns):
        oxycol = 'dissolved_oxy (mg/l)'
    elif ('dissolved_oxy (ml/l)' in seaphox_df.columns):
        oxycol = 'dissolved_oxy (ml/l)'
        convert_mll_to_mgl = True
    elif ('dissovled_oxy' in seaphox_df.columns):
        oxycol = 'dissovled_oxy'
    elif 'doxy' in seaphox_df.columns:
        oxycol = 'doxy'
    else:
        oxycol = None
    if oxycol is None:
        blankoxy = [np.nan for ii in range(0,len(seaphox_df))]
        oxycol = 'dissolved_oxy (mg/l)'
        seaphox_df[oxycol] = blankoxy
    if instinfo['o2_unit'] == 'ml/L':
        convert_mll_to_mgl = True
    elif instinfo['o2_unit'] == 'mg * L^-1':
        convert_mll_to_mgl = False


    # Rename the hydrocat variable names
    seaphox_df = seaphox_df.rename({'time': 'time', 
                                    oxycol:'mass_concentration_of_oxygen_in_sea_water',
                                    'temperature': 'sea_water_temperature', 
                                    'pressure': 'sea_water_pressure',
                                    'salinity': 'sea_water_practical_salinity', 
                                    'external_pH': 'sea_water_ph_reported_on_total_scale'},
                                   axis=1)

    # Use temperature and salinity to calculate potential density
    cond = gsw.C_from_SP(seaphox_df['sea_water_practical_salinity'],
                         seaphox_df['sea_water_temperature'],
                         seaphox_df['sea_water_pressure'])
    SA = gsw.SA_from_SP(cond, seaphox_df['sea_water_pressure'],
                        depinfo['longitude'], depinfo['latitude'])
    CT = gsw.CT_from_t(SA, seaphox_df['sea_water_temperature'], 
                       seaphox_df['sea_water_pressure'])
    sigma0 = gsw.sigma0(SA, CT)

    # Per Sea-Bird Application Note 64 (Rev Jun 2013)
    # Data Conversions
    # Sea-Bird uses the following equations to convert oxygen to various engineering units:
    # [mg/L] = [ml/L] * 1.42903
    if instinfo['o2_unit'] == 'ml/L':
        seaphox_df['mass_concentration_of_oxygen_in_sea_water'] = 1.42903 * seaphox_df['mass_concentration_of_oxygen_in_sea_water']


    dosat_umolkg = gsw.O2sol(SA, CT, seaphox_df['sea_water_pressure'],
                             depinfo['longitude'], depinfo['latitude'])
    dosat_mgL = oxy_umolkg_to_mgL(dosat_umolkg, sigma0)
    dosat_per = 100*seaphox_df['mass_concentration_of_oxygen_in_sea_water']/dosat_mgL

    seaphox_df['sea_water_electrical_conductivity'] = [(ii*100/1000) for ii in cond]
    seaphox_df['sea_water_sigma_theta'] = sigma0
    seaphox_df['fractional_saturation_of_oxygen_in_sea_water'] = dosat_per

    return seaphox_df

def process_ctd_files(buoy_designation, ctdpaths, dep_starts, 
                      inst_pos=None, dep_info=None, buoy_info=None, ctd_pos='surfctd'):


    inst_info = dep_info['instruments'][inst_pos]
    inst_type = inst_info['model']
    
    buoy_name = buoy_info['buoy_name']
    buoy_depth = buoy_info['depth']
    buoy_depthstr = str(buoy_depth)
    buoy_lat = buoy_info['latitude']
    buoy_latstr = str(buoy_lat)
    buoy_lon = buoy_info['longitude']
    buoy_lonstr = str(buoy_lon)
    
    ctd_df = None
    for ind in range(0,len(ctdpaths)):
        if ctdpaths[ind] is None:
            continue
        else:
            temp_df = pd.read_csv(ctdpaths[ind])
        
        if ctd_df is None:
            ctd_df = temp_df.copy()
        else:
            ctd_df = pd.concat([ctd_df, temp_df]).reset_index(drop=True)
        
    inst_types = [inst_type for _ in range(len(ctd_df))]
    ctd_df['instrument_type'] = inst_types

    if 'sbe37' in inst_info['model'].lower():
        ctd_df = process_sbe37data(ctd_df, dep_info, inst_info)
    elif 'ecov2' in inst_info['model'].lower():
        ctd_df = process_ecov2data(ctd_df, dep_info, inst_info)
    elif 'seaphox' in inst_info['model'].lower():
        ctd_df = process_seaphoxdata(ctd_df, dep_info, inst_info)
    if ctd_df is None or ctd_df.shape[0] == 0:
        return None, None

    
    ctd_df, _ = add_missing_cols(ctd_df)
    ctd_df = drop_unnecessary_vars(ctd_df)
    ctd_df = round_vars(ctd_df)
    
    #############################################################
    # Perform QARTOD tests on the data on the full dataset
    # and null out the particularly bad datapoints
    depth_level = 1.0
    sample_rate=60*10
    sensor_names = ['sea_water_temperature',
                     'sea_water_electrical_conductivity',
                     'sea_water_pressure',
                     'sea_water_practical_salinity',
                     'sea_water_sigma_theta',
                     'mass_concentration_of_oxygen_in_sea_water',
                     'mass_concentration_of_chlorophyll_a_in_sea_water',
                     'sea_water_turbidity']
    ctd_qc_df = ctd_qc_df = process_hydroqartod_tests(ctd_df, sensor_names, 
                                                      depth_level, sample_rate)
    
    qc_vars = ['qc_aggregate', 
               'qc_gross_range_test', 'qc_rate_of_change_test',
               'qc_spike_test', 'qc_flat_line_test',
               'qc_manually_flagged_test']
    
    ctd_df, ctd_qc_df = add_missing_cols(ctd_df, ctd_qc_df, qc_vars=qc_vars)
    

    # Fill NaNs with fillval, if needed
    fillVal = -555
    ctd_df = ctd_df.fillna(fillVal)
    ctd_qc_df = ctd_qc_df.fillna(9)
    
    return ctd_df, ctd_qc_df


# %%


def round_vars(ctd_df):
    
    datacols = ctd_df.columns
    
    datavars = ['sea_water_temperature',
                'sea_water_electrical_conductivity',
                'sea_water_pressure',
                'mass_concentration_of_oxygen_in_sea_water',
                'sea_water_ph_reported_on_total_scale',
                'sea_water_turbidity',
                'mass_concentration_of_chlorophyll_a_in_sea_water',
                'sea_water_turbidity_std',
                'mass_concentration_of_chlorophyll_a_in_sea_water_std',
                'fractional_saturation_of_oxygen_in_sea_water',
                'sea_water_practical_salinity',
                'sea_water_sigma_theta']
    for col in datacols:
        if any([col == datavar for datavar in datavars]):
            ctd_df.loc[:,col] = np.round(ctd_df.loc[:,col],4)
            
    return ctd_df            


# %%


def drop_unnecessary_vars(data_df):
    
    unnecessary_vars = ['v0','v1','v2','v3','v4',
                        'salinity','timestamp']
    
    for var in unnecessary_vars:
        if any([var == col for col in data_df.columns]):
            data_df = data_df.drop(columns=[var])
            
    return data_df


# %%


def add_missing_cols(data_df, data_qcdf=None, qc_vars=None):
    
    df_cols = data_df.columns
    nrows = len(data_df)
    emptycol = np.nan*np.zeros(nrows)
    emptyqc = 9*np.ones(nrows).astype(int)
    
    needed_cols = ['sea_water_temperature',
                   'sea_water_electrical_conductivity',
                   'sea_water_pressure',
                   'mass_concentration_of_oxygen_in_sea_water',
                   'sea_water_ph_reported_on_total_scale',
                   'sea_water_turbidity',
                   'mass_concentration_of_chlorophyll_a_in_sea_water',
                   'sea_water_turbidity_std',
                   'mass_concentration_of_chlorophyll_a_in_sea_water_std',
                   'fractional_saturation_of_oxygen_in_sea_water',
                   'sea_water_practical_salinity',
                   'sea_water_sigma_theta']
    
    for col in needed_cols:
        if not(any([ii==col for ii in df_cols])):
            data_df.loc[:, col] = emptycol
            
            if data_qcdf is not None:
                for qccol in qc_vars:
                    data_qcdf.loc[:, col+'_'+qccol] = emptyqc
                    
    return data_df, data_qcdf        


# # QARTOD Functions

# %%


def process_hydroqartod_tests(var_data, sensor_names, depth_level,
                              sample_rate=60*10):
    
    qartod_valid_sensors = ['sea_water_temperature', 
                            'sea_water_electrical_conductivity', 
                            'sea_water_practical_salinity',
                            'sea_water_sigma_theta',
                            'sea_water_pressure', 
                            'mass_concentration_of_oxygen_in_sea_water',
                            'sea_water_ph_reported_on_total_scale', 
                            'sea_water_turbidity',
                            'mass_concentration_of_chlorophyll_a_in_sea_water']
    qartod_df = []
    NT = var_data.shape[0]
    
    reftime = datetime.datetime(1970,1,1)
    vardates = [pd.Timestamp(ii).to_pydatetime() 
                  for ii in var_data['time']]
    
    for sensor in sensor_names:
        if (sensor in qartod_valid_sensors):
            if not((all(var_data[sensor].isna())) or (all(var_data[sensor] == -555))):
            
                # Create a temporary data frame of the time variable, and the sensor to run tests on
                if ((sensor == 'mass_concentration_of_chlorophyll_a_in_sea_water')
                    or
                    (sensor == 'sea_water_turbidity')):
                    temp_vardata = np.array([np.log10(ii) if ii > 0 else np.nan for ii in np.ndarray.flatten(var_data.loc[:,sensor].to_numpy())])
                else:
                    temp_vardata = np.ndarray.flatten(var_data.loc[:,sensor].to_numpy())
                
                
                qc_sensorname = sensor
                test_df = pd.DataFrame(data=list(zip(vardates, temp_vardata)),
                                       columns=['time',sensor])

                
                
                # Perform the qartod tests for the specified sensor
                temp_qartod_df = run_qartod_tests(test_df, qc_sensorname, depth_level)


                # If necessary, rename the rollup qc column to include the sensor name
                if 'qartod_rollup_qc' in temp_qartod_df.columns:
                    temp_qartod_df = temp_qartod_df.rename({'qartod_rollup_qc':sensor+'_qc_aggregate'},
                                                           axis=1)

            
            else:
                # If all the data is bad, apply "fail" flags everywhere, and assign 9999 for the tests, indicating
                # that no tests were performed
                config = load_sensor_qartod_config(sensor)
                
                temp_qartod_df = pd.DataFrame(data=list(zip(4*np.ones(NT).astype(int))),
                                              columns=[sensor+'_qc_aggregate'])
                for qc_test in config[sensor]['qartod']:
                    temp_qartod_df[sensor + '_qartod_' + qc_test] = 4*np.ones(NT).astype(int)
                    
            temp_qartod_df[sensor + '_qartod_manually_flagged_test'] = 9*np.ones(NT).astype(int)
                
        
            
            # Append on the newly created qartod flags to the qartod dataframe
            if isinstance(qartod_df,list):
                qartod_df = temp_qartod_df
            else:
                qartod_df = pd.concat([qartod_df, temp_qartod_df],axis=1)
                
                
    qc_cols = qartod_df.columns
    qc_rename = {}
    for col in qc_cols:
        qc_rename[col] = col.replace('_qartod_','_qc_')
    qartod_df = qartod_df.rename(columns=qc_rename)
                
                
                
    return qartod_df


# %%


def run_qartod_tests(var_df, sensor, depth_level):
    
    
    # Ensure that the variable is a pandas dataframe
    if not(isinstance(var_df,pd.DataFrame)):
        var_df = pd.DataFrame(data=var_df,columns=['time',sensor])
    
    if depth_level <= 5:
        surf_hydro = True
    else:
        surf_hydro=False
    config = load_sensor_qartod_config(sensor, surf_hydro)
    if sensor == 'sea_water_pressure':
        if depth_level <= 5:
            config['sea_water_pressure']['qartod']['gross_range_test']['suspect_span'] = [0.1, 2.5]
            config['sea_water_pressure']['qartod']['gross_range_test']['fail_span'] = [0, 5]
        elif (depth_level > 5) and (depth_level <= 35):
            config['sea_water_pressure']['qartod']['gross_range_test']['suspect_span'] = [20, 30]
            config['sea_water_pressure']['qartod']['gross_range_test']['fail_span'] = [15, 35]
        elif (depth_level > 35) and (depth_level <= 53):
            config['sea_water_pressure']['qartod']['gross_range_test']['suspect_span'] = [42, 50]
            config['sea_water_pressure']['qartod']['gross_range_test']['fail_span'] = [39, 53]
        elif (depth_level > 53) and (depth_level <= 70):
            config['sea_water_pressure']['qartod']['gross_range_test']['suspect_span'] = [56, 64]
            config['sea_water_pressure']['qartod']['gross_range_test']['fail_span'] = [53, 67]
        elif (depth_level > 70) and (depth_level <= 90):
            config['sea_water_pressure']['qartod']['gross_range_test']['suspect_span'] = [81, 87]
            config['sea_water_pressure']['qartod']['gross_range_test']['fail_span'] = [75, 89]
        
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
    results_store[sensor + '_qc_aggregate'] = aggr_flags
    
    return results_store


# %%


def load_sensor_qartod_config(sensor, sample_rate=None, surf_hydro=False):
    
    config = {}
    
    ## Vaisala Wind
    if sensor == 'sea_water_temperature':
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
                        "tolerance": 0.01,
                        "suspect_threshold": 12*(60*60),
                        "fail_threshold": 24*(60*60)}
                }
            }
        }   
        
        
    elif sensor == 'sea_water_electrical_conductivity':
        config = {
            "sea_water_electrical_conductivity": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [0.5,3.8],
                        "fail_span": [0,4.5]},
                    "spike_test": {
                        "suspect_threshold": 0.5,
                        "fail_threshold": 1.0},
                    "rate_of_change_test": {
                        "threshold": 1.0/(30*60)},
                    "flat_line_test": {
                        "tolerance": 0.001,
                        "suspect_threshold": 12*(60*60),
                        "fail_threshold": 24*(60*60)}
                }
            }
        }
        
    elif sensor == 'sea_water_practical_salinity':
        config = {
            "sea_water_practical_salinity": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [5,35],
                        "fail_span": [0,40]},
                    "spike_test": {
                        "suspect_threshold": 0.5,
                        "fail_threshold": 1.0},
                    "rate_of_change_test": {
                        "threshold": 20.0/(60*60)},
                    "flat_line_test": {
                        "tolerance": 0.0005,
                        "suspect_threshold": 12*(60*60),
                        "fail_threshold": 24*(60*60)}
                }
            }
        }
        
    elif sensor == 'sea_water_sigma_theta':
        config = {
            "sea_water_sigma_theta": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [5,30],
                        "fail_span": [0,40]},
                    "spike_test": {
                        "suspect_threshold": 0.25,
                        "fail_threshold": 0.5},
                    "rate_of_change_test": {
                        "threshold": 5.0/(30*60)},
                    "flat_line_test": {
                        "tolerance": 0.0005,
                        "suspect_threshold": 12*(60*60),
                        "fail_threshold": 24*(60*60)}
                }
            }
        }
        
    elif sensor == 'sea_water_pressure':
        config = {
            "sea_water_pressure": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [0.25,30],
                        "fail_span": [0,35]},
                    "spike_test": {
                        "suspect_threshold": 0.25,
                        "fail_threshold": 0.5},
                    "rate_of_change_test": {
                        "threshold": 0.5/(20*60)},
                    "flat_line_test": {
                        "tolerance": 0.001,
                        "suspect_threshold": 12*(60*60),
                        "fail_threshold": 24*(60*60)}
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
                        "suspect_threshold": 12*(60*60),
                        "fail_threshold": 24*(60*60)}
                }
            }
        }
        
    elif sensor == 'sea_water_turbidity':
        config = {
            "sea_water_turbidity": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [np.log10(0.05), np.log10(100)],
                        "fail_span": [np.log10(0.001), np.log10(500)]},
                    "spike_test": {
                        "suspect_threshold": np.log10(50),
                        "fail_threshold": np.log10(100)},
                    "rate_of_change_test": {
                        "threshold": np.log10(1000)/(60*60)},
                    "flat_line_test": {
                        "tolerance": 0.005,
                        "suspect_threshold": (60*60),
                        "fail_threshold": 2*(60*60)}
                }
            }
        }
        
    elif sensor == 'mass_concentration_of_chlorophyll_a_in_sea_water':
        config = {
            "mass_concentration_of_chlorophyll_a_in_sea_water": {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": [np.log10(0.05), np.log10(30)],
                        "fail_span": [np.log10(0.007), np.log10(50)]},
                    "spike_test": {
                        "suspect_threshold": 0.25,
                        "fail_threshold": 0.5},
                    "rate_of_change_test": {
                        "threshold": 5.0/(30*60)},
                    "flat_line_test": {
                        "tolerance": 0.005,
                        "suspect_threshold": (60*60),
                        "fail_threshold": 2*(60*60)}
                }
            }
        }
        
    return config


# # Main wrapper functions

# %%


def get_filelists(buoy_designation, basedir, latest_date, end_date=None, ctd_pos='surfctd'):
    
    ctddir = os.path.join(basedir, ctd_pos)
    if not os.path.exists(ctddir):
        print('No data found in directory: ' + ctddir)
        return None, None

    ctdfiles = sorted([file for file in os.listdir(ctddir) if '.csv' in file])
    searchstr = f'_{ctd_pos}_'
    ctd_filedates = [datetime.datetime.strptime(file[file.find(searchstr)+len(searchstr):file.find('.csv')],'%Y%m%d') 
                     for file in ctdfiles]
    
    
    if latest_date is not None:
        latest_filedate = latest_date.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
        ctd_fileinds = np.where([latest_filedate <= filedate 
                                 for filedate in ctd_filedates])[0]
    else:
        ctd_fileinds = [int(ind) for ind in range(0,len(ctdfiles))]
        
    if end_date is not None:
        ctd_fileinds = np.where([end_date > filedate 
                                 for filedate in ctd_filedates])[0]
    else:
        ctd_fileinds = [int(ind) for ind in range(0,len(ctdfiles))]
        

    ctdfiles = [ctdfiles[ind] for ind in ctd_fileinds]
    ctd_filedates = [ctd_filedates[ind] for ind in ctd_fileinds]
    if end_date is not None:
        print('Last data file to access is: ' + str(ctdfiles[len(ctdfiles)-1]))
    
    mindate = min(ctd_filedates)
    maxdate = max(ctd_filedates)
    
            
    return ctddir, ctdfiles


# %%


def load_deployment_starts(buoy_designation, depdir, deployment_name=None):
    
    dep_file = buoy_designation.lower() + '_deployments.json'
    with open(os.path.join(depdir, dep_file),'r') as f:
        dep_info = json.load(f)

    if deployment_name is not None:
        deployment_name = deployment_name.lower()
        nemo_deps = [dep_info['deployments'][dep] for dep in dep_info['deployments'] 
                     if buoy_designation.lower() in dep and dep == deployment_name]
    else:
        nemo_deps = [dep_info['deployments'][dep] for dep in dep_info['deployments'] 
                     if buoy_designation.lower() in dep]

    dep_starts = [datetime.datetime.strptime(dep['start'],'%Y-%m-%dT%H:%M:%SZ')
                  for dep in nemo_deps]
    dep_ends = [datetime.datetime.strptime(dep['end'],'%Y-%m-%dT%H:%M:%SZ') 
                if dep['end'] is not None else datetime.datetime.now()+datetime.timedelta(days=1)
                 for dep in nemo_deps]
    
    return dep_starts, dep_ends


# %%


def find_latest_datafile(buoy_designation, ncdir, startdate=None, ctd_pos='surfctd'):
    
    nc_files = sorted([file for file in os.listdir(ncdir) if '.nc' in file])
    if len(nc_files) == 0:
        return None, None, None

    nc_files = [file for file in nc_files if f'_{ctd_pos}_' in file]
    if len(nc_files) == 0:
            return None, None, None
    
    strsearch = f'_{ctd_pos}_'
    nc_dates = [datetime.datetime.strptime(file[file.find(strsearch)+len(strsearch):file.find('.nc')],'%Y%m')
                for file in nc_files]
    
    if startdate is None:
        maxind = np.argmax(nc_dates)
    else:
        if any([nc_date == startdate for nc_date in nc_dates]):
            maxind = np.where([nc_date == startdate for nc_date in nc_dates])[0][0]
        else:            
            maxind = np.nanargmin([(nc_date - startdate).total_seconds()
                                   if (nc_date - startdate) >= datetime.timedelta(seconds=0)
                                   else np.nan
                                   for nc_date in nc_dates])
    recent_file = os.path.join(ncdir,nc_files[maxind])
    if startdate is not None:
        print('Starting data file to load is: ' + recent_file)
    
    if (maxind==0) or ((nc_dates[maxind]-nc_dates[maxind-1]) > datetime.timedelta(days=45)):
        prior_file = None
    else:
        prior_file = os.path.join(ncdir,nc_files[maxind-1])
    
    
    return nc_dates[maxind], recent_file, prior_file


# # netCDF generation Functions

# %%

def add_dataset_general_metadata(dataset):

    dataset.contributor_email = 'setht1@uw.edu, mickett@uw.edu, None'
    dataset.contributor_name = 'Seth Travis, John Mickett, NANOOS'
    dataset.contributor_role = 'collaborator, principalInvestigator, funder'
    dataset.contributor_role_vocabulary = 'https://vocab.nerc.ac.uk/collection/G04/current/'
    dataset.contributor_url = 'https://nwem.apl.washington.edu/, https://nwem.apl.washington.edu/, https//www.nanoos.org/'

    dataset.creator_address = 'Henderson Hall, 1013 NE 40th Street, Seattle, WA 98105'
    dataset.creator_city = 'Seattle'
    dataset.creator_country = 'USA'
    dataset.creator_email = 'setht1 at uw.edu'
    dataset.creator_institution = 'Applied Physics Lab - University of Washington (APL-UW)'
    dataset.creator_name = 'Seth Travis'
    dataset.creator_postalcode = '98105'
    dataset.creator_sector = 'academic'
    dataset.create_state = 'WA'
    dataset.creator_type = 'person'
    dataset.creator_url = 'https://nwem.apl.washington.edu/'

    dataset.publisher_address = 'Henderson Hall, 1013 NE 40th Street, Seattle, WA 98105'
    dataset.publisher_city = 'Seattle'
    dataset.publisher_country = 'USA'
    dataset.publisher_email = 'setht1 at uw.edu'
    dataset.publisher_institution = 'Applied Physics Lab - University of Washington (APL-UW)'
    dataset.publisher_name = 'Seth Travis'
    dataset.publisher_postalcode = '98105'
    dataset.publisher_sector = 'academic'
    dataset.publisher_state = 'WA'
    dataset.publisher_type = 'person'
    dataset.publisher_url = 'https://nwem.apl.washington.edu/'

    curyear = str(datetime.datetime.now().year)
    #dataset.citation = ('University of Washington NEMO Mooring Data, 2010-' + curyear + 
    #                    '; https://nwem.apl.washington.edu/, enabled by the ' + 
    #                    'Northwest Association of Networked Ocean Observing Systems ' +
    #                    '(NANOOS, https://www.nanoos.org)')
    dataset.institution = ('Northwest Environmental Moorings Group at ' + 
                           'University of Washington - Applied Physical Laboratory')
    dataset.ror = 'https://ror.org/01a258x16'

    return dataset


def make_new_ctd_xarray(buoy_designation, buoy_info, ctd_df, target_depth=0.5):
    
    nrows = len(ctd_df)
    # Buoy info
    buoy_name = buoy_info['buoy_name']
    buoy_depth = buoy_info['depth']
    buoy_depthstr = str(buoy_depth)
    buoy_lat = buoy_info['latitude']
    buoy_latstr = str(buoy_lat)
    buoy_lon = buoy_info['longitude']
    buoy_lonstr = str(buoy_lon)
    
    #
    fillVal = -555
    if 'chaba' not in buoy_designation.lower():
        wmo_code = '--'
        gtsflag = 'false'
    else:
        wmo_code = '46119'
        gtsflag = 'true'

    xr_new = xr.Dataset()    
    
    def make_ancvar_str(varname):
        # Define the general QC flag suffixes
        qc_vars = ['qc_aggregate', 'qc_gross_range_test', 'qc_rate_of_change_test',
                   'qc_spike_test', 'qc_flat_line_test']
        # Initialize an ancillary variable stirng
        ancvar_str = ''
        for qc_var in qc_vars:
            # Add each qc flag specific to the variable to the string
            ancvar_str = ancvar_str + varname + '_' + qc_var + ' '
        # Remove the last ", " from the string
        ancvar_str = ancvar_str[:-1]
        
        return ancvar_str
    
    
    xr_new.attrs['title'] = "NEMO - " + buoy_name + " - CTD Data, Near-realtime (incomplete data)"
    xr_new.attrs['description'] = ("CTD samples taken from near-realtime reporting at the " + buoy_name +
                                   " mooring. CTDs are located at multiple depths on a mooring line and regularly transmitting data.")
    xr_new.attrs['history'] = "Ingesting real-time data on " + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
    xr_new.attrs['conventions'] = 'CF-1.6, ACDD-1.3, IOOS-1.2'
    xr_new.attrs['buoy_designation'] = buoy_name
    xr_new.attrs['buoy_latitude'] = str(np.round(buoy_lat, 6)) +'N'
    xr_new.attrs['buoy_longitude'] = str(np.round(buoy_lon, 6)) + 'E'
    xr_new.attrs['buoy_depth'] = str(buoy_depth) + ' m'
    xr_new.attrs['instrument_target_depth'] = str(target_depth) + ' m'
    xr_new.attrs['author'] = 'Seth Travis'
    xr_new.attrs['institution'] = 'University of Washington - Applied Physical Laboratory - NorthWest Environmental Moorings (NWEM) Group'
    xr_new.attrs['contact'] = 'setht1@uw.edu'
    xr_new.attrs['_FillValue'] = fillVal
    xr_new.attrs['cdm_data_type'] = 'timeSeries'
    xr_new.attrs['cdm_timeseries_variables'] = 'location_id, latitude, longitude'
    xr_new.attrs['subsetVariables'] = 'depth, instrument_type'

    # The global attributes are needed to allow for ingesting by NDBC
    # into the GTS data harvester for models
    xr_new.attrs['wmo_platform_code'] = str(wmo_code)
    xr_new.attrs['gts_ingest'] = gtsflag

    xr_new.attrs['contributor_email'] = 'setht1@uw.edu, mickett@uw.edu, None'
    xr_new.attrs['contributor_name'] = 'Seth Travis, John Mickett, NANOOS'
    xr_new.attrs['contributor_role'] = 'collaborator, principalInvestigator, funder'
    xr_new.attrs['contributor_role_vocabulary'] = 'https://vocab.nerc.ac.uk/collection/G04/current/'
    xr_new.attrs['contributor_url'] = 'https://nwem.apl.washington.edu/, https://nwem.apl.washington.edu/, https//www.nanoos.org/'

    xr_new.attrs['creator_address'] = 'Henderson Hall, 1013 NE 40th Street, Seattle, WA 98105'
    xr_new.attrs['creator_city'] = 'Seattle'
    xr_new.attrs['creator_country'] = 'USA'
    xr_new.attrs['creator_email'] = 'setht1 at uw.edu'
    xr_new.attrs['creator_institution'] = 'Applied Physics Lab - University of Washington (APL-UW)'
    xr_new.attrs['creator_name'] = 'Seth Travis'
    xr_new.attrs['creator_postalcode'] = '98105'
    xr_new.attrs['creator_sector'] = 'academic'
    xr_new.attrs['create_state'] = 'WA'
    xr_new.attrs['creator_type'] = 'person'
    xr_new.attrs['creator_url'] = 'https://nwem.apl.washington.edu/'

    xr_new.attrs['publisher_address'] = 'Henderson Hall, 1013 NE 40th Street, Seattle, WA 98105'
    xr_new.attrs['publisher_city'] = 'Seattle'
    xr_new.attrs['publisher_country'] = 'USA'
    xr_new.attrs['publisher_email'] = 'setht1 at uw.edu'
    xr_new.attrs['publisher_institution'] = 'Applied Physics Lab - University of Washington (APL-UW)'
    xr_new.attrs['publisher_name'] = 'Seth Travis'
    xr_new.attrs['publisher_postalcode'] = '98105'
    xr_new.attrs['publisher_sector'] = 'academic'
    xr_new.attrs['publisher_state'] = 'WA'
    xr_new.attrs['publisher_type'] = 'person'
    xr_new.attrs['publisher_url'] = 'https://nwem.apl.washington.edu/'

    xr_new.attrs['ror'] = 'https://ror.org/01a258x16'
    xr_new.attrs['institution'] = ('Northwest Environmental Moorings Group at ' +
                                   'University of Washington - Applied Physical Laboratory')
    

    xr_new['time'] = xr.DataArray(ctd_df['time'], dims='time')
    if nrows > 0:
        xr_new['time'] = xr_new['time'].astype(np.datetime64)
    xr_new['time'].attrs["long_name"] = 'sample_time'
    xr_new['time'].attrs["description"] = 'time of sampling'  
    xr_new['time'].encoding["dtype"] = "float32"
    xr_new['time'].encoding["units"] = 'seconds since 1970-01-01 00:00:00' 
    xr_new['time'].encoding["calendar"] = 'gregorian'  
    xr_new['time'].encoding["timezone"] = 'UTC'  
    
    #return xr_new
    
    xr_new['location_id'] = xr.DataArray([buoy_name], dims='location_id')
    xr_new['location_id'].attrs["long_name"] = 'location_id'
    xr_new['location_id'].attrs["description"] = 'Buoy designation'
    xr_new['location_id'].attrs["cf_role"] = 'timeseries_id'
    xr_new['location_id'].attrs["units"] = '1'
    xr_new['location_id'].attrs["ioos_category"] = 'Identifier'

    if 'deployment_name' in ctd_df.columns:
        deployment_vals = ctd_df['deployment_name'].astype(str).to_numpy()
    elif 'deployment' in ctd_df.columns:
        deployment_vals = ctd_df['deployment'].astype(str).to_numpy()
    else:
        deployment_vals = np.array(['' for _ in range(nrows)])
    xr_new['deployment_name'] = xr.DataArray(deployment_vals, dims='time')
    xr_new['deployment_name'].attrs["long_name"] = 'deployment name'
    xr_new['deployment_name'].attrs["description"] = 'Deployment designation'
    xr_new['deployment_name'].attrs["cf_role"] = 'timeseries_id'

    xr_new['latitude'] = xr.DataArray([float(buoy_lat) for ii in range(0,nrows)], dims='time')
    xr_new['latitude'].attrs["long_name"] = 'latitude'
    xr_new['latitude'].attrs["units"] = 'degrees North'
    xr_new['latitude'].attrs["description"] = 'Latitude position in degrees North'
    
    xr_new['longitude'] = xr.DataArray([float(buoy_lon) for ii in range(0,nrows)], dims='time')
    xr_new['longitude'].attrs["long_name"] = 'longitude'
    xr_new['longitude'].attrs["units"] = 'degrees East'
    xr_new['longitude'].attrs["description"] = 'Longitude position in degrees East'

    if 'instrument_type' in ctd_df.columns:
        instrument_vals = ctd_df['instrument_type'].to_numpy()
    elif 'InstrumentType' in ctd_df.columns:
        instrument_vals = ctd_df['InstrumentType'].to_numpy()
    else:
        instrument_vals = np.array(['' for _ in range(nrows)])
    xr_new['instrument_type'] = xr.DataArray(instrument_vals, dims='time')
    xr_new['instrument_type'].attrs["long_name"] = 'instrument_type'
    xr_new['instrument_type'].attrs["description"] = 'Names of the instrument(s) used in collecting the sample data'

    xr_new['sea_water_temperature'] = xr.DataArray(ctd_df['sea_water_temperature'].to_numpy(), dims='time')
    xr_new['sea_water_temperature'].fillna(fillVal)
    xr_new['sea_water_temperature'].attrs["long_name"] = 'sea_water_temperature'
    xr_new['sea_water_temperature'].attrs["standard_name"] = 'sea_water_temperature'
    xr_new['sea_water_temperature'].attrs["description"] = 'In-situ temperature of water (T90 scale)'
    xr_new['sea_water_temperature'].attrs["units"] = 'degrees C'
    xr_new['sea_water_temperature'].attrs["missing_value"] = fillVal
    xr_new['sea_water_temperature'].attrs["_FillVal"] = fillVal
    xr_new['sea_water_temperature'].attrs["gts_ingest"] = gtsflag
    xr_new['sea_water_temperature'].attrs["ancillary_variables"] = make_ancvar_str('sea_water_temperature')
    
    xr_new['sea_water_electrical_conductivity'] = xr.DataArray(ctd_df['sea_water_electrical_conductivity'].to_numpy(), dims='time')
    xr_new['sea_water_electrical_conductivity'].fillna(fillVal)
    xr_new['sea_water_electrical_conductivity'].attrs["long_name"] = 'sea_water_electrical_conductivity'
    xr_new['sea_water_electrical_conductivity'].attrs["standard_name"] = 'sea_water_electrical_conductivity'
    xr_new['sea_water_electrical_conductivity'].attrs["description"] = 'Ability to pass electrical current. In water, it is a proxy from which to derive salinity.'
    xr_new['sea_water_electrical_conductivity'].attrs["units"] = 'S/m'
    xr_new['sea_water_electrical_conductivity'].attrs["missing_value"] = fillVal
    xr_new['sea_water_electrical_conductivity'].attrs["_FillVal"] = fillVal
    xr_new['sea_water_electrical_conductivity'].attrs["gts_ingest"] = gtsflag
    xr_new['sea_water_electrical_conductivity'].attrs["ancillary_variables"] = make_ancvar_str('sea_water_electrical_conductivity')
    
    xr_new['sea_water_pressure'] = xr.DataArray(ctd_df['sea_water_pressure'].to_numpy(), dims='time')
    xr_new['sea_water_pressure'].fillna(fillVal)
    xr_new['sea_water_pressure'].attrs["long_name"] = 'sea_water_pressure'
    xr_new['sea_water_pressure'].attrs["standard_name"] = 'sea_water_pressure'
    xr_new['sea_water_pressure'].attrs["description"] = 'Pressure exerted by overlying water, excluding air pressure.'
    xr_new['sea_water_pressure'].attrs["units"] = 'dbar'
    xr_new['sea_water_pressure'].attrs["missing_value"] = fillVal
    xr_new['sea_water_pressure'].attrs["_FillVal"] = fillVal
    xr_new['sea_water_pressure'].attrs["gts_ingest"] = gtsflag
    xr_new['sea_water_pressure'].attrs["ancillary_variables"] = make_ancvar_str('sea_water_pressure')
    
    xr_new['depth'] = xr.DataArray(ctd_df['depth'].to_numpy(), dims='time')
    xr_new['depth'].fillna(fillVal)
    xr_new['depth'].attrs["long_name"] = 'depth'
    xr_new['depth'].attrs["standard_name"] = 'depth'
    xr_new['depth'].attrs["description"] = 'Z-coordinate of observation in vertical distance below reference. Down is positive. (reference is sea surface)'
    xr_new['depth'].attrs["units"] = 'm'
    xr_new['depth'].attrs["missing_value"] = fillVal
    xr_new['depth'].attrs["_FillVal"] = fillVal
    xr_new['depth'].attrs["gts_ingest"] = gtsflag
    
    
    xr_new['sea_water_practical_salinity'] = xr.DataArray(ctd_df['sea_water_practical_salinity'].to_numpy(), dims='time')
    xr_new['sea_water_practical_salinity'].fillna(fillVal)
    xr_new['sea_water_practical_salinity'].attrs["long_name"] = 'sea_water_practical_salinity'
    xr_new['sea_water_practical_salinity'].attrs["standard_name"] = 'sea_water_practical_salinity'
    xr_new['sea_water_practical_salinity'].attrs["description"] = 'Salinity is to the salt content of a water sample or body of water. The measure of salt content of a water sample follows UNESCO standards known as the Practical Salinity Scale (PSS) as the conductivity ratio of a sea water sample to a standard KCl solution. PSS is a ratio and has no units.'
    xr_new['sea_water_practical_salinity'].attrs["units"] = 'psu'
    xr_new['sea_water_practical_salinity'].attrs["missing_value"] = fillVal
    xr_new['sea_water_practical_salinity'].attrs["_FillVal"] = fillVal
    xr_new['sea_water_practical_salinity'].attrs["reference"] = 'UNESCO 1983 - Conductivity Ratio to Salinity Conversion'
    xr_new['sea_water_practical_salinity'].attrs["gts_ingest"] = gtsflag
    xr_new['sea_water_practical_salinity'].attrs["ancillary_variables"] = make_ancvar_str('sea_water_practical_salinity')
    
    xr_new['sea_water_sigma_theta'] = xr.DataArray(ctd_df['sea_water_sigma_theta'].to_numpy(), dims='time')
    xr_new['sea_water_sigma_theta'].fillna(fillVal)
    xr_new['sea_water_sigma_theta'].attrs["long_name"] = 'sea_water_sigma_theta'
    xr_new['sea_water_sigma_theta'].attrs["standard_name"] = 'sea_water_sigma_theta'
    xr_new['sea_water_sigma_theta'].attrs["description"] = 'Potential density, referenced to 0 dbar, offset by 1000 kg * m^-3 (sigma-0).'
    xr_new['sea_water_sigma_theta'].attrs["units"] = 'kg * m-3 - 1000'
    xr_new['sea_water_sigma_theta'].attrs["missing_value"] = fillVal
    xr_new['sea_water_sigma_theta'].attrs["_FillVal"] = fillVal
    xr_new['sea_water_sigma_theta'].attrs["reference"] = 'UNESCO 1983 - EOS-80 (International Equation of State of Seawater)'
    xr_new['sea_water_sigma_theta'].attrs["gts_ingest"] = gtsflag
    xr_new['sea_water_sigma_theta'].attrs["ancillary_variables"] = make_ancvar_str('sea_water_sigma_theta')
    
    xr_new['mass_concentration_of_oxygen_in_sea_water'] = xr.DataArray(ctd_df['mass_concentration_of_oxygen_in_sea_water'].to_numpy(), dims='time')
    xr_new['mass_concentration_of_oxygen_in_sea_water'].fillna(fillVal)
    xr_new['mass_concentration_of_oxygen_in_sea_water'].attrs["long_name"] = 'mass_concentration_of_oxygen_in_sea_water'
    xr_new['mass_concentration_of_oxygen_in_sea_water'].attrs["standard_name"] = 'mass_concentration_of_oxygen_in_sea_water'
    xr_new['mass_concentration_of_oxygen_in_sea_water'].attrs["description"] = 'Concentration of dissolved oxygen in water'
    xr_new['mass_concentration_of_oxygen_in_sea_water'].attrs["units"] = 'mg * L^-1'
    xr_new['mass_concentration_of_oxygen_in_sea_water'].attrs["missing_value"] = fillVal
    xr_new['mass_concentration_of_oxygen_in_sea_water'].attrs["_FillVal"] = fillVal
    xr_new['mass_concentration_of_oxygen_in_sea_water'].attrs["gts_ingest"] = gtsflag
    xr_new['mass_concentration_of_oxygen_in_sea_water'].attrs["ancillary_variables"] = make_ancvar_str('mass_concentration_of_oxygen_in_sea_water')
    
    xr_new['fractional_saturation_of_oxygen_in_sea_water'] = xr.DataArray(ctd_df['fractional_saturation_of_oxygen_in_sea_water'].to_numpy(), dims='time')
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].fillna(fillVal)
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].attrs["long_name"] = 'fractional_saturation_of_oxygen_in_sea_water'
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].attrs["standard_name"] = 'fractional_saturation_of_oxygen_in_sea_water'
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].attrs["description"] = 'Concentration of dissolved oxygen in water, as a percentage of the concentration of dissolved oxygen in water at saturation. Dissolved oxygen saturation is the concentration of dissolved oxygen at saturation levels at the same temperature and salinity in a water sample.'
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].attrs["units"] = 'percent (%)'
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].attrs["missing_value"] = fillVal
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].attrs["_FillVal"] = fillVal
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].attrs["gts_ingest"] = gtsflag
    xr_new['fractional_saturation_of_oxygen_in_sea_water'].attrs["ancillary_variables"] = make_ancvar_str('fractional_saturation_of_oxygen_in_sea_water')
    
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'] = xr.DataArray(ctd_df['mass_concentration_of_chlorophyll_a_in_sea_water'].to_numpy(), dims='time')
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].fillna(fillVal)
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].attrs["long_name"] = 'mass_concentration_of_chlorophyll_a_in_sea_water'
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].attrs["standard_name"] = 'mass_concentration_of_chlorophyll_a_in_sea_water'
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].attrs["description"] = 'Mass concentration of chlorophyll-a in sea water. Measurement is based upon fluoroescence taken from fluorometer'
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].attrs["units"] = 'mg * m^-3'
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].attrs["missing_value"] = fillVal
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].attrs["_FillVal"] = fillVal
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].attrs["gts_ingest"] = gtsflag
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water'].attrs["ancillary_variables"] = make_ancvar_str('mass_concentration_of_chlorophyll_a_in_sea_water')
    
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water_std'] = xr.DataArray(ctd_df['mass_concentration_of_chlorophyll_a_in_sea_water_std'].to_numpy(), dims='time')
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water_std'].fillna(fillVal)
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water_std'].attrs["long_name"] = 'mass_concentration_of_chlorophyll_a_in_sea_water_std'
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water_std'].attrs["description"] = 'Standard deviation of the mass concentration of chlorophyll-a in sea water. Measurement is based upon fluoroescence taken from fluorometer'
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water_std'].attrs["units"] = 'mg * m^-3'
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water_std'].attrs["missing_value"] = fillVal
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water_std'].attrs["_FillVal"] = fillVal
    xr_new['mass_concentration_of_chlorophyll_a_in_sea_water_std'].attrs["gts_ingest"] = 'false'
    
    xr_new['sea_water_turbidity'] = xr.DataArray(ctd_df['sea_water_turbidity'].to_numpy(), dims='time')
    xr_new['sea_water_turbidity'].fillna(fillVal)
    xr_new['sea_water_turbidity'].attrs["long_name"] = 'sea_water_turbidity'
    xr_new['sea_water_turbidity'].attrs["standard_name"] = 'sea_water_turbidity'
    xr_new['sea_water_turbidity'].attrs["description"] = 'Measure of light scattering due to suspended material in water. Measure of the cloudiness of water. Higher turbidity levels are often associated with higher levels of disease-causing microorganisms such as viruses, parasites and some bacteria. Turbidity is measured in nephelometric turbidity units (NTU)'
    xr_new['sea_water_turbidity'].attrs["units"] = 'NTU'
    xr_new['sea_water_turbidity'].attrs["missing_value"] = fillVal
    xr_new['sea_water_turbidity'].attrs["_FillVal"] = fillVal
    xr_new['sea_water_turbidity'].attrs["gts_ingest"] = gtsflag
    xr_new['sea_water_turbidity'].attrs["ancillary_variables"] = make_ancvar_str('sea_water_turbidity')
    
    xr_new['sea_water_turbidity_std'] = xr.DataArray(ctd_df['sea_water_turbidity_std'].to_numpy(), dims='time')
    xr_new['sea_water_turbidity_std'].fillna(fillVal)
    xr_new['sea_water_turbidity_std'].attrs["long_name"] = 'sea_water_turbidity_std'
    xr_new['sea_water_turbidity_std'].attrs["description"] = 'Standard deviation of the measure of light scattering due to suspended material in water. Measure of the cloudiness of water. Higher turbidity levels are often associated with higher levels of disease-causing microorganisms such as viruses, parasites and some bacteria. Turbidity is measured in nephelometric turbidity units (NTU)'
    xr_new['sea_water_turbidity_std'].attrs["units"] = 'NTU'
    xr_new['sea_water_turbidity_std'].attrs["missing_value"] = fillVal
    xr_new['sea_water_turbidity_std'].attrs["_FillVal"] = fillVal
    xr_new['sea_water_turbidity_std'].attrs["gts_ingest"] = 'false'
    
    xr_new['sea_water_ph_reported_on_total_scale'] = xr.DataArray(ctd_df['sea_water_ph_reported_on_total_scale'].to_numpy(), dims='time')
    xr_new['sea_water_ph_reported_on_total_scale'].fillna(fillVal)
    xr_new['sea_water_ph_reported_on_total_scale'].attrs["long_name"] = 'sea_water_ph_reported_on_total_scale'
    xr_new['sea_water_ph_reported_on_total_scale'].attrs["standard_name"] = 'sea_water_ph_reported_on_total_scale'
    xr_new['sea_water_ph_reported_on_total_scale'].attrs["description"] = 'measure of acidity of seawater, defined as the negative logarithm of the concentration of dissolved hydrogen ions'
    xr_new['sea_water_ph_reported_on_total_scale'].attrs["units"] = '1'
    xr_new['sea_water_ph_reported_on_total_scale'].attrs["missing_value"] = fillVal
    xr_new['sea_water_ph_reported_on_total_scale'].attrs["_FillVal"] = fillVal
    xr_new['sea_water_ph_reported_on_total_scale'].attrs["gts_ingest"] = 'false'
    
    
    return xr_new


# %%


def ctd_qc_process(buoy_designation, ctd_qc, datatime):
    

    fillVal = -555
    ctd_qc_xr = xr.Dataset()
    
    ctd_qc[ctd_qc.isna()] = fillVal
    ctd_qc_xr.attrs['_FillValue'] = fillVal

    ctd_qc_xr['time'] = xr.DataArray(datatime, dims='time')
    ctd_qc_xr['time'] = ctd_qc_xr['time'].astype(np.datetime64)
    ctd_qc_xr['time'].attrs["long_name"] = 'sample_time'
    ctd_qc_xr['time'].attrs["description"] = 'time of sampling'
    ctd_qc_xr['time'].encoding["units"] = 'seconds since 1970-01-01 00:00:00'
    ctd_qc_xr['time'].encoding["timezone"] = 'UTC'
    ctd_qc_xr['time'].encoding["calendar"] = 'gregorian'
    
    
     #############
    # QARTOD Data
    varnames = ['sea_water_temperature',
                'sea_water_electrical_conductivity',
                'sea_water_pressure',
                'sea_water_practical_salinity',
                'sea_water_sigma_theta',
                'mass_concentration_of_oxygen_in_sea_water',
                'mass_concentration_of_chlorophyll_a_in_sea_water',
                'sea_water_turbidity',
                'sea_water_ph_reported_on_total_scale']
    varlabs = ['Sea Water Temperature', 
               'Sea Water Electrical Conductivity', 
               'Sea Water Pressure', 
               'Sea Water Practical Salinity',
               'Sea Water Sigma Theta',
               'Mass Concentration of Oxygen in Sea Water', 
               'Mass Concentration of Chlorophyll_a in Sea Water', 
               'Sea Water Turbidity',
               'Sea Water pH']

    origqc_vars = ['qc_aggregate', 
                   'qc_gross_range_test', 'qc_rate_of_change_test',
                   'qc_spike_test', 'qc_flat_line_test',
                   'qc_manually_flagged_test']
    qc_vars = ['qc_aggregate', 
               'qc_gross_range_test', 'qc_rate_of_change_test',
               'qc_spike_test', 'qc_flat_line_test',
                   'qc_manually_flagged_test']
    qc_standards = ['aggregate_quality_flag', 
                    'gross_range_test_quality_flag', 
                    'rate_of_change_test_quality_flag', 'spike_test_quality_flag',
                    'flat_line_test_quality_flag',
                    'manually_flagged_test']
    qc_labs = ['Aggregate Flag', 'Gross Range Test Flag', 
               'Rate of Change Test Flag', 'Spike Test Flag',
               'Flat Line Test Flag',
               'Manually Flagged Test']
    blankflags = 9*np.ones(len(ctd_qc))

    for ii in range(0,len(varnames)):
        # Step through each QARTOD test for each variable, and write it to the file
        var = varnames[ii]
        varlabel = varlabs[ii]
        for jj in range(0,len(qc_vars)):
            origqc_var = origqc_vars[jj]
            qc_var = qc_vars[jj]
            qc_standard = qc_standards[jj]
            qc_lab = qc_labs[jj]

            # Create new variable
            if any([col == var + '_' + origqc_var for col in ctd_qc.columns]):
                ctd_qc_xr[var + '_' + qc_var] = xr.DataArray(ctd_qc[var + '_' + origqc_var].values, dims='time').astype(np.int32)                    
            else:
                ctd_qc_xr[var + '_' + qc_var] = xr.DataArray(blankflags, dims='time').astype(np.int32) 
            
            
            #temp_qcvar = dataset.createVariable(var + '_' + qc_var,'i4',('location_id','time'))
            ctd_qc_xr[var + '_' + qc_var].attrs['standard_name'] = qc_standard
            ctd_qc_xr[var + '_' + qc_var].attrs['long_name'] = var + '_' + qc_var
            ctd_qc_xr[var + '_' + qc_var].attrs['description'] = varlabel + ' ' + qc_lab
            ctd_qc_xr[var + '_' + qc_var].attrs['ioos_category'] = 'Quality Control'
            ctd_qc_xr[var + '_' + qc_var].attrs['units'] = '1'
            ctd_qc_xr[var + '_' + qc_var].attrs['coverage_content_type'] = 'qualityInformation'
            ctd_qc_xr[var + '_' + qc_var].attrs['flag_vals'] = '1, 2, 3, 4, 9'
            ctd_qc_xr[var + '_' + qc_var].attrs['flag_meanings'] = 'PASS NOT_EVALUATED SUSPECT FAIL MISSING'
            ctd_qc_xr[var + '_' + qc_var].attrs['_FillValue'] = fillVal
            ctd_qc_xr[var + '_' + qc_var].attrs["missing_value"] = fillVal
        
    return ctd_qc_xr


# %%


def write_ctd_netcdf(savedir, save_filename, ds):
    
    # Write the initial data
    print('   Saving file: ' + save_filename)
    ds.drop_vars('time', errors='ignore').to_netcdf(os.path.join(savedir,save_filename))
    
    # Calculate the shifted time values
    dstime = [pd.Timestamp(ii).to_pydatetime() for ii in ds['time'].data]
    ref_time = datetime.datetime(1970,1,1)
    ds_timeshifted = [(ii-ref_time).total_seconds() for ii in dstime]    
    
    # Add the correct time data
    update_datasettime(savedir, save_filename, ds_timeshifted)
    
    return

def update_datasettime(savedir, save_filename, ds_timeshift):
    
    dataset = Dataset(os.path.join(savedir,save_filename), 'r+')
    try:
        if 'time' in dataset.variables:
            datatime = dataset.variables['time']
        else:
            datatime = dataset.createVariable('time','f8',('time',))

        datatime.long_name = 'sample_time'
        datatime.description = 'time of sampling'
        datatime.units = 'seconds since 1970-01-01 00:00:00'
        datatime.calendar = 'gregorian'

        datatime[:] = np.asarray(ds_timeshift, dtype=np.float64)
        dataset.sync()
    except Exception as e:
        print('Correcting xarray timestamps did not work')
        print(e)

    dataset.close()


# # Main Functions

# %%

def get_deployment_instruments(buoy_designation, dep_info):

    instruments = dep_info.get('instruments', None)
    if instruments is None:
        return [], [], [], []

    insttypes = []
    instids = []
    instrealtimeids = []
    for inst in instruments:
        if instruments[inst].get('realtime_id', None) is not None:
            instids.append(inst)
            insttypes.append(instruments[inst].get('model', None))
            instrealtimeids.append(instruments[inst].get('realtime_id', None))
    return instruments, insttypes, instids, instrealtimeids

def drop_outside_deployment(ctd_data, ctd_qc, dep_starts, dep_ends):
    # Check that all data falls within the deployment start and end times
    dropinds = []
    for depind in range(0,len(dep_starts)):
        if depind == 0:
            if any(ctd_data.loc[:,'time'] < dep_starts[depind]):
                dropinds.extend(np.where(ctd_data.loc[:,'time']<dep_starts[depind])[0])
            elif ((dep_ends[depind] is not None) and 
                  any(np.logical_and(ctd_data.loc[:,'time']<dep_starts[depind],
                                     ctd_data.loc[:,'time']>dep_ends[depind-1]))):
                dropinds.extend(np.where(np.logical_and(ctd_data.loc[:,'time']<dep_starts[depind],
                                                        ctd_data.loc[:,'time']>dep_ends[depind-1]))[0])

    ctd_data = ctd_data.drop(dropinds).reset_index(drop=True)
    ctd_qc = ctd_qc.drop(dropinds).reset_index(drop=True) 
    return ctd_data, ctd_qc

def merge_ctd_datasets(ctd_data, ctd_qc, ctd_latest):


    if ctd_latest is not None:
        latest_ctd = ctd_latest.loc[:, ctd_data.columns].copy()
        latest_qc = ctd_latest.loc[:, ctd_qc.columns].copy()

        finalctd_df = (pd.concat([latest_ctd, ctd_data])
                        .reset_index(drop=True).sort_values(by='time'))
        final_inds = finalctd_df.index
        finalctd_qcdf = (pd.concat([latest_qc, ctd_qc])
                            .reset_index(drop=True).loc[final_inds,:]
                            .reset_index(drop=True))
    else:
        finalctd_df = ctd_data.reset_index(drop=True).sort_values(by='time')
        final_inds = finalctd_df.index
        finalctd_qcdf = (ctd_qc
                            .reset_index(drop=True).loc[final_inds,:]
                            .reset_index(drop=True))

    for col in finalctd_df.columns:
        if 'time' in col:
            continue
        fill_val = 'unknown' if pd.api.types.is_string_dtype(finalctd_df[col].dtype) else -555
        if fill_val == 'unknown':
            print('Fill with string for column', col)
        finalctd_df.loc[finalctd_df[col].isna(),col] = fill_val

    return finalctd_df, finalctd_qcdf



def process_newdata_wrapper(buoy_designation, startdate=None, enddate=None, deployment_name=None, log=None):
    
    [procdir, basedir, datadir, savedir, depdir] = get_procdirs(buoy_designation.lower())

    deployments = nemo.get_deployment_info(buoy_designation, procdir)
    dep_info = None
    if deployment_name is not None:
        if deployment_name not in deployments['deployments']:
            raise ValueError(f"Deployment name {deployment_name} not found in deployment info")
        dep_info = deployments['deployments'][deployment_name]
    else:
        if startdate is not None:
            startdate = pd.Timestamp(startdate).to_pydatetime()
            if enddate is None:
                enddate = datetime.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)+datetime.timedelta(days=1)
        else:
            startdate = datetime.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)-datetime.timedelta(days=7)
            enddate = datetime.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)+datetime.timedelta(days=1)    
        for dep in deployments['deployments']:
            dep_info = deployments['deployments'][dep]
            dep_start = pd.Timestamp(dep_info['start']).to_pydatetime()
            dep_end = pd.Timestamp(dep_info['end']).to_pydatetime() if 'end' in dep_info else None
            if startdate >= dep_start and (enddate <= dep_end if dep_end is not None else True):
                deployment_name = dep
                break
        if deployment_name is None:
            raise ValueError("No deployment found for the specified start and end dates")
        else:
            dep_info = deployments['deployments'][deployment_name]
    
    dep_starts, dep_ends = load_deployment_starts(buoy_designation, depdir, deployment_name)



    

    [instruments, insttypes, 
     instids, instrealtimeids] = get_deployment_instruments(buoy_designation, dep_info)

    buoy_info = {"buoy_name": deployments['buoy_name'],
                "latitude": dep_info['latitude'],
                "longitude": dep_info['longitude'],
                "depth": dep_info['depth']}

    for ind, instid in enumerate(instids):
        insttype = insttypes[ind]
        inst_pos = instrealtimeids[ind]
        inst_info = dep_info['instruments'][instid]
        

        ctd_latest, latest_date = load_latest_file(buoy_designation, savedir, startdate, ctd_pos=inst_pos)

        inst_datadir = os.path.join(datadir, instid)
        ctddir, ctdfiles = get_filelists(buoy_designation, datadir, latest_date, enddate, ctd_pos=inst_pos)
        if (ctdfiles is None) or (len(ctdfiles) == 0):
            print('No CTD files found for instrument position: ' + inst_pos)
            continue

        
        ctd_data, ctd_qc = process_ctd_files(buoy_designation,
                                            [os.path.join(ctddir, file) for file in ctdfiles],
                                            dep_starts, instid, dep_info, buoy_info, ctd_pos=inst_pos)

        if ctd_data is None or ctd_data.shape[0] == 0:
            print('No valid CTD data found for instrument position: ' + inst_pos)
            continue

        ctd_data, ctd_qc = drop_outside_deployment(ctd_data, ctd_qc, dep_starts, dep_ends)

        finalctd_df, finalctd_qcdf = merge_ctd_datasets(ctd_data, ctd_qc, ctd_latest)     

        buoy_name = deployments['buoy_name']
        buoy_lat = dep_info['latitude']
        buoy_lon = dep_info['longitude']

        ctd_xr = make_new_ctd_xarray(buoy_designation, buoy_info, finalctd_df, target_depth=inst_info['instrument_depth'])
        ctd_qc_xr = ctd_qc_process(buoy_designation, finalctd_qcdf, finalctd_df['time'])

        ctdall_xr = xr.merge([ctd_xr, ctd_qc_xr], compat='override')
        if latest_date is not None:
            ctdall_xr = ctdall_xr.sel(time=slice(latest_date.replace(day=1,hour=0,minute=0,
                                                                    second=0,microsecond=0),None))
        ctdall_xr = ctdall_xr.drop_duplicates(dim='time')

        # Ensure all non-dimension variables include both location_id and time dimensions.
        # Re-attach location_id as a coordinate with attrs preserved because expand_dims
        # with raw .values can recreate it without metadata.
        loc_coord = ctdall_xr['location_id'].copy(deep=True)
        for var_name in list(ctdall_xr.data_vars):
            if (var_name == 'time') or (var_name == 'location_id'):
                continue
            if ctdall_xr[var_name].dims == ('time',):
                ctdall_xr[var_name] = ctdall_xr[var_name].expand_dims(
                    {'location_id': loc_coord.values}
                ).transpose('location_id', 'time')

        ctdall_xr = ctdall_xr.assign_coords(location_id=loc_coord)


        ctd_yrs = np.unique(ctdall_xr['time.year'])
        ctdyr_grouped = ctdall_xr.groupby('time.year')
        for yr in ctd_yrs:
            ctd_year = ctdyr_grouped[yr]

            ctd_months = np.unique(ctd_year['time.month'])
            ctdmnth_grouped = ctd_year.groupby('time.month')

            for month in ctd_months:
                ctd_month = ctdmnth_grouped[month]

                # Create a filename for the data to be saved
                if month < 10:
                    save_filename = buoy_designation.lower() + '_' + inst_pos + '_' + str(yr) + '0' + str(month) + '.nc'
                else:
                    save_filename = buoy_designation.lower() + '_' + inst_pos + '_' + str(yr) + str(month) + '.nc'

                write_ctd_netcdf(savedir, save_filename, ctd_month)
            
    return


# %%


def main():
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hu:b:s:e:d:", ["help", "buoy=", "start=", "end=", "deployment="])
    except Exception as inst:
        # print help information and exit:
        print('Error in getting options: ' + str(inst)) # will print something like "option -a not recognized"
        sys.exit(2)
        
    
    # step through command-line options and their arguments
    buoyName = False
    startFlag = False
    endFlag = False
    deploymentFlag = False
    
    # For all the command-line arguments, update a flag to indicate if the option was given,
    # and extract the value of the command-line argument
    for o, a in opts:
        if o == "-v":
            bVerbose = True
            # but currently verbose run is not implemented!
        elif o in ("-h", "--help"):
            sys.exit()
        elif o in ("-b", "--buoy"):
            buoyName = True
            buoy_designation = a
        elif o in ("-s", "--startdate"):
            startFlag = True
            startName = a
        elif o in ("-e", "--enddate"):
            endFlag = True
            endName = a
        elif o in ("-d", "--deployment"):
            deploymentFlag = True
            deploymentName = a
        else:
            assert False, "unhandled option"
                 

    ## Open a log file
    logging = False
    if buoyName and logging:
        logfile = ('ProcessCTDData_' + buoy_designation + '_' +  
                   datetime.datetime.now().strftime('%Y%b%dT%H%M') + '.txt')
        if os.path.exists(logfile):
            os.rename(logfile, logfile[:logfile.find('.txt')] + '_old.txt')
        print('Opening new log file:', logfile)
        log = open(logfile,'w')
    else:
        log = None
    
    
    
    ##    
        
    # If the buoy designation was given, ensure that the buoy designation is in uppercase,
    # and that it matches one of the valid buoy designations
    valid_buoys = ['chaba','nemoss']
    if buoyName:
        buoy_designation = buoy_designation.lower()
        if buoy_designation in valid_buoys:
            print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') 
                  + ' - Begin processing the met data for:' + buoy_designation, file=log)
        else:
            print('Invalid buoy designation. Check name, and retry.', file=log)
            sys.exit(2)
    else:
        print('No buoy designation given. Try again with options: "-b *buoy_designation*"')
        sys.exit(2)
        

    # If a deployment was specified, ensure that it is valid
    valid_deployments = ['chaba25b', 'chaba26a', 'chaba26b']
    if deploymentFlag:
        if deploymentName not in valid_deployments:
            print('Invalid deployment name. Check name, and retry.', file=log)
            sys.exit(2)

        # If a start and end date are given, ignore them
        startFlag = False
        endFlag = False

    # Start Date
    startDate = None
    if startFlag:        
        try:
            startDate = datetime.datetime.strptime(startName, '%Y%m')
            print('Start Date Flag Given: Process data starting at ' + 
                  startDate.strftime('%Y-%b-%dT%H:%M:%SZ'))
        except:
            print('Start date is not given in a valid format.')
            print('Dates must be given in a YYYYMM format: e.g., 202505 for May 2025')
            print('Restart, and try again.')
            sys.exit(2)
            
    # End Date
    endDate = None
    if endFlag: 
        if not(startFlag):
            print('A starting date must be given if an end date is to be used.')
            print('Restart, including a starting date ("-s" argument)')
            sys.exit(2)
            
        try:
            endDate = datetime.datetime.strptime(endName, '%Y%m')
        except:
            print('End date is not given in a valid format.')
            print('Dates must be given in a YYYYMM format: e.g., 202505 for May 2025')
            print('Restart, and try again.')
            sys.exit(2)
            
        if endDate < startDate:
            print('The ending date given is earlier than the starting date.')
            print('Starting date: ' + startDate.strftime('%Y%m'))
            print('Ending date:   ' + endDate.strftime('%Y%m'))
            print('Restart, and try again.')
            sys.exit(2)
        else:
            endDate_yr = endDate.timetuple().tm_year
            endDate_mon = endDate.timetuple().tm_mon
            if endDate_mon == 12:
                endDate_yr = endDate_yr + 1
                endDate_mon = 1
            else:
                endDate_mon = endDate_mon + 1
            endDate = datetime.datetime(endDate_yr, endDate_mon, 1)
        print('End Date Flag Given:   Process data up to       ' 
              + (endDate-datetime.timedelta(seconds=1)).strftime('%Y-%b-%dT%H:%M:%SZ'))
            
        
    
    ##################################################################
    # Begin processing new data for a given buoy
    print('Begin processing surface CTD data: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'),file=log)
    try:
        process_newdata_wrapper(buoy_designation, startDate, endDate, deploymentName, log)
        print('New data processing completed: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'),file=log)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('Error occurred in new data processing: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'),file=log)
        print(e,file=log)
        print('Error occurs on line ' + str(exc_tb.tb_lineno), file=log)
        
            
        sendEmail = False
        if sendEmail:
            failmsg = ('An error occurred while processing new surface CTD data for ' + buoy_designation + '.' + 
                       '\n\nProcessing failed at ' + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') + 
                       '\n\nPlease check processing logs to diagnose error.')
            send_emailreport(msgtxt = failmsg,
                                  subj = buoy_designation + ' - New Surface CTD Data Processing Failure')
    
    if logging:
        log.close()
        
        # Move the log to the appropriate folder
        curdir = os.getcwd()
        logdir = nemo.get_info_json()['logs']
        shutil.move(os.path.join(curdir,logfile), 
                    os.path.join(logdir, buoy_designation.lower(), logfile))

    sys.exit(0)


# %%


if __name__ == "__main__":
    main()
