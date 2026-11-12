#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
import pandas as pd
import xarray as xr
import datetime
import dateutil
import json

import ioos_qc

import re

import os
import sys
import getopt
import gc
import shutil

import netCDF4
from netCDF4 import Dataset

import requests
from urllib import request

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



# # General Functions

# %%


def get_procdirs(buoy_designation):

    
    procdir, basedir, datadir = nemo.get_mpkpaths(buoy_designation)
    savedir = nemo.get_info_json()['data_savedir']
    savedir = os.path.join(savedir, 'realtime', buoy_designation, 'met')
    depdir = os.path.join(procdir, 'info_jsons', 'deployments')
    
    return procdir, basedir, datadir, savedir, depdir

# %%


def average_datadf(data_df, varnames, average_window):
    
    # Convert the data timestamp into python datetime
    # objects
    datatime = np.array([pd.Timestamp(ii).to_pydatetime() 
                        for ii in data_df['time'].to_numpy()])
    # Use the reference time, and convert the data time
    # into seconds since the reference time
    reftime = datetime.datetime(1970,1,1)
    datatimestamp = np.array([(ii-reftime).total_seconds() 
                              for ii in datatime])

    #
    data_difftime = np.diff(datatime)
    data_jumps = np.where(data_difftime > 
                          datetime.timedelta(seconds=average_window))[0].astype(int)

    def average_variable(var, jumps, varname):
        # Average a variable across each jump
        meanvar = []
        def average_chunk(varchunk, varname):
            if all(np.isnan(varchunk)):
                return np.nan
            
            if 'direction' in varname:
                if any(np.isnan(varchunk)):
                    meanchunk = varchunk.copy()
                    varinds = np.where(np.logical_not(np.isnan(varchunk)))[0]
                    var_unwrapped = np.unwrap(np.array(varchunk)[varinds], period=360).tolist()
                    meanchunk[varinds] = var_unwrapped
                    meanchunk = np.nanmean(meanchunk) % 360
                else:
                    var_unwrapped = np.unwrap(varchunk, period=360)
                    meanchunk = np.nanmean(var_unwrapped) % 360
            elif varname == 'wind_speed_of_gust':
                varchunk_smooth = pd.Series(varchunk).rolling(window=5,center=True).mean()
                meanchunk = np.nanmax(varchunk_smooth)
            else:
                meanchunk = np.nanmean(varchunk)
            return np.round(meanchunk, 4)
        
        meanvar.append(average_chunk(var[:jumps[0]+1], varname))
        for ii in range(1, len(jumps)-1):            
            meanvar.append(average_chunk(var[jumps[ii]+1:jumps[ii+1]], varname))
        meanvar.append(average_chunk(var[jumps[-1]+1:], varname))
        return meanvar
        
    # Average the timestamp variable for each data jump
    data_meantimestamp = average_variable(datatimestamp, data_jumps, 'time')
    data_meantime = [(reftime+datetime.timedelta(seconds=ii)) 
                     for ii in data_meantimestamp]
    data_meantime = [(ii.replace(microsecond=0) + 
                      datetime.timedelta(seconds=round(ii.microsecond/1e6)))
                     for ii in data_meantime]
    
    data_meandf = pd.DataFrame(columns=['time'], data=data_meantime)
    for var in varnames:
        datavar = data_df[var].to_numpy()
        data_meanvar = average_variable(datavar, data_jumps, var)
        data_meandf[var] = data_meanvar
    
    return data_meandf


# %%


def datetime_toordinal_withseconds(origdate):
    
    import numpy as np
    import datetime
    
    year = origdate.year
    month = origdate.month
    day = origdate.day
    hour = origdate.hour
    minute = origdate.minute
    second = origdate.second
    microsec = origdate.microsecond
    
    return (datetime.datetime.toordinal(datetime.datetime(year,month,day)) +
            (hour + (minute + (second + (microsec/1e6))/60)/60)/24)


# %%


def datetime_fromordinal_withseconds(origdate):
    
    import datetime
    import numpy as np
    
    # Take the base integer value, and use this to create a base datetime object
    floordate = int(origdate)
    basedate = datetime.datetime.fromordinal(floordate)
    
    # Convert the remaining datevalue above the floor date to hours
    remhours = 24*(origdate - floordate)
    # Take the base integer value, and use this as the number of hours
    hours = int(remhours)
    
    # Convert the remaining datevalue above the floor hours to minutes
    remminutes = 60*(remhours - hours)
    # Take the base integer value, and use this as the number of minutes
    minutes = int(remminutes)
    
    # Convert the remaining datevalue above the floor minutes to seconds
    remseconds = 60*(remminutes - minutes)
    # Take the base integer value, and use this as the number of seconds
    seconds = int(remseconds)
    
    # Count the number of remaining microseconds
    # Note, original data precision limits this
    # to increments of 10 microseconds accuracy
    rem_parseconds = int(np.round(100000*(remseconds - seconds))*10)
    
    # Create a datetime object using the derived values
    finaltime = datetime.datetime(basedate.year, basedate.month, basedate.day,
                                  hours, minutes, seconds)
    finaltime = finaltime + datetime.timedelta(microseconds=rem_parseconds)
    
    return finaltime    


# # Met station functions

# %%


def process_met_files(buoy_designation, metpaths):
    
    met_df = None
    for ind in range(0,len(metpaths)):
        if metpaths[ind] is None:
            continue
        else:
            temp_df = pd.read_csv(metpaths[ind])
        
        if met_df is None:
            met_df = temp_df.copy()
        else:
            met_df = pd.concat([met_df, temp_df]).reset_index(drop=True)
        
    if met_df is None:
        print('No met data found.')
        return None, None
        
    reftime = datetime.datetime(1970,1,1)
    met_df['time'] = [(reftime + datetime.timedelta(microseconds=float(ii)/1e3)) 
                      for ii in met_df['timestamp']]
    met_df = met_df.sort_values(by='time').reset_index(drop=True)
    
    # Rename columns
    met_df_sensors = ['pressure','humidity','temperature', 'dewpoint']
    met_df_newname = ['air_pressure_at_mean_sea_level', 
                      'relative_humidity',
                      'air_temperature', 
                      'dew_point_temperature']
    newname_dict = {}
    for ii in range(0,len(met_df_sensors)):
        newname_dict[met_df_sensors[ii]] = met_df_newname[ii]

    emptycol = np.nan*np.zeros(len(met_df))
    for col in met_df_sensors:
        if col not in met_df.columns:
            met_df[col] = emptycol
    met_df.rename(columns=newname_dict, inplace=True)
    
    met_df = check_missing_vars(met_df, datatype='met')
    
    
    #############################################################
    # Perform QARTOD tests on the data on the full dataset
    # and null out the particularly bad datapoints
    met_qc_df = process_qartod_tests(buoy_designation, met_df, 
                                     sensor_type='met', fulldata_flag=True)
    for col in met_df.columns[1:]:
        try:
            met_df.loc[met_qc_df[col+'_qc_aggregate']>=4, col] = np.nan
        except:
            if not((('datenum' in col) or ('volts' in col))):
                print('Skipping column type: ' + col)
                
    
    ############################################################
    # Average the data
    averaging_window = (5 - 1.5*(0.5)) * 60
    met_varnames = ['air_pressure_at_mean_sea_level', 'air_temperature', 
                    'relative_humidity','dew_point_temperature',
                    'surface_downwelling_photosynthetic_photon_flux_in_air',
                    'rainfall_amount']
    met_meandf = average_datadf(met_df, met_varnames, averaging_window)

    met_meandf = met_meandf.drop_duplicates()
    met_meandf = met_meandf.reset_index(drop=True)

    
    ############################################################
    met_qc_meandf = process_qartod_tests(buoy_designation, met_meandf, 
                                         sensor_type='met', fulldata_flag=False)
    

    # Fill NaNs with fillval, if needed
    fillVal = -555
    met_meandf = met_meandf.fillna(fillVal)
    met_qc_df = met_qc_df.fillna(9)
    
    return met_meandf, met_qc_meandf


# %%


def process_rad_files(buoy_designation, shortradpaths, longradpaths):
    
    # Load in shortwave radiation data
    shortrad_df = None
    for ind in range(0,len(shortradpaths)):
        if shortradpaths[ind] is None:
            continue
        else:
            temp_df = pd.read_csv(shortradpaths[ind])

        if shortrad_df is None:
            shortrad_df = temp_df.copy()
        else:
            shortrad_df = pd.concat([shortrad_df, temp_df]).reset_index(drop=True)

    # Long in longwave radiation data
    longrad_df = None
    for ind in range(0,len(longradpaths)):
        if longradpaths[ind] is None:
            continue
        else:
            temp_df = pd.read_csv(longradpaths[ind])

        if longrad_df is None:
            longrad_df = temp_df.copy()
        else:
            longrad_df = pd.concat([longrad_df, temp_df]).reset_index(drop=True)

            
    if (shortrad_df is None) and (longrad_df is None):
        print('No short- or longwave radiation data found.')
        return None, None
    elif shortrad_df is None:
        print('No shortwave radiation data found.')
    elif longrad_df is None:
        print('No longwave radiation data found.')

    reftime = datetime.datetime(1970,1,1)
    
    def rename_dict(sensor_names, sensor_newnames, sensor_df):
        newname_dict = {}
        for ii in range(0,len(sensor_names)):
            newname_dict[sensor_names[ii]] = sensor_newnames[ii]

        emptycol = np.nan*np.zeros(len(sensor_df))
        for col in sensor_names:
            if col not in sensor_df.columns:
                sensor_df[col] = emptycol
        sensor_df.rename(columns=newname_dict, inplace=True)

        return sensor_df
    
    #####################################
    # Shortwave Radiation 
    #####################################
    
    if shortrad_df is not None:
        # Ensure the shortwave radiation is sorted by time
        shortrad_df['time'] = [(reftime + datetime.timedelta(microseconds=float(ii)/1e3)) 
                          for ii in shortrad_df['timestamp']]
        shortrad_df = shortrad_df.sort_values(by='time').reset_index(drop=True)
        sortinds = np.argsort(shortrad_df['timestamp'])
        shortrad_df = shortrad_df.loc[sortinds,:]
        
        # Rename the shortwave and longwave radiation sensors
        # The shortwave sensor only has downwelling flux
        shortrad_df_sensors = ['sensor1']
        shortrad_df_newname = ['surface_downwelling_shortwave_flux_in_air']

        shortrad_df = rename_dict(shortrad_df_sensors, shortrad_df_newname, shortrad_df)

        # Round the shortwave radiation timestamps to the nearest second
        shortrad_dates = [nemo.round_seconds(ii) for ii in shortrad_df['time']]
        shortrad_df['time'] = shortrad_dates

        # Null out shortwave radiation less than zero
        shortrad_df.loc[shortrad_df['surface_downwelling_shortwave_flux_in_air']<0,'surface_downwelling_shortwave_flux_in_air'] = np.nan
        shortrad_df = shortrad_df.loc[:,['time','timestamp','surface_downwelling_shortwave_flux_in_air']]

    
    #####################################
    # Longwave Radiation 
    #####################################
    
    if longrad_df is not None:
    
        # Ensure the longwave radiation is sorted by time
        longrad_df['time'] = [(reftime + datetime.timedelta(microseconds=float(ii)/1e3)) 
                          for ii in longrad_df['timestamp']]
        longrad_df = longrad_df.sort_values(by='time').reset_index(drop=True)
        sortinds = np.argsort(longrad_df['timestamp'])
        longrad_df = longrad_df.loc[sortinds,:]


        # Rename the shortwave and longwave radiation sensors
        # The longwave sensor has downwelling and new flux

        longrad_df_sensors = ['sensor1','sensor2']
        longrad_df_newname = ['surface_net_longwave_flux_in_air',
                              'surface_downwelling_longwave_flux_in_air']

        longrad_df = rename_dict(longrad_df_sensors, longrad_df_newname, longrad_df)

        # Round the shortwave radiation timestamps to the nearest second
        longrad_dates = [nemo.round_seconds(ii) for ii in longrad_df['time']]
        longrad_df['time'] = longrad_dates


        # Null out longwave radiation reading at 0
        longrad_df.loc[longrad_df['surface_downwelling_longwave_flux_in_air']==0,'surface_downwelling_longwave_flux_in_air'] = np.nan
        longrad_df.loc[longrad_df['surface_net_longwave_flux_in_air']==0,'surface_net_longwave_flux_in_air'] = np.nan

        longrad_df = longrad_df.loc[:,['time','timestamp','surface_downwelling_longwave_flux_in_air','surface_net_longwave_flux_in_air']]

    
    if (shortrad_df is not None) and (longrad_df is not None):
        rad_df = shortrad_df.copy().merge(longrad_df.copy(), how='left', on=['time']).drop_duplicates().drop(columns=['timestamp_x','timestamp_y'])
    elif (shortrad_df is None):
        rad_df = longrad_df.copy().drop_duplicates()
    elif (longrad_df is None):
        rad_df = shortrad_df.copy().drop_duplicates()
    rad_df = check_missing_vars(rad_df, datatype='rad')


    #############################################################
    # Perform QARTOD tests on the data on the full dataset
    # and null out the particularly bad datapoints
    rad_qc_df = process_qartod_tests(buoy_designation, rad_df, 
                                     sensor_type='rad', fulldata_flag=True)
    for col in rad_df.columns[1:]:
        try:
            rad_df[col][rad_qc_df[col+'_qc_aggregate']>=4] = np.nan
        except:
            if not((('datenum' in col) or ('volts' in col))):
                print('Skipping column type: ' + col)


    ############################################################
    # Average the data
    averaging_window = (5 - 1.5*(0.5)) * 60
    rad_varnames = ['surface_downwelling_longwave_flux_in_air',
                    'surface_downwelling_shortwave_flux_in_air']
    rad_meandf = average_datadf(rad_df.copy(), rad_varnames, averaging_window)

    rad_meandf = rad_meandf.drop_duplicates()
    rad_meandf = rad_meandf.reset_index(drop=True)


    ############################################################
    rad_qc_meandf = process_qartod_tests(buoy_designation, rad_meandf, 
                                         sensor_type='rad', fulldata_flag=False)


    # Fill NaNs with fillval, if needed
    fillVal = -555
    rad_meandf = rad_meandf.fillna(fillVal)
    rad_qc_meandf = rad_qc_meandf.fillna(9)
    
    return rad_meandf, rad_qc_meandf


# # Wind sensor functions

# %%


def get_magnetic_declination(buoy_lat, buoy_lon, datadate):

    datayear = datadate.year
    datamonth = datadate.month
    dataday = datadate.day

    magdec_model = 'IGRF'
    key = 'zNEw7'

    baseurl = 'https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination'

    url_find = (baseurl + '?key=' + key 
                + '&lat1=' + str(buoy_lat) + '&lon1=' + str(buoy_lon) 
                + '&model=' + magdec_model + '&startYear=' + str(datayear)
                + '&startMonth=' + str(datamonth) + '&startDay=' + str(dataday)
                + '&resultFormat=json')


    response = request.urlopen(url_find)
    data_json = json.loads(response.read())
    mag_dec = data_json['result'][0]['declination']
    
    return mag_dec


# %%


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def radians_2_degrees_truenorth(angles):
    
    [x,y] = pol2cart(1, angles)
    
    # Rotate from 0 degrees being in the +x-direction,
    # to 0 degrees being in the +y-direction
    NewAng = np.arctan2(y, -x) - (3*np.pi/2)
    
    angleInDegrees = np.degrees(NewAng)
    angleInDegrees = np.array([ii%360 for ii in angleInDegrees])
    
    return angleInDegrees


# %%


def rotate_vaisala_winds(buoy_designation, wind_filename, ica_filename):
    
    # Read in the CSV
    wind_df = pd.read_csv(wind_filename)  
    ica_df = pd.read_csv(ica_filename)  
    
    wind_df = wind_df.drop_duplicates()
    ica_df = ica_df.drop_duplicates()
    
    if (len(wind_df) == 0) or (len(ica_df)==0):
        return None
    
    wind_df = wind_df.reset_index(drop=True)
    ica_df = ica_df.reset_index(drop=True)
    
    if not((len(wind_df) > 0) and (len(ica_df) > 0)):
        return None
    
    ref_time = datetime.datetime(1970,1,1)
    wind_df['time'] = [(ref_time + datetime.timedelta(microseconds=float(ii)/1e3)) 
                      for ii in wind_df['timestamp']]
    ica_df['time'] = [(ref_time + datetime.timedelta(microseconds=float(ii)/1e3)) 
                      for ii in ica_df['timestamp']]
    
    wind_dates = [nemo.round_seconds(ii) for ii in wind_df['time']]
    ica_dates = [nemo.round_seconds(ii) for ii in ica_df['time']]
    
    # Null out bad compass data
    ica_df.loc[np.logical_and(np.logical_and(ica_df['psi']==0, ica_df['phi']==0),
                              np.logical_and(ica_df['tilt']==0, ica_df['mag']==0),
                              ica_df['inc']==90),
               ica_df.columns.drop('timestamp')] = np.nan

    
    # Convert wind in cartesian components into polar components
    # NOTE: this is polar direction in rad with 0 to EAST!  *wx is NORTH for Vaisala!!* 
    [R, theta] = cart2pol(wind_df['wy'].to_numpy().squeeze(), 
                          wind_df['wx'].to_numpy().squeeze())

    
    # Apply corrections to the compass direction
    # for magnetic declination and error to find
    # direction in regards to "true" north
    buoy_lat = 48.0
    buoy_lon = -125.0

    # Get a magnetic declination for the start and end of the month
    mag_dec_startdate = ica_dates[0].replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    if ica_dates[0].timetuple().tm_mon == 12:
        mag_dec_enddate = (ica_dates[0].replace(year=ica_dates[0].timetuple().tm_year+1,
                                                month=1, day=1, hour=0,
                                                minute=0, second=0, microsecond=0))
    else:
        mag_dec_enddate = (ica_dates[0].replace(month=ica_dates[0].timetuple().tm_mon+1,
                                                day=1,hour=0,minute=0,second=0,microsecond=0))    
    mag_dec_start = get_magnetic_declination(buoy_lat, buoy_lon, mag_dec_startdate)
    mag_dec_end = get_magnetic_declination(buoy_lat, buoy_lon, mag_dec_enddate)

    # Shift the compass time stamps to "seconds since reference time"
    ica_timestamps = [float(ii)/1e9 for ii in ica_df['timestamp']]
    mag_dec_starttimestamp = (mag_dec_startdate - ref_time).total_seconds()
    mag_dec_endtimestamp = (mag_dec_enddate - ref_time).total_seconds()

    # Interpolate the magnetic declinations from the start and end of the month
    # to a declination for each time stamp
    mag_decs = np.interp(ica_timestamps, 
                         np.array([mag_dec_starttimestamp, mag_dec_endtimestamp]),
                         np.array([mag_dec_start, mag_dec_end]))
    
    err = 0
    YdirT = [(ica_df['psi'][ii] + mag_decs[ii] - err)%360 
             for ii in range(0,len(ica_df))]

    # If compass is pointed north, rad. offset is zero. 
    # If east, then rad offset is -pi/2
    pointNorth = True
    if pointNorth:
        angleShift = 0
    else:
        angleShift = 90
    YdirTrad = [-np.radians(ii-angleShift) for ii in YdirT]
    radOffset = [ii - np.radians(angleShift) for ii in YdirTrad]
    
    radOffset_unwrap = np.array(radOffset.copy())
    radOffset_inds = np.where(np.logical_not(np.isnan(radOffset)))[0]
    radOffset_unwrap[radOffset_inds] = np.unwrap(np.array(radOffset)[radOffset_inds], 
                                                 period=2*np.pi)

    # Drop nan-dates
    ica_dates = [ica_dates[ii] for ii in radOffset_inds]
    radOffset_unwrap = radOffset_unwrap[radOffset_inds]
    ica_df = ica_df.iloc[radOffset_inds].reset_index(drop=True)


    # Interpolate the compass radian offset from the compass
    # time onto the wind time
    wind_timeref = [(ii-wind_dates[0]).total_seconds() for ii in wind_dates]
    ica_timeref = [(ii-wind_dates[0]).total_seconds() for ii in ica_dates]
    radOffset_interp = np.interp(wind_timeref, ica_timeref, radOffset_unwrap)
    radOffset_interp = np.array([ii%(2*np.pi) for ii in radOffset_interp])


    # Apply the rotation correction, and shift by 180 degrees
    # to ensure that we have "winds from" rather than "winds toward"
    rotated_theta = theta + radOffset_interp
    degT_rot = radians_2_degrees_truenorth(rotated_theta)
    degT_rot = np.array([ii%360 for ii in degT_rot])

    # Note that these directions are transposed in order from original input
    wu_rot, wv_rot = pol2cart(R, rotated_theta)
    
    
    ## Smooth the data:
    ## Data comes in in 2-minute blocks every 15 minutes,
    ## i.e., 2 minutes of data, 13 minutes of no data, etc.
    
    #wu_rot_smooth = pd.Series(wu_rot, index=wind_dates).rolling(window='120s',center=True).median().to_numpy().squeeze()
    #wv_rot_smooth = pd.Series(wv_rot, index=wind_dates).rolling(window='120s',center=True).median().to_numpy().squeeze()
    #degT_rot_smooth = pd.Series(degT_rot, index=wind_dates).rolling(window='120s',center=True).median().to_numpy().squeeze()
    
    wu_rot = np.round(wu_rot,2)
    wv_rot = np.round(wv_rot,2)
    degT_rot = np.round(degT_rot,2)
    
    #
    wind_df2 = wind_df.copy()
    wind_df2['wu'] = wu_rot
    wind_df2['wv'] = wv_rot
    wind_df2['wd'] = degT_rot
    
    return wind_df2


# %%


def process_wind_files(buoy_designation, windpaths, compasspaths):
    
    wind_df = None
    for ind in range(0,len(windpaths)):
        if (windpaths[ind] is None) or (compasspaths[ind] is None):
            continue
        else:
            temp_df = rotate_vaisala_winds(buoy_designation, windpaths[ind], compasspaths[ind])
        
        if wind_df is None:
            wind_df = temp_df.copy()
        else:
            wind_df = pd.concat([wind_df, temp_df]).reset_index(drop=True)
    wind_df = wind_df.sort_values(by='time').reset_index(drop=True)    
        
    if wind_df is None:
        print('No wind and/or compass data found.')
        return None, None
    
    # Rename columns
    wind_df_sensors = ['ws','wu','wv','gu', 
                       'temp', 'wd']
    wind_df_newname = ['wind_speed', 'eastward_wind', 
                       'northward_wind', 'wind_speed_of_gust', 
                       'air_temperature', 'wind_from_direction']
    
    newname_dict = {}
    for ii in range(0,len(wind_df_sensors)):
        newname_dict[wind_df_sensors[ii]] = wind_df_newname[ii]

    emptycol = np.nan*np.zeros(len(wind_df))
    for col in wind_df_sensors:
        if col not in wind_df.columns:
            wind_df[col] = emptycol
    wind_df.rename(columns=newname_dict, inplace=True)
    
    wind_df = check_missing_vars(wind_df, datatype='wind')
    
    
    #############################################################
    # Perform QARTOD tests on the data on the full dataset
    # and null out the particularly bad datapoints
    wind_qc_df = process_qartod_tests(buoy_designation, wind_df, 
                                      sensor_type='wind', fulldata_flag=True)
    for col in wind_df.columns[1:]:
        try:
            wind_df.loc[wind_qc_df[col+'_qc_aggregate']>=4, col] = np.nan
        except:
            if not((('datenum' in col) or ('volts' in col))):
                print('Skipping column type: ' + col)
                
    
    ############################################################
    # Average the data
    averaging_window = (5 - 1.5*(0.5)) * 60
    wind_varnames = ['eastward_wind', 'northward_wind', 
                     'wind_speed_of_gust', 'air_temperature']
    wind_meandf = average_datadf(wind_df, wind_varnames, averaging_window)
    # Use the averaged eastward and northward wind components to calculate wind speed and direction
    wind_meandf['wind_speed'] = np.sqrt(wind_meandf['eastward_wind']**2 + 
                                        wind_meandf['northward_wind']**2)
    # Calculate wind direction from the averaged wind components
    [R, THETA] = cart2pol(wind_meandf['eastward_wind'], wind_meandf['northward_wind'])
    true_THETA = radians_2_degrees_truenorth(THETA)
    wind_meandf['wind_from_direction'] = true_THETA

    wind_meandf = wind_meandf.drop_duplicates()
    wind_meandf = wind_meandf.reset_index(drop=True)

    
    ############################################################
    wind_qc_meandf = process_qartod_tests(buoy_designation, wind_meandf, 
                                          sensor_type='wind', fulldata_flag=False)
    
    return wind_meandf, wind_qc_meandf


# # QARTOD Functions

# %%


def process_qartod_tests(buoy_designation, data_df, 
                         sensor_type='met', fulldata_flag=False):
    
    if sensor_type == 'met':
        qartod_valid_sensors = ['air_pressure_at_mean_sea_level', 'air_temperature', 
                                'relative_humidity', 'dew_point_temperature',
                                'surface_downwelling_photosynthetic_photon_flux_in_air',
                                'rainfall_amount']
    elif sensor_type == 'wind':
        qartod_valid_sensors = ['wind_speed', 'eastward_wind', 
                                 'northward_wind', 'wind_speed_of_gust', 
                                 'air_temperature', 'wind_from_direction']
    elif sensor_type == 'rad':
        qartod_valid_sensors = ['surface_downwelling_shortwave_flux_in_air', 
                                'surface_downwelling_longwave_flux_in_air']
        
    if fulldata_flag:
        # For full data, the sample rate is 1 Hz
        sample_rate = 1
    else:
        # For the averaged data, the sample rate is 1 per 5 minutes (1/300 Hz)
        sample_rate = 1/(60*5)
        
        
    qartod_df = []
    NT = data_df.shape[0]
    
    # Calculate qartod tests
    sensor_names = [col for col in data_df.columns 
                    if (('time' not in col) and ('date' not in col))]
    for sensor in sensor_names:
        if (sensor in qartod_valid_sensors):
            qc_sensor_name = sensor
            if not((all(data_df[sensor].isna())) or 
                   (all(data_df[sensor] == -555))):              

                if 'direction' in sensor:
                    datavar = data_df[sensor].to_numpy()
                    if any(np.isnan(datavar)):
                        datavar_inds = np.where(np.logical_not(np.isnan(datavar)))[0]
                        datavar_unwrap = datavar.copy()
                        datavar_unwrap[datavar_inds] = np.unwrap(np.array(datavar)[datavar_inds], 
                                                                 period=360)
                    else:
                        datavar_unwrap = np.unwrap(datavar, period=360)
                    
                    temp_df = pd.DataFrame(data = list(zip(np.ndarray.flatten(datavar),
                                                           np.ndarray.flatten(data_df['time'].to_numpy()))),
                                           columns=['wind_from_direction','time'])
                    temp_df_unwrap = pd.DataFrame(data = list(zip(np.ndarray.flatten(datavar_unwrap),
                                                           np.ndarray.flatten(data_df['time'].to_numpy()))),
                                           columns=['wind_from_direction','time'])
                    qc_sensor_name = 'wind_from_direction'
                    
                    # Perform the qartod tests for the unwrapped data
                    # to apply gross range tests
                    temp_qartod_df = nemo_qc.run_qartod_tests(temp_df_unwrap, None,qc_sensor_name, 
                                                              None, sample_rate, platform='orca20_met')
                    
                    # Perform the qartod tests for regular data
                    temp_qartod_df2 = nemo_qc.run_qartod_tests(temp_df, None, qc_sensor_name, 
                                                               None, sample_rate, platform='orca20_met')
                    
                    # Overwrite the gross range test results for the 
                    # unwrapped data with the results from the regular data
                    gross_var = 'wind_from_direction_qartod_gross_range_test'
                    spike_var = 'wind_from_direction_qartod_spike_test'
                    temp_qartod_df[gross_var] = temp_qartod_df2[gross_var].to_numpy().squeeze()
                    newaggr_flags = ioos_qc.qartod.qartod_compare(temp_qartod_df.loc[:,[gross_var,
                                                                                        spike_var]].to_numpy().T)
                    temp_qartod_df['wind_from_direction_qartod_rollup_qc'] = newaggr_flags
                    
                    
                else:
                    
                    if ('wind_speed' in sensor) or ('eastward' in sensor) or ('northward' in sensor):
                        temp_df = pd.DataFrame(data = list(zip(np.ndarray.flatten(abs(data_df[sensor].to_numpy())),
                                                               np.ndarray.flatten(data_df['time'].to_numpy()))),
                                               columns=['wind_speed','time'])
                        qc_sensor_name = 'wind_speed'
                        
                    else:
                    
                
                        temp_df = pd.DataFrame(data = list(zip(np.ndarray.flatten(data_df[sensor].to_numpy()),
                                                               np.ndarray.flatten(data_df['time'].to_numpy()))),
                                               columns=[sensor,'time'])
                        qc_sensor_name = sensor


                    # Perform the qartod tests for the specified sensor
                    temp_qartod_df = nemo_qc.run_qartod_tests(temp_df, None, qc_sensor_name, 
                                                              None, sample_rate, platform='orca20_met')
                    
                    
                    
                temp_results = nemo_qc.concat_test_results_into_string(temp_qartod_df)

                # Append the concatenated test results to the data frame
                temp_qartod_final = temp_qartod_df.copy()
                temp_qartod_final[qc_sensor_name+'_qc_aggregate'] = temp_qartod_df[qc_sensor_name+'_qartod_rollup_qc'].to_numpy()
                temp_qartod_final[qc_sensor_name+'_qc_tests'] = temp_results
                
                if ('wind_speed' in sensor) or ('eastward' in sensor) or ('northward' in sensor):
                    qc_cols = temp_qartod_final.columns
                    new_col_dict = {}
                    for col in qc_cols:
                        new_col_dict[col] = sensor + col[len(qc_sensor_name):]
                    temp_qartod_final.rename(columns=new_col_dict, inplace=True)
                

            else:
                # If all the data is bad, apply "fail" flags everywhere, and assign 99 for the tests, indicating
                # that no tests were performed
                flag_val = 9999
                temp_qartod_final = pd.DataFrame(data=list(zip(4*np.ones(NT).astype(int),
                                                               4*np.ones(NT).astype(int),
                                                               4*np.ones(NT).astype(int),
                                                               4*np.ones(NT).astype(int),
                                                               4*np.ones(NT).astype(int),
                                                               flag_val*np.ones(NT).astype(int))),
                                                 columns=[sensor+'_qc_aggregate',
                                                          sensor+'_qc_gross_range_test',
                                                          sensor+'_qc_spike_test',
                                                          sensor+'_qc_rate_of_change_test',
                                                          sensor+'_qc_flat_line_test',
                                                          sensor+'_qc_tests']) 
                
        
            
            # Append on the newly created qartod flags to the qartod dataframe
            if isinstance(qartod_df,list):
                qartod_df = temp_qartod_final
            else:
                qartod_df = pd.concat([qartod_df, temp_qartod_final],axis=1)
    
                
                
    return qartod_df


# # Data processing

# %%


def check_missing_vars(met_df, datatype='met'):
    
    def add_qc_types(req_vars):
        all_vars = []
        for var in req_vars:
            all_vars.append(var)
            all_vars.append(var+'_qc_aggregate')
            all_vars.append(var+'_qc_gross_range_test')
            all_vars.append(var+'_qc_spike_test')
            all_vars.append(var+'_qc_rate_of_change_test')
            all_vars.append(var+'_qc_flat_line_test')
            all_vars.append(var+'_qc_tests')
        return all_vars
        
    
    if 'met' in datatype:
        required_vars = ['air_pressure_at_mean_sea_level', 'air_temperature',
                         'relative_humidity', 'dew_point_temperature', 
                         'surface_downwelling_photosynthetic_photon_flux_in_air',
                         'rainfall_amount']
        
    elif 'wind' in datatype:
        required_vars = ['wind_speed', 'wind_speed_of_gust',
                         'eastward_wind', 'northward_wind', 
                         'wind_from_direction']
    elif 'rad' in datatype:
        required_vars = ['surface_downwelling_shortwave_flux_in_air', 
                         'surface_downwelling_longwave_flux_in_air']
    else:
        print('Invalid data type selected. Do not check for missing variables.')
        return met_df
    
    if '_qc' in datatype:
        required_vars = add_qc_types(required_vars)
        
    met_cols = met_df.columns
    
    nrows = len(met_df)
    blankvar = np.nan*np.ones(nrows)
    for col in required_vars:
        if col == 'time':
            continue
        elif col not in met_cols:
            met_df.loc[:, col] = blankvar
            
    return met_df


# %%


def merge_with_met(met_data, new_data, met_qc, new_qc,
                   merge_tol=pd.Timedelta(minutes=2), new_suffix='wind'):

    met_merged = pd.merge_asof(met_data.copy().reset_index(), 
                               new_data.copy().reset_index(), 
                               on='time', tolerance=merge_tol,
                               direction='nearest', suffixes=('_met','_'+new_suffix))
    new_merged = pd.merge_asof(new_data.copy().reset_index(), 
                               met_data.copy().reset_index(), 
                               on='time', tolerance=merge_tol,
                               direction='nearest', suffixes=('_'+new_suffix,'_met'))

    new_matched = met_merged['index_'+new_suffix].dropna().to_numpy().squeeze()
    metnew_matched = met_merged['index_met'].dropna().to_numpy().squeeze()
    met_matched = new_merged['index_met'].dropna().to_numpy().squeeze()
    metnew_matched = new_merged['index_'+new_suffix].dropna().to_numpy().squeeze()

    new_missing = list(set(new_data.index) - set(new_matched))
    met_missing = list(set(met_data.index) - set(met_matched))

    if (len(new_data) != len(new_matched) + len(new_missing)):
        print('Error in matching ' + new_suffix + '  indices')
    if (len(met_data) != len(met_matched) + len(met_missing)):
        print('Error in matching met indices')


    if any(['air_temperature' in col for col in new_data.columns]):
        new_data = new_data.copy().rename(columns={'air_temperature': 'air_temperature_'+new_suffix})
        
    if len(new_missing) > 0:
        met_combined = (pd.concat([met_merged, 
                                   new_data.copy().loc[new_missing,:]])
                        .sort_values(by=['time'])
                        .reset_index(drop=True)
                       )
    else:
        met_combined = met_merged.copy()
        
    if any(['air_temperature_met' in col for col in met_combined.columns]):
        met_combined = met_combined.drop(columns=['air_temperature_'+new_suffix]).rename(columns={'air_temperature_met':'air_temperature'})
    
    
    # Merge the QC data the same way
    met_qc.loc[:,'time'] = met_data.loc[:,'time']
    new_qc.loc[:,'time'] = new_data.loc[:,'time']
    new_qc = new_qc.drop(columns=[col for col in new_qc.columns
                          if 'air_temperature' in col])
    
    met_qc_merged = pd.merge_asof(met_qc.reset_index(), 
                                  new_qc.reset_index(), 
                                  on='time', tolerance=merge_tol,
                                  direction='nearest', suffixes=('_met','_'+new_suffix))
    met_qc_combined = (pd.concat([met_qc_merged.reset_index(), 
                                  new_qc.loc[new_missing,:].reset_index()])
                        .sort_values(by=['time'])
                        .reset_index(drop=True)
                        .drop(columns=['index', 'index_met', 'index_'+new_suffix, 'time'])
    )
    
    new_cols = {}
    for col in met_qc_combined.columns:
        if '_qartod_' in col:
            new_cols[col] = col.replace('_qartod_', '_qc_')
    met_qc_combined = met_qc_combined.rename(columns=new_cols)

    drop_cols = []
    for col in met_qc_combined.columns:
        if ('_rollup' in col) or ('_tests' in col):
            drop_cols.append(col)
    met_qc_combined = met_qc_combined.drop(columns=drop_cols)
    
    # Drop index columns
    if 'index_met' in met_combined.columns:
        met_combined = met_combined.drop(columns=['index_met'])
    if 'index_wind' in met_combined.columns:
        met_combined = met_combined.drop(columns=['index_wind'])
    if 'index_rad' in met_combined.columns:
        met_combined = met_combined.drop(columns=['index_rad'])
    
    return met_combined, met_qc_combined


# %%


def find_latest_datafile(buoy_designation, ncdir, latest_date=None):

    ctd_pos = "MET"
    nc_files = sorted([file for file in os.listdir(ncdir) if '.nc' in file])
    if len(nc_files) == 0:
        return None, None, None

    nc_files = [file for file in nc_files if f'_{ctd_pos}_' in file]
    if len(nc_files) == 0:
            return None, None, None
    
    strsearch = f'_{ctd_pos}_'
    nc_dates = [datetime.datetime.strptime(file[file.find(strsearch)+len(strsearch):file.find('.nc')],'%Y%m')
                for file in nc_files]

    if latest_date is None:
        maxind = np.argmax(nc_dates)
    else:
        maxind = np.argmin([abs(nc_date-latest_date) for nc_date in nc_dates])
    recent_file = os.path.join(ncdir,nc_files[maxind])
    if (maxind==0) or ((nc_dates[maxind]-nc_dates[maxind-1]) > datetime.timedelta(days=45)):
        prior_file = None
    else:
        prior_file = os.path.join(ncdir,nc_files[maxind-1])
    
    
    return nc_dates[maxind], recent_file, prior_file


# %%


def load_latest_file(buoy_designation, savedir, latest_date=None):
    def _load_met_nc_as_time_df(nc_path):
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
    
    latest_date, latest_path, prior_path = find_latest_datafile(buoy_designation, savedir, latest_date)
    if latest_date is None:
        return None, None

    print('Loading latest file for buoy: ' + buoy_designation)
    print('Latest file path: ' + latest_path)
    if prior_path is not None:
        print('Prior file path: ' + prior_path)

    ds_latest, met_latest = _load_met_nc_as_time_df(latest_path)
    met_latest = (met_latest.drop_duplicates(subset='time')
                  .sort_values(by='time')
                  .reset_index(drop=True))
    
    if prior_path is not None:
        _, met_prior = _load_met_nc_as_time_df(prior_path)
        met_prior = (met_prior.drop_duplicates(subset='time')
                      .sort_values(by='time')
                      .reset_index(drop=True))
        
        met_latest = (pd.concat([met_prior, met_latest]).drop_duplicates(subset='time')
                      .sort_values(by='time')
                      .reset_index(drop=True))
        
    # Drop out data that is within the last 3 hours
    checktime = (datetime.datetime.now() + datetime.timedelta(hours=7)
                 - datetime.timedelta(hours=3))
    met_latest = met_latest.loc[met_latest.loc[:,'time']<=checktime,:].reset_index(drop=True)
        
    # Do a double check to remove duplicate points
    met_latest = met_latest.drop_duplicates()
    met_latest = met_latest.reset_index(drop=True)
        
    
    if (('eastward_wind' not in met_latest.columns) or
        ('northward_wind' not in met_latest.columns)):
        
        uwind = (-met_latest.loc[:,'wind_speed'] * 
                 np.sin(np.radians(met_latest.loc[:,'wind_from_direction'])))
        vwind = (-met_latest.loc[:,'wind_speed'] * 
                 np.cos(np.radians(met_latest.loc[:,'wind_from_direction'])))
        blankwind = np.nan*np.ones(len(uwind))
        met_latest.loc[:,'eastward_wind'] = uwind
        met_latest.loc[:,'northward_wind'] = vwind
        met_latest.loc[:,'wind_speed_of_gust'] = blankwind
        
        qc_cols = [col[col.find('_qc'):] for col in met_latest.columns
                   if 'wind_speed_qc' in col]
        for qc_col in qc_cols:
            met_latest.loc[:,'eastward_wind'+qc_col] = met_latest.loc[:,'wind_speed'+qc_col]
            met_latest.loc[:,'northward_wind'+qc_col] = met_latest.loc[:,'wind_speed'+qc_col]
            met_latest.loc[:,'wind_speed_of_gust'+qc_col] = 9*np.ones(len(uwind))
    
    # Backward compatibility for previously written files that used
    # eastward_wind_speed_* / northward_wind_speed_* QC names.
    legacy_qc_rename = {}
    for col in met_latest.columns:
        if col.startswith('eastward_wind_speed_qc_'):
            new_col = col.replace('eastward_wind_speed_qc_', 'eastward_wind_qc_')
            if new_col not in met_latest.columns:
                legacy_qc_rename[col] = new_col
        elif col.startswith('northward_wind_speed_qc_'):
            new_col = col.replace('northward_wind_speed_qc_', 'northward_wind_qc_')
            if new_col not in met_latest.columns:
                legacy_qc_rename[col] = new_col
    if len(legacy_qc_rename) > 0:
        met_latest = met_latest.rename(columns=legacy_qc_rename)
            
            
    required_vars = ['air_pressure_at_mean_sea_level', 'air_temperature',
                     'relative_humidity', 'dew_point_temperature', 
                     'rainfall_amount',
                     'surface_downwelling_shortwave_flux_in_air',
                     'surface_downwelling_longwave_flux_in_air',
                     'surface_downwelling_photosynthetic_photon_flux_in_air']
    req_qcvar = ['qc_gross_range_test', 'qc_spike_test',
                 'qc_rate_of_change_test', 'qc_flat_line_test',
                 'qc_aggregate']
    met_latest_cols = met_latest.columns
    blankqc = 9*np.ones(len(met_latest))
    for var in required_vars:
        if var not in met_latest_cols:
            met_latest.loc[:,var] = blankqc
        for qc_var in req_qcvar:
            if (var+'_'+qc_var) not in met_latest_cols:
                met_latest.loc[:,var+'_'+qc_var] = blankqc                
    
    return met_latest, met_latest.loc[len(met_latest)-1,'time'].to_pydatetime()


# %%


def get_filelists(buoy_designation, basedir, start_date, end_date):
    
    # Get the instrument directory paths
    metdir = os.path.join(basedir, 'met')
    winddir = os.path.join(basedir, 'wind')
    compassdir = os.path.join(basedir, 'compass')
    
    shortraddir = None
    if os.path.exists(os.path.join(basedir, 'pyranometer_shortwave')):
        shortraddir = os.path.join(basedir, 'pyranometer_shortwave')
    longraddir = None
    if os.path.exists(os.path.join(basedir, 'pyrgeometer_longwave')):
        longraddir = os.path.join(basedir, 'pyrgeometer_longwave')
    

    # Build a list of met files
    metfiles = sorted([file for file in os.listdir(metdir) if '.csv' in file])
    filestr = '_met_'
    met_filedates = [datetime.datetime.strptime(file[file.find(filestr)+len(filestr):file.find('.csv')],'%Y%m%d') 
                     for file in metfiles]
    
    # Build a list of wind and compass files
    windfiles = sorted([file for file in os.listdir(winddir) if '.csv' in file])
    compassfiles = sorted([file for file in os.listdir(compassdir) if '.csv' in file])
    
    filestr = '_wind_'
    wind_filedates = [datetime.datetime.strptime(file[file.find(filestr)+len(filestr):file.find('.csv')],'%Y%m%d')
                     for file in windfiles]
    filestr = '_compass_'
    compass_filedates = [datetime.datetime.strptime(file[file.find(filestr)+len(filestr):file.find('.csv')],'%Y%m%d')
                     for file in compassfiles]
    
    # Build a list of short- and longwave radiation files
    if shortraddir is not None:
        shortradfiles = sorted([file for file in os.listdir(shortraddir) if '.csv' in file])
        filestr = '_pyranometer_shortwave_'
        shortrad_filedates = [datetime.datetime.strptime(file[file.find(filestr)+len(filestr):file.find('.csv')],'%Y%m%d')
                              for file in shortradfiles]
    else:
        shortradfiles = []
        shortrad_filedates = []
        
    if longraddir is not None:
        longradfiles = sorted([file for file in os.listdir(longraddir) if '.csv' in file])
        filestr = '_pyrgeometer_longwave_'
        longrad_filedates = [datetime.datetime.strptime(file[file.find(filestr)+len(filestr):file.find('.csv')],'%Y%m%d')
                             for file in longradfiles]
    else:
        longradfiles = []
        longrad_filedates = []
        

    #######################################################
    # Restrict the files to just the time range of interest
    
    # Met files
    if start_date is not None:
        met_fileinds = np.where([(start_date <= filedate)
                                 for filedate in met_filedates])[0]
        metfiles = [metfiles[ind] for ind in met_fileinds]
        met_filedates = [met_filedates[ind] for ind in met_fileinds]
        
        # Wind and compass files
        wind_fileinds = np.where([(start_date <= filedate) 
                                  for filedate in wind_filedates])[0]
        compass_fileinds = np.where([(start_date <= filedate)
                                     for filedate in compass_filedates])[0]

    if end_date is not None:
        met_fileinds = np.where([(met_filedates[ind] <= end_date)
                                 for ind in range(len(met_filedates))])[0]
        metfiles = [metfiles[ind] for ind in met_fileinds]
        met_filedates = [met_filedates[ind] for ind in met_fileinds]

        wind_fileinds = np.where([(wind_filedates[ind] <= end_date) 
                                  for ind in range(len(wind_filedates))])[0]
        compass_fileinds = np.where([(compass_filedates[ind] <= end_date)
                                     for ind in range(len(compass_filedates))])[0]

    windfiles = [windfiles[ind] for ind in wind_fileinds]
    compassfiles = [compassfiles[ind] for ind in compass_fileinds]

    wind_filedates = [wind_filedates[ind] for ind in wind_fileinds]
    compass_filedates = [compass_filedates[ind] for ind in compass_fileinds]

    # Short and longwave rad files
    if shortraddir is not None:
        shortrad_fileinds = np.where([((start_date <= filedate) and (end_date >= filedate))
                                      for filedate in shortrad_filedates])[0]
        shortradfiles = [shortradfiles[ind] for ind in shortrad_fileinds]
        shortrad_filedates = [shortrad_filedates[ind] for ind in shortrad_fileinds]
    if longraddir is not None:
        longrad_fileinds = np.where([((start_date <= filedate) and (end_date >= filedate)) 
                                     for filedate in longrad_filedates])[0]
        longradfiles = [longradfiles[ind] for ind in longrad_fileinds]
        longrad_filedates = [longrad_filedates[ind] for ind in longrad_fileinds]
    
    
    # Identify the min and max file ranges
    mindate = min([min(met_filedates, default=pd.NaT), 
                   min(wind_filedates, default=pd.NaT), min(compass_filedates, default=pd.NaT),
                   min(shortrad_filedates, default=pd.NaT), min(longrad_filedates, default=pd.NaT)])
    maxdate = max([max(met_filedates, default=pd.NaT), 
                   max(wind_filedates, default=pd.NaT), max(compass_filedates, default=pd.NaT),
                   max(shortrad_filedates, default=pd.NaT), max(longrad_filedates, default=pd.NaT)])
    if mindate is pd.NaT:
        mindate = start_date
    if maxdate is pd.NaT:
        maxdate = end_date
    
    # Use the file min and max dates to build out a 
    # file date range, and append any files
    # that are in that range
    file_daterange = pd.date_range(start=mindate, end=maxdate, freq='D')
    for dateind in range(0,len(file_daterange)):
        if file_daterange[dateind] not in met_filedates:
            metfiles.insert(dateind,None)
            windfiles.insert(dateind,None)
            compassfiles.insert(dateind,None)  
        if (shortraddir is not None) and (file_daterange[dateind] not in shortrad_filedates):
            shortradfiles.insert(dateind,None)
        if (longraddir is not None) and (file_daterange[dateind] not in longrad_filedates):
            longradfiles.insert(dateind,None)
            
    return (metdir, winddir, compassdir, shortraddir, longraddir,
            metfiles, windfiles, compassfiles, shortradfiles, longradfiles)


# %%


def load_deployment_starts(buoy_designation, depdir, startDate=None, endDate=None):
    
    dep_file = buoy_designation.lower() + '_deployments.json'
    with open(os.path.join(depdir, dep_file),'r') as f:
        dep_info = json.load(f)

    all_deps = [dep_info['deployments'][dep] for dep in dep_info['deployments']]
    if startDate is None and endDate is None:
        nemo_deps = [nemo_dep for nemo_dep in all_deps if nemo_dep['status'] == 'realtime']
    else:
        nemo_deps = []
        for nemo_dep in all_deps:
            dep_start = datetime.datetime.strptime(nemo_dep['start'], '%Y-%m-%dT%H:%M:%SZ')
            dep_end = (datetime.datetime.strptime(nemo_dep['end'], '%Y-%m-%dT%H:%M:%SZ')
                       if nemo_dep['end'] is not None else None)

            overlaps_start = ((startDate is None) or (dep_end is None) or (dep_end >= startDate))
            overlaps_end = ((endDate is None) or (dep_start <= endDate))
            if overlaps_start and overlaps_end:
                nemo_deps.append(nemo_dep)

    if len(nemo_deps) == 0:
        nemo_deps = all_deps

    dep_starts = [datetime.datetime.strptime(dep['start'],'%Y-%m-%dT%H:%M:%SZ')
                  for dep in nemo_deps]
    dep_ends = [datetime.datetime.strptime(dep['end'],'%Y-%m-%dT%H:%M:%SZ') 
                if dep['end'] is not None else datetime.datetime.now()+datetime.timedelta(days=1)
                 for dep in nemo_deps]
    
    return nemo_deps, dep_starts, dep_ends


# %%


def make_new_met_xarray(buoy_designation, buoy_name, 
                        buoy_lat, buoy_lon, package_name, 
                        met_df):
    
    nrows = len(met_df)
    fillVal = -555
    wmo_code = 46119

    xr_new = xr.Dataset()

    xr_new.attrs["title"] = "Cha'ba: Meteorological Station Data"
    xr_new.attrs["description"] = "Meteorological Data taken from the Cha'ba mooring"
    xr_new.attrs["history"] = "Process historic data on " + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
    xr_new.attrs['conventions'] = 'CF-1.6, ACDD-1.3, IOOS-1.2'
    xr_new.attrs['buoy_designation'] = buoy_designation
    xr_new.attrs['buoy_latitude'] = buoy_lat
    xr_new.attrs['buoy_longitude'] = buoy_lon
    xr_new.attrs['author'] = 'Seth Travis'
    xr_new.attrs['institution'] = 'University of Washington - Applied Physical Laboratory - NorthWest Environmental Moorings (NWEM) Group'
    xr_new.attrs['contact'] = 'setht1@uw.edu'
    xr_new.attrs['_FillValue'] = fillVal
    xr_new.attrs['cdm_data_type'] = 'timeSeries'
    xr_new.attrs['cdm_timeseries_variables'] = 'location_id,deployment_name,latitude,longitude'
    xr_new.attrs['subsetVariables'] = 'deployment_name'
    
    # The global attributes are needed to allow for ingesting by NDBC
    # into the GTS data harvester for models
    xr_new.attrs['wmo_platform_code'] = str(wmo_code)
    xr_new.attrs['gts_ingest'] = 'true'
    
    xr_new.attrs['contributor_email'] = 'setht1@uw.edu, mickett@uw.edu'
    xr_new.attrs['contributor_name'] = 'Seth Travis, John Mickett'
    xr_new.attrs['contributor_role'] = 'author, principalInvestigatorr'
    xr_new.attrs['contributor_role_vocabulary'] = 'https://vocab.nerc.ac.uk/collection/G04/current/'
    xr_new.attrs['contributor_url'] = 'https://nwem.apl.washington.edu/, https//www.nanoos.org/, https://nwem.apl.washington.edu/'
    
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
    
    xr_new.attrs['publisher_address'] = 'Henderson Hall, 616 NE Northlake Place, Seattle, WA 98105'
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
    
    
    
    curyear = str(datetime.datetime.now().year)
    xr_new.attrs['citation'] = ('University of Washington NWEM Mooring Data, 2005-' + curyear + 
                        '; https://nwem.apl.washington.edu/, enabled by the Northwest Association of Networked Ocean Observing Systems ' +
                        '(NANOOS, https://www.nanoos.org, https://ror.org/01a258x16) and the Washington State Ocean Acidification Center ' +
                        '(WOAC, https://oceanacidification.uw.edu/)')
    xr_new.attrs['institution'] = 'Northwest Environmental Moorings Group at University of Washington - Applied Physical Laboratory'
    
    
    
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

    xr_new['time'] = xr.DataArray(met_df['time'], dims='time')
    if nrows > 0:
        xr_new['time'] = xr_new['time'].astype(np.datetime64)
    xr_new['time'].attrs["long_name"] = 'sample_time'
    xr_new['time'].attrs["description"] = 'time of sampling'  
    xr_new['time'].encoding["dtype"] = "float32"
    xr_new['time'].encoding["units"] = 'seconds since 1970-01-01 00:00:00' 
    xr_new['time'].encoding["calendar"] = 'gregorian'  
    
    #return xr_new
    
    xr_new['location_id'] = xr.DataArray([buoy_name], dims='location_id')
    xr_new['location_id'].attrs["long_name"] = 'location_id'
    xr_new['location_id'].attrs["description"] = 'Buoy Label Identifier'
    xr_new['location_id'].attrs["cf_role"] = 'timeseries_id'
    xr_new['location_id'].attrs["units"] = '1'
    xr_new['location_id'].attrs["ioos_category"] = 'Identifier'

    xr_new['deployment_name'] = xr.DataArray([buoy_designation for ii in range(0,nrows)], dims='time')
    xr_new['deployment_name'].attrs["long_name"] = 'deployment_name'
    xr_new['deployment_name'].attrs["description"] = 'Deployment designation'

    xr_new['buoy_name'] = xr.DataArray([buoy_designation for ii in range(0,nrows)], dims='time')
    xr_new['buoy_name'].attrs["long_name"] = 'buoy_name'
    xr_new['buoy_name'].attrs["description"] = buoy_designation + ': buoy located at ' + buoy_name
    xr_new['buoy_name'].attrs["cf_role"] = 'timeseries_id'
    
    xr_new['latitude'] = xr.DataArray([float(buoy_lat) for ii in range(0,nrows)], dims='time')
    xr_new['latitude'].attrs["long_name"] = 'latitude'
    xr_new['latitude'].attrs["units"] = 'degrees North'
    
    xr_new['longitude'] = xr.DataArray([float(buoy_lon) for ii in range(0,nrows)], dims='time')
    xr_new['longitude'].attrs["long_name"] = 'longitude'
    xr_new['longitude'].attrs["units"] = 'degrees East'



    xr_new['met_package'] = xr.DataArray(package_name, dims='time')
    xr_new['met_package'].attrs["long_name"] = 'met_package'
    xr_new['met_package'].attrs["description"] = 'Meteorological Instrumentation Package Label'
    
    xr_new['air_pressure_at_mean_sea_level'] = xr.DataArray(met_df['air_pressure_at_mean_sea_level'].to_numpy(), dims='time')
    xr_new['air_pressure_at_mean_sea_level'].fillna(fillVal)
    xr_new['air_pressure_at_mean_sea_level'].attrs["long_name"] = 'air_pressure_at_mean_sea_level'
    xr_new['air_pressure_at_mean_sea_level'].attrs["standard_name"] = 'air_pressure_at_mean_sea_level'
    xr_new['air_pressure_at_mean_sea_level'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/air_pressure_at_mean_sea_level'
    xr_new['air_pressure_at_mean_sea_level'].attrs["description"] = 'Air pressure at sea level is the quantity often abbreviated as MSLP or PMSL. Air pressure is the force per unit area which would be exerted when the moving gas molecules of which the air is composed strike a theoretical surface of any orientation. "Mean sea level" means the time mean of sea surface elevation at a given location over an arbitrary period sufficient to eliminate the tidal signals.'
    xr_new['air_pressure_at_mean_sea_level'].attrs["units"] = 'hPa'
    xr_new['air_pressure_at_mean_sea_level'].attrs["missing_value"] = fillVal
    xr_new['air_pressure_at_mean_sea_level'].attrs["_FillValue"] = fillVal 
    xr_new['air_pressure_at_mean_sea_level'].attrs['gts_ingest'] = 'true'
    xr_new['air_pressure_at_mean_sea_level'].attrs['ancillary_variables'] = make_ancvar_str('air_pressure_at_mean_sea_level')

    xr_new['relative_humidity'] = xr.DataArray(met_df['relative_humidity'].to_numpy(), dims='time')
    xr_new['relative_humidity'].fillna(fillVal)
    xr_new['relative_humidity'].attrs["long_name"] = 'relative_humidity'
    xr_new['relative_humidity'].attrs["standard_name"] = 'relative_humidity'
    xr_new['relative_humidity'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/relative_humidity'
    xr_new['relative_humidity'].attrs["description"] = 'percent mass fraction of water vapor to air'
    xr_new['relative_humidity'].attrs["units"] = 'percent'
    xr_new['relative_humidity'].attrs["missing_value"] = fillVal 
    xr_new['relative_humidity'].attrs["_FillValue"] = fillVal 
    xr_new['relative_humidity'].attrs['gts_ingest'] = 'true'
    xr_new['relative_humidity'].attrs['ancillary_variables'] = make_ancvar_str('relative_humidity')

    xr_new['air_temperature'] = xr.DataArray(met_df['air_temperature'].to_numpy(), dims='time')
    xr_new['air_temperature'].fillna(fillVal)
    xr_new['air_temperature'].attrs["long_name"] = 'air_temperature'
    xr_new['air_temperature'].attrs["standard_name"] = 'air_temperature'
    xr_new['air_temperature'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/air_temperature'
    xr_new['air_temperature'].attrs["description"] = 'bulk temperature of the air'
    xr_new['air_temperature'].attrs["units"] = 'degrees C'
    xr_new['air_temperature'].attrs["missing_value"] = fillVal 
    xr_new['air_temperature'].attrs["_FillValue"] = fillVal 
    xr_new['air_temperature'].attrs['gts_ingest'] = 'true'
    xr_new['air_temperature'].attrs['ancillary_variables'] = make_ancvar_str('air_temperature')


    xr_new['dew_point_temperature'] = xr.DataArray(met_df['dew_point_temperature'].to_numpy(), dims='time')
    xr_new['dew_point_temperature'].fillna(fillVal)
    xr_new['dew_point_temperature'].attrs["long_name"] = 'dew_point_temperature'
    xr_new['dew_point_temperature'].attrs["standard_name"] = 'dew_point_temperature'
    xr_new['dew_point_temperature'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/dew_point_temperature'
    xr_new['dew_point_temperature'].attrs["description"] = 'temperature at which of parcel of air reaches saturation upon being cooled at constant pressure and specific humidity'
    xr_new['dew_point_temperature'].attrs["units"] = 'degrees C'
    xr_new['dew_point_temperature'].attrs["missing_value"] = fillVal 
    xr_new['dew_point_temperature'].attrs["_FillValue"] = fillVal 
    xr_new['dew_point_temperature'].attrs['gts_ingest'] = 'true'
    xr_new['dew_point_temperature'].attrs['ancillary_variables'] = make_ancvar_str('dew_point_temperature')

    if 'rainfall_amount' in met_df.columns:
        rain_data = met_df['rainfall_amount'].to_numpy()
    else:
        rain_data = fillVal*np.ones(nrows)
    xr_new['rainfall_amount'] = xr.DataArray(rain_data, dims='time')
    xr_new['rainfall_amount'].fillna(fillVal)
    xr_new['rainfall_amount'].attrs["long_name"] = 'rainfall_amount'
    xr_new['rainfall_amount'].attrs["description"] = 'mass of rain per unit area'
    xr_new['rainfall_amount'].attrs["units"] = 'kg * m^-2'
    xr_new['rainfall_amount'].attrs["missing_value"] = fillVal
    xr_new['rainfall_amount'].attrs["_FillValue"] = fillVal
    xr_new['rainfall_amount'].attrs['gts_ingest'] = 'false'
    xr_new['rainfall_amount'].attrs['ancillary_variables'] = make_ancvar_str('rainfall_amount')

    xr_new['wind_speed'] = xr.DataArray(met_df['wind_speed'].to_numpy(), dims='time')
    xr_new['wind_speed'].fillna(fillVal)
    xr_new['wind_speed'].attrs["long_name"] = 'wind_speed'
    xr_new['wind_speed'].attrs["standard_name"] = 'wind_speed'
    xr_new['wind_speed'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/wind_speed'
    xr_new['wind_speed'].attrs["description"] = 'magnitude of the horizontal wind velocity'
    xr_new['wind_speed'].attrs["units"] = 'm * s^-1'
    xr_new['wind_speed'].attrs["missing_value"] = fillVal 
    xr_new['wind_speed'].attrs["_FillValue"] = fillVal 
    xr_new['wind_speed'].attrs['gts_ingest'] = 'true'

    xr_new['wind_speed_of_gust'] = xr.DataArray(met_df['wind_speed_of_gust'].to_numpy(), dims='time')
    xr_new['wind_speed_of_gust'].fillna(fillVal)
    xr_new['wind_speed_of_gust'].attrs["long_name"] = 'wind_speed_of_gust'
    xr_new['wind_speed_of_gust'].attrs["standard_name"] = 'wind_speed_of_gust'
    xr_new['wind_speed_of_gust'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/wind_speed'
    xr_new['wind_speed_of_gust'].attrs["description"] = 'magnitude of the horizontal wind velocity'
    xr_new['wind_speed_of_gust'].attrs["units"] = 'm * s^-1'
    xr_new['wind_speed_of_gust'].attrs["missing_value"] = fillVal 
    xr_new['wind_speed_of_gust'].attrs["_FillValue"] = fillVal 
    xr_new['wind_speed_of_gust'].attrs['gts_ingest'] = 'true'

    xr_new['eastward_wind'] = xr.DataArray(met_df['eastward_wind'].to_numpy(), dims='time')
    xr_new['eastward_wind'].fillna(fillVal)
    xr_new['eastward_wind'].attrs["long_name"] = 'eastward_wind'
    xr_new['eastward_wind'].attrs["standard_name"] = 'eastward_wind'
    xr_new['eastward_wind'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/wind_speed'
    xr_new['eastward_wind'].attrs["description"] = 'magnitude of the horizontal wind velocity'
    xr_new['eastward_wind'].attrs["units"] = 'm * s^-1'
    xr_new['eastward_wind'].attrs["missing_value"] = fillVal 
    xr_new['eastward_wind'].attrs["_FillValue"] = fillVal 
    xr_new['eastward_wind'].attrs['gts_ingest'] = 'true'

    xr_new['northward_wind'] = xr.DataArray(met_df['northward_wind'].to_numpy(), dims='time')
    xr_new['northward_wind'].fillna(fillVal)
    xr_new['northward_wind'].attrs["long_name"] = 'northward_wind'
    xr_new['northward_wind'].attrs["standard_name"] = 'northward_wind'
    xr_new['northward_wind'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/wind_speed'
    xr_new['northward_wind'].attrs["description"] = 'magnitude of the horizontal wind velocity'
    xr_new['northward_wind'].attrs["units"] = 'm * s^-1'
    xr_new['northward_wind'].attrs["missing_value"] = fillVal 
    xr_new['northward_wind'].attrs["_FillValue"] = fillVal 
    xr_new['northward_wind'].attrs['gts_ingest'] = 'true'

    xr_new['wind_from_direction'] = xr.DataArray(met_df['wind_from_direction'].to_numpy(), dims='time')
    xr_new['wind_from_direction'].fillna(fillVal)
    xr_new['wind_from_direction'].attrs["long_name"] = 'wind_from_direction'
    xr_new['wind_from_direction'].attrs["standard_name"] = 'wind_from_direction'
    xr_new['wind_from_direction'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/wind_from_direction'
    xr_new['wind_from_direction'].attrs["description"] = 'direction that the wind is blowing from, measured as positive clockwise from due north'
    xr_new['wind_from_direction'].attrs["units"] = 'degrees'
    xr_new['wind_from_direction'].attrs["missing_value"] = fillVal 
    xr_new['wind_from_direction'].attrs["_FillValue"] = fillVal 
    xr_new['wind_from_direction'].attrs['gts_ingest'] = 'true'

    xr_new['surface_downwelling_shortwave_flux_in_air'] = xr.DataArray(met_df['surface_downwelling_shortwave_flux_in_air'].to_numpy(), dims='time')
    xr_new['surface_downwelling_shortwave_flux_in_air'].fillna(fillVal)
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs["long_name"] = 'surface_downwelling_shortwave_flux_in_air'
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs["standard_name"] = 'surface_downwelling_shortwave_flux_in_air'
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/surface_downwelling_shortwave_flux_in_air'
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs["description"] = 'surface downwelling shortwave radiation is the sum of direct and diffuse solar radiation incident on the surface'
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs["units"] = 'W * m^-2'
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs["missing_value"] = fillVal 
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs["_FillValue"] = fillVal 
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs['gts_ingest'] = 'true'
    xr_new['surface_downwelling_shortwave_flux_in_air'].attrs['ancillary_variables'] = make_ancvar_str('surface_downwelling_shortwave_flux_in_air')
    
    xr_new['surface_downwelling_longwave_flux_in_air'] = xr.DataArray(met_df['surface_downwelling_longwave_flux_in_air'].to_numpy(), dims='time')
    xr_new['surface_downwelling_longwave_flux_in_air'].fillna(fillVal)
    xr_new['surface_downwelling_longwave_flux_in_air'].attrs["long_name"] = 'surface_downwelling_longwave_flux_in_air'
    xr_new['surface_downwelling_longwave_flux_in_air'].attrs["standard_name"] = 'surface_downwelling_longwave_flux_in_air'
    xr_new['surface_downwelling_longwave_flux_in_air'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/surface_downwelling_longwave_flux_in_air'
    xr_new['surface_downwelling_longwave_flux_in_air'].attrs["description"] = 'surface downwelling longwave radiation is the sum of direct and diffuse solar radiation incident on the surface'
    xr_new['surface_downwelling_longwave_flux_in_air'].attrs["units"] = 'W * m^-2'
    xr_new['surface_downwelling_longwave_flux_in_air'].attrs["missing_value"] = fillVal 
    xr_new['surface_downwelling_longwave_flux_in_air'].attrs["_FillValue"] = fillVal 
    xr_new['surface_downwelling_longwave_flux_in_air'].attrs['gts_ingest'] = 'true'

    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'] = xr.DataArray(met_df['surface_downwelling_photosynthetic_photon_flux_in_air'].to_numpy(), dims='time')
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].fillna(fillVal)
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].attrs["long_name"] = 'surface_downwelling_photosynthetic_photon_flux_in_air'
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].attrs["standard_name"] = 'surface_downwelling_photosynthetic_photon_flux_in_air'
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].attrs["standard_name_url"] = 'https://mmisw.org/ont/cf/parameter/surface_downwelling_photosynthetic_photon_flux_in_air'
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].attrs["description"] = 'PAR (photosynthetically available radiation) - the radiation in the wavelength band of 400-700 nm'
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].attrs["units"] = 'umol m^-2 s^-1 '
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].attrs["missing_value"] = fillVal 
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].attrs["_FillValue"] = fillVal
    xr_new['surface_downwelling_photosynthetic_photon_flux_in_air'].attrs['gts_ingest'] = 'true'
    
    return xr_new


# %%


def met_qc_process(buoy_designation, met_qc, datatime):
    

    fillVal = -555
    met_qc_xr = xr.Dataset()
    
    met_qc[met_qc.isna()] = fillVal
    met_qc_xr.attrs['_FillValue'] = fillVal

    met_qc_xr['time'] = xr.DataArray(datatime, dims='time')
    met_qc_xr['time'] = met_qc_xr['time'].astype(np.datetime64)
    met_qc_xr['time'].attrs["long_name"] = 'sample_time'
    met_qc_xr['time'].attrs["description"] = 'time of sampling'
    met_qc_xr['time'].encoding["units"] = 'seconds since 1970-01-01 00:00:00'
    met_qc_xr['time'].encoding["timezone"] = 'UTC'
    met_qc_xr['time'].encoding["calendar"] = 'gregorian'
    
    
     #############
    # QARTOD Data

    varnames = ['air_pressure_at_mean_sea_level',
                'relative_humidity',
                'air_temperature',
                'dew_point_temperature',
                'surface_downwelling_shortwave_flux_in_air',
                'surface_downwelling_longwave_flux_in_air',
                'surface_downwelling_photosynthetic_photon_flux_in_air',
                'wind_speed',
                'wind_speed_of_gust',
                'eastward_wind',
                'northward_wind',
                'wind_from_direction']
    varlabs = ['Surface Air Pressure', 'Relative Humidity', 
               'Air Temperature', 'Dew Point Temperature',
               'Surface Downwelling Shortwave Flux in Air',
               'Surface Downwelling Longwave Flux in Air',
               'Surface Downwelling Photosynthetic Photon Flux in Air',
               'Wind Speed', 'Wind Speed of Gust',
               'Eastward Wind', 'Northward Wind', 'Wind From Direction']

    origqc_vars = ['qc_aggregate', 
                   'qc_gross_range_test', 'qc_rate_of_change_test',
                   'qc_spike_test', 'qc_flat_line_test']
    qc_vars = ['qc_aggregate', 
               'qc_gross_range_test', 'qc_rate_of_change_test',
               'qc_spike_test', 'qc_flat_line_test']
    qc_standards = ['aggregate_quality_flag', 
                    'gross_range_test_quality_flag', 
                    'rate_of_change_test_quality_flag', 'spike_test_quality_flag',
                    'flat_line_test_quality_flag']
    qc_labs = ['Aggregate Flag', 'Gross Range Test Flag', 
               'Rate of Change Test Flag', 'Spike Test Flag',
               'Flat Line Test Flag']
    
    def _qc_col_name(base_var, qc_suffix):
        preferred = base_var + '_' + qc_suffix
        if preferred in met_qc.columns:
            return preferred
        legacy = base_var + '_speed_' + qc_suffix
        if legacy in met_qc.columns:
            return legacy
        return preferred

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
            source_col = _qc_col_name(var, origqc_var)
            target_col = var + '_' + qc_var
            met_qc_xr[target_col] = xr.DataArray(met_qc[source_col].values, dims='time').astype(np.int32)
            
            #temp_qcvar = dataset.createVariable(var + '_' + qc_var,'i4',('location_id','time'))
            met_qc_xr[target_col].attrs['standard_name'] = qc_standard
            met_qc_xr[target_col].attrs['long_name'] = target_col
            met_qc_xr[target_col].attrs['description'] = varlabel + ' ' + qc_lab
            met_qc_xr[target_col].attrs['ioos_category'] = 'Quality Control'
            met_qc_xr[target_col].attrs['units'] = '1'
            met_qc_xr[target_col].attrs['coverage_content_type'] = 'qualityInformation'
            met_qc_xr[target_col].attrs['flag_vals'] = '1, 2, 3, 4, 9'
            met_qc_xr[target_col].attrs['flag_meanings'] = 'PASS NOT_EVALUATED SUSPECT FAIL MISSING'
            met_qc_xr[target_col].attrs['_FillValue'] = fillVal
            met_qc_xr[target_col].attrs["missing_value"] = fillVal
            #met_qc_xr[var + '_' + qc_var] = met_qc[var + '_' + origqc_var]

    for col in varnames:
        source_col = _qc_col_name(col, 'qc_aggregate')
        target_col = col + '_qc_aggregate'
        met_qc_xr[target_col] = xr.DataArray(met_qc[source_col].values, dims='time').astype(np.int32)

        met_qc_xr[target_col].attrs['standard_name'] = qc_standard
        met_qc_xr[target_col].attrs['long_name'] = target_col
        met_qc_xr[target_col].attrs['description'] = 'QARTOD Aggregate flag for ' + col.replace('_',' ')
        met_qc_xr[target_col].attrs['ioos_category'] = 'Quality Control'
        met_qc_xr[target_col].attrs['units'] = '1'
        met_qc_xr[target_col].attrs['coverage_content_type'] = 'qualityInformation'
        met_qc_xr[target_col].attrs['flag_vals'] = '1, 2, 3, 4, 9'
        met_qc_xr[target_col].attrs['flag_meanings'] = 'PASS NOT_EVALUATED SUSPECT FAIL MISSING'
        met_qc_xr[target_col].attrs['_FillValue'] = fillVal
        met_qc_xr[col + '_qc_aggregate'].attrs["missing_value"] = fillVal
        
    return met_qc_xr


# %%


def write_met_netcdf(savedir, save_filename, ds):
    
    # Write the initial data
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


# %%


def add_blank_data(met_data, wind_data, rad_data, 
                   met_qc, wind_qc, rad_qc):
    
    # If all the dataframes are None, return None
    if (met_data is None) and (wind_data is None) and (rad_data is None):
        return None, None
    
    
    # Check each of the dataframes to see if they exist,
    # and extract out the data "time" variable to use in
    # populating a new dataframe
    if met_data is None:
        if (wind_data is not None):
            datatime = wind_data['time'].to_numpy()
        elif (rad_data is not None):
            datatime = rad_data['time'].to_numpy()
        else:
            return None, None
    elif wind_data is None:
        if (met_data is not None):
            datatime = met_data['time'].to_numpy()
        elif (rad_data is not None):
            datatime = rad_data['time'].to_numpy()
        else:
            return None, None
    elif rad_data is None:
        if (met_data is not None):
            datatime = met_data['time'].to_numpy()
        elif (wind_data is not None):
            datatime = wind_data['time'].to_numpy()
        else:
            return None, None
    
    # Build a dataframe that only has data "time" as an entry
    blank_df = pd.DataFrame(data=datatime, columns=['time'])
    
    # Check each of the data types (met, wind, rad) to see if they exist.
    # If they don't, use a copy of the blank_df as a template to 
    # build out complete dataframes for that data type, full of blnak
    # values for the entries
    if met_data is None:
        met_data = check_missing_vars(blank_df.copy(), datatype='met')
        met_qc = process_qartod_tests('ORCA', met_data, 
                                      sensor_type='met', fulldata_flag=False)
        
    if wind_data is None:
        wind_data = check_missing_vars(blank_df.copy(), datatype='wind')
        wind_qc = process_qartod_tests('ORCA', wind_data, 
                                       sensor_type='wind', fulldata_flag=False)
        
    if rad_data is None:
        rad_data = check_missing_vars(blank_df.copy(), datatype='rad')
        rad_qc = process_qartod_tests('ORCA', rad_data, 
                                      sensor_type='rad', fulldata_flag=False)
        
    
    # Using the now developed dataframes, rebuilt if necessary,
    # merge the dataframes together
    merged_met, merged_qc = merge_with_met(met_data.copy(), wind_data.copy(), 
                                           met_qc.copy(), wind_qc.copy(),
                                           new_suffix='wind')
    merged_met, merged_qc = merge_with_met(merged_met.copy(), rad_data.copy(), 
                                           merged_qc.copy(), rad_qc.copy(),
                                           new_suffix='rad')
        
    return merged_met, merged_qc    


# %%


def process_data_wrapper(buoy_designation, startDate=None, endDate=None, log=None):
    
    # Get processing directories
    procdir, basedir, datadir, savedir, depdir = get_procdirs(buoy_designation.lower())
    
    # Load in the most recent file
    met_latest, latest_date = load_latest_file(buoy_designation, savedir, latest_date=startDate)

    # Load in deployment information
    nemo_deps, dep_starts, dep_ends = load_deployment_starts(buoy_designation, depdir, startDate, endDate)
    
    # If start and end dates are provided, restrict/expand the start/end dates
    # to include a full month of data
    # If the latest data file is earlier than
    if startDate is not None:
        if (latest_date is not None) and (startDate > latest_date):
            startDate = latest_date.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
        else:
            startDate = startDate.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    elif latest_date is not None:
        startDate = latest_date.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    else:
        startDate = min(dep_starts)
        print('Using the first valid deployment start date as the start date: ', startDate)
    
    if endDate is None:
        endDate = datetime.datetime.now()+datetime.timedelta(hours=8)
    else:
        end_dateyr = endDate.timetuple().tm_year
        end_datemon = endDate.timetuple().tm_mon
        if end_datemon == 12:
            endDate = (endDate.replace(year=end_dateyr+1, month=1, 
                                         day=1,hour=0,minute=0,second=0,microsecond=0)
                        - datetime.timedelta(days=1))
        else:
            endDate = (endDate.replace(year=end_dateyr, month=end_datemon+1, 
                                         day=1,hour=0,minute=0,second=0,microsecond=0)
                        - datetime.timedelta(days=1))

    if endDate < startDate:
        print('Updating end date to one day past the start date')
        endDate = startDate + datetime.timedelta(days=1)


    print('Start date:', startDate)
    print('End date:', endDate)
    [metdir, winddir, compassdir, 
     shortraddir, longraddir,
     metfiles, windfiles, compassfiles, 
     shortradfiles, longradfiles] = get_filelists(buoy_designation, datadir, 
                                                  startDate, endDate)

    met_data, met_qc = process_met_files(buoy_designation,
                                         [os.path.join(metdir, file) for file in metfiles if file is not None])
    wind_data, wind_qc = process_wind_files(buoy_designation, 
                                            [os.path.join(winddir, file) for file in windfiles if file is not None],
                                            [os.path.join(compassdir, file) for file in compassfiles if file is not None])
    rad_data, rad_qc = process_rad_files(buoy_designation, 
                                         [os.path.join(shortraddir, file) if file is not None else None 
                                          for file in shortradfiles if file is not None], 
                                         [os.path.join(longraddir, file) if file is not None else None 
                                          for file in longradfiles if file is not None])

    if ((met_data is not None) and
        (wind_data is not None) and
        (rad_data is not None)):
        merged_met, merged_qc = merge_with_met(met_data, wind_data, 
                                               met_qc, wind_qc,  new_suffix='wind')
        merged_met, merged_qc = merge_with_met(merged_met, rad_data, 
                                               merged_qc, rad_qc,  new_suffix='rad')
    elif ((met_data is not None) or
          (wind_data is not None) or
          (rad_data is not None)):
        merged_met, merged_qc = add_blank_data(met_data, wind_data, rad_data,
                                               met_qc, wind_qc, rad_qc)
    else:
        print('No data before merging')
        return
    
    if merged_met is None:
        print('No data after merging')
        return

    # Check that all data falls within the deployment start and end times
    dropinds = []
    for depind in range(0,len(dep_starts)):
        if depind == 0:
            if any(merged_met.loc[:,'time'] < dep_starts[depind]):
                dropinds.extend(np.where(merged_met.loc[:,'time']<dep_starts[depind])[0])
            elif ((dep_ends[depind] is not None) and 
                  any(np.logical_and(merged_met.loc[:,'time']<dep_starts[depind],
                                     merged_met.loc[:,'time']>dep_ends[depind-1]))):
                dropinds.extend(np.where(np.logical_and(merged_met.loc[:,'time']<dep_starts[depind],
                                                        merged_met.loc[:,'time']>dep_ends[depind-1]))[0])

    merged_met = merged_met.drop(dropinds).reset_index(drop=True)
    merged_qc = merged_qc.drop(dropinds).reset_index(drop=True)   

    if met_latest is not None:
        latest_met = met_latest.loc[:, merged_met.columns].copy()
        latest_qc = met_latest.loc[:, merged_qc.columns].copy()

        finalmet_df = (pd.concat([latest_met, merged_met])
                       .reset_index(drop=True).sort_values(by='time'))
        final_inds = finalmet_df.index
        finalmet_qcdf = (pd.concat([latest_qc, merged_qc])
                         .reset_index(drop=True).loc[final_inds,:]
                         .reset_index(drop=True))
    else:
        finalmet_df = merged_met.copy().reset_index(drop=True).sort_values(by='time')
        final_inds = finalmet_df.index
        finalmet_qcdf = (merged_qc.copy()
                         .reset_index(drop=True).loc[final_inds,:]
                         .reset_index(drop=True))

    for col in finalmet_df.columns:
        if 'time' in col:
            continue
        naninds = finalmet_df[col].isna()
        if any(naninds):
            finalmet_df.loc[np.where(naninds)[0],col] = -555


    buoy_lat = nemo_deps[0]['latitude']
    buoy_lon = nemo_deps[0]['longitude']
    buoy_name = "Cha'ba"
    met_xr = make_new_met_xarray(buoy_designation, buoy_name, buoy_lat, buoy_lon,
                                 ['GillMetConnect' for ii in range(0,len(finalmet_df))], 
                                 finalmet_df)
    met_qc_xr = met_qc_process(buoy_designation, finalmet_qcdf, finalmet_df['time'])

    metall_xr = xr.merge([met_xr, met_qc_xr], compat='override')
    if latest_date is not None:
        metall_xr = metall_xr.sel(time=slice(startDate.replace(day=1,hour=0,minute=0,
                                                               second=0,microsecond=0),None))
    metall_xr = metall_xr.drop_duplicates(dim='time')

    # Ensure all non-dimension variables include both location_id and time dimensions.
    # Re-attach location_id as a coordinate with attrs preserved because expand_dims
    # with raw .values can recreate it without metadata.
    loc_coord = metall_xr['location_id'].copy(deep=True)
    for var_name in list(metall_xr.data_vars):
        if (var_name == 'time') or (var_name == 'location_id'):
            continue
        if metall_xr[var_name].dims == ('time',):
            metall_xr[var_name] = metall_xr[var_name].expand_dims(
                {'location_id': loc_coord.values}
            ).transpose('location_id', 'time')

    metall_xr = metall_xr.assign_coords(location_id=loc_coord)


    met_yrs = np.unique(metall_xr['time.year'])
    metyr_grouped = metall_xr.groupby('time.year')
    for yr in met_yrs:
        met_year = metyr_grouped[yr]

        met_months = np.unique(met_year['time.month'])
        metmnth_grouped = met_year.groupby('time.month')

        for month in met_months:
            met_month = metmnth_grouped[month]

            # Create a filename for the data to be saved
            if month < 10:
                save_filename = 'NEMO_' + buoy_designation + '_MET_' + str(yr) + '0' + str(month) + '.nc'
            else:
                save_filename = 'NEMO_' + buoy_designation + '_MET_' + str(yr) + str(month) + '.nc'

            print('Saving MET data for:', save_filename)
            write_met_netcdf(savedir, save_filename, met_month)
            
    return


# %%


def main():
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hu:b:s:e:", ["help", "buoy=","start=","end="])
    except Exception as inst:
        # print help information and exit:
        print('Error in getting options: ' + str(inst)) # will print something like "option -a not recognized"
        sys.exit(2)
        
    
    # step through command-line options and their arguments
    buoyFlag = False
    startFlag = False
    endFlag = False
    
    # For all the command-line arguments, update a flag to indicate if the option was given,
    # and extract the value of the command-line argument
    for o, a in opts:
        if o == "-v":
            bVerbose = True
            # but currently verbose run is not implemented!
        elif o in ("-h", "--help"):
            sys.exit()
        elif o in ("-b", "--buoy"):
            buoyFlag = True
            buoy_designation = a
        elif o in ("-s", "--start"):
            startFlag = True
            startDate = a
        elif o in ("-e", "--end"):
            endFlag = True
            endDate = a
        else:
            assert False, "unhandled option"
                 

    ## Open a log file
    if buoyFlag:
        logfile = ('ProcessMetData_' + buoy_designation + '_' +  
                   datetime.datetime.now().strftime('%Y%b%dT%H%M') + '.txt')
        if os.path.exists(logfile):
            os.rename(logfile, logfile[:logfile.find('.txt')] + '_old.txt')
        print('Opening new log file:', logfile)
        log = open(logfile,'w')
        
    # Get the current processing time
    curDate = datetime.datetime.now()
    
    
    
    ##    
        
    # If the buoy designation was given, ensure that the buoy designation is in uppercase,
    # and that it matches one of the valid buoy designations
    valid_buoys = ['chaba', 'nemoss']
    if buoyFlag:
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
    
    # Get a start date, if applicable
    if startFlag:
        try:
            startDate = dateutil.parser.parse(startDate)
        except:
            print("Unable to parse start date: " + startDate)
            print("Please try again, and enter a date format of YYYY-MM-DD")
            print("Example: 2025-06-13")
            sys.exit(2)
            
        curDate = datetime.datetime.now()
        if startDate > curDate:
            print("Start date cannot be later than current date:")
            print("Start date  : " + startDate.strftime('%Y-%M-%D'))
            print("Current date: " + curDate.strftime('%Y-%M-%D'))
            print("Please try again with a valid date")
            sys.exit(2)
    else:
        startDate = None
        
    # Get an end date, if applicable
    if endFlag:
        try:
            endDate = dateutil.parser.parse(endDate)
        except:
            print("Unable to parse start date: " + endDate)
            print("Please try again, and enter a date format of YYYY-MM-DD")
            print("Example: 2025-06-13")
            sys.exit(2)
            
        if endDate > curDate:
            print("End date cannot be later than current date:")
            print("End date    : " + endDate.strftime('%Y-%M-%D'))
            print("Current date: " + curDate.strftime('%Y-%M-%D'))
            print("The current date will be used as the end date")
            endDate = curDate.replace(hour=0,minute=0,second=0,microsecond=0)
            
        if (startDate is not None) and (endDate < startDate):
            print("End date cannot be earlier than start date:")
            print("End date  : " + endDate.strftime('%Y-%M-%D'))
            print("Start date: " + startDate.strftime('%Y-%M-%D'))
            print("Please try again with valid start/end dates.")
            sys.exit()
    else:
        endDate = None
        
    
    ##################################################################
    # Begin processing new data for a given buoy
    print('Begin processing met data: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'),file=log)
    try:
        process_data_wrapper(buoy_designation, startDate=startDate, endDate=endDate, log=log)
        print('New data processing completed: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'),file=log)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('Error occurred in new data processing: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'),file=log)
        print(e,file=log)
        print('Error occurs on line ' + str(exc_tb.tb_lineno), file=log)
        
            
        sendEmail = False
        if sendEmail:
            failmsg = ('An error occurred while processing new data for ' + buoy_designation + '.' + 
                       '\n\nProcessing failed at ' + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') + 
                       '\n\nPlease check processing logs to diagnose error.')
            send_emailreport(msgtxt = failmsg,
                                  subj = buoy_designation + ' - New Data Processing Failure')
    
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


# %%
