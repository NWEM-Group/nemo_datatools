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

import matplotlib
from matplotlib import pyplot as plt

import struct
import codecs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ADCP_Functions as adcp_funcs
import nemo_general_functions as nemo_funcs
import nemo_deployments

# # ADCP Functions

def _parse_deployment_time(time_str):
    
    if time_str is None:
        return None
    
    parsed_time = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    if parsed_time.tzinfo is not None:
        parsed_time = parsed_time.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    
    return parsed_time



def _load_adcp_deployment_window(buoy_name, deployment_name):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{buoy_name}_deployments.json')
    
    with open(json_path, 'r') as f:
        deployment_info = json.load(f)
    
    deployment_data = deployment_info.get('deployments', {}).get(deployment_name)
    if deployment_data is None:
        raise KeyError(f"Deployment '{deployment_name}' not found for {buoy_name}.")
    
    lat = deployment_data.get('latitude')
    lon = deployment_data.get('longitude')
    depth = deployment_data.get('depth')
    start_deployment = _parse_deployment_time(deployment_data.get('start'))
    end_deployment = _parse_deployment_time(deployment_data.get('end'))
    status = deployment_data.get('status', 'unknown')
    
    return lat, lon, depth, start_deployment, end_deployment, status
  

def recalc_xducer_depth(depth_xducer, instr_params, inst_lat):
    
    import numpy as np
    
    # Invert the calculation to get pressure from the
    # instrument transducer depth
    press = [ii / (1.02 - 0.00069*instr_params['estimated_salinity'])
           for ii in depth_xducer]
    
    # Calculate "depth" from pressure at a given latitude
    c1 = 9.72659
    c2 = -2.2512E-5
    c3 = 2.279E-10
    c4 = -1.82E-15
    gam_dash = 2.184e-6

    X = np.sin(np.radians(inst_lat))**2
    bot_line = [9.780318 * (1.0 + (5.2788e-3 + 2.36e-5*X)*X) + gam_dash*0.5*prr
                for prr in press]
    top_line = [(((c4*prr + c3)*prr + c2)*prr + c1)*prr 
                for prr in press]
    newdepth = [top_line[ii] / bot_line[ii] for ii in range(0,len(press))]
    
    return newdepth

def get_magnetic_declination(buoy_lat, buoy_lon, datadate):

    from urllib import request
    import json
    
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


    try:
        response = request.urlopen(url_find)
        data_json = json.loads(response.read())
        mag_dec = data_json['result'][0]['declination']
    except Exception as e:
        print('Error occurred when getting the magnetic declination.')
        print(e)
        print('Using default mag_dec of 17 degrees')
        mag_dec = 17.0
    
    return mag_dec


# In[ ]:


def datetime_toordinal_withseconds(origdate):
    
    # Function: datetime_toordinal_withseconds
    # This function changes a datetime object
    # into an ordinal time (i.e., days since Jan-01-0000),
    # but includes resolution, to the second.
    
    import numpy as np
    import pandas as pd
    import datetime
    
    if pd.isna(origdate):
        return np.nan
    
    year = origdate.year
    month = origdate.month
    day = origdate.day
    hour = origdate.hour
    minute = origdate.minute
    second = origdate.second
    microsec = origdate.microsecond
    
    return (datetime.datetime.toordinal(datetime.datetime(year,month,day)) +
            (hour + (minute + (second + (microsec/1e6))/60)/60)/24)


# In[ ]:


def cart2pol(x, y):
    
    import numpy as np
    
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    
    import numpy as np
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def radians_2_degrees_truenorth(angles):
    
    import numpy as np
    
    [x,y] = pol2cart(1, angles)
    
    NewAng = np.arctan2(y, -x) + (3*np.pi/2)
    
    angleInDegrees = np.degrees(NewAng)
    angleInDegrees = np.array([ii%360 for ii in angleInDegrees])
    
    return angleInDegrees

def apply_magnetic_declination(datadates, buoy_lat, buoy_lon, 
                               vel_beam1, vel_beam2):
    
    import numpy as np
    import datetime
    
    ############################################################################
    # Get a magnetic declination for each unique day of year within the dataset,
    # and then rotate the horizontal data by the magnetic declination
    
    data_days = np.unique([ii.replace(hour=0,minute=0,second=0,microsecond=0) 
                           for ii in datadates])
    if len(data_days) == 0:
        print('No days available. No magnetic declination found.')
        return None
    elif len(data_days) == 1:
        data_days = np.append(data_days, data_days[0]+datetime.timedelta(days=1))
    
    mag_decs = []
    for datadate in data_days:
        mag_decs.append(get_magnetic_declination(buoy_lat, buoy_lon, datadate))
        
    # Convert the dates into ordinal times, with fractional units
    data_orddates = np.array([datetime_toordinal_withseconds(ii) for ii in datadates])
    data_orddays = np.array([datetime_toordinal_withseconds(ii) for ii in data_days])
        
    # Interpolate the magnetic declination onto the actual date stamps
    mag_decs = np.interp(data_orddates, data_orddays, mag_decs)
    avg_mag_dec = np.mean(mag_decs)
    
    
    ##############################
    # Convert the horizontal velocity into direction, and then rotate
    
    # Calculate the direction of the velocity
    # Note that 0 is from straight north, and increases clockwise
    vel_ang = np.degrees(np.arctan2(vel_beam1, vel_beam2))
    
    
    
    # Calculate the magnitude of the velocity
    vel_mag = np.sqrt(vel_beam1**2 + vel_beam2**2)

    # Step through each ensemble, and rotate by the magnetic declination
    vel_ang_rot = vel_ang.copy() 
    for ii in range(0,len(mag_decs)):
        vel_ang_rot[ii,:] = vel_ang[ii,:] + mag_decs[ii]

    # Convert the velocity back into rotated u-v components
    vel_beam1_rot = np.sin(np.radians(vel_ang_rot)) * vel_mag
    vel_beam2_rot = np.cos(np.radians(vel_ang_rot)) * vel_mag
    
    return vel_beam1_rot, vel_beam2_rot, avg_mag_dec

def adcp_regrid_ensemble(zgrid, zs_adcp, depth_xducer, 
                         vars_to_regrid, orientation='upfacing'):
    
    import numpy as np
    
    
    # Extract out the names of all the variables to regrid
    
    # Get the shape of the new data to regrid 
    # (NT from the variable, and NZ from the new zgrid)
    var_names = [ii for ii in vars_to_regrid.keys()]
    [NT,NZ] = vars_to_regrid[var_names[0]].shape
    NZ = len(zgrid)
    
    # Define an interpolation function, where only
    # non-NaN values are included in the interpolation
    def interp_prof(x, xp, fp, printflag=False):
        if not(isinstance(xp, np.ndarray)):
            xp = np.array(xp)
        if not(isinstance(fp, np.ndarray)):
            fp = np.array(fp)
        goodinds = np.logical_not(np.isnan(fp))
        f = np.nan*np.zeros(len(x))
        if sum(goodinds) >= 2:
            f = np.interp(x, xp[goodinds], fp[goodinds], left=np.nan, right=np.nan)
        return f
    
    
    # Define a new dictionary to store the regridded variables
    regridded_vars = {}
    
    # Step through each of the variables to be regridded
    for var in var_names:
        
        # Initialize a new empty array to store the regridded values
        var_regrid = np.nan*np.zeros((NT,NZ))
        
        # Step through each time step
        for tt in range(0,NT):
            
            # Create an accurate z level, based upon the
            # distance from the ADCP head and the 
            # adjusted transducer depth
            z_adcp = zs_adcp[tt]
            if orientation == 'upfacing':
                zadjust_adcp = depth_xducer[tt] - z_adcp
                
                # Regrid the variable
                var_regrid[tt,:] = interp_prof(zgrid, zadjust_adcp[::-1], 
                                               vars_to_regrid[var][tt,::-1])
            else:
                zadjust_adcp = depth_xducer[tt] + z_adcp
                
                # Regrid the variable
                var_regrid[tt,:] = interp_prof(zgrid, zadjust_adcp, 
                                               vars_to_regrid[var][tt,:])

            
            
        # If the data is correlation or echo data, 
        # ensure that it remains of the "int" type
        if ('corr' in var) or ('echo' in var):
            var_regrid = var_regrid.astype(int)
            
        # Store the new regridded data
        regridded_vars[var] = var_regrid
    
    return regridded_vars



def extract_ensemble_data(adcp_records):
    
    n_records = len(adcp_records)
    adcp_record = adcp_records[0]

    z_adcp = np.arange(adcp_record.fixed_data.bin_1_distance/100,
                       (adcp_record.fixed_data.bin_1_distance/100 + 
                        adcp_record.fixed_data.number_of_cells*adcp_record.fixed_data.depth_cell_length/100),
                        adcp_record.fixed_data.depth_cell_length/100)

    if adcp_record.sysconfig.beam_facing == 1:
        beam_dir = 'upfacing'
    else:
        beam_dir = 'downfacing'
    if adcp_record.sysconfig.beam_pattern == 1:
        beam_pat = 'convex'
    else:
        beam_pat = 'concave'

    if adcp_record.coord_transform.coord_transform == 0:
        coord_xform = 'beam'
    elif adcp_record.coord_transform.coord_transform == 1:
        coord_xform = 'instrument'
    elif adcp_record.coord_transform.coord_transform == 2:
        coord_xform = 'ship'
    elif adcp_record.coord_transform.coord_transform == 3:
        coord_xform = 'earth'

    if adcp_record.fixed_data.system_bandwidth == 0:
        bandwidth = 'broad'
    else:
        bandwidth = 'narrow'
        
    threebeam_flag = False
    if adcp_record.coord_transform.three_beam_used == 1:
        threebeam_flag = True
        

    sys_config = {'frequency': str(adcp_record.sysconfig.frequency) + 'kHz',
                  'beam_direction': beam_dir,
                  'beam_pattern': beam_pat,
                  'beam_angle': adcp_record.fixed_data.beam_angle,
                  'beam_transform': coord_xform,
                  'three_beams': threebeam_flag,
                  'serial_number': adcp_record.fixed_data.serial_number}

    instr_params = {'transmit_pulse_length': adcp_record.fixed_data.transmit_pulse_length/100,
                    'low_correlation_threshold': adcp_record.fixed_data.low_corr_threshold,
                    'velocity_error_threshold': adcp_record.fixed_data.error_velocity_max,
                    'min_percent_good': adcp_record.fixed_data.minimum_percentage,
                    'heading_bias': adcp_record.fixed_data.heading_bias,
                    'estimated_salinity': adcp_record.variable_data.salinity,
                    'bandwidth': bandwidth}

    ens_no = []

    dtnum = []
    heading = []
    pitch = []
    roll = []
    hdg_std = []
    pitch_std = []
    roll_std = []
    depth_xducer = []
    soundvel = []
    xducer_temp = []
    
    zadcps = []

    corr_beam1 = []
    corr_beam2 = []
    corr_beam3 = []
    corr_beam4 = []

    echo_beam1 = []
    echo_beam2 = []
    echo_beam3 = []
    echo_beam4 = []

    vel_beam1 = []
    vel_beam2 = []
    vel_beam3 = []
    vel_beam4 = []


    for ii in range(0,n_records):

        ens_no.append(adcp_records[ii].variable_data.ensemble_number)
        
        zadcp = np.arange(adcp_records[ii].fixed_data.bin_1_distance/100,
                          (adcp_records[ii].fixed_data.bin_1_distance/100 + 
                           adcp_records[ii].fixed_data.number_of_cells*adcp_records[ii].fixed_data.depth_cell_length/100),
                           adcp_records[ii].fixed_data.depth_cell_length/100)
        zadcps.append(zadcp)

        dtnum.append(datetime.datetime(2000+adcp_records[ii].variable_data.rtc_year,
                                       adcp_records[ii].variable_data.rtc_month,
                                       adcp_records[ii].variable_data.rtc_day,
                                       adcp_records[ii].variable_data.rtc_hour,
                                       adcp_records[ii].variable_data.rtc_minute,
                                       adcp_records[ii].variable_data.rtc_second,
                                       adcp_records[ii].variable_data.rtc_hundredths*1000))

        heading.append(adcp_records[ii].variable_data.heading/100)
        pitch.append(adcp_records[ii].variable_data.pitch/100)
        roll.append(adcp_records[ii].variable_data.roll/100)
        hdg_std.append(adcp_records[ii].variable_data.heading_standard_deviation)
        pitch_std.append(adcp_records[ii].variable_data.pitch_standard_deviation/10)
        roll_std.append(adcp_records[ii].variable_data.roll_standard_deviation/10)
        depth_xducer.append(adcp_records[ii].variable_data.depth_of_transducer/10)
        soundvel.append(adcp_records[ii].variable_data.speed_of_sound)
        xducer_temp.append(adcp_records[ii].variable_data.temperature/100)

        corr_beam1.append(adcp_records[ii].correlation_magnitudes.beam1)
        corr_beam2.append(adcp_records[ii].correlation_magnitudes.beam2)
        corr_beam3.append(adcp_records[ii].correlation_magnitudes.beam3)
        

        echo_beam1.append(adcp_records[ii].echo_intensity.beam1)
        echo_beam2.append(adcp_records[ii].echo_intensity.beam2)
        echo_beam3.append(adcp_records[ii].echo_intensity.beam3)

        vel_beam1.append([np.nan if abs(ii) == 32768 else ii/1000 
                          for ii in adcp_records[ii].velocities.beam1])
        vel_beam2.append([np.nan if abs(ii) == 32768 else ii/1000 
                          for ii in adcp_records[ii].velocities.beam2])
        vel_beam3.append([np.nan if abs(ii) == 32768 else ii/1000 
                          for ii in adcp_records[ii].velocities.beam3])
        
        
        if threebeam_flag:
            corr_beam4.append(adcp_records[ii].correlation_magnitudes.beam4)
            echo_beam4.append(adcp_records[ii].echo_intensity.beam4)
            vel_beam4.append([np.nan if abs(ii) == 32768 else ii/1000 
                              for ii in adcp_records[ii].velocities.beam4])
        else:
            corr_beam4.append([np.nan for ii in adcp_records[ii].correlation_magnitudes.beam1])
            echo_beam4.append([np.nan for ii in adcp_records[ii].echo_intensity.beam1])
            vel_beam4.append([np.nan for ii in adcp_records[ii].velocities.beam1])
            



    corr_beam1_array = np.array([np.array(ii) for ii in corr_beam1])
    corr_beam2_array = np.array([np.array(ii) for ii in corr_beam2])
    corr_beam3_array = np.array([np.array(ii) for ii in corr_beam3])
    corr_beam4_array = np.array([np.array(ii) for ii in corr_beam4])

    echo_beam1_array = np.array([np.array(ii) for ii in echo_beam1])
    echo_beam2_array = np.array([np.array(ii) for ii in echo_beam2])
    echo_beam3_array = np.array([np.array(ii) for ii in echo_beam3])
    echo_beam4_array = np.array([np.array(ii) for ii in echo_beam4])

    vel_beam1_array = np.array([np.array(ii) for ii in vel_beam1])
    vel_beam2_array = np.array([np.array(ii) for ii in vel_beam2])
    vel_beam3_array = np.array([np.array(ii) for ii in vel_beam3])
    vel_beam4_array = np.array([np.array(ii) for ii in vel_beam4])
    
    
    return (sys_config, instr_params, zadcps, 
            ens_no, dtnum, heading, 
            pitch, roll, hdg_std, pitch_std, roll_std,
            depth_xducer, soundvel, xducer_temp, 
            corr_beam1_array, corr_beam2_array, corr_beam3_array, corr_beam4_array,
            echo_beam1_array, echo_beam2_array, echo_beam3_array, echo_beam4_array,
            vel_beam1_array, vel_beam2_array, vel_beam3_array, vel_beam4_array
    )



def make_adcp_xr(adcp_filename, buoy_name, deployment_name, lat, lon, depth, 
                 timestamp, sys_config, nemo_info, instr_params, z_adcp, 
                 ens_no, dtnum, heading, 
                 pitch, roll, hdg_std, pitch_std, roll_std,
                 depth_xducer, soundvel, xducer_temp, 
                 corr_beam1_array, corr_beam2_array, corr_beam3_array, corr_beam4_array,
                 echo_beam1_array, echo_beam2_array, echo_beam3_array, echo_beam4_array,
                 vel_beam1_array, vel_beam2_array, vel_beam3_array, vel_beam4_array):
    
    if sys_config['beam_transform'] == 'earth':
        beam_names = ['eastward_sea_water_velocity', 'northward_sea_water_velocity',
                      'upward_sea_water_velocity', 'upward_sea_water_velocity_error']
        beam_desc = ['velocity of sea water in the eastward direction (westward is negative)', 
                     'velocity of sea water in the northward direction (southward is negative)',
                     'velocity of sea water in the upward direction; i.e., towards the surface (downward is negative)',
                     'error velocity of sea water in the upward direction; i.e., towards the surface (downward is negative)']
    else:
        beam_names = ['velocity_beam1', 'velocity_beam2',
                      'velocity_beam3', 'velocity_beam4']
        beam_desc = ['velocity as measured by beam 1', 
                     'velocity as measured by beam 2',
                     'velocity as measured by beam 3',
                     'velocity as measured by beam 4']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{buoy_name}_deployments.json')
    with open(json_path, 'r') as f:
        deployment_info = json.load(f)
    institution_info = deployment_info.get('institution_info', {})

    xr_new = xr.Dataset()
    
    fillVal = -(2**15)
    fillVal_int = -555
    
    xr_new.attrs["title"] = deployment_info['buoy_title'] + ' - ADCP Data'
    xr_new.attrs["description"] = 'Acoustic Doppler Current Profiler (ADCP) data on the ' + buoy_name + ' platform'
    xr_new.attrs["history"] = "File created on " + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
    xr_new.attrs['conventions'] = 'CF-1.6, ACDD-1.3, IOOS-1.2'
    xr_new.attrs['designation'] = buoy_name + '_ADCP'
    xr_new.attrs['latitude'] = str(np.round(lat,6)) + 'N'
    xr_new.attrs['longitude'] = str(np.round(lon,6)) + 'E'
    xr_new.attrs['deployment_depth'] = str(depth) + 'm'
    xr_new.attrs['infoUrl'] = deployment_info['info_url']
    xr_new.attrs['institution'] = institution_info.get('institution', '')
    
    ## These are fields required by NCEI by ingestion to the sensor map
    xr_new.attrs['creator_name'] = institution_info.get('name', '')
    xr_new.attrs['creator_institution'] = institution_info.get('institution', '')
    xr_new.attrs['creator_url'] = institution_info.get('url', '')
    xr_new.attrs['creator_country'] = institution_info.get('country', '')
    xr_new.attrs['creator_sector'] = institution_info.get('sector', '')
    xr_new.attrs['creator_type'] = institution_info.get('type', '')
    
    xr_new.attrs['contributor_name'] = institution_info.get('name', '') + ', NANOOS, NWEM'
    xr_new.attrs['contributor_role_vocabulary'] = 'https://vocab.nerc.ac.uk/collection/G04/current/'
    xr_new.attrs['contributor_role'] =  'owner, funder, publisher'
    xr_new.attrs['contributor_role_url'] = institution_info.get('url', '') + ', https://www.nanoos.org/, https://nwem.apl.washington.edu/'
    
    xr_new.attrs['publisher_name'] = 'NorthWest Environmental Moorings (NWEM) Group'
    xr_new.attrs['publisher_email'] = 'setht1@uw.edu'
    xr_new.attrs['publisher_institution'] = 'University of Washington - Applied Physics Laboratory'
    xr_new.attrs['publisher_type'] = 'group'
    xr_new.attrs['publisher_url'] = 'https://nwem.apl.washington.edu/'
    xr_new.attrs['publisher_country'] = 'United Stated'
    
    
    xr_new.attrs['author'] = 'Seth Travis'
    xr_new.attrs['contact'] = 'setht1@uw.edu'
    xr_new.attrs['_FillValue'] = fillVal
    xr_new.attrs['cdm_data_type'] = 'timeSeries'
    xr_new.attrs['cdm_timeseries_variables'] = 'buoy_name,deployment_name,latitude,longitude'
    
    xr_new.attrs['serial_number'] = sys_config['serial_number']
    xr_new.attrs['frequency'] = sys_config['frequency']
    xr_new.attrs['beam_direction'] = sys_config['beam_direction']
    xr_new.attrs['beam_pattern'] = sys_config['beam_pattern']
    xr_new.attrs['beam_angle'] = sys_config['beam_angle']
    xr_new.attrs['coordinate_transformation'] = sys_config['beam_transform']
    
    xr_new.attrs['transmit_pulse_length'] = instr_params['transmit_pulse_length']
    xr_new.attrs['low_correlation_threshold'] = instr_params['low_correlation_threshold']
    xr_new.attrs['velocity_error_threshold'] = instr_params['velocity_error_threshold']
    xr_new.attrs['min_percent_good'] = instr_params['min_percent_good']
    xr_new.attrs['heading_bias'] = instr_params['heading_bias']
    xr_new.attrs['bandwidth'] = instr_params['bandwidth']
    xr_new.attrs['estimated_salinity'] = instr_params['estimated_salinity']
    xr_new.attrs['mag_dec_applied'] = nemo_info.get('avg_mag_dec', '')

    xr_new.attrs['mooring_diagram_baseurl'] = deployment_info.get('mooring_diagram_base', '')
    xr_new.attrs['mooring_diagram_url'] = deployment_info.get('mooring_diagram_base', '') + deployment_name + '_mooring_diagram_final.pdf'
    xr_new.attrs['time_drift_seconds'] = deployment_info.get('time_drift_seconds', '')
    xr_new.attrs['time_drift_description'] = 'Number of seconds that the instrument clock drifted from UTC time during the deployment. This is calculated by comparing the instrument clock to the GPS time at the start and end of the deployment. Positive values indicate that the instrument clock was ahead of UTC time, while negative values indicate that the instrument clock was behind UTC time.'
    
    
    
    
    xr_new['time'] = xr.DataArray(dtnum, dims='time')
    xr_new['time'] = xr_new['time'].astype(np.datetime64)
    xr_new['time'].attrs["long_name"] = 'time'
    xr_new['time'].attrs["description"] = 'time of sampling' 
    xr_new['time'].attrs["units"] = 'seconds since 1970-01-01 00:00:00' 
    xr_new['time'].attrs["calendar"] = 'gregorian' 
    xr_new['time'].encoding["dtype"] = "float32"
    xr_new['time'].encoding["units"] = 'seconds since 1970-01-01 00:00:00' 
    xr_new['time'].encoding["calendar"] = 'gregorian'  
    
    xr_new['controller_timestamp'] = xr.DataArray(timestamp, dims='time')
    xr_new['controller_timestamp'] = xr_new['controller_timestamp'].astype(np.datetime64)
    xr_new['controller_timestamp'].attrs["long_name"] = 'controller_time'
    xr_new['controller_timestamp'].attrs["description"] = 'time of the controller when the ensemble was collected' 
    xr_new['controller_timestamp'].attrs["units"] = 'seconds since 1970-01-01 00:00:00' 
    xr_new['controller_timestamp'].attrs["calendar"] = 'gregorian' 
    xr_new['controller_timestamp'].encoding["dtype"] = "float32"
    xr_new['controller_timestamp'].encoding["units"] = 'seconds since 1970-01-01 00:00:00' 
    xr_new['controller_timestamp'].encoding["calendar"] = 'gregorian'  
    
    xr_new['depth'] = xr.DataArray(z_adcp, dims='depth')
    xr_new['depth'].attrs["long_name"] = 'depth'
    xr_new['depth'].attrs["description"] = 'depth below water surface'
    xr_new['depth'].attrs["units"] = 'm'
    xr_new['depth'].attrs["positive_direction"] = 'down'

    xr_new['buoy_name'] = xr.DataArray(
        np.array([buoy_name for ii in range(0, len(dtnum))], dtype=object),
        dims='time'
    )
    xr_new['buoy_name'].attrs["long_name"] = 'buoy_name'
    xr_new['buoy_name'].attrs["description"] = 'lander description name'

    xr_new['deployment_name'] = xr.DataArray(
        np.array([deployment_name for ii in range(0, len(dtnum))], dtype=object),
        dims='time'
    )
    xr_new['deployment_name'].attrs["long_name"] = 'deployment_name'
    xr_new['deployment_name'].attrs["description"] = 'deployment identifier name'
    
    
    xr_new['ensemble_number'] = xr.DataArray(ens_no, dims='time')
    xr_new['ensemble_number'].attrs["long_name"] = 'ensemble_number'
    xr_new['ensemble_number'].attrs["description"] = 'ensemble number'
    xr_new['ensemble_number'].attrs["cf_role"] = 'timeseries_id'
    
    
    
    ################################
    # Assign the 1-D variables
    
    xr_new['heading'] = xr.DataArray(heading, dims='time')
    xr_new['heading'].attrs["long_name"] = 'heading'
    xr_new['heading'].attrs["description"] = 'orientation of the instrument'
    xr_new['heading'].attrs["units"] = 'degrees'
    
    xr_new['pitch'] = xr.DataArray(pitch, dims='time')
    xr_new['pitch'].fillna(fillVal)
    xr_new['pitch'].attrs["long_name"] = 'pitch'
    xr_new['pitch'].attrs["description"] = '--'
    xr_new['pitch'].attrs["units"] = 'degrees'
    xr_new['pitch'].attrs["missing_value"] = fillVal
    
    xr_new['roll'] = xr.DataArray(roll, dims='time')
    xr_new['roll'].fillna(fillVal)
    xr_new['roll'].attrs["long_name"] = 'roll'
    xr_new['roll'].attrs["description"] = '--'
    xr_new['roll'].attrs["units"] = 'degrees'
    xr_new['roll'].attrs["missing_value"] = fillVal
    
    xr_new['heading_std'] = xr.DataArray(hdg_std, dims='time')
    xr_new['heading_std'].fillna(fillVal)
    xr_new['heading_std'].attrs["long_name"] = 'heading_standard_deviation'
    xr_new['heading_std'].attrs["description"] = '--'
    xr_new['heading_std'].attrs["units"] = 'degrees'
    xr_new['heading_std'].attrs["missing_value"] = fillVal
    
    xr_new['pitch_std'] = xr.DataArray(pitch_std, dims='time')
    xr_new['pitch_std'].fillna(fillVal)
    xr_new['pitch_std'].attrs["long_name"] = 'pitch_standard_deviation'
    xr_new['pitch_std'].attrs["description"] = '--'
    xr_new['pitch_std'].attrs["units"] = 'degrees'
    xr_new['pitch_std'].attrs["missing_value"] = fillVal
    
    xr_new['roll_std'] = xr.DataArray(roll_std, dims='time')
    xr_new['roll_std'].fillna(fillVal)
    xr_new['roll_std'].attrs["long_name"] = 'roll_standard_deviation'
    xr_new['roll_std'].attrs["description"] = '--'
    xr_new['roll_std'].attrs["units"] = 'degrees'
    xr_new['roll_std'].attrs["missing_value"] = fillVal
    
    xr_new['transducer_depth'] = xr.DataArray(depth_xducer, dims='time')
    xr_new['transducer_depth'].fillna(fillVal)
    xr_new['transducer_depth'].attrs["long_name"] = 'transducer_depth'
    xr_new['transducer_depth'].attrs["description"] = '--'
    xr_new['transducer_depth'].attrs["units"] = 'm'
    xr_new['transducer_depth'].attrs["missing_value"] = fillVal
    
    xr_new['sound_velocity'] = xr.DataArray(soundvel, dims='time')
    xr_new['sound_velocity'].fillna(fillVal)
    xr_new['sound_velocity'].attrs["long_name"] = 'sound_velocity'
    xr_new['sound_velocity'].attrs["description"] = 'velocity of sound in water'
    xr_new['sound_velocity'].attrs["units"] = 'm / s'
    xr_new['sound_velocity'].attrs["missing_value"] = fillVal
    
    xr_new['sea_water_tempertaure'] = xr.DataArray(xducer_temp, dims='time')
    xr_new['sea_water_tempertaure'].fillna(fillVal)
    xr_new['sea_water_tempertaure'].attrs["long_name"] = 'sea_water_tempertaure'
    xr_new['sea_water_tempertaure'].attrs["description"] = 'temperature of sea water at the transducer'
    xr_new['sea_water_tempertaure'].attrs["units"] = 'degrees C'
    xr_new['sea_water_tempertaure'].attrs["missing_value"] = fillVal
    
    
    ################################
    # Assign the 2-D variables
    
    # Correlations
    
    xr_new['correlation_beam1'] = xr.DataArray(corr_beam1_array, dims=['time','depth'])
    xr_new['correlation_beam1'].fillna(fillVal)
    xr_new['correlation_beam1'].attrs["long_name"] = 'correlation_beam1'
    xr_new['correlation_beam1'].attrs["description"] = 'correlation magnitude for beam 1; 255 = perfect correlation'
    xr_new['correlation_beam1'].attrs["units"] = '--'
    xr_new['correlation_beam1'].attrs["range"] = '0-255'
    xr_new['correlation_beam1'].attrs["missing_value"] = fillVal
    
    xr_new['correlation_beam2'] = xr.DataArray(corr_beam2_array, dims=['time','depth'])
    xr_new['correlation_beam2'].fillna(fillVal)
    xr_new['correlation_beam2'].attrs["long_name"] = 'correlation_beam2'
    xr_new['correlation_beam2'].attrs["description"] = 'correlation magnitude for beam 2; 255 = perfect correlation'
    xr_new['correlation_beam2'].attrs["units"] = '--'
    xr_new['correlation_beam1'].attrs["range"] = '0-255'
    xr_new['correlation_beam2'].attrs["missing_value"] = fillVal
    
    xr_new['correlation_beam3'] = xr.DataArray(corr_beam3_array, dims=['time','depth'])
    xr_new['correlation_beam3'].fillna(fillVal)
    xr_new['correlation_beam3'].attrs["long_name"] = 'correlation_beam3'
    xr_new['correlation_beam3'].attrs["description"] = 'correlation magnitude for beam 3; 255 = perfect correlation'
    xr_new['correlation_beam3'].attrs["units"] = '--'
    xr_new['correlation_beam1'].attrs["range"] = '0-255'
    xr_new['correlation_beam3'].attrs["missing_value"] = fillVal
    
    xr_new['correlation_beam4'] = xr.DataArray(corr_beam4_array, dims=['time','depth'])
    xr_new['correlation_beam4'].fillna(fillVal)
    xr_new['correlation_beam4'].attrs["long_name"] = 'correlation_beam4'
    xr_new['correlation_beam4'].attrs["description"] = 'correlation magnitude for beam 4; 255 = perfect correlation'
    xr_new['correlation_beam4'].attrs["units"] = '--'
    xr_new['correlation_beam1'].attrs["range"] = '0-255'
    xr_new['correlation_beam4'].attrs["missing_value"] = fillVal
    
    
    # Echo intensity
    
    xr_new['echo_intensity_beam1'] = xr.DataArray(echo_beam1_array, dims=['time','depth'])
    xr_new['echo_intensity_beam1'].fillna(fillVal)
    xr_new['echo_intensity_beam1'].attrs["long_name"] = 'echo_intensity_beam1'
    xr_new['echo_intensity_beam1'].attrs["description"] = 'echo intensity for beam 1, at about 0.45 db per count'
    xr_new['echo_intensity_beam1'].attrs["units"] = '--'
    xr_new['echo_intensity_beam1'].attrs["missing_value"] = fillVal
    
    xr_new['echo_intensity_beam2'] = xr.DataArray(echo_beam2_array, dims=['time','depth'])
    xr_new['echo_intensity_beam2'].fillna(fillVal)
    xr_new['echo_intensity_beam2'].attrs["long_name"] = 'echo_intensity_beam2'
    xr_new['echo_intensity_beam2'].attrs["description"] = 'echo intensity for beam 2, at about 0.45 db per count'
    xr_new['echo_intensity_beam2'].attrs["units"] = '--'
    xr_new['echo_intensity_beam2'].attrs["missing_value"] = fillVal
    
    
    xr_new['echo_intensity_beam3'] = xr.DataArray(echo_beam3_array, dims=['time','depth'])
    xr_new['echo_intensity_beam3'].fillna(fillVal)
    xr_new['echo_intensity_beam3'].attrs["long_name"] = 'echo_intensity_beam3'
    xr_new['echo_intensity_beam3'].attrs["description"] = 'echo intensity for beam 3, at about 0.45 db per count'
    xr_new['echo_intensity_beam3'].attrs["units"] = '--'
    xr_new['echo_intensity_beam3'].attrs["missing_value"] = fillVal 
    
    
    xr_new['echo_intensity_beam4'] = xr.DataArray(echo_beam4_array, dims=['time','depth'])
    xr_new['echo_intensity_beam4'].fillna(fillVal)
    xr_new['echo_intensity_beam4'].attrs["long_name"] = 'echo_intensity_beam4'
    xr_new['echo_intensity_beam4'].attrs["description"] = 'echo intensity for beam 4, at about 0.45 db per count'
    xr_new['echo_intensity_beam4'].attrs["units"] = 'db-'
    xr_new['echo_intensity_beam4'].attrs["missing_value"] = fillVal
    
    
    # Beam velocities
    
    xr_new[beam_names[0]] = xr.DataArray(np.round(vel_beam1_array,6), dims=['time','depth'])
    xr_new[beam_names[0]].fillna(fillVal)
    xr_new[beam_names[0]].attrs["long_name"] = beam_names[0]
    xr_new[beam_names[0]].attrs["description"] = beam_desc[0]
    xr_new[beam_names[0]].attrs["units"] = 'm/s'
    xr_new[beam_names[0]].attrs["missing_value"] = fillVal 
    
    xr_new[beam_names[1]] = xr.DataArray(np.round(vel_beam2_array,6), dims=['time','depth'])
    xr_new[beam_names[1]].fillna(fillVal)
    xr_new[beam_names[1]].attrs["long_name"] = beam_names[1]
    xr_new[beam_names[1]].attrs["description"] = beam_desc[1]
    xr_new[beam_names[1]].attrs["units"] = 'm/s'
    xr_new[beam_names[1]].attrs["missing_value"] = fillVal
    
    xr_new[beam_names[2]] = xr.DataArray(np.round(vel_beam3_array,6), dims=['time','depth'])
    xr_new[beam_names[2]].fillna(fillVal)
    xr_new[beam_names[2]].attrs["long_name"] = beam_names[2]
    xr_new[beam_names[2]].attrs["description"] = beam_desc[2]
    xr_new[beam_names[2]].attrs["units"] = 'm/s'
    xr_new[beam_names[2]].attrs["missing_value"] = fillVal
    
    xr_new[beam_names[3]] = xr.DataArray(np.round(vel_beam4_array,6), dims=['time','depth'])
    xr_new[beam_names[3]].fillna(fillVal)
    xr_new[beam_names[3]].attrs["long_name"] = beam_names[3]
    xr_new[beam_names[3]].attrs["description"] = beam_desc[3]
    xr_new[beam_names[3]].attrs["units"] = 'm/s'
    xr_new[beam_names[3]].attrs["missing_value"] = fillVal

    return xr_new


# In[ ]:


def _find_adcp_files(datadir, buoy_name, deployment_name, serial_num=None, status=None):
    """Discover ADCP data files in datadir matching deployment and optional serial."""
    if not os.path.isdir(datadir):
        print(datadir + ' is not a valid directory. Cannot search for ADCP files.')
        return []

    if deployment_name is None or str(deployment_name).strip() == '':
        raise ValueError('deployment_name is required to search for ADCP files.')

    required_suffix = '.000' if status == 'recovered' else '_raw.dat'
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
        if 'adcp' not in flower:
            continue
        if all(token in flower for token in required_tokens):
            files.append(fname)
    return sorted(files)

def extract_adcp_full_records(adcp_files: str, datadir: str, rd_bytes: int = None):

    adcp_records = {}
    for adcp_file in adcp_files:
        print('   ...Processing file: ' + adcp_file)

        rd_bytes = None

        # Ensure that there is any data in the file to process
        if os.path.getsize(os.path.join(datadir, adcp_file)) <= 0:
            print('No data in ADCP file: ' + adcp_file + '\n   ...Continue on...')
            continue

        
        # Read in the file
        if rd_bytes is None:
            with open(os.path.join(datadir,adcp_file), mode='rb') as f:
                test = adcp_funcs.AdcpPd0Record(f.read(10000))
            rd_bytes = test.header.num_bytes + 2
        lines = []
        
        with open(os.path.join(datadir,adcp_file), mode='rb') as f:
            while True:
                try:
                    templine = f.read(rd_bytes)
                    if len(templine) > 0:
                        lines.append(templine)
                    else:
                        break
                except struct.error:
                    break
        
        adcp_temprecords = {}
        total_lines = len(lines)
        rec_count = 0
        print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') + ': Begin processing...')
        print('   ...Total lines: ' + str(total_lines))
        for ii in range(0,len(lines)):
            line = lines[ii]
            try:
                adcp_records[rec_count] = adcp_funcs.AdcpPd0Record(line)
                rec_count = rec_count + 1
                if rec_count % 5000 == 0:
                    print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') + 
                        ': Line ' + str(rec_count) + ' out of ' + str(total_lines) + ' completed...')
            except:
                continue
        print('   ...finished reading in data: ' + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S'))

        adcp_records.update(adcp_temprecords)

    return adcp_records

def extract_adcp_realtime_records(adcp_files: list, datadir: str,
                                  deployment_info: dict = None):

    # Extract out information from the deployment info, if it is provided
    if deployment_info is not None:
        start_deployment = deployment_info.get('start_deployment', None)
        end_deployment = deployment_info.get('end_deployment', None)

    # Read and concatenate all found files
    frames = []
    for fname in adcp_files:
        fpath = os.path.join(datadir, fname)
        try:
            frames.append(read_adcp_buoy_dataframe(fpath))
        except Exception:
            print(f'   Reading {fname} as a dataframe did not work, trying line parsing.')
            frames.append(read_adcp_buoy_lineparsing(fpath))

    if len(frames) == 1:
        adcp_rawdf = frames[0]
    else:
        adcp_rawdf = pd.concat(frames, ignore_index=True)

    ###################################################
    # Filter the data and load any existing processed file

    # Filter out only the valid ADCP ensembles
    adcp_rawdf = adcp_rawdf.query('ValidStr == True' )
        
        
    # Load in the previously processed datafiles, if they exist
    savedir = deployment_info.get('savedir', None)
    savefile = deployment_info.get('savefile', None)
    if os.path.exists(os.path.join(savedir, savefile)):
        adcp_xrold = xr.load_dataset(os.path.join(savedir, savefile))
    else:
        adcp_xrold = None
        
    
    # If there is an existsing ADCP file, isolate only the
    # ADCP records that are new
    if adcp_xrold is not None:
        last_ensemble = pd.Timestamp(adcp_xrold['controller_timestamp'].data[-1]).to_pydatetime()
        adcp_timestamps = [ii.to_pydatetime() for ii in adcp_rawdf['Timestamp'].to_list()]
        adcp_rawdf = adcp_rawdf[[ii > last_ensemble for ii in adcp_timestamps]]
        
    # Keep data only within the deployment window.
    valid_time_mask = ~adcp_rawdf['Timestamp'].isna()
    if start_deployment is not None:
        valid_time_mask = valid_time_mask & (adcp_rawdf['Timestamp'] >= start_deployment)
    if end_deployment is not None:
        valid_time_mask = valid_time_mask & (adcp_rawdf['Timestamp'] <= end_deployment)
    adcp_rawdf = adcp_rawdf[valid_time_mask]
        
    # Ensure that the indexing of the new ADCP dataframe is correct
    if len(adcp_rawdf) > 0:
        adcp_rawdf = adcp_rawdf.reset_index(drop=True)
        adcp_timestamps = [ii.to_pydatetime() for ii in adcp_rawdf['Timestamp'].to_list()]
    
    if len(adcp_rawdf) == 0:
        return None, None, adcp_xrold
    
    
    ###############################################################
    # Convert the ADCP Hex data
    ###############################################################
    
    # Step through each record, and convert the ADCP Hex data into
    # human-readable data
    adcp_records = {}
    for ii in range(0,adcp_rawdf.shape[0]):
        teststr_raw = adcp_rawdf.loc[ii,'ADCP_Raw'].strip(' ')
        teststr = bytes.fromhex(teststr_raw)

        adcp_records[ii] = adcp_funcs.AdcpPd0Record(teststr)

    return adcp_records, adcp_timestamps, adcp_xrold


def _build_adcp_average_file_name(base_savefile, minutes):

    if base_savefile.lower().endswith('.nc'):
        return base_savefile[:-3] + f'_{minutes}min.nc'
    return base_savefile + f'_{minutes}min.nc'


def _make_adcp_time_average(orig_xr, minutes, deployment_info):
    """Create a time-averaged ADCP dataset while preserving metadata."""
    if 'time' not in orig_xr.coords:
        raise KeyError("ADCP dataset missing required 'time' coordinate.")

    avg_rule = f'{int(minutes)}min'
    adcp_sorted = orig_xr.sortby('time')

    averaged_vars = {}
    for var_name, data_array in adcp_sorted.data_vars.items():
        if np.issubdtype(data_array.dtype, np.number):
            averaged_vars[var_name] = data_array.resample(time=avg_rule).mean(skipna=True, keep_attrs=True)
        else:
            averaged_vars[var_name] = data_array.resample(time=avg_rule).first(keep_attrs=True)

    adcp_avg = xr.Dataset(averaged_vars, attrs=adcp_sorted.attrs.copy())

    # Keep static string metadata variables object-typed after averaging so
    # downstream NetCDF writing logic treats them consistently.
    for static_name in ('buoy_name', 'deployment_name'):
        if static_name in adcp_avg.data_vars and static_name in adcp_sorted.data_vars:
            static_source = adcp_sorted[static_name]
            static_attrs = static_source.attrs.copy()
            time_len = adcp_avg.sizes.get('time', 0)
            static_values = np.empty(time_len, dtype=object)
            if static_source.size > 0 and time_len > 0:
                static_values[:] = static_source.values[0]
            adcp_avg[static_name] = xr.DataArray(
                static_values,
                dims=('time',),
                coords={'time': adcp_avg['time']},
                attrs=static_attrs,
            )
    
    adcp_avg.attrs["title"] = deployment_info['buoy_title'] + ' - ADCP Data (' + str(minutes) + '-minute averages)'
    adcp_avg.attrs["description"] = 'Acoustic Doppler Current Profiler (ADCP) data on the ' + deployment_info['buoy_name'] + ' platform, averaged over ' + str(minutes) + ' minutes'

    adcp_avg.attrs['time_averaging_minutes'] = int(minutes)
    adcp_avg.attrs['time_averaging_description'] = f'Time-averaged to {int(minutes)}-minute bins.'

    return adcp_avg


def buoy_adcp_wrapper(buoy_name, deployment_name, instrument_name='adcp', serial_num=None, average_windows=None):
    
    ###########################################
    # Get basic lander information
    lat, lon, depth, start_deployment, end_deployment, status = _load_adcp_deployment_window(
        buoy_name,
        deployment_name
    )
    if not((status == 'realtime') or (status == 'recovered')):
        print(f"Deployment status is '{status}'. Skipping processing for {buoy_name} - {deployment_name}.")
        return
    else:
        print(f"Deployment status is '{status}'. Proceeding with processing for {buoy_name} - {deployment_name}.")
    datadir, savedir = nemo_funcs.get_datalocations(buoy_name, deployment_name, 'adcp', status=status)
    if status == 'recovered':
        datadir = os.path.join(datadir, 'adcp')

    # Load deployment metadata for serial lookup and output metadata.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{buoy_name}_deployments.json')
    with open(json_path, 'r') as f:
        deployment_info = json.load(f)
    deployment_data = deployment_info.get('deployments', {}).get(deployment_name, {})
    target_depth = deployment_data.get('target_depth', None)
    instrument_info = {}

    if serial_num is None:
        instruments = deployment_data.get('instruments', {})
        adcp_serials = []

        if isinstance(instruments, dict):
            for _, instrument_record in instruments.items():
                if not isinstance(instrument_record, dict):
                    continue

                model_name = str(instrument_record.get('model', '')).strip().lower()
                serial_value = instrument_record.get('sn')
                if serial_value is None:
                    continue
                if 'adcp' not in model_name:
                    continue

                serial_text = str(serial_value).strip()
                if serial_text == '':
                    continue
                if serial_text not in adcp_serials:
                    adcp_serials.append(serial_text)

        if len(adcp_serials) == 0:
            print(
                f"No ADCP serial numbers were found in deployment info for "
                f"{buoy_name} - {deployment_name}."
            )
            return

        print(
            f"No serial number provided. Found {len(adcp_serials)} ADCP serial(s): "
            f"{', '.join(adcp_serials)}"
        )
        for current_serial in adcp_serials:
            print(f"\nProcessing ADCP serial {current_serial}...")
            buoy_adcp_wrapper(buoy_name, deployment_name, instrument_name, current_serial, average_windows)
        return
    
    

    instrument_info = nemo_deployments.find_matching_deployment_instrument(
        buoy_name,
        deployment_name,
        'adcp',
        serial_num
    )
    if instrument_info is None:
        print(
            f"   No matching deployment instrument found for ADCP serial {serial_num}. "
            'Proceeding without deployment-specific instrument metadata.'
        )
        instrument_info = {}
    else:
        print(instrument_info)
        if instrument_info.get('instrument_depth', None) is not None:
            target_depth = instrument_info.get('instrument_depth')

    # Discover matching data files
    adcp_files = _find_adcp_files(datadir, buoy_name, deployment_name, serial_num, status)
    if len(adcp_files) == 0:
        print(f'   No ADCP files found in {datadir}')
        return
    print(f'   Found {len(adcp_files)} ADCP file(s): {adcp_files}')


    #####################
    # Realtime processing
    savefile = buoy_name.lower() + '_' + deployment_name.lower() + '_adcp_sn' + serial_num + '.nc'
    if status == 'realtime':
        deployment_dict = {
            'start_deployment': start_deployment,
            'end_deployment': end_deployment,
            'savedir': savedir,
            'savefile': savefile
        }
        adcp_records, adcp_timestamps, adcp_xrold = extract_adcp_realtime_records(adcp_files, datadir, deployment_dict)
    elif status == 'recovered':
        adcp_records = extract_adcp_full_records(adcp_files, datadir)
        adcp_xrold = None
    else:
        print(f"Deployment status is '{status}'. Skipping processing for {buoy_name} - {deployment_name}.")
        return
    
    # If status is realtime, and there is no new data, end the function
    if status == 'realtime' and adcp_records is None:
        print('   No new ADCP data to process. Continue on.')
        return
        
        
    # Extract all of the data out of each ADCP ensemble
    [sys_config, instr_params, zs_adcp, 
     ens_no, dtnum, heading, 
     pitch, roll, hdg_std, pitch_std, roll_std,
     depth_xducer, soundvel, xducer_temp, 
     corr_beam1, corr_beam2, corr_beam3, corr_beam4,
     echo_beam1, echo_beam2, echo_beam3, echo_beam4,
     vel_beam1, vel_beam2, vel_beam3, vel_beam4] = extract_ensemble_data(adcp_records)   
    if status == 'recovered':
        adcp_timestamps = dtnum 


    #######################################
    # Adjust the depth/location of the data
    #

    depth_xducer = recalc_xducer_depth(depth_xducer, instr_params, lat)
    z_adcp = zs_adcp[0]
    if sys_config['beam_direction'] == 'upfacing':
        zadjust_adcp = np.median(depth_xducer) - z_adcp
    elif sys_config['beam_direction'] == 'downfacing':
        zadjust_adcp = np.median(depth_xducer) + z_adcp
    maxz = np.max(zadjust_adcp)


    print('   ...Null out the bad values when the signal is contaminated by beam reflection')
    if sys_config['beam_direction'] == 'upfacing':
        # Identify a distance below the surface where we can
        # assume beam reflection is contaminating the signal,
        # and find all indices above this depth
        reflect_depthinds = np.where(zadjust_adcp < 0)

        corr_beam1[:, reflect_depthinds[0][0]:] = 0
        corr_beam2[:, reflect_depthinds[0][0]:] = 0
        corr_beam3[:, reflect_depthinds[0][0]:] = 0
        corr_beam4[:, reflect_depthinds[0][0]:] = 0

        echo_beam1[:, reflect_depthinds[0][0]:] = 0
        echo_beam2[:, reflect_depthinds[0][0]:] = 0
        echo_beam3[:, reflect_depthinds[0][0]:] = 0
        echo_beam4[:, reflect_depthinds[0][0]:] = 0

        vel_beam1[:, reflect_depthinds[0][0]:] = np.nan
        vel_beam2[:, reflect_depthinds[0][0]:] = np.nan
        vel_beam3[:, reflect_depthinds[0][0]:] = np.nan
        vel_beam4[:, reflect_depthinds[0][0]:] = np.nan
    elif sys_config['beam_direction'] == 'downfacing':

        # Identify a distance near the bottom where we can
        # assume beam reflection is contaminating the signal,
        # and find all indices above this depth
        reflect_depthinds = np.where(zadjust_adcp > depth)

        corr_beam1[:, reflect_depthinds[0][0]:] = 0
        corr_beam2[:, reflect_depthinds[0][0]:] = 0
        corr_beam3[:, reflect_depthinds[0][0]:] = 0
        corr_beam4[:, reflect_depthinds[0][0]:] = 0

        echo_beam1[:, reflect_depthinds[0][0]:] = 0
        echo_beam2[:, reflect_depthinds[0][0]:] = 0
        echo_beam3[:, reflect_depthinds[0][0]:] = 0
        echo_beam4[:, reflect_depthinds[0][0]:] = 0

        vel_beam1[:, reflect_depthinds[0][0]:] = np.nan
        vel_beam2[:, reflect_depthinds[0][0]:] = np.nan
        vel_beam3[:, reflect_depthinds[0][0]:] = np.nan
        vel_beam4[:, reflect_depthinds[0][0]:] = np.nan



    #####################################################
    # Null out the bad values when the signal correlation
    # is too "weak"
    print('   ...Null out the bad values when the signal correlation is too weak')
    bad_corr = 10
    vel_beam1[corr_beam1 < bad_corr] = np.nan
    vel_beam2[corr_beam2 < bad_corr] = np.nan
    vel_beam3[corr_beam3 < bad_corr] = np.nan
    vel_beam4[corr_beam4 < bad_corr] = np.nan


    ############################################################
    # Rotate the velocity beams by the magnetic declination
    # Convert horizontal velocity in cartesian components into polar components
    print('   ...Rotate the velocity beams by the magnetic declination')
    vel_beam1, vel_beam2, avg_mag_dec = apply_magnetic_declination(dtnum, lat, lon, 
                                                                   vel_beam1, vel_beam2)


    ###########################################
    # Interpolate the beams onto gridded depths

    print('   ...Interpolate the beams onto gridded depths')
    zstep = np.nanmax([1,np.round(np.mean(np.diff(zs_adcp)),1)])
    if target_depth is not None:
        ztarget = target_depth
    else:
        ztarget = np.ceil(maxz/zstep)*zstep
    zgrid = np.arange(0,np.ceil(ztarget/zstep)+1)*zstep

    vars_to_regrid = {'vel_beam1': vel_beam1,
                      'vel_beam2': vel_beam2,
                      'vel_beam3': vel_beam3,
                      'vel_beam4': vel_beam4,
                      'echo_beam1': echo_beam1, 
                      'echo_beam2': echo_beam2,
                      'echo_beam3': echo_beam3,
                      'echo_beam4': echo_beam4,
                      'corr_beam1': corr_beam1,
                      'corr_beam2': corr_beam2,
                      'corr_beam3': corr_beam3,
                      'corr_beam4': corr_beam4}

    regridded_vars = adcp_regrid_ensemble(zgrid, zs_adcp, depth_xducer, 
                                          vars_to_regrid, sys_config['beam_direction'])

    vel_beam1_regrid = regridded_vars['vel_beam1']
    vel_beam2_regrid = regridded_vars['vel_beam2']
    vel_beam3_regrid = regridded_vars['vel_beam3']
    vel_beam4_regrid = regridded_vars['vel_beam4']

    echo_beam1_regrid = regridded_vars['echo_beam1']
    echo_beam2_regrid = regridded_vars['echo_beam2']
    echo_beam3_regrid = regridded_vars['echo_beam3']
    echo_beam4_regrid = regridded_vars['echo_beam4']

    corr_beam1_regrid = regridded_vars['corr_beam1']
    corr_beam2_regrid = regridded_vars['corr_beam2']
    corr_beam3_regrid = regridded_vars['corr_beam3']
    corr_beam4_regrid = regridded_vars['corr_beam4']       


    ##########################################################
    # Null out values above the reflection depth  
    print('   ...Null out values above the reflection depth')
    if sys_config['beam_direction'] == 'upfacing':
        z_abs = [np.median(depth_xducer) - ii for ii in zs_adcp[0]]
        reflect_depth = 0.07*np.median(depth_xducer)
        reflect_depthinds = np.where(zgrid < reflect_depth)    

        vel_beam1_regrid[:, :reflect_depthinds[0][-1]+1] = np.nan
        vel_beam2_regrid[:, :reflect_depthinds[0][-1]+1] = np.nan
        vel_beam3_regrid[:, :reflect_depthinds[0][-1]+1] = np.nan
        vel_beam4_regrid[:, :reflect_depthinds[0][-1]+1] = np.nan    

        echo_beam1_regrid[:, :reflect_depthinds[0][-1]+1] = 0
        echo_beam2_regrid[:, :reflect_depthinds[0][-1]+1] = 0
        echo_beam3_regrid[:, :reflect_depthinds[0][-1]+1] = 0
        echo_beam4_regrid[:, :reflect_depthinds[0][-1]+1] = 0    

        corr_beam1_regrid[:, :reflect_depthinds[0][-1]+1] = 0
        corr_beam2_regrid[:, :reflect_depthinds[0][-1]+1] = 0
        corr_beam3_regrid[:, :reflect_depthinds[0][-1]+1] = 0
        corr_beam4_regrid[:, :reflect_depthinds[0][-1]+1] = 0

    elif sys_config['beam_direction'] == 'downfacing':

        z_abs = [np.median(depth_xducer) + ii for ii in zs_adcp[0]]
        reflect_depth = (1-0.07)*np.median(depth)
        reflect_depthinds = np.where(zgrid > reflect_depth)   

        vel_beam1_regrid[:, reflect_depthinds[0][-1]:] = np.nan
        vel_beam2_regrid[:, reflect_depthinds[0][-1]:] = np.nan
        vel_beam3_regrid[:, reflect_depthinds[0][-1]:] = np.nan
        vel_beam4_regrid[:, reflect_depthinds[0][-1]:] = np.nan    

        echo_beam1_regrid[:, reflect_depthinds[0][-1]:] = 0
        echo_beam2_regrid[:, reflect_depthinds[0][-1]:] = 0
        echo_beam3_regrid[:, reflect_depthinds[0][-1]:] = 0
        echo_beam4_regrid[:, reflect_depthinds[0][-1]:] = 0    

        corr_beam1_regrid[:, reflect_depthinds[0][-1]:] = 0
        corr_beam2_regrid[:, reflect_depthinds[0][-1]:] = 0
        corr_beam3_regrid[:, reflect_depthinds[0][-1]:] = 0
        corr_beam4_regrid[:, reflect_depthinds[0][-1]:] = 0
    
    
    ###############################################################
    # Write all of the extract ADCP data into an xarray dataset

    print('   ...Write all of the extract ADCP data into an xarray dataset')
    nemo_info = {'BuoyName': buoy_name,
                 'BuoyTitle': deployment_info.get('buoy_title', deployment_info.get('nemo_title', '')),
                 'info_url': deployment_info.get('info_url', ''),
                 'DeploymentName': deployment_name,
                 'Latitude': lat,
                 'Longitude': lon,
                 'Depth': depth,
                 "avg_mag_dec": np.round(avg_mag_dec,4),
                 'institution_info': deployment_info.get('institution_info', {}),
                 'InstrumentType': 'sbe56',
                 'InstrumentInfo': instrument_info}

    adcp_xrnew = make_adcp_xr(savefile, buoy_name, deployment_name, lat, lon, depth,
                              adcp_timestamps, sys_config, nemo_info, instr_params, zgrid,
                              ens_no, dtnum, heading, 
                              pitch, roll, hdg_std, pitch_std, roll_std,
                              depth_xducer, soundvel, xducer_temp, 
                              corr_beam1_regrid, corr_beam2_regrid, corr_beam3_regrid, corr_beam4_regrid,
                              echo_beam1_regrid, echo_beam2_regrid, echo_beam3_regrid, echo_beam4_regrid,
                              vel_beam1_regrid, vel_beam2_regrid, vel_beam3_regrid, vel_beam4_regrid)
    adcp_xrnew = adcp_xrnew.drop_duplicates(dim='time', keep='last')
    
    
    #######################################
    # Save the ADCP data
    #######################################
    if adcp_xrold is not None:
        adcp_xr = xr.concat([adcp_xrold, adcp_xrnew], dim='time', 
                            coords=['time'], compat='override', 
                            combine_attrs='drop_conflicts')
        adcp_xr = adcp_xr.drop_duplicates(dim='time', keep='last')
    else:
        adcp_xr = adcp_xrnew


    # Apply time drift correction if necessary
    if (status == 'recovered') and (instrument_info.get('time_drift_secs', False)):
        drift_seconds = instrument_info.get('time_drift_secs', 0)
        inst_start_time = datetime.datetime.strptime(instrument_info.get('start_time', None), '%Y-%m-%dT%H:%M:%SZ') if instrument_info.get('start_time', None) else None
        inst_end_time = datetime.datetime.strptime(instrument_info.get('end_time', None), '%Y-%m-%dT%H:%M:%SZ') if instrument_info.get('end_time', None) else None
        sample_time = adcp_xr['time'].to_pandas()
        timerange = (inst_end_time - inst_start_time).total_seconds() if inst_start_time and inst_end_time else 0
        drift_rate = drift_seconds / timerange if timerange > 0 else 0
        print(f'   Applying time drift correction of {drift_seconds} seconds over {timerange} seconds ({drift_rate} seconds per second).')

        # Apply a linear drift correction across the time range of the data
        sample_time = sample_time - pd.to_timedelta(drift_rate * (sample_time - inst_start_time).total_seconds(), unit='s')
        adcp_xr['time'] = ('time', sample_time)

    # Restrict the dataset to the deployment window
    if start_deployment is not None:
        adcp_xr = adcp_xr.where(adcp_xr['time'] >= np.datetime64(start_deployment), drop=True)
    if end_deployment is not None:
        adcp_xr = adcp_xr.where(adcp_xr['time'] <= np.datetime64(end_deployment), drop=True)
    
    
    
    adcp_write_netcdf(adcp_xr.copy(), savedir, savefile)

    if average_windows is not None and len(average_windows) > 0:
        for minutes in average_windows:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Averaging ADCP data to ' + str(minutes) + ' minutes...')
            avg_savefile = _build_adcp_average_file_name(savefile, minutes)
            adcp_avg = _make_adcp_time_average(adcp_xr.copy(), minutes, deployment_info)
            adcp_write_netcdf(adcp_avg, savedir, avg_savefile)
    
    return


# In[ ]:


def adcp_write_netcdf(data_xr, savedir, savefile):
    
    # Write a netCDF file, using the netCDF4 Dataset library,
    # and building the data from an xarray dataset
    
    tmpsavepath = os.path.join(savedir, 'nemo_adcptemp.nc')
    try:
        with data_xr as src, Dataset(tmpsavepath, 'w') as dst:
            # copy global attributes all at once via dictionary
            dst.setncatts(data_xr.attrs)
            # copy dimensions
            for name, dimension in src.dims.items():
                dst.createDimension(
                    name, (dimension))
            # copy all file data except for the excluded
            for name, variable in src.variables.items():
                if 'time' in name:
                    timeoffset = datetime.datetime.strptime(variable.attrs['units'][variable.attrs['units'].find('since ')+6:],
                                                            '%Y-%m-%d %H:%M:%S')
                    vardata = [(pd.Timestamp(ii).to_pydatetime(warn=False) - timeoffset).total_seconds() 
                               for ii in variable.data]
                    datatype = 'f8'
                    x = dst.createVariable(name, datatype, variable.dims)
                    dst[name][:] = vardata
                elif variable.dtype == object:
                    vardata = [ii for ii in variable.data]
                    x = dst.createVariable(name, str, variable.dims)
                    for i, val in enumerate(vardata):
                        dst[name][i] = val
                else:
                    vardata = [ii for ii in variable.data]
                    datatype = variable.dtype
                    x = dst.createVariable(name, datatype, variable.dims)
                    dst[name][:] = vardata
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].attrs)


        ##################################################
        # Move and rename the netCDF
        # Note: this is done to ensure that the file
        #       is properly identified by the ERDDAP
        #       server as a viable file quickly, rather
        #       than needing a full dataset reload
        #       See here:
        #       https://coastwatch.pfeg.noaa.gov/erddap/download/setupDatasetsXml.html#updateEveryNMillis
        shutil.move(tmpsavepath, os.path.join(savedir, savefile))
        print('   ...Saved ADCP netCDF file: ' + os.path.join(savedir, savefile))
    except Exception as e:
        print('Error occurred with making ADCP netCDF')
        print(e)
        if e.__traceback__ is not None: 
            print(f'Error occurred on line number: {e.__traceback__.tb_lineno}')
        os.remove(tmpsavepath)
        
        
    return


def regrid_ensemble(z_adcp, depth_xduce, var):
    
    print('Doing nothing')
    
    return


def _parse_average_windows(average_text):

    if average_text is None:
        return []

    entries = [entry.strip() for entry in str(average_text).split(',') if entry.strip() != '']
    if len(entries) == 0:
        return []

    windows = []
    for entry in entries:
        try:
            minutes = int(entry)
        except ValueError:
            raise ValueError(f"Invalid averaging window '{entry}'. Use integer minutes like '10,60'.")

        if minutes <= 0:
            raise ValueError(f"Averaging window must be positive minutes. Got: {minutes}")

        if minutes not in windows:
            windows.append(minutes)

    return windows


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process ADCP data for a lander deployment')
    parser.add_argument('--buoy', type=str,
                        help='Buoy name to process')
    parser.add_argument('--lander', type=str,
                        help='Lander name (legacy alias for --buoy)')
    parser.add_argument('--deployment', type=str, required=True,
                        help='Deployment name')
    parser.add_argument('--instrument', type=str,
                        help='Instrument name to process')
    parser.add_argument('--serial', type=str,
                        help='Instrument serial number to process')
    parser.add_argument('--averages', type=str,
                        help='Optional comma-separated averaging windows in minutes (e.g., 10,60)')
    args = parser.parse_args()
    buoy_name = args.buoy if args.buoy is not None else args.lander
    if buoy_name is None:
        raise ValueError('Either --buoy or --lander is required.')
    average_windows = _parse_average_windows(args.averages)
    buoy_adcp_wrapper(buoy_name, args.deployment, args.instrument, args.serial, average_windows)


if __name__ == '__main__':
    main()