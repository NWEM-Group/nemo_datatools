#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import getopt

import traceback

import pandas as pd
import numpy as np

import datetime

import matplotlib
from matplotlib import pyplot as plt

import json


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


# In[ ]:


def round_to_nearest_second(dt):
    if dt.microsecond >= 500_000:
        dt += datetime.timedelta(seconds=1)
    return dt.replace(microsecond=0)


# In[ ]:


def make_gpsjson(savedir, gpsdata, lookback_days=10):
    
    # Find the start time of the json to create
    gps_timedata = gpsdata['time (seconds)'].to_numpy().squeeze()
    curtime = datetime.datetime.now()
    lasttime = (datetime.datetime(1970,1,1) + 
                datetime.timedelta(seconds=int(gps_timedata[len(gps_timedata)-1])))
    if curtime - lasttime > datetime.timedelta(days=lookback_days):
        curtime = (lasttime.replace(hour=0,minute=0,second=0,microsecond=0) + 
                   datetime.timedelta(days=1))
        
    starttime = (curtime - datetime.timedelta(days=lookback_days) 
                 - datetime.datetime(1970,1,1)).total_seconds()

    # Based upong the start time, find the first
    # indice of the gps data
    if any(gps_timedata >= starttime):
        gps_firstind = np.where(gps_timedata >= starttime)[0][0]
    else:
        gps_firstind = 0

        
    # Initialize an empty dictionary
    gps_dict = {'readings': {}}

    # Build a list of gps coordinate, and populate
    # that list with the data in the gps data
    gps_readings = []
    for ii in range(gps_firstind,len(gpsdata)):
        reading = {}
        reading['esn'] = '2-3095448'
        reading['timestamp'] = gpsdata.loc[ii,'time']
        if (pd.isna(gpsdata.loc[ii,'longitude (degrees)']) or
            pd.isna(gpsdata.loc[ii,'latitude (degrees)'])):
            reading['position'] = None
        else:
            reading['position'] = {}
            reading['position']['coordinates'] = [gpsdata.loc[ii,'longitude (degrees)'],
                                                  gpsdata.loc[ii,'latitude (degrees)']]

        gps_readings.append(reading)

    # Add the list of the gps readings to the
    # dictionary
    gps_dict['readings'] = gps_readings
    
    
    # Write the gps dictionary to a json file
    with open(os.path.join(savedir,
                           'gps-GF.json'),'w') as gpsfile:
        json.dump(gps_dict, gpsfile)
        
    return


# # Processing Directories and Buoy Info

# In[ ]:


def get_basedir(buoy_designation: str):
    
    basepath = "/home/nemo/data/realtime/chaba/"
        
    return basepath


# In[ ]:


def get_webdir(buoy_designation: str):
    
    web_basepath = "/var/www/htdocs/LiveFiles-CHABA/"
    webdir = web_basepath + buoy_designation.upper() + '/'
    
    return webdir


# In[ ]:


def get_buoy_prefix(buoy_designation: str):

    if 'chaba' in buoy_designation.lower():
        buoyprefix = 'chaba'
    elif 'nemo' in buoy_designation.lower():
        buoyprefix = 'nemo'
    else:
        buoyprefix = buoy_designation.lower()
        
    return buoyprefix


# In[ ]:


def get_buoyname(buoy_designation: str):
    
    buoyname_dict = {
        "CHABA": "Cha'ba",
        "NEMO": "NEMO-Subsurface"
    }
    
    return buoyname_dict[buoy_designation.upper()]


# In[ ]:


def get_buoycontroller(buoy_designation: str):
    
    buoyname_dict = {
        "CHABA": "chaba1",
        "NEMO": "nemo"
    }
    
    return buoyname_dict[buoy_designation]


# # Update latest contact info

# In[ ]:


def get_latestmpk_filedate(buoy_designation: str):
    
    mpkdir = os.path.join("/hdd0/orca20_data/", get_buoycontroller(buoy_designation), "INBOX")
    mpkfiles = [file for file in os.listdir(mpkdir) if '.mpk' in file] 
    mpkdates = [datetime.datetime.strptime(file[file.find('orcadata_')+len('orcadata_'):file.find('.mpk')],
                                           '%Y%m%d-%H') for file in mpkfiles]
    latest_ind = np.argmax(mpkdates)
    latest_mpkfile = mpkfiles[latest_ind]
    latest_filetime = os.path.getmtime(os.path.join(mpkdir,latest_mpkfile))

    latest_filedate = round_to_nearest_second(datetime.datetime(1970,1,1)+datetime.timedelta(seconds=latest_filetime))
    utc_offset = datetime.datetime.now().astimezone().tzinfo.utcoffset(None)
    latest_filedate = latest_filedate + utc_offset
    # Shift latest file date from UTC to local
    
    
    return latest_filedate


# In[ ]:


def update_latestcontact(buoy_prefix: str, buoy_designation: str, webdir: str):
    
    latest_filedate = get_latestmpk_filedate(buoy_designation)
    html_content = latest_filedate.strftime("%a, %b-%d-%Y %H:%M:%S")

    contact_filename = buoy_prefix + '-lastcontact.html'
    with open(os.path.join(webdir, contact_filename), "w", encoding="utf-8") as f:
        f.write(html_content)

    return


# # Buoy Engineering Data Diagnostics

# In[ ]:


def load_buoy_datafiles(basepath, filetype: str = 'sunsaver', lookback_days: int = 30):

        
    datapath = os.path.join(basepath, "nemo_instruments", filetype)
    if not(os.path.exists(datapath)):
        print('No folders exist for filetype: ' + filetype)
        return None
    datafiles = sorted(os.listdir(datapath))
    
    datadates = [datetime.datetime.strptime(file[file.find(filetype)+len(filetype)+1:file.find('.csv')],
                                            '%Y%m%d') for file in datafiles]
    curdate = (datetime.datetime.now()+datetime.timedelta(hours=7)).replace(hour=0,minute=0,second=0,microsecond=0)
    keepinds = [datadate >= (curdate-datetime.timedelta(days=lookback_days)) for datadate in datadates]
    if any(keepinds):
        datafiles = [datafiles[ii] for ii in range(0,len(datafiles)) if keepinds[ii]]
    else:
        print('No valid datafiles found in the lookback time range of ' + str(lookback_days) + ' days.')
        print('Lookback range: ' + (curdate-lookback_days).strftime('%Y-%b-%d') + ' to ' + curdate.strftime('%Y-%b-%d'))
        return None
    

    data_df = None
    for file in datafiles:
        temp_df = pd.read_csv(os.path.join(datapath, file))
            
        if data_df is None:
            data_df = temp_df.copy()
        else:
            data_df = pd.concat([data_df, temp_df]).reset_index(drop=True)
            
    data_df = data_df.sort_values(by='timestamp').reset_index(drop=True)
    
    ref_date = datetime.datetime(1970,1,1)
    data_df['date'] = [(ref_date + datetime.timedelta(seconds=round(float(ii)/1e9))) for ii in data_df['timestamp']]
    data_df['date_local'] = [datadate-datetime.timedelta(hours=7) for datadate in data_df['date']]

    return data_df


# In[ ]:


def create_smooted_var(data_df, varnames: list, smoothing_minutes: int):
    
    for varname in varnames:
        data_df[varname+'_smooth'] = pd.Series(data_df[varname].tolist(), index=data_df['date']).rolling(window=str(smoothing_minutes)+'T', center=True).mean().tolist()
    
    return data_df


# In[ ]:


def convert_batt_volt_to_percent(voltdata, doubbatt_flag=False):
    
    batt_volt = [11.0, 11.41, 11.71, 11.86, 12.01, 12.15,
                 12.28, 12.4, 12.55, 12.7, 12.85]
    doubbatt_volt = [ii*2 for ii in batt_volt]
    batt_pers = [0, 10.0, 20.0, 30.0, 40.0, 50.0, 
                 60.0, 70.0, 80.0, 90.0, 100.0]

    if doubbatt_flag:
        batt_interp = np.interp(voltdata, doubbatt_volt, batt_pers)
    else:
        batt_interp = np.interp(voltdata, batt_volt, batt_pers)
    
    return batt_interp


# In[ ]:


def make_chaba20_eng_plot(buoy_designation: str, sundata, engdata, lookback_days: int):
    
    xmin = 0.1
    ymin = 0.075
    xwidth = 0.375
    yheight = 0.2125
    xoffset = 0.1125
    yoffset = 0.0875
    
    # Get some info on the time range
    if sundata is not None:
        lasttime_sun = sundata['date_local'][len(sundata)-1]
    else:
        lasttime_sun = None
    if engdata is not None:
        lasttime_eng = engdata['date_local'][len(engdata)-1]
    else:
        lasttime_eng = lasttime_sun
    
    lasttime = pd.Timestamp(max([lasttime_sun, lasttime_eng])).to_pydatetime()
    
    curtime = datetime.datetime.now()
    if curtime-lasttime > datetime.timedelta(days=lookback_days):
        curtime = (lasttime.replace(hour=0,minute=0,second=0,microsecond=0) + 
                   datetime.timedelta(days=1))
    

    # Extract the sunsaver data
    if sundata is not None:
        # Extract out the variables            
        timedata_sun = [pd.Timestamp(ii).to_pydatetime() for ii in sundata['date_local'].to_numpy().squeeze()]
        timedata_sundiffhrs = np.array([ii.total_seconds()/3600 for ii in np.diff(timedata_sun)])
        timedata_sundiffhrs = np.append(timedata_sundiffhrs[0],timedata_sundiffhrs)
        load_current = sundata['Adc_il_f_smooth'].to_numpy().squeeze()
        charge_current = sundata['Adc_ic_f_smooth'].to_numpy().squeeze()
        load_volt = sundata['Adc_vl_f_smooth'].to_numpy().squeeze()
        charge_volt = sundata['Adc_vb_f_smooth'].to_numpy().squeeze()
        load_watt = sundata['power_load'].to_numpy().squeeze()
        charge_watt = sundata['power_charge'].to_numpy().squeeze()
        charge_percent = sundata['battery_percent'].to_numpy().squeeze()
        
        # Calculate a rolling power window consumption/charge
        load_watthr = np.array(pd.Series(data=(load_watt*timedata_sundiffhrs).tolist(), 
                                index=timedata_sun).rolling(window='24H',
                                                            min_periods=int(60*18),center=True).sum().to_list())
        charge_watthr = np.array(pd.Series(data=(charge_watt*timedata_sundiffhrs).tolist(), 
                                  index=timedata_sun).rolling(window='24H',
                                                              min_periods=int(60*18),center=True).sum().to_list())
        
        # Find the time range of interest
        timebools_sun = [ii >= (curtime.replace(hour=0,minute=0,second=0,microsecond=0)-
                                datetime.timedelta(days=lookback_days)) for ii in timedata_sun]
        if any(timebools_sun):
            firstind_sun = np.where(timebools_sun)[0][0]
        else:
            print('No recent sunsaver data found')
            firstind_sun = None
            
        # Get the starting and ending sunsaver dates
        startdate_sun = timedata_sun[firstind_sun].replace(hour=0,minute=0,second=0,microsecond=0)    
        enddate_sun = (lasttime_sun.replace(hour=0,minute=0,second=0,microsecond=0)
                       + datetime.timedelta(days=1))
        
        # Restrict the variables to the timeframe of interest
        timedata_sun = timedata_sun[firstind_sun:]        
        load_current = load_current[firstind_sun:]
        charge_current = charge_current[firstind_sun:]
        load_volt = load_volt[firstind_sun:]
        charge_volt = charge_volt[firstind_sun:]
        load_watt = load_watt[firstind_sun:]
        charge_watt = charge_watt[firstind_sun:]
        charge_percent = charge_percent[firstind_sun:]
        load_watthr = load_watthr[firstind_sun:]
        charge_watthr = charge_watthr[firstind_sun:]
        
    
    
    # Extract the engineering data
    if engdata is not None:
        # Extract out the variables
        timedata_eng = [pd.Timestamp(ii).to_pydatetime() for ii in engdata['date_local'].to_numpy().squeeze()]
        temper = engdata['temperature_smooth'].to_numpy().squeeze()
        humid = engdata['humidity_smooth'].to_numpy().squeeze()

        # Find the time range of interest
        timebools_eng = [ii >= (curtime.replace(hour=0,minute=0,second=0,microsecond=0)-
                                datetime.timedelta(days=lookback_days)) for ii in timedata_eng]
        if any(timebools_eng):
            firstind_eng = np.where(timebools_eng)[0][0]
        else:
            print('No recent engineering data found')
            firstind_eng = None

        # Get the starting and ending engineering dates
        startdate_eng = timedata_eng[firstind_eng].replace(hour=0,minute=0,second=0,microsecond=0)
        enddate_eng = (lasttime_eng.replace(hour=0,minute=0,second=0,microsecond=0)
                       + datetime.timedelta(days=1))
        
        
        # Restrict the variables to the timeframe of interest
        timedata_eng = timedata_eng[firstind_eng:]
        temper = temper[firstind_eng:]
        humid = humid[firstind_eng:]
    else:
        startdate_eng = startdate_sun
        enddate_eng = enddate_sun
        
    
    
        

    
    
    
    # Determine the xlimit range as the start and end date of the whole dataset
    startdate = min([startdate_eng, startdate_sun])
    enddate = max([enddate_eng, enddate_sun])
        
      
    ###########
    if lookback_days <= 3:
        maj_xticks = matplotlib.dates.DateFormatter('%H:%M')
        min_xticks = matplotlib.ticker.MultipleLocator(0.25)
    else:
        maj_xticks = matplotlib.dates.DateFormatter('%b-%d')
        min_xticks = matplotlib.ticker.MultipleLocator(0.25)
  

        
    #####################
    #
    
    def apply_ticklabs(figax, lookback_days):
        if lookback_days <= 3:
            ticks = figax.get_xticks()
            ticklabs = figax.get_xticklabels()
            for ii in range(0,len(ticks)):
                if (ticks[ii]%1 == 0):
                    ticklabs[ii] = (datetime.datetime(1970,1,1) + 
                                    datetime.timedelta(days=ticks[ii])).strftime('%b-%d')
            figax.set_xticklabels(ticklabs)
        
        return figax
    
    def plot_baddata(figax, timevar, datavar):
        badinds = np.where(np.isnan(datavar))[0].astype(int)
        ylims = figax.get_yticks()
        ylim_min = min(ylims)
        ylim_max = max(ylims)
        figax.plot([timevar[ii] for ii in badinds], ylim_min*np.ones(len(badinds)),
                   'ro',markersize=2,label='Bad data',zorder=10, clip_on=False)
        figax.set_ylim([ylim_min,ylim_max])
        figax.legend(loc='best',fontsize=6)
        
        return figax
    
    def set_grid_and_labels(figax):
        
        figax.grid(visible=True, which='major',axis='both',
                   linewidth=0.5, color='grey', alpha = 0.5, linestyle='--')
        figax.grid(visible=True, which='minor',axis='both',
                   linewidth=0.25, color='grey', alpha = 0.25, linestyle='--')
        for label in figax.get_xticklabels(which='major'):
            label.set(rotation=25, fontsize=8, horizontalalignment='center')
        for label in figax.get_yticklabels(which='major'):
            label.set(fontsize=8, horizontalalignment='right')
            
        return figax
        
        
        
    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(6.5,8), dpi=160)
    

    ################################
    # Temperature (Engineering)
    if engdata is not None:
        ax[0].plot(timedata_eng, temper, '*',markersize=2)
        if any(np.isnan(temper)):
            ax[0] = plot_baddata(ax[0], timedata_eng, temper)        
        ax[0].set_xlim([startdate, enddate])
        ax[0].set_ylabel(r'$^o$C')
        ax[0].set_title('System Temperature')
        ax[0].set_position([xmin, ymin+2*(yheight+yoffset), xwidth, yheight])
        ax[0].xaxis.set_major_formatter(maj_xticks)
        ax[0].xaxis.set_minor_locator(min_xticks)
        ax[0] = apply_ticklabs(ax[0], lookback_days)
        ax[0] = set_grid_and_labels(ax[0])
    else:
        fig.delaxes(ax[0])
        fig.text(
            xmin+(xwidth/2), ymin+2*(yheight+yoffset)+(yheight/2), 
            "No system temperature data",
            ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='w', edgecolor='0.3', boxstyle='round,pad=0.4')
        )


    
    #################################
    # Humidity (Engineering)
    if engdata is not None:
        ax[1].plot(timedata_eng, humid, '*',markersize=2)
        if any(np.isnan(humid)):
            ax[1] = plot_baddata(ax[1], timedata_eng, humid)  
        ax[1].set_xlim([startdate, enddate])
        ax[1].set_ylabel('%')
        ax[1].set_title('Relative Humidity')
        ax[1].set_position([xmin+xwidth+xoffset, ymin+2*(yheight+yoffset), xwidth, yheight])
        ax[1].xaxis.set_major_formatter(maj_xticks)
        ax[1].xaxis.set_minor_locator(min_xticks)
        ax[1] = apply_ticklabs(ax[1], lookback_days)
        ax[1] = set_grid_and_labels(ax[1])
    else:
        fig.delaxes(ax[1])
        fig.text(
            xmin+xwidth+xoffset+(xwidth/2), ymin+2*(yheight+yoffset)+(yheight/2), 
            "No sytem relative\nhumidity data",
            ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='w', edgecolor='0.3', boxstyle='round,pad=0.4')
        )

        
    #################################
    # Power (sunsaver)
    ax[2].plot(timedata_sun, load_watt, '*',markersize=2, 
               label='Load: Avg='+str(np.round(np.nanmean(load_watt),1))+'W')
    if any(np.isnan(load_watt)):
        ax[2] = plot_baddata(ax[2], timedata_sun, load_watt)  
    ax[2].plot(timedata_sun, charge_watt, '*',markersize=2,
               label='Charge: Avg='+str(np.round(np.nanmean(charge_watt),1))+'W')
    if any(np.isnan(load_watt)):
        ax[2] = plot_baddata(ax[2], timedata_sun, charge_watt)  
    ax[2].legend(loc='best',fontsize=6)
    ax[2].set_xlim([startdate, enddate])
    ax[2].set_ylabel('Watts')
    ax[2].set_title('Power')
    ax[2].set_position([xmin, ymin+yheight+yoffset, xwidth, yheight])
    ax[2].xaxis.set_major_formatter(maj_xticks)
    ax[2].xaxis.set_minor_locator(min_xticks)
    ax[2] = apply_ticklabs(ax[2], lookback_days)
    ax[2] = set_grid_and_labels(ax[2])
           

    #####################################
    # Current (sunsaver)
    ax[3].plot(timedata_sun, load_current, '*',markersize=2)
    if any(np.isnan(load_current)):
        ax[3] = plot_baddata(ax[3], timedata_sun, load_current)  
    ax[3].set_xlim([startdate, enddate])
    ax[3].set_ylabel('Amperes')
    ax[3].set_title('Internal Load Current')
    ax[3].set_position([xmin+xwidth+xoffset, ymin+yheight+yoffset, xwidth, yheight])
    ax[3].xaxis.set_major_formatter(maj_xticks)
    ax[3].xaxis.set_minor_locator(min_xticks)
    ax[3] = apply_ticklabs(ax[3], lookback_days)
    ax[3] = set_grid_and_labels(ax[3])
    
    # Rolling cumulative energy usage/charge
    load_watthr_avg = np.round(np.nanmean(load_watthr),0)
    if np.isnan(load_watthr_avg):
        load_watthr_avgstr = 'NaN'
    else:
        load_watthr_avgstr = str(int(load_watthr_avg))
    charge_watthr_avg = np.round(np.nanmean(charge_watthr),0)
    if np.isnan(charge_watthr_avg):
        charge_watthr_avgstr = 'NaN'
    else:
        charge_watthr_avgstr = str(int(charge_watthr_avg))

    ax[4].plot(timedata_sun, load_watthr, '*',markersize=2, 
               label='Load: Avg='+load_watthr_avgstr+'Wh')
    
    ax[4].plot(timedata_sun, charge_watthr, '*',markersize=2, 
               label='Charge: Avg='+charge_watthr_avgstr+'Wh')
    ax[4].legend(loc='best',fontsize=6)
    ax[4].set_xlim([startdate, enddate])
    ax[4].set_xlabel('Time (PDT/UTC-07)')
    ax[4].set_ylabel('Watt-hrs')
    ax[4].set_title('Cum. power (in 24-hr period)')
    ax[4].set_position([xmin, ymin, xwidth, yheight])
    ax[4].xaxis.set_major_formatter(maj_xticks)
    ax[4].xaxis.set_minor_locator(min_xticks)
    ax[4] = apply_ticklabs(ax[4], lookback_days)
    ax[4] = set_grid_and_labels(ax[4])

        
    ################################
    # Battery voltage
    ax[5].plot(timedata_sun, charge_volt, '*',markersize=2)
    if any(np.isnan(charge_volt)):
        ax[5] = plot_baddata(ax[4], timedata_sun, charge_volt)  
    ax[5].set_xlim([startdate, enddate])
    ax[5].set_xlabel('Time (PDT/UTC-07)')
    ax[5].set_title('Battery voltage/percentage')
    ax[5].set_position([xmin+xwidth+xoffset, ymin, xwidth, yheight])
    ax[5].xaxis.set_major_formatter(maj_xticks)
    ax[5].xaxis.set_minor_locator(min_xticks)
    ax[5] = apply_ticklabs(ax[5], lookback_days)
    ax[5] = set_grid_and_labels(ax[5])
    volt_lims = ax[5].get_yticks()
    voltlabs = ax[5].get_yticklabels()
    volt_newlabs = []
    for lab in voltlabs:
        labtext = lab.get_text()
        labval = lab.get_position()[1]
        labper = convert_batt_volt_to_percent(labval)
        if np.round(labper,1) > 99.9:
            labper = int(np.round(labper,0))
        else:
            labper = np.round(labper,1)
        labtext = labtext + "V /\n"+ str(labper) + '%'
        volt_newlabs.append(labtext)
    ax[5].set_yticklabels(volt_newlabs)
        
        
    if lookback_days == 1:
        fig.suptitle((get_buoyname(buoy_designation) + ' ('+buoy_designation+') Engineering Parameters\nLast ' 
                      + str(lookback_days) + ' day'),
                     x=0.5,y=0.975, horizontalalignment='center', fontsize=12, fontweight='bold')
    else:
        fig.suptitle((get_buoyname(buoy_designation) + ' ('+buoy_designation+') Engineering Parameters\nLast ' 
                      + str(lookback_days) + ' days'),
                     x=0.5,y=0.975, horizontalalignment='center', fontsize=12, fontweight='bold')
        
    return fig


# In[ ]:


def get_recent_controller_diagnostic_vals(sundata, engdata):
    
    # Determine recent battery voltage and percentage
    latest_suntime = pd.Timestamp(sundata.loc[:,'date_local'].max()).to_pydatetime()
    start_ind = np.where([pd.Timestamp(ii)>=(latest_suntime-datetime.timedelta(hours=1)) for ii in sundata['date_local']])[0][0]

    battery_voltage = np.nanmean(sundata.loc[start_ind:,'Vb_f'])
    battery_percent = convert_batt_volt_to_percent(battery_voltage)
    battery_voltage = np.round(battery_voltage,1)
    battery_percent = np.round(battery_percent,1)

    if engdata is not None:
        latest_engtime = pd.Timestamp(engdata.loc[:,'date_local'].max()).to_pydatetime()
        start_ind = np.where([pd.Timestamp(ii)>=(latest_engtime-datetime.timedelta(hours=1)) for ii in engdata['date_local']])[0][0]

        temperature = np.round(np.nanmean(engdata.loc[start_ind:,'temperature']),1)
        humidity = np.round(np.nanmean(engdata.loc[start_ind:,'humidity']),1)
    else:
        latest_engtime = latest_suntime
        temperature = None
        humidity = None
    
    latest_time = max([latest_suntime, latest_engtime])
    latest_time = latest_time.strftime('%Y-%b-%d %H:%M:%S')
    
    return latest_time, battery_voltage, battery_percent, temperature, humidity


# In[ ]:


def save_recent_batteryvoltages(webdir:str, buoyprefix:str, sundata, lookback_days:int = 15):
    
    # Determine recent battery voltage and percentage
    latest_suntime = pd.Timestamp(sundata.loc[:,'date_local'].max().replace(hour=0,minute=0,second=0,microsecond=0)).to_pydatetime()
    start_ind = np.where([pd.Timestamp(ii)>=(latest_suntime-datetime.timedelta(days=lookback_days)) 
                          for ii in sundata['date_local']])[0][0]

    sun_tempdata = sundata.copy().set_index(sundata['date_local']).iloc[start_ind:,:].resample('30T').mean()
    sun_tempdata = sun_tempdata.loc[:,['Vb_f_smooth','battery_percent']].reset_index()
    sun_tempdata['Vb_f_smooth'] = [np.round(ii,2) for ii in sun_tempdata['Vb_f_smooth']]
    sun_tempdata['battery_percent'] = [np.round(ii,2) for ii in sun_tempdata['battery_percent']]

    sun_tempdata = sun_tempdata.rename(columns={'Vb_f_smooth':'voltage','battery_percent': 'percent'})

    battfile = os.path.join(webdir, buoyprefix+'-recentbattery.csv')
    sun_tempdata.to_csv(battfile, index=False)
    
    return


# In[ ]:


def make_recenteng_json(webdir:str, buoyprefix:str, sundata, engdata):
    
    [latest_diagtime, battvolt, 
     battper, temper, humid] = get_recent_controller_diagnostic_vals(sundata, engdata)

    chaba20_recentdiag = {'latest_time': latest_diagtime,
                         'battery_voltage': battvolt,
                         'battery_percent': battper,
                         'temperature': temper,
                         'humidity': humid}

    diagpath = os.path.join(webdir,buoyprefix+'_eng.json')
    with open(diagpath,'w') as jsonfile:
        json.dump(chaba20_recentdiag, jsonfile)
    
    return


# In[ ]:


def update_diagnostic_engineering(buoy_designation: str):
    
    ##################################
    # Get the basepaths
    basepath = get_basedir(buoy_designation.lower())
    buoyprefix = get_buoy_prefix(buoy_designation.lower())
    webdir = get_webdir(buoyprefix.upper())
    
    ##################################
    # Load in the data files, and
    # smooth some of the variables,
    # and calculate power and
    # battery percentages
    sun_df = load_buoy_datafiles(basepath, 'sunsaver', lookback_days=30)
    sun_df = create_smooted_var(sun_df, ['Adc_vb_f','Adc_vl_f','Adc_ic_f','Adc_il_f','Vb_f'], smoothing_minutes=30)

    sun_df['power_load'] = sun_df['Adc_vl_f_smooth'] * sun_df['Adc_il_f_smooth']
    sun_df['power_charge'] = sun_df['Adc_vb_f_smooth'] * sun_df['Adc_ic_f_smooth']
    sun_df['battery_percent'] = convert_batt_volt_to_percent(sun_df['Adc_vb_f_smooth'])

    engineer_df = load_buoy_datafiles(basepath, 'engineering', lookback_days=30)
    if engineer_df is not None:
        engineer_df = create_smooted_var(engineer_df, ['temperature','humidity'], smoothing_minutes=30)
    
    
    
    ###################################
    # Make the static engineering pages
    lookback_days = [1,10,30]
    for lookback in lookback_days:
        fig = make_chaba20_eng_plot(buoy_designation, sun_df, engineer_df, lookback)
        filepath = os.path.join(webdir, buoyprefix+'_eng_'+str(int(lookback))+'days.png')
        fig.savefig(filepath, format='png', dpi=160)
        plt.close(fig)
        
        
    ###################################
    # Save the recent battery voltages
    save_recent_batteryvoltages(webdir, buoyprefix, sun_df, lookback_days=15)
    
    # Save the recent engineering data json
    make_recenteng_json(webdir, buoyprefix, sun_df, engineer_df)
    
    # Update the file latest contact
    update_latestcontact(buoyprefix, buoy_designation, webdir)
    
    return

def main():
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hu:b:", 
                                   ["help", "buoy="])
    except Exception as inst:
        # print help information and exit:
        print('Error in getting options: ' + str(inst)) # will print something like "option -a not recognized"
        sys.exit(2)
        
        
    # step through command-line options and their arguments
    buoyName = False
    
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
        else:
            assert False, "unhandled option"
    
    
    ##    
        
    # If the buoy designation was given, ensure that the buoy designation is in uppercase,
    # and that it matches one of the valid buoy designations
    valid_buoys = ['CHABA', 'NEMO']
    if buoyName:
        buoy_designation = buoy_designation.upper()
        if buoy_designation in valid_buoys:
            print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S') 
                  + ' - Begin updating diagnostic information for buoy:' + buoy_designation)
        else:
            print('Invalid buoy designation. Check name, and retry.')
            sys.exit(2)
    else:
        print('No buoy designation given. Try again with options: "-b *buoy_designation*"')
        sys.exit(2)
        

    ##################################################################
    # Begin processing new data for a given buoy
    print('Update the diagnostics information: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S') + '\n')
    try:
        update_diagnostic_engineering(buoy_designation)
        print('Diagnostic information updated: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'))
    except Exception as e:
        print('Error occured in updating the diagnostic information: ' + datetime.datetime.now().strftime('%b-%d %H:%M:%S'))
        print(e)
        traceback.print_exc()
    sys.exit(0)


# In[ ]:


if __name__ == "__main__":
    main()

