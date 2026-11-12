#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from erddapy import ERDDAP
import json

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patheffects

import cartopy
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs

import cmocean

import numpy as np
import pandas as pd
import xarray
import datetime

import os
import re
import sys
import getopt
import argparse
import gc

import time
import paramiko

import nemo_general_functions as nemo


# In[ ]:


def great_circle_calc(lat, lon):
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    R = 6370*1e3 # Radius of earth, in m
    
    dlon = lon_rad[1:] - lon_rad[:-1]
    
    fac1 = np.cos(lat_rad[1:]) * np.cos(lat_rad[:-1]) * np.cos(dlon)
    fac2 = np.sin(lat_rad[1:]) * np.sin(lat_rad[:-1])
    
    dist = R * np.arccos(fac1 + fac2)
    
    # Returns distance from the next point, in meters
    return dist


# In[ ]:


def load_gpsfile(time_lookback=3):

    _, _, nemo_instruments_dir = nemo.get_mpkpaths('chaba')
    gps_dir = os.path.join(nemo_instruments_dir, 'gps')
    if not os.path.exists(gps_dir):
        raise FileNotFoundError('Unable to find gps instrument directory: ' + gps_dir)

    gps_files = sorted([file for file in os.listdir(gps_dir)
                        if file.lower().endswith('.csv')])
    if len(gps_files) == 0:
        raise FileNotFoundError('No gps csv files found in: ' + gps_dir)

    lookback_start = datetime.datetime.now() - datetime.timedelta(days=time_lookback)
    min_filedate = lookback_start.date() - datetime.timedelta(days=1)
    gpsfile_dates = []
    for gps_file in gps_files:
        date_match = re.search(r'(\d{8})', gps_file)
        if date_match is not None:
            gpsfile_dates.append(datetime.datetime.strptime(date_match.group(1), '%Y%m%d').date())
        else:
            gpsfile_dates.append(None)

    selected_files = [gps_files[ii] for ii in range(0, len(gps_files))
                      if gpsfile_dates[ii] is None or gpsfile_dates[ii] >= min_filedate]
    
    send_emailflag = False
    if len(selected_files) == 0:
        selected_files = [gps_files[-1]]

        send_emailflag = True

    if send_emailflag:
        print('Sending email to report missing gps')
        latest_gpsfile = gps_files[-1]
        latest_gpsdate = re.search(r'(\d{8})', latest_gpsfile)
        if latest_gpsdate is not None:
            latest_gpsdate = datetime.datetime.strptime(latest_gpsdate.group(1), '%Y%m%d').date()
        msgtxt = 'No GPS files found within the lookback period. Most recent GPS file: ' + gps_files[-1]
        if latest_gpsdate is not None:
            msgtxt += ' (Date: ' + str(latest_gpsdate) + ')'
        subj = 'Chaba Realtime - Missing GPS Files'
        email_targets = "setht1@uw.edu"
        send_emailreport(msgtxt, subj, attachment_loc=None,
                         toaddr=email_targets)

    gps = None
    for gps_file in selected_files:
        gps_filedf = pd.read_csv(os.path.join(gps_dir, gps_file))
        if gps is None:
            gps = gps_filedf.copy()
        else:
            gps = pd.concat([gps, gps_filedf], ignore_index=True)

    if {'lon', 'lat', 'timestamp'}.issubset(gps.columns):
        gps_lon = gps['lon'].to_numpy(dtype=float).squeeze()
        gps_lat = gps['lat'].to_numpy(dtype=float).squeeze()
        gps_hdop = gps['hdop'].to_numpy(dtype=float).squeeze()
        gps_timesecs = (gps['timestamp'].to_numpy(dtype=float).squeeze())/1e9
    elif {'longitude (degrees)', 'latitude (degrees)', 'time (seconds)'}.issubset(gps.columns):
        gps_lon = gps['longitude (degrees)'].to_numpy(dtype=float).squeeze()
        gps_lat = gps['latitude (degrees)'].to_numpy(dtype=float).squeeze()
        gps_timesecs = gps['time (seconds)'].to_numpy(dtype=float).squeeze()
    else:
        raise ValueError('Unable to parse gps csv columns in files from: ' + gps_dir)

    sort_inds = np.argsort(gps_timesecs)
    gps_lon = gps_lon[sort_inds]
    gps_lat = gps_lat[sort_inds]
    gps_timesecs = gps_timesecs[sort_inds]
    gps_hdop = gps_hdop[sort_inds]

    # The HDOP gives the horizontal dilution of precision, which indicates the quality of the GPS fix.
    # Lower HDOP values correspond to higher accuracy, while higher values indicate lower accuracy.
    # Filter out GPS points with poor HDOP values (e.g., HDOP > 5)
    gps_lon = gps_lon[gps_hdop <= 5]
    gps_lat = gps_lat[gps_hdop <= 5]
    gps_timesecs = gps_timesecs[gps_hdop <= 5]
    gps_hdop = gps_hdop[gps_hdop <= 5]


    # Convert GPS timestamps from seconds since epoch to datetime objects
    gps_time = np.array([datetime.datetime(1970, 1, 1) + 
                         datetime.timedelta(seconds=float(ii))
                         for ii in gps_timesecs])


    # Determine the starting index based on the lookback start time
    time_startinds = np.where(gps_time > lookback_start)[0]
    if len(time_startinds) > 0:
        time_startind = time_startinds[0]
    else:
        time_startind = 0

    time_secrange = gps_timesecs[-1] - gps_timesecs[time_startind]
    if time_secrange <= 0:
        gps_timescale = np.array([1 for ii in range(0, len(gps_timesecs))], dtype=float)
    else:
        gps_timescale = (gps_timesecs - gps_timesecs[time_startind])*(30/time_secrange)

    return (gps_lon[time_startind:], gps_lat[time_startind:],
            gps_time[time_startind:], gps_timescale[time_startind:], gps_hdop[time_startind:])


# In[ ]:


def check_buoy_and_plot(ChaBa_lon, ChaBa_lat, testdist=150):
    
    # Extract the gps data
    gps_lon, gps_lat, gps_time, gps_timescale, gps_hdop = load_gpsfile()
    
    # Calculate distance from target deployment
    gps_dists = np.zeros(len(gps_lon))
    for ii in range(0,len(gps_lon)):
        gps_dists[ii] = great_circle_calc([gps_lat[ii], ChaBa_lat],
                                          [gps_lon[ii], ChaBa_lon])[0]
    
    
    # Define the colors for the GPS scatter plot, based upon time
    colors = matplotlib.dates.date2num(gps_time)
    norm = matplotlib.colors.Normalize(vmin=colors.min(), 
                                       vmax=colors.max())

    # Choose a number of ticks to be shown on the colorbar
    N = 6
    ticks = np.linspace(colors.min(), colors.max(), N)
    # Convert the ticks back to date format
    labels = [matplotlib.dates.num2date(t).strftime("%b-%d\n%H:%M") 
              for t in ticks]

    gps_lonmin = np.nanmin(gps_lon)
    gps_lonmax = np.nanmax(gps_lon)
    gps_latmin = np.nanmin(gps_lat)
    gps_latmax = np.nanmax(gps_lat)
    
    # Define boundary conditions for plotting region of interest
    minLon = np.min([np.floor(gps_lonmin*10)/10, -125.5])
    maxLon = np.max([np.ceil(gps_lonmax*10)/10, -123.75])
    minLat = np.min([np.floor(gps_latmin*10)/10, 47])
    maxLat = np.max([np.ceil(gps_latmax*10)/10, 48.9])


    # Define ChaBa buoy location
    lonrad = testdist*1e-3/(111*np.cos(np.radians(ChaBa_lat)))
    lon_range = lonrad * 2 * 4
    latrad = testdist*1e-3/111
    lat_range = latrad * 2 * 4


    # Create bounds & constraints variables, used to
    # facilitate erddap data pulls, and plotting
    bounds = {'main': [minLon, maxLon, minLat, maxLat],
              'chaba': [np.min([np.floor(gps_lonmin*10000)/10000, 
                                np.floor(10000*(ChaBa_lon-(lon_range/2)))/10000]), 
                        np.max([np.ceil(gps_lonmax*10000)/10000,
                                np.ceil(10000*(ChaBa_lon+(lon_range/2)))/10000]), 
                        np.min([np.floor(gps_latmin*10000)/10000,
                                np.floor(10000*(ChaBa_lat-(lat_range/2)))/10000]), 
                        np.max([np.ceil(gps_latmax*10000)/10000,
                                np.ceil(10000*(ChaBa_lat+(lat_range/2)))/10000])]}

    constraints = {
        "longitude<=": str(maxLon+0.5),
        "longitude>=": str(minLon-0.5),
        "latitude<=": str(maxLat+0.5),
        "latitude>=": str(minLat-0.5)
    }

    # Define the url of the ERDDAP
    erddap_url = 'https://coastwatch.pfeg.noaa.gov/erddap/'


    #########################
    # Get topopgraphic info

    # Identify the dataset
    topo_dataset = 'etopo180'

    # Get the data
    print(datetime.datetime.now().strftime('%Y-%b-%d %H:%M') + ': Downloading topographic data...')
    topo_snapshot = nemo.get_erddap_data(erddap_url, topo_dataset, 
                                          data_protocol='griddap', 
                                          variables=None, constraints=constraints)
    
    
    # Define ChaBa buoy location

    # La Push location
    lapush_lat = 47.908619
    lapush_lon = -124.632486
    
    
    
    # Prepare the data for plotting
    
    plot_colormap = matplotlib.cm.jet
    m_proj = ccrs.UTM(zone=10, southern_hemisphere=False)
    m_transform = ccrs.PlateCarree()

    gps_locs_transformed = m_proj.transform_points(m_transform,
                            np.array([gps_lon[ii] for ii in range(0,len(gps_lon))]),
                            np.array([gps_lat[ii] for ii in range(0,len(gps_lat))]))
    gps_lontrans = [ii[0] for ii in gps_locs_transformed]
    gps_lattrans = [ii[1] for ii in gps_locs_transformed]
    
    
    
    ###############################################################
    # Find the circles a given distance away from the target ChaBa
    # deployment, and transform them to the correct projection
    
    degs = np.arange(0,365,2.5)
    
    # Define the first circle
    circ_xrange = [ChaBa_lon + lonrad*np.cos(np.radians(ii)) for ii in degs]
    circ_yrange = [ChaBa_lat + latrad*np.sin(np.radians(ii)) for ii in degs]
    circ_locs_transformed = m_proj.transform_points(m_transform,
                            np.array([circ_xrange[ii] for ii in range(0,len(circ_xrange))]),
                            np.array([circ_yrange[ii] for ii in range(0,len(circ_xrange))]))
    circ_xtrans = [ii[0] for ii in circ_locs_transformed]
    circ_ytrans = [ii[1] for ii in circ_locs_transformed]

    # Define the second circle
    circ2_xrange = [ChaBa_lon + 2*lonrad*np.cos(np.radians(ii)) for ii in degs]
    circ2_yrange = [ChaBa_lat + 2*latrad*np.sin(np.radians(ii)) for ii in degs]
    circ2_locs_transformed = m_proj.transform_points(m_transform,
                            np.array([circ2_xrange[ii] for ii in range(0,len(circ2_xrange))]),
                            np.array([circ2_yrange[ii] for ii in range(0,len(circ2_xrange))]))
    circ2_xtrans = [ii[0] for ii in circ2_locs_transformed]
    circ2_ytrans = [ii[1] for ii in circ2_locs_transformed]

    
    
    
    ################################################
    # Begin plotting
    ################################################

    fig, ax = plt.subplots(figsize=(7.5, 5.75), dpi=160)
    fig.delaxes(ax)
    ax1 = plt.subplot(311, projection=m_proj)
    ax2 = plt.subplot(312, projection=m_proj)
    ax3 = plt.subplot(313)



    #####################################################
    # Main plot w/ topography

    ax1.set_extent(bounds['main'])
    ax1.set_facecolor('grey')

    # Plot the gps positions (using transformed locs)
    ax1.scatter(gps_lontrans, gps_lattrans, s=gps_timescale, 
                c=colors,zorder=10, cmap=plot_colormap)

    # Plot the bathymetric contours, and label
    cs1 = topo_snapshot['altitude'].plot.contour(ax=ax1,
                                           transform=m_transform,
                                           levels=[-2000,-1000,-200,-100],
                                           linewidths=[0.1,0.25,0.4,0.5], colors='white',
                                           vmin=-2000,vmax=0)
    manual_locations = [
        (-125.8, 46.9), (-125.5, 44.5), 
        (-124.8, 45.5), (-125.25, 47.5),
        (-124.3, 45.2), (-124.75, 46.8), 
        (-124.25, 45.6), (-124.8, 47.7)]
    man_locs_transformed = m_proj.transform_points(m_transform,
                            np.array([ii[0] for ii in manual_locations]),
                            np.array([ii[1] for ii in manual_locations]))
    man1_locs = [(ii[0], ii[1]) for ii in man_locs_transformed]
    ax1.clabel(cs1, fontsize=4, inline=True, manual=man1_locs)

    # Mask out the land in the plot
    ax1.add_feature(cartopy.feature.LAND, zorder=100, 
                    edgecolor='goldenrod', linewidth = 0.5, facecolor='darkgoldenrod')


    # Plot Location Data

    # ChaBa Buoy
    chaba_label = 'ESP and\nNANOOS\nmoorings'
    ax1.scatter(ChaBa_lon, ChaBa_lat, 
                color='red', edgecolor='black', zorder=10,
                s=75, marker='*', linewidths=0.5, transform = m_transform,
                label=chaba_label)

    # La Push
    ax1.scatter(lapush_lon, lapush_lat, marker='d', 
                color='black', edgecolor='silver', 
                s=15, linewidths=0.5, transform = m_transform, zorder=101)
    ax1.text(lapush_lon+0.07, lapush_lat, 'La Push', 
             color='white', size=6, fontweight='bold',ha='left', va='center',
             transform=m_transform,zorder=101,
             path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")])


    # Ensure the plotting transform is applied, and add gridlines
    ax1.set_transform(m_proj)
    gl1 = ax1.gridlines(crs = m_transform, draw_labels=True,
                       linewidth=0.5, color='darkgrey', alpha = 0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.ylocator = matplotlib.ticker.FixedLocator([47.2, 47.6, 48.0, 48.4, 48.8])
    gl1.xlocator = matplotlib.ticker.FixedLocator([-125.5, -125.25, -125, -124.75, -124.5, -124.25,
                                                   -124, -123.75, -123.5])
    gl1.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl1.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl1.xlabel_style = {'size': 8, 'color': 'k'}
    gl1.ylabel_style = {'size': 8, 'color': 'k'}






    #####################################################
    # Zoomed in plot

    ax2.set_extent(bounds['chaba'])

    ax2.set_facecolor('grey')

    # Likely not needed, but add mask out the land in the plot
    ax2.add_feature(cartopy.feature.LAND, zorder=100, 
                    edgecolor='goldenrod', linewidth = 0.5, facecolor='darkgoldenrod')

    # Plot target location of the ChaBa buoy
    ax2.scatter(ChaBa_lon, ChaBa_lat, 
                color='red', edgecolor='black', zorder=15,
                s=30, marker='*', linewidths=0.5, transform = m_transform,
                label='ChaBa Target')


    # Plot the geofence radius circles
    ax2.plot(circ_xtrans, circ_ytrans,'--w', 
             label='150m radius')
    ax2.plot(circ2_xtrans, circ2_ytrans,'--w', 
             linewidth=0.5, label='300m radius')

    # Add the GPS locations
    cb=ax2.scatter(gps_lontrans, gps_lattrans, s=gps_timescale, 
                   c=colors,zorder=10, cmap=plot_colormap)
    ax2.plot(gps_lontrans, gps_lattrans,'--k',
             linewidth=0.5,alpha=0.5,zorder=2)
    ax2.legend(loc='lower left',fontsize=8,
               facecolor='lightblue')

    # Ensure the plotting transform is applied, and add gridlines
    ax2.set_transform(m_proj)
    gl2 = ax2.gridlines(crs = m_transform, draw_labels=True,
                       linewidth=0.5, color='darkgrey', alpha = 0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.ylocator = matplotlib.ticker.FixedLocator(np.arange(np.floor(1000*bounds['chaba'][2])/1000,
                                                            np.ceil(1000*bounds['chaba'][3])/1000+0.003,0.003).squeeze().tolist())
    gl2.xlocator = matplotlib.ticker.FixedLocator(np.arange(np.floor(1000*bounds['chaba'][0])/1000,
                                                            np.ceil(1000*bounds['chaba'][1])/1000+0.003,0.003).squeeze().tolist())
    gl2.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl2.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl2.xlabel_style = {'size': 8, 'color': 'k'}
    gl2.ylabel_style = {'size': 8, 'color': 'k', 'rotation': 25}
    
    
    
    # Time series of distance from deployment
    ax3.plot(gps_time, gps_dists, '*--', markersize=2, linewidth=0.5)
    ax3.plot([gps_time[0], gps_time[-1]], [150, 150], '--', linewidth=0.5)
    ax3.plot([gps_time[0], gps_time[-1]], [300, 300], '--', linewidth=1.5)
    ax3.set_xlim([gps_time[0], gps_time[-1]])
    ax3.set_ylim([0, max([350,np.ceil(max(gps_dists)/10)*10])])
    ax3.set_ylabel('Distance from\ndeployment location\n(meters)', 
                   fontsize=8)
    ax3.yaxis.set_tick_params(labelsize=8)
    ax3.yaxis.set_label_position('right')
    ax3.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,12]))
    ax3.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=[6,18]))
    ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b-%#d/%H'))
    ax3.xaxis.set_tick_params(labelsize=8)
    ax3.set_xlabel('Date/Hour (UTC)',fontsize=8)
    ax3.grid(which='both', axis='x',linewidth=0.25, alpha=0.5)



    ###############################################
    # Fix figure plot formatting

    # Set axes positions
    fullheight = 0.5875
    halfheight = 0.5875
    shortheight = 0.2
    xmin = 0.0625
    ymin = 0.075
    xoffset = 0.0875
    yoffset = 0.05
    cbar_width = 0.0125
    cbar_offset = 0.01

    # Determine the ax1 scaling
    ax1_pos = ax1.get_position()
    ax1_width = ax1_pos.x1-ax1_pos.x0
    ax1_height = ax1_pos.y1-ax1_pos.y0
    ax1_ratio = ax1_width/ax1_height
    ax1_height = fullheight
    ax1_width = ax1_ratio*ax1_height

    # Determine the ax2 scaling
    ax2_pos = ax2.get_position()
    ax2_width = ax2_pos.x1-ax2_pos.x0
    ax2_height = ax2_pos.y1-ax2_pos.y0
    ax2_ratio = ax2_width/ax2_height
    ax2_height = fullheight
    ax2_width = ax2_ratio*ax2_height
    

    # Set the axes positions
    ax1.set_position([xmin, ymin+yoffset+shortheight, 
                      ax1_width, ax1_height])
    ax2.set_position([xmin+ax1_width+xoffset, ymin+yoffset+shortheight, 
                      ax2_width, ax2_height])
    ax3.set_position([xmin, ymin, ax1_width+xoffset+ax2_width, shortheight])


    # Add the colorbar
    cbar_ax = fig.add_axes([xmin+ax1_width+ax2_width+xoffset+cbar_offset,
                            ymin+yoffset+shortheight, cbar_width, ax2_height])
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=plot_colormap),
        ticks=ticks,
        cax=cbar_ax,
        ax=ax2,
        label="Date (UTC)"
    )
    cbar.ax.set_yticklabels(labels)
    cbar.ax.tick_params(labelsize=8)


    # Add a figure title
    title_str = ('ChaBa Position - Last 3 days\n' +
                 gps_time[0].strftime('%Y-%b-%d %H:%M') + ' - ' +
                 gps_time[-1].strftime('%Y-%b-%d %H:%M'))
    fig.text(0.5, ymin+shortheight+yoffset+fullheight+0.075, 
             title_str, fontsize=12,
             ha='center',va='top')


    # Save the figure
    if '/hdd0/orca/' in os.getcwd():
        savedir = '/var/www/htdocs/figures/'
    else:
        savedir = 'C:/Users/APLUser/NEMO_Data/'
    figname = 'ChaBa_LatestPosition.png'
    fig.savefig(os.path.join(savedir,figname), 
                format='png', dpi=160)
    
    
    
    #############################################
    # Lastly, check the buoy position, 
    # and if necessary, send an email to the team
    check_buoy_position(gps_lon, gps_lat, gps_time, gps_hdop, 
                        ChaBa_lon, ChaBa_lat, 
                        attachment_file=os.path.join(savedir,figname))


# In[ ]:


def check_buoy_position(gps_lon, gps_lat, gps_time, gps_hdop,
                        ChaBa_lon, ChaBa_lat, 
                        attachment_file=None, target_dist=150):
    
    #email_targets = 'nwem_team@uw.edu'
    email_targets = 'setht1@uw.edu'
    
    gps_dists = np.zeros(len(gps_lon))
    for ii in range(0,len(gps_lon)):
        gps_dists[ii] = great_circle_calc([gps_lat[ii], ChaBa_lat],
                                          [gps_lon[ii], ChaBa_lon])[0]
        
    
    send_message = False
    msgtxt = ""
    subj = ""    
    checktime = datetime.datetime.now() - datetime.timedelta(hours = 3)
    
    if any(gps_dists > 2*target_dist):
        
        faroutside_inds = np.where(gps_dists > 2*target_dist)[0]
        if len(faroutside_inds) > 0:
            good_faroutside_inds = faroutside_inds[gps_hdop[faroutside_inds] < 2]
        else:
            good_outside_inds = []
        
        outside_inds = np.where(gps_dists > target_dist)[0]
        if len(outside_inds) > 0:
            good_outside_inds = outside_inds[gps_hdop[outside_inds] < 2]
        else:
            good_outside_inds = []

        if any(gps_hdop[faroutside_inds] < 5):
            send_message = True
        elif any(gps_hdop[outside_inds] < 2):
            send_message = True
        
        if gps_dists[-1] > 2*target_dist:
            send_message = True
        elif ((len(faroutside_inds) > 3 and gps_time[faroutside_inds[-1]] > checktime)
              or
              (len(outside_inds) > 6 and gps_time[outside_inds[-1] > checktime])):
            send_message = True
            
        msgtxt = ("""
        According to recent updates, CHABA has broken free from its mooring, and is more than """ + 
                  str(int(target_dist)) + """ meters from is deployed location. 
        \n\nPlease gather recovery materials and setup CHABA recovery ops IMMEDIATELY.
        
        \n\nFor more information, please view attachment, or view the file online at:\n
        https://nwem.apl.washington.edu/figures/ChaBa_LatestPosition.png""")
        
        subj = 'ALERT: CHABA HAS BROKEN FREE (>300 meters from target)'
        
    elif any(gps_dists > target_dist):
        
        outside_inds = np.where(gps_dists > target_dist)[0]
        if len(outside_inds) > 0:
            good_outside_inds = outside_inds[gps_hdop[outside_inds] < 2]
        else:
            good_outside_inds = []
        
        if (gps_dists[-1] > target_dist) and (gps_hdop[-1] < 5):
            send_message = True
        elif len(good_outside_inds) > 3 and gps_time[good_outside_inds[-1]] > checktime:
            send_message = True
        
        msgtxt = ("""
        According to recent updates, CHABA has likely broken free from its mooring, and is more than """ + 
                  str(int(target_dist)) + """ meters from is deployed location. 
        \n\nPlease gather recovery materials and setup CHABA recovery ops IMMEDIATELY.
        
        \n\nFor more information, please view attachment, or view the file online at:\n
        https://nwem.apl.washington.edu/figures/ChaBa_LatestPosition.png""")
        
        subj = 'ALERT: CHABA HAS LIKELY BROKEN FREE (>150 meters from target)'
    
    
    
    if send_message:
        print('Sending email report')
        send_emailreport(msgtxt, subj, 
                         attachment_loc=attachment_file,
                         toaddr=email_targets)
        
        
    
    return


# In[ ]:


def send_emailreport(msgtxt, subj, 
                     attachment_loc=None,
                     fromaddr='nwemdata@uw.edu', toaddr='nwemdata@uw.edu',
                     login='nwemdata@gamail.uw.edu', passwd='nwemDa8&', 
                     smtpserver='smtp.uw.edu:587'):
    
    # Function: send_emailreport
    # This function is used to send automatic emails to the data manager
    
    
    import smtplib
    from email.message import EmailMessage
    
    # Create the email message
    msg = EmailMessage()
    msg['Subject'] = subj
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg.set_content(msgtxt, subtype='html')
    
    # Attach a file, if necessary
    if attachment_loc is not None:
        with open(attachment_loc, 'rb') as f:
            file_data = f.read()
        msg.add_attachment(file_data, maintype='image', subtype='png', 
                           filename='ChaBa_LatestPosition.png')
    
    # Login to the email server
    server = smtplib.SMTP(smtpserver)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(login, passwd)
    
    
    # Send the message
    problems = server.send_message(msg)
    
    # Logout of the email server
    server.quit()
    
    return


# In[ ]:

def load_chaba_deployment_info(deployment_name):
    basedir = os.path.dirname(os.path.abspath(__file__))
    deployment_info = nemo.get_deployment_info('chaba', basedir)
    deployments = deployment_info.get('deployments', {})
    if deployment_name not in deployments:
        raise KeyError('Unable to find deployment "' + deployment_name +
                       '" in chaba deployment info.')
    ChaBa_lon = deployments[deployment_name]['longitude']
    ChaBa_lat = deployments[deployment_name]['latitude']
    return ChaBa_lon, ChaBa_lat

def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Create ChaBa buoy position plots and geofence alerts.',
        epilog='Example: python chaba_buoy_position.py --deployment chaba26b',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--deployment',
        type=str,
        required=True,
        help='ChaBa deployment name used to load target latitude/longitude '
             '(e.g., chaba26b).'
    )
    parser.add_argument(
        '--geofence-radius',
        type=float,
        default=150,
        help='Allowable distance from target location in meters.'
    )
    return parser.parse_args()

def main():

    args = parse_inputs()

    # Define the position of the ChaBa buoy
    
    # Load in the buoy position from the Chaba deployment info json
    ChaBa_lon, ChaBa_lat = load_chaba_deployment_info(
        deployment_name=args.deployment)
    print('Loaded ChaBa buoy position: lon =', ChaBa_lon, 'lat =', ChaBa_lat)
    
    # Allowable distance from the target buoy location,
    # given in meters
    geofence_radius = args.geofence_radius
    print('Using geofence radius:', geofence_radius, 'meters')
    
    check_buoy_and_plot(ChaBa_lon, ChaBa_lat, geofence_radius)
    
    # 
    sys.exit(0)


if __name__ == "__main__":
    main()
