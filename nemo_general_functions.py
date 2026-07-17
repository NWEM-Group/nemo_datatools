
#############################################
# Buoy and instrument information functions #
#############################################


def get_datalocations(buoy_name, deployment_name, inst_type, status=None):

    import os
    import json
    
    if inst_type not in ['ctd', 'adcp']:
        print('Invalid instrument type:', inst_type)
        return None, None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pathdirs_path = os.path.join(script_dir, 'info_jsons', 'pathdirs.json')
    
    with open(pathdirs_path, 'r') as f:
        pathdirs_info = json.load(f)
    
    datadir = pathdirs_info.get('data_basedir')
    savedir = pathdirs_info.get('data_savedir')


    if datadir is None or savedir is None:
        raise KeyError("'data_basedir' and 'data_savedir' are required in info_jsons/pathdirs.json")


    if status is not None:
        status_norm = str(status).strip().lower()
        if status_norm == 'recovered':
                if buoy_name == 'chaba':
                    datadir = os.path.join(datadir, 'chaba', deployment_name)
                    savedir = os.path.join(savedir, 'chaba', deployment_name, inst_type)
                elif buoy_name == 'nemoss':
                    datadir = os.path.join(datadir, 'nemoss', deployment_name)
                    savedir = os.path.join(savedir, 'nemoss', deployment_name, inst_type)

                if not(os.path.exists(savedir)):
                    print(f"Creating directory for saving data: {savedir}")
                    os.makedirs(savedir, exist_ok=True)




    
    return datadir, savedir

##################################
# netCDF Functions               #
##################################

# CTD Function
def make_ctd_netCDF(nemo_info, savedir, sbe_df, sbe_qcdf=None):

    from netCDF4 import Dataset
    import os
    import numpy as np
    import datetime
    
    #######
    # Extract out the lander information
    
    nemo_name = nemo_info['BuoyName']
    nemo_title = nemo_info['BuoyTitle']
    info_url = nemo_info['info_url']
    deployment_name = nemo_info['DeploymentName']
    lat = nemo_info['Latitude']
    lon = nemo_info['Longitude']
    depth = nemo_info['Depth']
    institution_info = nemo_info['institution_info']
    instrument_info = nemo_info['InstrumentInfo']
    
    fillVal = -555
    NT = len(sbe_df)
    
    ##################################################
    # Save as a netCDF
    ##################################################
    
    ctd_ncfile = nemo_name.lower() + '_' + deployment_name.lower() + '_ctd_' + instrument_info.get('model', '').lower() + '_' + instrument_info.get('sn', '').lower() + '.nc'
    dataset = Dataset(os.path.join(savedir,ctd_ncfile),
                      'w',format='NETCDF4')
    
    try:
    
        dataset.title = nemo_title + ' - CTD Data'
        dataset.description = 'Water property data from a ' + instrument_info.get('model', '') + ' on the ' + nemo_name + ' buoy during the ' + deployment_name + ' deployment. Data processed by NWEM.'
        dataset.history = "File created on " + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
        dataset.conventions = 'CF-1.6, ACDD-1.3, IOOS-1.2'
        dataset.designation = nemo_name
        dataset.latitude = str(np.round(lat,6)) + 'N'
        dataset.longitude = str(np.round(lon,6)) + 'E'
        dataset.buoy_depth = str(depth) + 'm'
        dataset.insturment_depth = str(instrument_info.get('depth', '')) + 'm'
        dataset.infoUrl = info_url
        dataset.institution = institution_info.get('institution', '')

        ## These are fields required by NCEI by ingestion to the sensor map
        dataset.creator_name = institution_info.get('name', '')
        dataset.creator_institution = institution_info.get('institution', '')
        dataset.creator_url = institution_info.get('url', '')
        dataset.creator_country = institution_info.get('country', '')
        dataset.creator_sector = institution_info.get('sector', '')
        dataset.creator_type = institution_info.get('type', '')

        dataset.contributor_name = institution_info.get('name', '') + ', John Mickett, NANOOS, NWEM'
        dataset.contributor_role_vocabulary = 'https://vocab.nerc.ac.uk/collection/G04/current/'
        dataset.contributor_role =  'owner, principalInvestigator, funder, publisher'
        dataset.contributor_role_url = institution_info.get('url', '') + ', --, https://www.nanoos.org/, https://nwem.apl.washington.edu/'

        dataset.publisher_name = 'NorthWest Environmental Moorings (NWEM) Group'
        dataset.publisher_email = 'setht1@uw.edu'
        dataset.publisher_institution = 'University of Washington - Applied Physics Laboratory'
        dataset.publisher_type = 'group'
        dataset.publisher_url = 'https://nwem.apl.washington.edu/'
        dataset.publisher_country = 'United States'

        dataset.author = 'Seth Travis'
        dataset.contact = 'setht1@uw.edu'
        dataset._FillValue = fillVal
        dataset.cdm_data_type = 'timeSeries'
        dataset.cdm_timeseries_variables = 'buoy_name,latitude,longitude'

        dataset.mooring_diagram_baseurl = nemo_info.get('mooring_diagram_base', '')
        dataset.mooring_diagram_url = nemo_info.get('mooring_diagram_base', '') + deployment_name + '_mooring_diagram_final.pdf'
        dataset.time_drift_seconds = instrument_info.get('time_drift_seconds', '')
        dataset.time_drift_description = 'Number of seconds that the instrument clock drifted from UTC time during the deployment. This is calculated by comparing the instrument clock to the GPS time at the start and end of the deployment. Positive values indicate that the instrument clock was ahead of UTC time, while negative values indicate that the instrument clock was behind UTC time.'


        #############################################
        # Create coordinate and identifying variables
        
        dataset.createDimension('time',NT)
        
        time = dataset.createVariable('time','f8',('time',))
        time.long_name = 'time'
        time.description = 'time of sampling'
        time.units = 'seconds since 1970-01-01 00:00:00'
        time.timezone = 'UTC'
        time.calendar = 'gregorian'

        buoy_name = dataset.createVariable('buoy_name','S9',('time',))
        buoy_name.long_name = 'buoy_name'
        buoy_name.description = 'buoy description name'
        buoy_name.cf_role = 'timeseries_id'

        deployment = dataset.createVariable('deployment_name','S32',('time',))
        deployment.long_name = 'deployment_name'
        deployment.description = 'deployment identifier name'

        latitude = dataset.createVariable('latitude','f8',('time',))
        latitude.long_name = 'latitude'
        latitude.units = 'degrees North'

        longitude = dataset.createVariable('longitude','f8',('time',))
        longitude.long_name = 'longitude'
        longitude.units = 'degrees East'

        ################################
        # Assign variables


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

        inst_time = dataset.createVariable('instrument_time','f8',('time',))
        inst_time.long_name = 'instrument_time'
        inst_time.description = 'time of sampling, taken from the instrument'
        inst_time.units = 'seconds since 1970-01-01 00:00:00'
        inst_time.timezone = 'UTC'
        inst_time.calendar = 'gregorian'  

        recordnum = dataset.createVariable('sample_number','i4',('time',))
        recordnum.long_name = 'deployment_sample_record_number'
        recordnum.description = 'the record number of the sample taken for a given deployment'
        recordnum.units = '--'
        recordnum.missing_value = fillVal   

        pres = dataset.createVariable('sea_water_pressure','f8',('time',))
        pres.long_name = 'sea_water_pressure'
        pres.standard_name = 'sea_water_pressure'
        pres.description = 'Pressure exerted by overlying water, excluding air pressure.'
        pres.units = 'dbar'
        pres.missing_value = fillVal
        pres.ancillary_variables = make_ancvar_str('sea_water_pressure')

        depth = dataset.createVariable('depth','f8',('time',))
        depth.long_name = 'depth'
        depth.standard_name = 'depth'
        depth.description = 'Depth of the water column at the measurement location. If pressure is measured, depth is calculated from pressure using the UNESCO 1983 algorithm. If pressure is not measured, depth is calculated assuming hydrostatic pressure. If appropriate variables are not present, depth is given as the deployed instrument depth, and does not account for variation over time.'
        depth.units = 'm'
        depth.missing_value = fillVal

        temp = dataset.createVariable('sea_water_temperature','f8',('time',))
        temp.long_name = 'sea_water_temperature'
        temp.standard_name = 'sea_water_temperature'
        temp.description = 'In-situ temperature of water (T90 scale)'
        temp.units = 'degrees C'
        temp.missing_value = fillVal 
        temp.ancillary_variables = make_ancvar_str('sea_water_temperature')

        cond = dataset.createVariable('sea_water_electrical_conductivity','f8',('time',))
        cond.long_name = 'sea_water_electrical_conductivity'
        cond.standard_name = 'sea_water_electrical_conductivity'
        cond.description = 'Ability to pass electrical current. In water, it is a proxy from which to derive salinity.'
        cond.units = 'S/m'
        cond.missing_value = fillVal
        cond.ancillary_variables = make_ancvar_str('sea_water_electrical_conductivity')

        salt = dataset.createVariable('sea_water_practical_salinity','f8',('time',))
        salt.long_name = 'sea_water_practical_salinity'
        salt.standard_name = 'sea_water_practical_salinity'
        salt.description = 'Salinity is to the salt content of a water sample or body of water. The measure of salt content of a water sample follows UNESCO standards known as the Practical Salinity Scale (PSS) as the conductivity ratio of a sea water sample to a standard KCl solution. PSS is a ratio and has no units.'
        salt.units = 'psu'
        salt.missing_value = fillVal
        salt.ancillary_variables = make_ancvar_str('sea_water_practical_salinity')

        dens = dataset.createVariable('sea_water_sigma_theta','f8',('time',))
        dens.long_name = 'sea_water_sigma_theta'
        dens.standard_name = 'sea_water_sigma_theta'
        dens.description = 'Potential density, referenced to 0 dbar, offset by 1000 kg * m^-3 (sigma-0).'
        dens.units = 'kg * m-3 - 1000'
        dens.missing_value = fillVal
        dens.ancillary_variables = make_ancvar_str('sea_water_sigma_theta')

        oxyc = dataset.createVariable('mass_concentration_of_oxygen_in_sea_water','f8',('time',))
        oxyc.long_name = 'mass_concentration_of_oxygen_in_sea_water'
        oxyc.standard_name = 'mass_concentration_of_oxygen_in_sea_water'
        oxyc.description = 'Concentration of dissolved oxygen in water'
        oxyc.units = 'mg * L^-1'
        oxyc.missing_value = fillVal
        oxyc.ancillary_variables = make_ancvar_str('mass_concentration_of_oxygen_in_sea_water')

        oxyf = dataset.createVariable('fractional_saturation_of_oxygen_in_sea_water','f8',('time',))
        oxyf.long_name = 'fractional_saturation_of_oxygen_in_sea_water'
        oxyf.standard_name = 'fractional_saturation_of_oxygen_in_sea_water'
        oxyf.description = 'Concentration of dissolved oxygen in water, as a percentage of the concentration of dissolved oxygen in water at saturation. Dissolved oxygen saturation is the concentration of dissolved oxygen at saturation levels at the same temperature and salinity in a water sample.'
        oxyf.units = '%'
        oxyf.missing_value = fillVal

        svel = dataset.createVariable('speed_of_sound_in_sea_water','f8',('time',))
        svel.long_name = 'speed_of_sound_in_sea_water'
        svel.standard_name = 'speed_of_sound_in_sea_water'
        svel.description = 'Speed of sound in sea water; speed is the magnitude of velocity.'
        svel.units = 'm * s^-1'
        svel.missing_value = fillVal

        ph = dataset.createVariable('sea_water_ph_reported_on_total_scale','f8',('time',))
        ph.long_name = 'sea_water_ph_reported_on_total_scale'
        ph.standard_name = 'sea_water_ph_reported_on_total_scale'
        ph.description = 'measure of acidity of seawater, defined as the negative logarithm of the concentration of dissolved hydrogen ions'
        ph.units = '1'
        ph.missing_value = fillVal
        ph.ancillary_variables = make_ancvar_str('sea_water_ph_reported_on_total_scale')

        chl = dataset.createVariable('mass_concentration_of_chlorophyll_a_in_sea_water','f8',('time',))
        chl.long_name = 'mass_concentration_of_chlorophyll_a_in_sea_water'
        chl.standard_name = 'mass_concentration_of_chlorophyll_a_in_sea_water'
        chl.description = 'Mass concentration of chlorophyll-a in sea water. Measurement is based upon fluoroescence taken from fluorometer'
        chl.units = 'mg * m^-3'
        chl.missing_value = fillVal
        chl.ancillary_variables = make_ancvar_str('mass_concentration_of_chlorophyll_a_in_sea_water')

        turb = dataset.createVariable('sea_water_turbidity','f8',('time',))
        turb.long_name = 'sea_water_turbidity'
        turb.standard_name = 'sea_water_turbidity'
        turb.description = 'Measure of light scattering due to suspended material in water. Measure of the cloudiness of water. Higher turbidity levels are often associated with higher levels of disease-causing microorganisms such as viruses, parasites and some bacteria. Turbidity is measured in nephelometric turbidity units (NTU)'
        turb.units = 'NTU'
        turb.missing_value = fillVal
        turb.ancillary_variables = make_ancvar_str('sea_water_turbidity')
        
        
        ############################
        # Assign the data
        
        # Assign the time variables
        reftime = datetime.datetime(1970,1,1)
        nemo_time = [ii for ii in sbe_df['time']]
        nemo_timeref = np.array([(ii.to_pydatetime(warn=False) - reftime).total_seconds()
                                   for ii in nemo_time])
        time[:] = nemo_timeref
        
        nemo_insttime =  [ii for ii in sbe_df['instrument_timestamp']]
        nemo_insttimeref = np.array([(ii.to_pydatetime(warn=False) - reftime).total_seconds()
                                   for ii in nemo_insttime])
        inst_time[:] = nemo_insttimeref
        
        # Assign the identifying variables
        buoy_name[:] = np.array(sbe_df['buoyname'].fillna('').tolist(), dtype=object)
        deployment[:] = np.array([deployment_name] * NT, dtype=object)
        latitude[:] = [float(lat) for ii in range(0,NT)]
        longitude[:] = [float(lon) for ii in range(0,NT)]
        recordnum[:] = sbe_df['record_number'].fillna(fillVal).to_numpy().squeeze()
        
        # Assign the sensor variables
        pres[:] = sbe_df['sea_water_pressure'].fillna(fillVal).to_numpy().squeeze()
        depth[:] = sbe_df['depth'].fillna(fillVal).to_numpy().squeeze()
        temp[:] = sbe_df['sea_water_temperature'].fillna(fillVal).to_numpy().squeeze()
        cond[:] = sbe_df['sea_water_electrical_conductivity'].fillna(fillVal).to_numpy().squeeze()
        salt[:] = sbe_df['sea_water_practical_salinity'].fillna(fillVal).to_numpy().squeeze()
        dens[:] = sbe_df['sea_water_sigma_theta'].fillna(fillVal).to_numpy().squeeze()
        oxyc[:] = sbe_df['mass_concentration_of_oxygen_in_sea_water'].fillna(fillVal).to_numpy().squeeze()
        oxyf[:] = sbe_df['fractional_saturation_of_oxygen_in_sea_water'].fillna(fillVal).to_numpy().squeeze()
        svel[:] = sbe_df['speed_of_sound_in_sea_water'].fillna(fillVal).to_numpy().squeeze()
        ph[:] = sbe_df['sea_water_ph_reported_on_total_scale'].fillna(fillVal).to_numpy().squeeze()
        chl[:] = sbe_df['mass_concentration_of_chlorophyll_a_in_sea_water'].fillna(fillVal).to_numpy().squeeze()
        turb[:] = sbe_df['sea_water_turbidity'].fillna(fillVal).to_numpy().squeeze()


        ####################
        # QARTOD variables #
        ####################
        if sbe_qcdf is None:
            print('No QARTOD data frame provided. Skipping QARTOD variables.')
        else:
            print('Writing QARTOD variables to netCDF.')
            varnames = ['sea_water_pressure',
                        'sea_water_temperature',
                        'sea_water_electrical_conductivity',
                        'sea_water_practical_salinity',
                        'sea_water_sigma_theta',
                        'mass_concentration_of_oxygen_in_sea_water',
                        'sea_water_ph_reported_on_total_scale',
                        'mass_concentration_of_chlorophyll_a_in_sea_water',
                        'sea_water_turbidity']
            varlabs = ['Sea Water Pressure', 'Sea Water Temperature', 
                    'Sea Water Electrical Conductivity', 'Sea Water Practical Salinity',
                    'Sea Water Sigma Theta', 'Mass Concentration of Oxygen in Sea Water',
                    'Sea Water pH Reported on Total Scale', 'Mass Concentration of Chlorophyll a in Sea Water',
                    'Sea Water Turbidity']
            
            origqc_vars = ['qartod_rollup_qc', 
                        'qartod_gross_range_test', 'qartod_rate_of_change_test',
                        'qartod_spike_test', 'qartod_flat_line_test']
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
                    temp_qcvar = dataset.createVariable(var + '_' + qc_var,'i4',('time'))
                    temp_qcvar.standard_name = qc_standard
                    temp_qcvar.long_name = var + '_' + qc_var
                    temp_qcvar.description = varlabel + ' ' + qc_lab
                    temp_qcvar.ioos_category = 'Quality Control'
                    temp_qcvar.units = '1'
                    temp_qcvar.coverage_content_type = 'qualityInformation'
                    temp_qcvar.flag_vals = '1, 2, 3, 4, 9'
                    temp_qcvar.flag_meanings = 'PASS NOT_EVALUATED SUSPECT FAIL MISSING'
                    temp_qcvar[:] = sbe_qcdf[var + '_' + origqc_var]
        
    except Exception as e:
        print('Error occurred while writing the netCDF')
        print(e)
        
    dataset.close()
    
    return





# CTD Function
def make_thermistor_netCDF(nemo_info, savedir, sbe_df, sbe_qcdf=None):

    from netCDF4 import Dataset
    import os
    import numpy as np
    import datetime
    
    #######
    # Extract out the lander information
    
    nemo_name = nemo_info['BuoyName']
    nemo_title = nemo_info['BuoyTitle']
    info_url = nemo_info['info_url']
    deployment_name = nemo_info['DeploymentName']
    lat = nemo_info['Latitude']
    lon = nemo_info['Longitude']
    depth = nemo_info['Depth']
    institution_info = nemo_info['institution_info']
    instrument_info = nemo_info['InstrumentInfo']
    
    fillVal = -555
    NT = len(sbe_df)
    
    ##################################################
    # Save as a netCDF
    ##################################################
    
    ctd_ncfile = nemo_name.lower() + '_' + deployment_name.lower() + '_thermistor_' + instrument_info.get('model', '').lower() + '_' + instrument_info.get('sn', '').lower() + '.nc'
    dataset = Dataset(os.path.join(savedir,ctd_ncfile),
                      'w',format='NETCDF4')
    
    try:
    
        dataset.title = nemo_title + ' - Thermistor Data'
        dataset.description = 'Water property data from a ' + instrument_info.get('model', '') + ' on the ' + nemo_name + ' buoy during the ' + deployment_name + ' deployment. Data processed by NWEM.'
        dataset.history = "File created on " + datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
        dataset.conventions = 'CF-1.6, ACDD-1.3, IOOS-1.2'
        dataset.designation = nemo_name
        dataset.latitude = str(np.round(lat,6)) + 'N'
        dataset.longitude = str(np.round(lon,6)) + 'E'
        dataset.buoy_depth = str(depth) + 'm'
        dataset.insturment_depth = str(instrument_info.get('depth', '')) + 'm'
        dataset.infoUrl = info_url
        dataset.institution = institution_info.get('institution', '')

        ## These are fields required by NCEI by ingestion to the sensor map
        dataset.creator_name = institution_info.get('name', '')
        dataset.creator_institution = institution_info.get('institution', '')
        dataset.creator_url = institution_info.get('url', '')
        dataset.creator_country = institution_info.get('country', '')
        dataset.creator_sector = institution_info.get('sector', '')
        dataset.creator_type = institution_info.get('type', '')

        dataset.contributor_name = institution_info.get('name', '') + ', John Mickett, NANOOS, NWEM'
        dataset.contributor_role_vocabulary = 'https://vocab.nerc.ac.uk/collection/G04/current/'
        dataset.contributor_role =  'owner, principalInvestigator, funder, publisher'
        dataset.contributor_role_url = institution_info.get('url', '') + ', --, https://www.nanoos.org/, https://nwem.apl.washington.edu/'

        dataset.publisher_name = 'NorthWest Environmental Moorings (NWEM) Group'
        dataset.publisher_email = 'setht1@uw.edu'
        dataset.publisher_institution = 'University of Washington - Applied Physics Laboratory'
        dataset.publisher_type = 'group'
        dataset.publisher_url = 'https://nwem.apl.washington.edu/'
        dataset.publisher_country = 'United States'

        dataset.author = 'Seth Travis'
        dataset.contact = 'setht1@uw.edu'
        dataset._FillValue = fillVal
        dataset.cdm_data_type = 'timeSeries'
        dataset.cdm_timeseries_variables = 'buoy_name,latitude,longitude'

        dataset.mooring_diagram_baseurl = nemo_info.get('mooring_diagram_base', '')
        dataset.mooring_diagram_url = nemo_info.get('mooring_diagram_base', '') + deployment_name + '_mooring_diagram_final.pdf'
        dataset.time_drift_seconds = instrument_info.get('time_drift_seconds', '')
        dataset.time_drift_description = 'Number of seconds that the instrument clock drifted from UTC time during the deployment. This is calculated by comparing the instrument clock to the GPS time at the start and end of the deployment. Positive values indicate that the instrument clock was ahead of UTC time, while negative values indicate that the instrument clock was behind UTC time.'


        #############################################
        # Create coordinate and identifying variables
        
        dataset.createDimension('time',NT)
        
        time = dataset.createVariable('time','f8',('time',))
        time.long_name = 'time'
        time.description = 'time of sampling'
        time.units = 'seconds since 1970-01-01 00:00:00'
        time.timezone = 'UTC'
        time.calendar = 'gregorian'

        buoy_name = dataset.createVariable('buoy_name','S9',('time',))
        buoy_name.long_name = 'buoy_name'
        buoy_name.description = 'buoy description name'
        buoy_name.cf_role = 'timeseries_id'

        deployment = dataset.createVariable('deployment_name','S9',('time',))
        deployment.long_name = 'deployment_name'
        deployment.description = 'deployment identifier name'

        latitude = dataset.createVariable('latitude','f8',('time',))
        latitude.long_name = 'latitude'
        latitude.units = 'degrees North'

        longitude = dataset.createVariable('longitude','f8',('time',))
        longitude.long_name = 'longitude'
        longitude.units = 'degrees East'

        ################################
        # Assign variables


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

        inst_time = dataset.createVariable('instrument_time','f8',('time',))
        inst_time.long_name = 'instrument_time'
        inst_time.description = 'time of sampling, taken from the instrument'
        inst_time.units = 'seconds since 1970-01-01 00:00:00'
        inst_time.timezone = 'UTC'
        inst_time.calendar = 'gregorian'  

        recordnum = dataset.createVariable('sample_number','i4',('time',))
        recordnum.long_name = 'deployment_sample_record_number'
        recordnum.description = 'the record number of the sample taken for a given deployment'
        recordnum.units = '--'
        recordnum.missing_value = fillVal   

        depth = dataset.createVariable('depth','f8',('time',))
        depth.long_name = 'depth'
        depth.standard_name = 'depth'
        depth.description = 'Depth of the water column at the measurement location. If pressure is measured, depth is calculated from pressure using the UNESCO 1983 algorithm. If pressure is not measured, depth is calculated assuming hydrostatic pressure. If appropriate variables are not present, depth is given as the deployed instrument depth, and does not account for variation over time.'
        depth.units = 'm'
        depth.missing_value = fillVal

        temp = dataset.createVariable('sea_water_temperature','f8',('time',))
        temp.long_name = 'sea_water_temperature'
        temp.standard_name = 'sea_water_temperature'
        temp.description = 'In-situ temperature of water (T90 scale)'
        temp.units = 'degrees C'
        temp.missing_value = fillVal 
        temp.ancillary_variables = make_ancvar_str('sea_water_temperature')
        
        
        ############################
        # Assign the data
        
        # Assign the time variables
        reftime = datetime.datetime(1970,1,1)
        nemo_time = [ii for ii in sbe_df['time']]
        nemo_timeref = np.array([(ii.to_pydatetime(warn=False) - reftime).total_seconds()
                                   for ii in nemo_time])
        time[:] = nemo_timeref
        
        nemo_insttime =  [ii for ii in sbe_df['instrument_timestamp']]
        nemo_insttimeref = np.array([(ii.to_pydatetime(warn=False) - reftime).total_seconds()
                                   for ii in nemo_insttime])
        inst_time[:] = nemo_insttimeref
        
        # Assign the identifying variables
        buoy_name[:] = np.array(sbe_df['buoyname'].fillna('').tolist(), dtype=object)
        deployment[:] = np.array([deployment_name] * NT, dtype=object)
        latitude[:] = [float(lat) for ii in range(0,NT)]
        longitude[:] = [float(lon) for ii in range(0,NT)]
        recordnum[:] = sbe_df['record_number'].fillna(fillVal).to_numpy().squeeze()
        
        # Assign the sensor variables
        depth[:] = sbe_df['depth'].fillna(fillVal).to_numpy().squeeze()
        temp[:] = sbe_df['sea_water_temperature'].fillna(fillVal).to_numpy().squeeze()


        ####################
        # QARTOD variables #
        ####################
        if sbe_qcdf is None:
            print('No QARTOD data frame provided. Skipping QARTOD variables.')
        else:
            print('Writing QARTOD variables to netCDF.')
            varnames = ['sea_water_temperature']
            varlabs = ['Sea Water Temperature']
            
            origqc_vars = ['qartod_rollup_qc', 
                        'qartod_gross_range_test', 'qartod_rate_of_change_test',
                        'qartod_spike_test', 'qartod_flat_line_test']
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
                    temp_qcvar = dataset.createVariable(var + '_' + qc_var,'i4',('time'))
                    temp_qcvar.standard_name = qc_standard
                    temp_qcvar.long_name = var + '_' + qc_var
                    temp_qcvar.description = varlabel + ' ' + qc_lab
                    temp_qcvar.ioos_category = 'Quality Control'
                    temp_qcvar.units = '1'
                    temp_qcvar.coverage_content_type = 'qualityInformation'
                    temp_qcvar.flag_vals = '1, 2, 3, 4, 9'
                    temp_qcvar.flag_meanings = 'PASS NOT_EVALUATED SUSPECT FAIL MISSING'
                    temp_qcvar[:] = sbe_qcdf[var + '_' + origqc_var]
        
    except Exception as e:
        print('Error occurred while writing the netCDF')
        print(e)
        
    dataset.close()
    
    return