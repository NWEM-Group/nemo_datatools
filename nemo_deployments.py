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

VALID_DEPLOYMENT_STATUSES = ['planned', 'realtime', 'recovered']


# # General Functions



def load_nemo_position(buoy_name, deployment_name):
    
    deployment_info = load_deployment_info(buoy_name)
    deployment_data = deployment_info.get('deployments', {}).get(deployment_name)
    
    if deployment_data is None:
        raise KeyError(f"Deployment '{deployment_name}' not found for {buoy_name}.")
    
    lat = deployment_data.get('latitude')
    lon = deployment_data.get('longitude')
    depth = deployment_data.get('depth')
    
    start_str = deployment_data.get('start')
    if start_str is not None:
        # Support ISO timestamps with trailing Z (UTC).
        start_deployment = datetime.datetime.fromisoformat(start_str.replace('Z', '+00:00'))
    else:
        start_deployment = None
        
    return lat, lon, depth, start_deployment



def load_deployment_info(buoy_name):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{buoy_name}_deployments.json')
    
    with open(json_path, 'r') as f:
        deployment_info = json.load(f)
    
    return deployment_info



def load_nemo_info():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'nemo_info.json')
    
    with open(json_path, 'r') as f:
        nemo_info = json.load(f)
    
    return nemo_info



def save_nemo_info(nemo_info):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'nemo_info.json')
    
    with open(json_path, 'w') as f:
        json.dump(nemo_info, f, indent=2)



def _get_deployment_json_path(buoy_name):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'info_jsons', 'deployments', f'{buoy_name}_deployments.json')


def _get_deployment_csv_path(buoy_name, deployment_name):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'info_csvs', f'{buoy_name}_{deployment_name}_instruments.csv')



def _create_empty_deployment_info_json(buoy_name, shorthand=None):
    
    if shorthand is None or shorthand.strip() == '':
        shorthand = buoy_name
    
    new_deployment_info = {
        'buoy_name': buoy_name,
        'shorthand': shorthand,
        'deployments': {}
    }
    
    json_path = _get_deployment_json_path(buoy_name)
    with open(json_path, 'w') as f:
        json.dump(new_deployment_info, f, indent=2)



def _ensure_deployment_info_file(buoy_name, interactive=False, action='add'):
    
    json_path = _get_deployment_json_path(buoy_name)
    if os.path.exists(json_path):
        return True
    
    nemo_info = load_nemo_info()
    valid_buoy_names = nemo_info.get('valid_buoy_names', [])
    
    if buoy_name in valid_buoy_names:
        shorthand = None
        if interactive:
            print(f"No deployment info JSON exists yet for '{buoy_name}'.")
            shorthand = _prompt_optional_text('Shorthand for new deployment file (blank uses lander name): ')
        _create_empty_deployment_info_json(buoy_name, shorthand=shorthand)
        return True
    
    if not interactive:
        raise ValueError(
            f"No deployment file for '{buoy_name}', and it is not in info_jsons/nemo_info.json valid list."
        )
    
    print(f"Buoy '{buoy_name}' is not in the valid buoy list.")
    print('Current valid buoy names:')
    if len(valid_buoy_names) == 0:
        print('  - None')
    else:
        for name in valid_buoy_names:
            print(f'  - {name}')
    
    should_add = input(f"Add '{buoy_name}' to valid buoy names? (y/n): ").strip().lower()
    if should_add not in ['y', 'yes']:
        print(f"Deployment {action} canceled.")
        return False
    
    nemo_info.setdefault('valid_buoy_names', []).append(buoy_name)
    save_nemo_info(nemo_info)
    print(f"Added '{buoy_name}' to info_jsons/nemo_info.json valid list.")
    
    shorthand = _prompt_optional_text('Shorthand for new deployment file (blank uses buoy name): ')
    _create_empty_deployment_info_json(buoy_name, shorthand=shorthand)
    return True



def _ensure_deployment_info_file_for_add(buoy_name, interactive=False):
    
    return _ensure_deployment_info_file(buoy_name, interactive=interactive, action='add')



def _ensure_deployment_info_file_for_update(buoy_name, interactive=False):
    
    return _ensure_deployment_info_file(buoy_name, interactive=interactive, action='update')



def _validate_deployment_status(status_value, required=False):
    
    if status_value is None:
        if required:
            raise ValueError(
                f"Deployment status is required. Valid statuses: {', '.join(VALID_DEPLOYMENT_STATUSES)}"
            )
        return None
    
    if not isinstance(status_value, str):
        raise ValueError(
            f"Deployment status must be a string. Valid statuses: {', '.join(VALID_DEPLOYMENT_STATUSES)}"
        )
    
    normalized_status = status_value.strip().lower()
    if normalized_status not in VALID_DEPLOYMENT_STATUSES:
        raise ValueError(
            f"Invalid deployment status '{status_value}'. "
            f"Valid statuses: {', '.join(VALID_DEPLOYMENT_STATUSES)}"
        )
    
    return normalized_status


def _normalize_csv_field_name(field_name):
    
    normalized_name = field_name.strip().lower()
    normalized_name = normalized_name.replace(' (utc)', '')
    normalized_name = normalized_name.replace(' (seconds)', '_seconds')
    normalized_name = normalized_name.replace(' ', '_')
    normalized_name = normalized_name.replace('-', '_')
    normalized_name = normalized_name.replace('/', '_')
    normalized_name = normalized_name.replace('__', '_')
    return normalized_name


def _coerce_csv_value(value):
    
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if float(value).is_integer():
            return int(value)
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value == '':
            return None
        return value
    return value


def _parse_csv_datetime(value):

    coerced_value = _coerce_csv_value(value)
    if coerced_value is None:
        return None

    if isinstance(coerced_value, datetime.datetime):
        return coerced_value

    # Support common CSV formats first, then fall back to pandas parser.
    if isinstance(coerced_value, str):
        for dt_format in ['%m/%d/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
            try:
                return datetime.datetime.strptime(coerced_value, dt_format)
            except ValueError:
                continue

    parsed_value = pd.to_datetime(coerced_value, errors='coerce')
    if pd.isna(parsed_value):
        return None
    return parsed_value.to_pydatetime()


def _datetime_to_iso_utc_z(value):

    parsed_value = _parse_csv_datetime(value)
    if parsed_value is None:
        return None
    return parsed_value.strftime('%Y-%m-%dT%H:%M:%SZ')


def instrument_key(instrument_name):

    if instrument_name is None:
        raise ValueError('Instrument name cannot be empty.')

    normalized_name = str(instrument_name).strip().lower()
    valid_instruments = {
        'ctd': ['wqm', 'hcat', 'sbe37', 'sbe39', 'sbe44', 'sbe56', 'seaphox', 'seafet'],
        'adcp': ['adcp']
    }

    for type_prefix, names in valid_instruments.items():
        if normalized_name in names:
            return type_prefix

    raise ValueError(
        f"Invalid instrument name '{instrument_name}'. "
        f"Valid names: {', '.join([name for names in valid_instruments.values() for name in names])}"
    )


def _instrument_model_name(instrument_name):

    normalized_name = str(instrument_name).strip().lower()
    model_lookup = {
        "wqm": "WQM",
        'hcat': 'HydroCat-EP',
        'sbe37': 'SBE37-ODO-IMP',
        'sbe39': 'SBE39',
        'sbe44': 'SBE44',
        'sbe56': 'SBE56',
        'seaphox': 'SeaPHOX',
        'adcp': 'ADCP'
    }
    return model_lookup.get(normalized_name, str(instrument_name).strip())


def _normalize_instrument_type(instrument_type):

    if instrument_type is None:
        raise ValueError('Instrument type cannot be empty.')

    normalized_type = str(instrument_type).strip().lower()
    if normalized_type in ['ctd', 'adcp']:
        return normalized_type

    return instrument_key(normalized_type)


def _normalize_serial_number(serial_number):

    coerced_serial = _coerce_csv_value(serial_number)
    if coerced_serial is None:
        return None

    return str(coerced_serial).strip().lower()


def _stringify_instrument_serial_numbers(instruments):

    if not isinstance(instruments, dict):
        return instruments

    for instrument_key_name, instrument_record in instruments.items():
        if not isinstance(instrument_record, dict):
            continue

        serial_value = instrument_record.get('sn')
        if serial_value is None:
            continue
        instrument_record['sn'] = str(serial_value).strip()

    return instruments


def _coerce_deployment_instrument_serial_numbers(deployment_data):

    if not isinstance(deployment_data, dict):
        return deployment_data

    instruments = deployment_data.get('instruments')
    if instruments is not None:
        deployment_data['instruments'] = _stringify_instrument_serial_numbers(instruments)

    return deployment_data


def _infer_instrument_type_from_key(instrument_key_name, instrument_record):

    key_name = str(instrument_key_name).strip().lower()
    if '/' in key_name:
        return key_name.split('/')[0]
    if key_name.startswith('ctd'):
        return 'ctd'
    if key_name.startswith('adcp'):
        return 'adcp'
    if key_name in ['ctd', 'adcp']:
        return key_name

    model_name = None
    if isinstance(instrument_record, dict):
        model_name = instrument_record.get('model')

    if model_name is None:
        return 'unknown'

    try:
        return instrument_key(model_name)
    except ValueError:
        return 'unknown'


def review_deployment_instruments(buoy_name, deployment_name):

    deployment_info = load_deployment_info(buoy_name)
    deployment_data = deployment_info.get('deployments', {}).get(deployment_name)
    if deployment_data is None:
        raise KeyError(f"Deployment '{deployment_name}' not found for {buoy_name}.")

    instruments = deployment_data.get('instruments', {})
    if not isinstance(instruments, dict):
        raise ValueError(
            f"Deployment '{deployment_name}' instruments must be a dictionary."
        )

    reviewed_instruments = []
    for instrument_name_key, instrument_record in instruments.items():
        if not isinstance(instrument_record, dict):
            continue

        reviewed_record = dict(instrument_record)
        reviewed_record['instrument_key'] = instrument_name_key
        reviewed_record['instrument_type'] = _infer_instrument_type_from_key(
            instrument_name_key,
            instrument_record
        )
        reviewed_record['sn_normalized'] = _normalize_serial_number(
            instrument_record.get('sn')
        )
        reviewed_instruments.append(reviewed_record)

    return reviewed_instruments


def find_matching_deployment_instrument(
    buoy_name,
    deployment_name,
    instrument_type,
    serial_number
):

    target_type = _normalize_instrument_type(instrument_type)
    target_serial = _normalize_serial_number(serial_number)
    if target_serial is None:
        raise ValueError('Serial number cannot be empty.')

    reviewed_instruments = review_deployment_instruments(buoy_name, deployment_name)
    matching_instruments = [
        instrument_record for instrument_record in reviewed_instruments
        if (
            instrument_record.get('instrument_type') == target_type
            and instrument_record.get('sn_normalized') == target_serial
        )
    ]

    if len(matching_instruments) == 0:
        return None

    if len(matching_instruments) > 1:
        instrument_keys = [record.get('instrument_key', '<unknown>') for record in matching_instruments]
        raise ValueError(
            f"Multiple instruments match type '{target_type}' and serial '{serial_number}': "
            f"{', '.join(instrument_keys)}"
        )

    return matching_instruments[0]
    



def import_deployment_csv_to_json(buoy_name, deployment_name):
    
    deployment_info = load_deployment_info(buoy_name)
    deployment_data = deployment_info.get('deployments', {}).get(deployment_name)
    if deployment_data is None:
        raise KeyError(f"Deployment '{deployment_name}' not found for {buoy_name}.")
    
    csv_path = _get_deployment_csv_path(buoy_name, deployment_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Deployment CSV not found: {csv_path}")
    
    csv_data = pd.read_csv(csv_path)
    required_columns = [
        'Instrument', 'SN', 'InstrumentDepth',
        'StartTime (UTC)', 'EndTime (UTC)', 'Instrument EndTime (UTC)'
    ]
    missing_columns = [column for column in required_columns if column not in csv_data.columns]
    if len(missing_columns) > 0:
        raise ValueError(
            f"Deployment CSV is missing required columns: {', '.join(missing_columns)}"
        )

    imported_instruments = {}
    instrument_counts = {}
    notes_column = 'Notes' if 'Notes' in csv_data.columns else None

    for _, row in csv_data.iterrows():
        instrument_name = _coerce_csv_value(row['Instrument'])
        if instrument_name is None:
            continue

        instrument_type = instrument_key(instrument_name)

        serial_value = _coerce_csv_value(row['SN'])
        instrument_record = {
            'model': _instrument_model_name(instrument_name),
            'sn': str(serial_value).strip() if serial_value is not None else None,
            'instrument_depth': _coerce_csv_value(row['InstrumentDepth'])
        }
        if notes_column is not None:
            notes_value = _coerce_csv_value(row[notes_column])
            if notes_value is not None:
                instrument_record['notes'] = str(notes_value).strip()

        start_time_value = row['StartTime (UTC)']
        end_time_value = row['EndTime (UTC)']
        instrument_end_time_value = row['Instrument EndTime (UTC)']

        iso_start_time = _datetime_to_iso_utc_z(start_time_value)
        iso_end_time = _datetime_to_iso_utc_z(end_time_value)
        iso_instrument_end_time = _datetime_to_iso_utc_z(instrument_end_time_value)

        if iso_start_time is not None:
            instrument_record['start_time'] = iso_start_time
        if iso_end_time is not None:
            instrument_record['end_time'] = iso_end_time
        if iso_instrument_end_time is not None:
            instrument_record['instrument_end_time'] = iso_instrument_end_time

        parsed_end_time = _parse_csv_datetime(end_time_value)
        parsed_instrument_end_time = _parse_csv_datetime(instrument_end_time_value)
        if parsed_end_time is not None and parsed_instrument_end_time is not None:
            instrument_record['time_drift_secs'] = (
                parsed_instrument_end_time - parsed_end_time
            ).total_seconds()

        instrument_counts[instrument_type] = instrument_counts.get(instrument_type, 0) + 1
        instrument_name_key = f"{instrument_type}/{instrument_counts[instrument_type]}"
        imported_instruments[instrument_name_key] = instrument_record
    
    if len(imported_instruments) == 0:
        raise ValueError(f"No instrument rows with valid names were found in {os.path.basename(csv_path)}.")
    
    deployment_info['deployments'][deployment_name]['instruments'] = imported_instruments
    
    json_path = _get_deployment_json_path(buoy_name)
    with open(json_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    return deployment_info['deployments'][deployment_name]['instruments']



def add_deployment(buoy_name, deployment_key, deployment_data):
    
    _ensure_deployment_info_file_for_add(buoy_name, interactive=False)
    
    deployment_info = load_deployment_info(buoy_name)
    
    if deployment_key in deployment_info['deployments']:
        raise ValueError(f"Deployment '{deployment_key}' already exists for {buoy_name}. Use update_deployment to modify it.")
    
    deployment_data['status'] = _validate_deployment_status(
        deployment_data.get('status'),
        required=True
    )
    deployment_data = _coerce_deployment_instrument_serial_numbers(deployment_data)
    
    deployment_info['deployments'][deployment_key] = deployment_data
    deployment_info['deployments'] = dict(
        sorted(
            deployment_info['deployments'].items(),
            key=lambda item: item[0].lower()
        )
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{buoy_name}_deployments.json')
    
    with open(json_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)



def update_deployment(buoy_name, deployment_key, updates):
    
    _ensure_deployment_info_file_for_update(buoy_name, interactive=False)
    
    deployment_info = load_deployment_info(buoy_name)
    
    if deployment_key not in deployment_info['deployments']:
        raise KeyError(f"Deployment '{deployment_key}' not found for {buoy_name}.")
    
    if 'status' in updates:
        updates['status'] = _validate_deployment_status(updates.get('status'), required=False)
    updates = _coerce_deployment_instrument_serial_numbers(updates)
    
    deployment_info['deployments'][deployment_key].update(updates)
    deployment_info['deployments'] = dict(
        sorted(
            deployment_info['deployments'].items(),
            key=lambda item: item[0].lower()
        )
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{buoy_name}_deployments.json')
    
    with open(json_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)



def print_nemo_info(buoy_name, deployment_name=None):
    
    deployment_info = load_deployment_info(buoy_name)
    deployments = deployment_info.get('deployments', {})
    deployment_names = list(deployments.keys())
    
    if deployment_name is None:
        print(f"Buoy Name: {deployment_info.get('buoy_name', buoy_name)}")
        print(f"Total Deployments: {len(deployment_names)}")
        print("Deployment Names:")
        
        if len(deployment_names) == 0:
            print("  - None")
        else:
            for name in deployment_names:
                print(f"  - {name}")
        
        return
    
    if deployment_name not in deployments:
        raise KeyError(f"Deployment '{deployment_name}' not found for {buoy_name}.")
    
    deployment_data = deployments[deployment_name]
    print(f"Buoy Name: {deployment_info.get('buoy_name', buoy_name)}")
    print(f"Deployment Name: {deployment_name}")
    print(json.dumps(deployment_data, indent=2))



def _prompt_required_text(prompt_text):
    
    while True:
        value = input(prompt_text).strip()
        if value != '':
            return value
        print('Value is required. Please try again.')



def _prompt_float(prompt_text, required=False):
    
    while True:
        value = input(prompt_text).strip()
        if value == '':
            if required:
                print('Value is required. Please try again.')
                continue
            return None
        try:
            return float(value)
        except ValueError:
            print('Please enter a valid number.')



def _prompt_int(prompt_text, required=False):
    
    while True:
        value = input(prompt_text).strip()
        if value == '':
            if required:
                print('Value is required. Please try again.')
                continue
            return None
        try:
            return int(value)
        except ValueError:
            print('Please enter a valid integer.')



def _prompt_optional_text(prompt_text):
    
    value = input(prompt_text).strip()
    if value == '':
        return None
    return value



def _prompt_deployment_status(required=False):
    
    status_prompt = f"Deployment status ({'/'.join(VALID_DEPLOYMENT_STATUSES)})"
    if required:
        status_prompt = status_prompt + ': '
    else:
        status_prompt = status_prompt + ' (blank to keep): '
    
    while True:
        status_value = input(status_prompt).strip()
        if status_value == '':
            if required:
                print('Value is required. Please try again.')
                continue
            return None
        
        try:
            return _validate_deployment_status(status_value, required=required)
        except ValueError as e:
            print(e)



def _prompt_instruments():
    
    instruments = {}
    include_instruments = input('Add instrument info? (y/n): ').strip().lower()
    if include_instruments not in ['y', 'yes']:
        return instruments
    
    adcp_model = _prompt_optional_text('ADCP model (blank to skip): ')
    adcp_sn = _prompt_optional_text('ADCP serial number (blank to skip): ')
    adcp_notes = _prompt_optional_text('ADCP notes (blank to skip): ')
    if adcp_model is not None or adcp_sn is not None or adcp_notes is not None:
        instruments['adcp'] = {}
        if adcp_model is not None:
            instruments['adcp']['model'] = adcp_model
        if adcp_sn is not None:
            instruments['adcp']['sn'] = adcp_sn
        if adcp_notes is not None:
            instruments['adcp']['notes'] = adcp_notes
    
    ctd_model = _prompt_optional_text('CTD model (blank to skip): ')
    ctd_sn = _prompt_optional_text('CTD serial number (blank to skip): ')
    ctd_notes = _prompt_optional_text('CTD notes (blank to skip): ')
    if ctd_model is not None or ctd_sn is not None or ctd_notes is not None:
        instruments['ctd'] = {}
        if ctd_model is not None:
            instruments['ctd']['model'] = ctd_model
        if ctd_sn is not None:
            instruments['ctd']['sn'] = ctd_sn
        if ctd_notes is not None:
            instruments['ctd']['notes'] = ctd_notes
    
    return instruments



def interactive_add_deployment():
    
    print('--- Add Deployment (Interactive) ---')
    
    buoy_name = _prompt_required_text('Buoy name: ')
    
    try:
        can_continue = _ensure_deployment_info_file_for_add(buoy_name, interactive=True)
        if not can_continue:
            return
    except Exception as e:
        print(f'Failed to initialize deployment info file: {e}')
        return
    
    deployment_key = _prompt_required_text('Deployment name/key: ')
    
    deployment_data = {
    }
    deployment_data['start'] = _prompt_required_text('Start time (ISO, e.g., 2024-07-10T17:45:00Z): ')
    end_time = _prompt_optional_text('End time (ISO, blank if active): ')
    if end_time is not None:
        deployment_data['end'] = end_time
    deployment_data['status'] = _prompt_deployment_status(required=True)
    deployment_data['latitude'] = _prompt_float('Latitude: ', required=True)
    deployment_data['longitude'] = _prompt_float('Longitude: ', required=True)
    deployment_data['depth'] = _prompt_int('Depth (m): ', required=True)
    
    
    
    instruments = _prompt_instruments()
    if len(instruments) > 0:
        deployment_data['instruments'] = instruments
    
    try:
        add_deployment(buoy_name, deployment_key, deployment_data)
        print(f"Added deployment '{deployment_key}' for {buoy_name}.")
        print_nemo_info(buoy_name, deployment_key)
    except Exception as e:
        print(f'Failed to add deployment: {e}')



def interactive_update_deployment():
    
    print('--- Update Deployment (Interactive) ---')
    
    buoy_name = _prompt_required_text('Buoy name: ')
    
    try:
        can_continue = _ensure_deployment_info_file_for_update(buoy_name, interactive=True)
        if not can_continue:
            return
    except Exception as e:
        print(f'Failed to initialize deployment info file: {e}')
        return
    
    deployment_key = _prompt_required_text('Deployment name/key: ')
    
    try:
        print('Current deployment info:')
        print_nemo_info(buoy_name, deployment_key)
    except Exception as e:
        print(f'Could not print current deployment info: {e}')
    
    print('Enter update JSON (blank to use guided prompts).')
    print('Example: {"depth": 55, "end": "2024-10-31T00:20:00Z"}')
    update_json = input('Updates JSON: ').strip()
    
    try:
        if update_json != '':
            updates = json.loads(update_json)
            if not isinstance(updates, dict):
                raise ValueError('Updates JSON must be an object/dictionary.')
        else:
            updates = {}
            start_time = _prompt_optional_text('New start time (blank to keep): ')
            status = _prompt_deployment_status(required=False)
            end_time = _prompt_optional_text('New end time (blank to keep): ')
            latitude = _prompt_float('New latitude (blank to keep): ', required=False)
            longitude = _prompt_float('New longitude (blank to keep): ', required=False)
            depth = _prompt_int('New depth (blank to keep): ', required=False)
            
            if start_time is not None:
                updates['start'] = start_time
            if status is not None:
                updates['status'] = status
            if end_time is not None:
                updates['end'] = end_time
            if latitude is not None:
                updates['latitude'] = latitude
            if longitude is not None:
                updates['longitude'] = longitude
            if depth is not None:
                updates['depth'] = depth
            
            instruments = _prompt_instruments()
            if len(instruments) > 0:
                updates['instruments'] = instruments
            
            if len(updates) == 0:
                print('No updates entered. Nothing to do.')
                return
        
        update_deployment(buoy_name, deployment_key, updates)
        print(f"Updated deployment '{deployment_key}' for {buoy_name}.")
        print_nemo_info(buoy_name, deployment_key)
    except Exception as e:
        print(f'Failed to update deployment: {e}')



def interactive_print_nemo_info():
    
    print('--- Buoy Info (Interactive) ---')
    buoy_name = _prompt_required_text('Buoy name: ')
    deployment_name = _prompt_optional_text('Deployment name (blank for high-level summary): ')
    
    try:
        if deployment_name is None:
            print_nemo_info(buoy_name)
        else:
            print_nemo_info(buoy_name, deployment_name)
    except Exception as e:
        print(f'Failed to load buoy info: {e}')


def interactive_import_deployment_instruments():

    print('--- Import Deployment Instruments (Interactive) ---')
    buoy_name = _prompt_required_text('Buoy name: ')
    deployment_name = _prompt_required_text('Deployment name/key: ')

    try:
        imported_instruments = import_deployment_csv_to_json(buoy_name, deployment_name)
        print(
            f"Imported {len(imported_instruments)} instrument records into "
            f"'{buoy_name}' deployment '{deployment_name}'."
        )
    except Exception as e:
        print(f'Failed to import deployment instruments: {e}')



def run_deployments_menu():
    
    while True:
        print('=== Buoy Deployments Menu ===')
        print('1) Show buoy info')
        print('2) Add deployment')
        print('3) Update deployment')
        print('4) Import deployment instruments from CSV')
        print('5) Exit')
        
        choice = input('Choose an option [1-5]: ').strip()
        
        if choice == '1':
            interactive_print_nemo_info()
        elif choice == '2':
            interactive_add_deployment()
        elif choice == '3':
            interactive_update_deployment()
        elif choice == '4':
            interactive_import_deployment_instruments()
        elif choice == '5':
            print('Exiting menu.')
            break
        else:
            print('Invalid choice. Please select 1, 2, 3, 4, or 5.')



if __name__ == '__main__':
    run_deployments_menu()