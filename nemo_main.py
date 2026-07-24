#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

from nemo_adcp import buoy_adcp_wrapper
from nemo_hydrocat import nemo_hydrocat_wrapper
from nemo_seaphox import nemo_seaphox_wrapper
from nemo_sbe37 import nemo_sbe37_wrapper
from nemo_sbe56 import nemo_sbe56_wrapper
from nemo_deployments import run_deployments_menu


def process_nemo_deployment(nemo_name, deployment_name):
    print(f"Processing lander '{nemo_name}' deployment '{deployment_name}'")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{nemo_name}_deployments.json')
    if not os.path.exists(json_path):
        alt_json_path = os.path.join(script_dir, 'info_jsons', 'deployments', f'{nemo_name.lower()}_deployments.json')
        if os.path.exists(alt_json_path):
            json_path = alt_json_path
        else:
            raise FileNotFoundError(f"Deployment info file not found: {json_path}")

    with open(json_path, 'r') as f:
        deployment_info = json.load(f)

    deployment_data = deployment_info.get('deployments', {}).get(deployment_name)
    if deployment_data is None:
        raise KeyError(f"Deployment '{deployment_name}' not found in {json_path}.")

    instruments = deployment_data.get('instruments', {})
    if not isinstance(instruments, dict) or len(instruments) == 0:
        print(f"No instruments listed for {nemo_name} - {deployment_name}; nothing to process.")
        return

    processors = {
        'adcp': buoy_adcp_wrapper,
        'sbe37': nemo_sbe37_wrapper,
        'sbe56': nemo_sbe56_wrapper,
        'hydrocat': nemo_hydrocat_wrapper,
        'seaphox': nemo_seaphox_wrapper,
    }

    selected_types = set()
    for instrument_id, instrument_record in instruments.items():
        instrument_id_text = str(instrument_id).strip().lower()
        model_name = ''
        if isinstance(instrument_record, dict):
            model_name = str(instrument_record.get('model', '')).strip().lower()

        if ('adcp' in model_name) or ('workhorse' in model_name) or ('adcp' in instrument_id_text):
            selected_types.add('adcp')
        elif 'sbe37' in model_name:
            selected_types.add('sbe37')
        elif 'sbe56' in model_name:
            selected_types.add('sbe56')
        elif ('hydrocat' in model_name) or ('hcat' in model_name):
            selected_types.add('hydrocat')
        elif ('seaphox' in model_name) or ('sphox' in model_name) or ('sea phox' in model_name):
            selected_types.add('seaphox')

    if len(selected_types) == 0:
        print(f"No supported instrument types found for {nemo_name} - {deployment_name}.")
        return

    ordered_types = ['adcp', 'sbe37', 'sbe56', 'hydrocat', 'seaphox']
    for instrument_type in ordered_types:
        if instrument_type not in selected_types:
            continue

        wrapper_fn = processors[instrument_type]
        print(f"\nRunning {instrument_type.upper()} processing for {nemo_name} - {deployment_name}")
        try:
            if instrument_type == 'adcp':
                wrapper_fn(nemo_name, deployment_name, instrument_name=instrument_type, average_windows=[10,60])
            else:
                wrapper_fn(nemo_name, deployment_name, instrument_name=instrument_type)
        except Exception as e:
            print(f"{instrument_type.upper()} processing failed for {nemo_name} - {deployment_name}: {e}")


def run_main_menu():
    while True:
        print('\n=== NEMO Processing Main Menu ===')
        print('1) Start interactive deployment menu')
        print('2) Process data for a buoy/deployment')
        print('3) Exit')

        choice = input('Choose an option [1-3]: ').strip()

        if choice == '1':
            run_deployments_menu()
        elif choice == '2':
            nemo_name = input('Buoy name: ').strip()
            deployment_name = input('Deployment name: ').strip()
            if nemo_name == '' or deployment_name == '':
                print('Buoy name and deployment name are required.')
                continue
            process_nemo_deployment(nemo_name, deployment_name)
        elif choice == '3':
            print('Exiting.')
            break
        else:
            print('Invalid choice. Please select 1, 2, or 3.')


def main():
    parser = argparse.ArgumentParser(description='Landers processing wrapper')
    parser.add_argument('--deployments-menu', action='store_true',
                        help='Start the interactive deployment menu')
    parser.add_argument('--process', action='store_true',
                        help='Process data for a buoy and deployment')
    parser.add_argument('--buoy', type=str,
                        help='Buoy name to process')
    parser.add_argument('--deployment', type=str,
                        help='Deployment name to process')
    parser.add_argument('--instrument', type=str,
                        help='Instrument name to process')
    parser.add_argument('--serial', type=str,
                        help='Instrument serial number to process')

    args = parser.parse_args()

    if args.deployments_menu:
        run_deployments_menu()
        return

    if args.process:
        if not args.buoy or not args.deployment:
            parser.error('--process requires --buoy and --deployment')
        process_nemo_deployment(args.buoy, args.deployment)
        return

    run_main_menu()


if __name__ == '__main__':
    main()
