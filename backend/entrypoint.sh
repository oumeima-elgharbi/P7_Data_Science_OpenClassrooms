#!/bin/bash

# Add script to download data fiels from bucket p7-data to /backend/resources
"/backend/resources"

python script_download_data_folder.py
#import subprocess
#subprocess.call(r'python3 script_download_data_folder.py', shell=True)

# shellcheck disable=SC2068
python $@
