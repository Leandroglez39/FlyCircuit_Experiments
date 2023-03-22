#!/bin/bash

python3 -m pip install virtualenv

virtualenv "envCDLib"
virtualenv "envNX"

source envCDLib/bin/activate
pip install -r "cdlib_requirements.txt"
deactivate

source envNX/bin/activate
pip install -r "nx_requirements.txt"
deactivate
