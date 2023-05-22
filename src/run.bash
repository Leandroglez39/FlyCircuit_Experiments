#!/bin/bash

source envNX/bin/activate
python3 src/matrix.py

deactivate

source encCDLib/bin/activate
python3 src/nmi.py

deactivate

