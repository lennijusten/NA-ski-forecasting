#!/bin/bash

set -e

ENVNAME=Climate
ENVDIR=$ENVNAME

export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

ls -alF

python3 wetbulb_generate_reference_grid.py
