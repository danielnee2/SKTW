#!/bin/bash
cd $(git rev-parse --show-toplevel)
cd data/international/italy/COVID-19
git submodule update --remote
mkdir -p dati-regioni-en
mkdir -p dati-province-en
cd ../ # necessary to get out of git submodule

cd $(git rev-parse --show-toplevel)
cd code/processing/raw_data_processing
python3 clean_italy.py