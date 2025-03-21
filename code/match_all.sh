#!/bin/bash

for file in "data/Grid Images"/*; do
    python code/template_match.py "$file"
done