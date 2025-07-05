#!/bin/bash

for file in "data/template_matching"/*; do
    python code/extract_cells.py "$file"
done