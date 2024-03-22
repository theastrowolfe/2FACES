#!/usr/bin/env python3

import os

WD = os.path.dirname(__file__)
cropped_path = os.path.join(WD, "cropped")
anc_path = os.path.abspath(os.path.join(WD, "..", "..", "anchor"))
pos_path = os.path.abspath(os.path.join(WD, "..", "..", "positive"))

for idx, cropped_file in enumerate(os.listdir(cropped_path)):
    old_path = os.path.join(cropped_path, cropped_file)
    new_path = os.path.join(anc_path if idx % 2 == 0 else pos_path, f"1_{cropped_file}")
    os.replace(old_path, new_path)