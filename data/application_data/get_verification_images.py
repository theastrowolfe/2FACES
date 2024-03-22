#!/usr/bin/env python3

import numpy as np
import os
import shutil

WD = os.path.dirname(__file__)
verification_dir = os.path.abspath(os.path.join(WD, "verification_images"))
positive_dir = os.path.abspath(os.path.join(WD, "..", "positive"))

for positive_image in np.random.choice(os.listdir(positive_dir), 25, replace=False):
    src = os.path.join(positive_dir, positive_image)
    dst = os.path.join(verification_dir, positive_image)
    shutil.copy2(src, dst)