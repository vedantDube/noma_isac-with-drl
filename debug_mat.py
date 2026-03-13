import scipy.io
import os
import numpy as np

M = 50
fpath = f'./results/ris_M{M}/results.mat'
if os.path.exists(fpath):
    try:
        data = scipy.io.loadmat(fpath)
        print(f"Keys in {fpath}: {data.keys()}")
        print(f"CRB_td3 type: {type(data['CRB_td3'])}")
        print(f"CRB_td3 value: {data['CRB_td3']}")
        val = np.atleast_1d(data['CRB_td3']).flatten()[0]
        print(f"Extracted value: {float(val)}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("File not found")
