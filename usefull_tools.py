import numpy as np
import pandas as pd


def test_raw_data_and_load_data(gestures) : 
    '''founction used to test if the raw data loaded with load_data_domain_4
      is correct by comparing it with the raw data read directly from the csv files.'''
    
    # to use this function you need to add the file path in the getsure dictionnary
    # => "file_path": file_path  #useful for debug 
    for g in gestures:
        df_raw = pd.read_csv(
        g["file_path"],
        skiprows=5,
        header=None,
        usecols=[0,1,2]
    )
    
        df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna()
    
        if not np.allclose(g["trajectory"], df_raw.values):
            print("Mismatch detected:", g["file_path"])
            break
    else:
        print("✅ All trajectories are correct!")
    