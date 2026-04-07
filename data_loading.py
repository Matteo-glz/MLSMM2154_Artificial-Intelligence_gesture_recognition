import glob
import os
import pandas as pd
from collections import defaultdict

def load_data_domain_1(directory_path):
    '''Load gesture data from CSV files and return a list of gesture dictionaries. Each dictionary contains:
     - "gesture_id": unique identifier for the gesture
     - "subject": subject number (e.g., 1 for "subject1")
     - "gesture_type": type of gesture (e.g., 1, 2, 3)
     - "repetition": repetition number (e.g., 1, 2, 3)
     - "trajectory": numpy array of shape (n_samples, 3) containing the x, y, z coordinates of the gesture trajectory
    '''
    gestures = []
    files = sorted(glob.glob(os.path.join(directory_path, "*.csv")))                # get all CSV files in the directory

    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path).replace('.csv', '')
        parts = filename.split('-') 
        
        df = pd.read_csv(file_path, header=None, names=['x','y','z','t'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        trajectory = df[['x','y','z']].values                               # only keep the spatial coordinates, ignore the time
        

        gestures.append({
            "gesture_id": i,
            "subject": int(parts[0][-1]),                                   # only take the subject's number (e.g., "subject1" -> 1)
            "gesture_type": int(parts[1]),
            "repetition": int(parts[2]),
            "trajectory": trajectory                                        #numpy array of shape (n_samples, 3)
        })
    return gestures


def load_data_domain_4(directory_path):
    '''Load gesture data from txt files and return a list of gesture dictionaries. Each dictionary contains:
     - "gesture_id": unique identifier for the gesture
     - "subject": subject number (e.g., 1 for "subject1")
     - "gesture_type": type of gesture (e.g., 1, 2, 3)
     - "repetition": repetition number (e.g., 1, 2, 3)
     - "trajectory": numpy array of shape (n_samples, 3) containing the x, y, z coordinates of the gesture trajectory
    '''

    gestures = []

    files = sorted(glob.glob(os.path.join(directory_path, "*.txt")))

    # Mapping class_id = gesture name
    gesture_map = {
        0: "Cuboid",
        1: "Cylinder",
        2: "Sphere",
        3: "Rectangular Pipe",
        4: "Hemisphere",
        5: "Cylinder Pipe",
        6: "Pyramid",
        7: "Tetrahedron",
        8: "Cone",
        9: "Toroid"
    }

    # managint the repetition (key = subject, gesture_type)
    repetition_counter = defaultdict(int)

    for i, file_path in enumerate(files):

        # Quick reading of the tree first lines (without loading all)
        with open(file_path, 'r') as f:
            line1 = f.readline()            # unusefull, juste the domain name (4)
            line2 = f.readline()
            line3 = f.readline()

        gesture_type = int(line2.split('=')[1].strip())-1
        subject = int(line3.split('=')[1].strip())

        # Compute the repetition
        key = (subject, gesture_type)
        repetition_counter[key] += 1
        repetition = repetition_counter[key]

        # Optimized reading of data
        df = pd.read_csv(
            file_path,
            skiprows=5,
            header=None,
            usecols=[0, 1, 2],          # ignore t → gain of time
            names=['x', 'y', 'z']
        )

        # cleaning
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        trajectory = df.values  

        gestures.append({
            "gesture_id": i,
            "subject": subject,
            "gesture_type": gesture_type,
            "gesture_name": gesture_map.get(gesture_type, "Unknown"),
            "repetition": repetition,
            "trajectory": trajectory
        })

    return gestures