import pandas as pd
import glob
import os
from sklearn.decomposition import PCA

def load_and_preprocess_2D(directory_path):
    all_chunks = []
    files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not files:
        print("Erreur : Aucun fichier trouvé.")
        return None

    # 1. Chargement et étiquetage
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path).replace('.csv', '')
        parts = filename.split('-') 
        
        df = pd.read_csv(file_path, header=None, names=['x','y','z','t'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        df['subject'] = float(parts[0][-1]) #pour ne pas garder le 'subject' dans la colonne et n'avoir que des floats
        df['gesture_type'] = float(parts[1])
        df['repetition'] = float(parts[2])
        df['gesture_id'] = float(i)  # Identifiant unique pour ce geste précis
        all_chunks.append(df)
    
    full_df = pd.concat(all_chunks, ignore_index=True)

    # 2. Fonction de Normalisation + Projection 2D
    def process_to_2d(group, gesture_id):
        cols = ['x', 'y', 'z']
        
        # Centrage local
        group_centered = group[cols] - group[cols].mean()
        
        # Standardisation locale (par axe)
        group_centered = group_centered / group_centered.std()
        
        # PCA 2D : projection 3D -> 2D
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(group_centered)
        
        # Reconstruction du DataFrame
        return pd.DataFrame({
            'PC1': coords_2d[:, 0],
            'PC2': coords_2d[:, 1],
            'subject': group['subject'].iloc[0],
            'gesture_type': group['gesture_type'].iloc[0],
            'repetition': group['repetition'].iloc[0],
            'gesture_id': gesture_id
            
        })
    print("Transformation 3D -> 2D (PCA) en cours...")

    dfs = []

    for gesture_id, group in full_df.groupby('gesture_id'):
        df_processed = process_to_2d(group, gesture_id)
        dfs.append(df_processed)

    df_final_2d = pd.concat(dfs, ignore_index=True)

    return df_final_2d

# --- TEST ---
path = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/Domain1_csv"
gesture_data = load_and_preprocess_2D(path)

print(gesture_data["PC1"])

