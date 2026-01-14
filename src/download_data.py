import os
import pandas as pd
import shutil
from pathlib import Path
import kagglehub

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def download_dataset():
    print("Téléchargement du dataset HAM10000...")
    
    try:
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print("Path to dataset files:", path)
        
        download_path = Path(path)
        
        if download_path.exists():
            print(f"Copie des fichiers depuis {download_path} vers {RAW_DIR}")
            for item in download_path.iterdir():
                dest = RAW_DIR / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
            print("Fichiers copiés avec succès.")
        else:
            print("Erreur: dossier de téléchargement non trouvé.")
            return False
            
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        return False
    
    return True


def prepare_dataset(sample_size=None):
    print("\nPréparation du dataset...")
    
    metadata_files = list(RAW_DIR.glob("HAM10000_metadata*.csv"))
    
    if not metadata_files:
        print("Erreur: fichier de métadonnées non trouvé.")
        return False
    
    df_list = []
    for file in metadata_files:
        df_list.append(pd.read_csv(file))
    df = pd.concat(df_list, ignore_index=True)
    
    print(f"Total d'images: {len(df)}")
    print(f"\nDistribution des classes:")
    print(df['dx'].value_counts())
    
    if sample_size:
        df_sampled = df.groupby('dx').apply(
            lambda x: x.sample(min(len(x), sample_size), random_state=42)
        ).reset_index(drop=True)
        df = df_sampled
        print(f"\nÉchantillon sélectionné: {len(df)} images")
    
    for class_name in df['dx'].unique():
        class_dir = PROCESSED_DIR / class_name
        class_dir.mkdir(exist_ok=True)
    
    image_dirs = [
        RAW_DIR / "HAM10000_images_part_1",
        RAW_DIR / "HAM10000_images_part_2"
    ]
    
    copied_count = 0
    for idx, row in df.iterrows():
        image_id = row['image_id']
        class_name = row['dx']
        
        image_found = False
        for img_dir in image_dirs:
            src_path = img_dir / f"{image_id}.jpg"
            if src_path.exists():
                dst_path = PROCESSED_DIR / class_name / f"{image_id}.jpg"
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                image_found = True
                break
        
        if not image_found:
            print(f"Attention: image {image_id}.jpg non trouvée")
        
        if (idx + 1) % 1000 == 0:
            print(f"Progression: {idx + 1}/{len(df)} images traitées")
    
    print(f"\n{copied_count} images copiées dans {PROCESSED_DIR}")
    
    df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)
    print(f"Métadonnées sauvegardées dans {PROCESSED_DIR / 'metadata.csv'}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TÉLÉCHARGEMENT ET PRÉPARATION DU DATASET HAM10000")
    print("=" * 60)
    
    # Télécharger le dataset
    if not download_dataset():
        print("\nSi le téléchargement automatique échoue, vous pouvez:")
        print("1. Télécharger manuellement depuis Kaggle")
        print("2. Extraire dans le dossier 'data/raw/'")
        print("3. Relancer ce script")
    
    # Préparer le dataset (utiliser tout le dataset)
    # Modification: sample_size=None pour utiliser toutes les images disponibles
    prepare_dataset(sample_size=None)
    
    print("\n" + "=" * 60)
    print("PRÉPARATION TERMINÉE!")
    if not download_dataset():
        print("\nSi le téléchargement automatique échoue, vous pouvez:")
        print("1. Télécharger manuellement depuis Kaggle")
        print("2. Extraire dans le dossier 'data/raw/'")
        print("3. Relancer ce script")
    