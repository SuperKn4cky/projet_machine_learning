import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AccidentDataProcessor:
    """
    Classe pour télécharger, nettoyer et préparer les données d'accidents de la route
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # URLs des données 2023 (mises à jour avec les liens actuels)
        self.data_urls_2023 = {
            'caracteristiques': 'https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241028-103125/caract-2023.csv',
            'lieux': 'https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153219/lieux-2023.csv',
            'vehicules': 'https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153253/vehicules-2023.csv',
            'usagers': 'https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153328/usagers-2023.csv'
        }
        
        # Mappings pour le recodage des variables
        self.gravite_mapping = {
            1: 'Indemne',
            2: 'Tué',
            3: 'Blessé hospitalisé', 
            4: 'Blessé léger'
        }
        
        self.meteo_mapping = {
            1: 'Normale',
            2: 'Pluie légère',
            3: 'Pluie forte',
            4: 'Neige - grêle',
            5: 'Brouillard - fumée',
            6: 'Vent fort - tempête',
            7: 'Temps éblouissant',
            8: 'Temps couvert',
            9: 'Autre'
        }
        
        self.lumiere_mapping = {
            1: 'Plein jour',
            2: 'Crépuscule ou aube',
            3: 'Nuit sans éclairage public',
            4: 'Nuit avec éclairage public non allumé',
            5: 'Nuit avec éclairage public allumé'
        }

    def download_data(self, year):
        """
        Télécharge les données pour une année donnée
        """
        if year == 2023:
            return self.download_data_2023()
        else:
            print(f"Données non disponibles pour l'année {year}")
            return False
    
    def download_data_2023(self):
        """
        Télécharge les 4 fichiers CSV pour 2023
        """
        year_dir = self.data_dir / "2023"
        year_dir.mkdir(exist_ok=True)
        
        try:
            print("Téléchargement des données 2023...")
            
            for file_type, url in self.data_urls_2023.items():
                filename = f"{file_type}-2023.csv"
                filepath = year_dir / filename
                
                print(f"  - Téléchargement {file_type}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"    {filename} téléchargé ({filepath.stat().st_size // 1024} KB)")
            
            print("Données 2023 téléchargées avec succès")
            return True
            
        except Exception as e:
            print(f"Erreur lors du téléchargement 2023: {e}")
            return False

    def load_yearly_data(self, year):
        """
        Charge les 4 fichiers CSV d'une année et les fusionne
        """
        year_dir = self.data_dir / str(year)
        
        # Noms des fichiers (peuvent varier selon l'année)
        file_patterns = {
            'caracteristiques': f'*caracteristiques*{year}*.csv',
            'lieux': f'*lieux*{year}*.csv', 
            'vehicules': f'*vehicules*{year}*.csv',
            'usagers': f'*usagers*{year}*.csv'
        }
        
        dataframes = {}
        
        for key, pattern in file_patterns.items():
            files = list(year_dir.glob(pattern))
            if files:
                print(f"Chargement {key} {year}: {files[0].name}")
                df = pd.read_csv(files[0], encoding='utf-8', sep=';', low_memory=False)
                dataframes[key] = df
            else:
                print(f"Fichier {key} non trouvé pour {year}")
                return None
        
        return self.merge_dataframes(dataframes, year)

    def merge_dataframes(self, dataframes, year):
        """
        Fusionne les 4 DataFrames sur les clés communes
        """
        # Fusion progressive des tables
        df = dataframes['caracteristiques']
        
        # Ajout des informations de lieu
        df = df.merge(dataframes['lieux'], on='Num_Acc', how='left')
        
        # Ajout des informations véhicules
        df = df.merge(dataframes['vehicules'], on='Num_Acc', how='left')
        
        # Ajout des informations usagers
        df = df.merge(dataframes['usagers'], on=['Num_Acc', 'id_vehicule'], how='left')
        
        df['annee'] = year
        print(f"Dataset {year} fusionné: {len(df)} lignes")
        
        return df

    def clean_data(self, df):
        """
        Nettoie et prépare les données
        """
        print("Nettoyage des données...")
        
        # Suppression des lignes avec des valeurs critiques manquantes
        df = df.dropna(subset=['grav'])
        
        # Recodage des variables catégorielles
        df['gravite_lib'] = df['grav'].map(self.gravite_mapping)
        
        # Gestion sécurisée des colonnes optionnelles
        if 'meteo' in df.columns:
            df['meteo_lib'] = df['meteo'].map(self.meteo_mapping)
        else:
            df['meteo_lib'] = 'Non renseigné'
            
        if 'lum' in df.columns:
            df['lumiere_lib'] = df['lum'].map(self.lumiere_mapping)
        else:
            df['lumiere_lib'] = 'Non renseigné'
        
        # Création de la variable cible binaire
        df['accident_mortel'] = (df['grav'] == 2).astype(int)
        
        # Nettoyage des heures
        if 'hrmn' in df.columns:
            df['heure'] = df['hrmn'].astype(str).str[:2].astype(int, errors='ignore')
            df['minute'] = df['hrmn'].astype(str).str[2:].astype(int, errors='ignore')
        
        # Gestion de l'âge
        if 'an_nais' in df.columns:
            df['age'] = df['annee'] - df['an_nais']
            df['age'] = df['age'].clip(0, 100)  # Valeurs aberrantes
            
        # Création de catégories d'âge
        df['tranche_age'] = pd.cut(df.get('age', 0), 
                                 bins=[0, 18, 25, 35, 50, 65, 100], 
                                 labels=['0-17', '18-24', '25-34', '35-49', '50-64', '65+'])
        
        # Conversion des coordonnées géographiques
        if 'lat' in df.columns and 'long' in df.columns:
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['long'] = pd.to_numeric(df['long'], errors='coerce')
        
        print(f"Données nettoyées: {len(df)} lignes")
        return df

    def create_features(self, df):
        """
        Crée des features supplémentaires pour l'analyse
        """
        print("Création de features...")
        
        # Features temporelles
        # Assurer que 'date' est en format datetime
        if 'date' not in df.columns:
            # Créer une colonne 'date' à partir des colonnes existantes
            df['date'] = pd.to_datetime({
                'year': df['an'],
                'month': df['mois'],
                'day': df['jour']
            }, errors='coerce')
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Créer les caractéristiques temporelles
        df['annee'] = df['date'].dt.year
        df['mois'] = df['date'].dt.month
        df['jour_semaine'] = df['date'].dt.dayofweek
        df['weekend'] = (df['jour_semaine'] >= 5).astype(int)
        
        # Encodage numérique pour les conditions météo
        if 'meteo' in df.columns:
            df['meteo_encoded'] = df['meteo'].astype(int)
        else:
            df['meteo_encoded'] = 0  # Valeur par défaut
            
        # Encodage numérique pour les conditions de lumière
        if 'lum' in df.columns:
            df['lum_encoded'] = df['lum'].astype(int)
        else:
            df['lum_encoded'] = 0  # Valeur par défaut
            
        # Features sur les conditions difficiles
        meteo_difficile = df['meteo_encoded'].isin([2, 3, 4, 5, 6])
        lum_difficile = df['lum_encoded'].isin([2, 3, 4])
        df['conditions_difficiles'] = (meteo_difficile | lum_difficile).astype(int)
        
        # Agrégation par accident
        accident_stats = df.groupby('Num_Acc').agg({
            'accident_mortel': 'max',  # Au moins un mort dans l'accident
            'grav': ['count', lambda x: (x == 2).sum()],  # Nb victimes et nb morts
            'id_vehicule': 'nunique',  # Nb véhicules impliqués
            'annee': 'first',  # Conserver l'année de l'accident
            'jour_semaine': 'first',  # Conserver le jour de la semaine
            'weekend': 'first',  # Conserver l'information weekend
            'meteo_encoded': 'first',  # Conserver la météo encodée
            'lum_encoded': 'first'  # Conserver les conditions de lumière encodées
        }).round(2)
        
        accident_stats.columns = ['accident_mortel', 'nb_victimes', 'nb_morts', 'nb_vehicules', 'annee', 'jour_semaine', 'weekend', 'meteo_encoded', 'lum_encoded']
        accident_stats = accident_stats.reset_index()
        accident_stats['accident_mortel'] = accident_stats['accident_mortel'].astype(int)
        
        return df, accident_stats

    def process_all_years(self, years=[2023]):
        """
        Traite toutes les années et combine les données
        """
        all_data = []
        
        for year in years:
            print(f"\n=== Traitement année {year} ===")
            
            # Téléchargement si nécessaire
            if not (self.data_dir / str(year)).exists():
                if not self.download_data(year):
                    continue
            
            # Chargement et nettoyage
            df = self.load_yearly_data(year)
            if df is not None:
                df = self.clean_data(df)
                all_data.append(df)
        
        if all_data:
            # Combinaison de toutes les années
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\nDataset final: {len(combined_df)} lignes")
            
            # Création des features
            final_df, accident_stats = self.create_features(combined_df)
            
            # Sauvegarde
            final_df.to_csv(self.data_dir / 'accidents_clean.csv', index=False)
            accident_stats.to_csv(self.data_dir / 'accidents_by_event.csv', index=False)
            
            print("Données sauvegardées:")
            print(f"- accidents_clean.csv: {len(final_df)} lignes")
            print(f"- accidents_by_event.csv: {len(accident_stats)} accidents")
            
            return final_df, accident_stats
        
        return None, None

# Utilisation
if __name__ == "__main__":
    processor = AccidentDataProcessor()
    
    # Traitement des données
    df_usagers, df_accidents = processor.process_all_years([2023])
    
    if df_usagers is not None:
        print("\n=== Aperçu des données ===")
        print(f"Shape: {df_usagers.shape}")
        print("\nColonnes disponibles:")
        print(df_usagers.columns.tolist())
        
        print("\nRépartition gravité:")
        print(df_usagers['gravite_lib'].value_counts())
        
        print("\nAccidents mortels par année:")
        print(df_usagers.groupby('annee')['accident_mortel'].agg(['count', 'sum', 'mean']))