#!/usr/bin/env python3
"""
Projet Machine Learning: Analyse des Accidents de la Route en France
====================================================================

Ce script principal orchestre l'ensemble du projet d'analyse des accidents
de la route, de la collecte des données à la modélisation prédictive.

Auteur: [Votre nom]
Date: [Date actuelle]
Cours: Machine Learning
"""

import os
import sys
import pandas as pd
import warnings
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Configuration des warnings
warnings.filterwarnings('ignore')

try:
    from data_preprocessing import AccidentDataProcessor
    from exploratory_analysis import AccidentExplorer  
    from ml_modeling import AccidentMLModeler
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous que tous les fichiers sont dans le même répertoire")
    sys.exit(1)

def setup_logging():
    """Configuration du système de logs"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"accident_analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class AccidentAnalysisProject:
    """
    Classe principale pour orchestrer le projet d'analyse des accidents
    """
    
    def __init__(self, data_dir="data", output_dir="outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Création des répertoires
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.processor = None
        self.explorer = None
        self.modeler = None
        
        # Résultats
        self.df_usagers = None
        self.df_accidents = None
        self.ml_results = None

    def step1_data_collection(self, years=[2021, 2022, 2023], force_download=False):
        """
        Étape 1: Collecte et préprocessing des données
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("ÉTAPE 1: COLLECTE ET PRÉPROCESSING DES DONNÉES")
        self.logger.info("\n" + "="*60)
        
        # Vérification si les données existent déjà
        clean_data_path = self.data_dir / 'accidents_clean.csv'
        accident_data_path = self.data_dir / 'accidents_by_event.csv'
        
        if clean_data_path.exists() and accident_data_path.exists() and not force_download:
            self.logger.info("📂 Données déjà présentes, chargement...")
            self.df_usagers = pd.read_csv(clean_data_path)
            self.df_accidents = pd.read_csv(accident_data_path)
            self.logger.info(f"✅ Données chargées: {len(self.df_usagers)} usagers, {len(self.df_accidents)} accidents")
        else:
            self.logger.info("🔄 Téléchargement et traitement des données...")
            self.processor = AccidentDataProcessor(self.data_dir)
            
            try:
                self.df_usagers, self.df_accidents = self.processor.process_all_years(years)
                if self.df_usagers is not None:
                    self.logger.info("✅ Préprocessing terminé avec succès")
                else:
                    self.logger.error("❌ Échec du préprocessing")
                    return False
            except Exception as e:
                self.logger.error(f"❌ Erreur lors du préprocessing: {e}")
                return False
        
        # Statistiques de base
        self.logger.info(f"📊 Dataset final:")
        self.logger.info(f"   - Usagers: {len(self.df_usagers):,}")
        self.logger.info(f"   - Accidents: {len(self.df_accidents):,}")
        self.logger.info(f"   - Période: {self.df_usagers['annee'].min()}-{self.df_usagers['annee'].max()}")
        
        return True

    def step2_exploratory_analysis(self, generate_visualizations=True):
        """
        Étape 2: Analyse exploratoire des données
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("ÉTAPE 2: ANALYSE EXPLORATOIRE")
        self.logger.info("\n" + "="*60)
        
        if self.df_usagers is None or self.df_accidents is None:
            self.logger.error("❌ Données non disponibles pour l'analyse")
            return False
        
        try:
            self.explorer = AccidentExplorer(self.df_usagers, self.df_accidents)
            
            if generate_visualizations:
                self.logger.info("📊 Génération de l'analyse exploratoire complète...")
                self.explorer.run_full_analysis()
            else:
                self.logger.info("📊 Génération des statistiques de base...")
                self.explorer.overview_statistics()
                self.explorer.generate_summary_report()
            
            self.logger.info("✅ Analyse exploratoire terminée")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'analyse exploratoire: {e}")
            return False

    def step3_machine_learning(self, balance_method='smote'):
        """
        Étape 3: Modélisation machine learning
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("ÉTAPE 3: MODÉLISATION MACHINE LEARNING")
        self.logger.info("\n" + "="*60)
        
        if self.df_usagers is None or self.df_accidents is None:
            self.logger.error("❌ Données non disponibles pour la modélisation")
            return False
        
        try:
            self.modeler = AccidentMLModeler(self.df_usagers, self.df_accidents)
            
            self.logger.info("🤖 Lancement du pipeline ML...")
            self.ml_results = self.modeler.run_full_pipeline(balance_method=balance_method)
            
            # Sauvegarde des résultats
            results_path = self.output_dir / 'ml_results_summary.txt'
            with open(results_path, 'w') as f:
                f.write("RÉSULTATS DE LA MODÉLISATION ML\n")
                f.write("="*40 + "\n\n")
                f.write(f"Meilleur modèle: {self.ml_results['best_model_name']}\n")
                f.write("\nPerformances par modèle:\n")
                for model, results in self.ml_results['results'].items():
                    f.write(f"{model}: F1={results['test_f1']:.3f}, AUC={results['test_auc']:.3f}\n")
            
            self.logger.info(f"📁 Résultats sauvegardés: {results_path}")
            self.logger.info("✅ Modélisation terminée avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la modélisation: {e}")
            return False

    def generate_final_report(self):
        """
        Génération du rapport final du projet
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("GÉNÉRATION DU RAPPORT FINAL")
        self.logger.info("\n" + "="*60)
        
        report_path = self.output_dir / 'rapport_final.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Projet Machine Learning: Analyse des Accidents de la Route en France\n\n")
            
            f.write("## 1. Problématique\n\n")
            f.write("**Question principale:** Quels sont les facteurs les plus liés aux accidents mortels sur la route en France ?\n\n")
            
            f.write("## 2. Données\n\n")
            f.write(f"- **Source:** data.gouv.fr - Base de données des accidents corporels\n")
            f.write(f"- **Période:** {self.df_usagers['annee'].min()}-{self.df_usagers['annee'].max()}\n")
            f.write(f"- **Volume:** {len(self.df_usagers):,} usagers impliqués dans {len(self.df_accidents):,} accidents\n")
            f.write(f"- **Taux de mortalité:** {self.df_accidents['accident_mortel'].mean()*100:.2f}% des accidents\n\n")
            
            f.write("## 3. Méthodologie\n\n")
            f.write("1. **Collecte et nettoyage:** Fusion des 4 fichiers CSV, gestion des valeurs manquantes\n")
            f.write("2. **Analyse exploratoire:** Visualisations temporelles, géographiques, par conditions\n")
            f.write("3. **Modélisation ML:** Test de 4 algorithmes avec validation croisée\n\n")
            
            if self.ml_results:
                f.write("## 4. Résultats\n\n")
                f.write(f"**Meilleur modèle:** {self.ml_results['best_model_name']}\n")
                best_results = self.ml_results['results'][self.ml_results['best_model_name']]
                f.write(f"- F1-Score: {best_results['test_f1']:.3f}\n")
                f.write(f"- AUC-ROC: {best_results['test_auc']:.3f}\n")
                f.write(f"- Accuracy: {best_results['test_accuracy']:.3f}\n\n")
                
                f.write("### Classement des modèles:\n")
                for i, (model, results) in enumerate(sorted(self.ml_results['results'].items(), 
                                                          key=lambda x: x[1]['test_f1'], reverse=True), 1):
                    f.write(f"{i}. {model}: F1={results['test_f1']:.3f}\n")
            
            f.write("\n## 5. Facteurs de Risque Identifiés\n\n")
            f.write("- **Temporels:** Heures de pointe, week-ends\n")
            f.write("- **Météorologiques:** Conditions dégradées (pluie, brouillard)\n") 
            f.write("- **Démographiques:** Âge des conducteurs\n")
            f.write("- **Géographiques:** Types de voies, zones urbaines/rurales\n\n")
            
            f.write("## 6. Limites et Perspectives\n\n")
            f.write("- **Limites:** Déséquilibre des classes, données manquantes\n")
            f.write("- **Améliorations:** Données météo précises, données comportementales\n")
            f.write("- **Applications:** Prévention, allocation des ressources de sécurité\n")
        
        self.logger.info(f"📄 Rapport final généré: {report_path}")

    def run_complete_project(self, years=[2021, 2022, 2023], 
                           force_download=False, 
                           skip_visualizations=False,
                           balance_method='smote'):
        """
        Exécution complète du projet
        """
        start_time = datetime.now()
        
        self.logger.info("🚀 DÉMARRAGE DU PROJET COMPLET")
        self.logger.info(f"Paramètres: années={years}, force_download={force_download}")
        
        # Étape 1: Données
        success = self.step1_data_collection(years, force_download)
        if not success:
            self.logger.error("❌ Échec à l'étape 1 - Arrêt du projet")
            return False
        
        # Étape 2: Analyse exploratoire
        success = self.step2_exploratory_analysis(not skip_visualizations)
        if not success:
            self.logger.error("❌ Échec à l'étape 2 - Arrêt du projet")
            return False
        
        # Étape 3: Machine Learning
        success = self.step3_machine_learning(balance_method)
        if not success:
            self.logger.error("❌ Échec à l'étape 3 - Arrêt du projet")
            return False
        
        # Rapport final
        self.generate_final_report()
        
        # Bilan
        duration = datetime.now() - start_time
        self.logger.info("\n" + "🎉" + "="*58 + "🎉")
        self.logger.info("           PROJET TERMINÉ AVEC SUCCÈS !")
        self.logger.info("\n" + "="*60)
        self.logger.info(f"⏱️  Durée totale: {duration}")
        self.logger.info("📁 Fichiers générés:")
        self.logger.info("   - data/accidents_clean.csv")
        self.logger.info("   - data/accidents_by_event.csv") 
        self.logger.info("   - best_accident_model.pkl")
        self.logger.info("   - outputs/rapport_final.md")
        self.logger.info("   - logs/accident_analysis_*.log")
        
        return True

def main():
    """Fonction principale avec arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description="Projet ML: Analyse des Accidents de la Route")
    
    parser.add_argument('--years', nargs='+', type=int, default=[2021, 2022, 2023],
                       help='Années à analyser (défaut: 2021 2022 2023)')
    parser.add_argument('--force-download', action='store_true',
                       help='Forcer le téléchargement même si les données existent')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Ignorer les visualisations (plus rapide)')
    parser.add_argument('--balance-method', choices=['smote', 'undersample', 'smoteenn', None],
                       default='smote', help='Méthode de rééquilibrage des classes')
    parser.add_argument('--step', choices=['data', 'explore', 'ml', 'all'], 
                       default='all', help='Étape à exécuter')
    
    args = parser.parse_args()
    
    # Configuration du logging
    logger = setup_logging()
    
    # Initialisation du projet
    project = AccidentAnalysisProject()
    
    # Exécution selon l'étape demandée
    if args.step == 'all':
        success = project.run_complete_project(
            years=args.years,
            force_download=args.force_download,
            skip_visualizations=args.skip_viz,
            balance_method=args.balance_method
        )
    elif args.step == 'data':
        success = project.step1_data_collection(args.years, args.force_download)
    elif args.step == 'explore':
        success = project.step1_data_collection(args.years, False)  # Chargement des données
        if success:
            success = project.step2_exploratory_analysis(not args.skip_viz)
    elif args.step == 'ml':
        success = project.step1_data_collection(args.years, False)  # Chargement des données
        if success:
            success = project.step3_machine_learning(args.balance_method)
    
    if not success:
        logger.error("❌ Échec de l'exécution")
        sys.exit(1)
    else:
        logger.info("✅ Exécution terminée avec succès")

if __name__ == "__main__":
    main()