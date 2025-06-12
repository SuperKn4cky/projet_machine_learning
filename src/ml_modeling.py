import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class AccidentMLModeler:
    """
    Classe pour la modélisation machine learning des accidents mortels
    """
    
    def __init__(self, df_usagers, df_accidents):
        self.df_usagers = df_usagers
        self.df_accidents = df_accidents
        self.models = {}
        self.results = {}
        self.best_model = None
        
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728'
        }

    def prepare_features(self):
        """
        Préparation des features pour le machine learning
        """
        print("🔧 Préparation des features...")
        
        df = self.df_accidents.copy()
        
        accident_features = self.df_usagers.groupby('Num_Acc').agg({
            'heure': 'first',
            'mois': 'first', 
            'jour_semaine': 'first',
            'weekend': 'first',
            'age': ['mean', 'min', 'max'],
            'meteo_encoded': 'first',
            'lum_encoded': 'first',
            'conditions_difficiles': 'first',
            'lat': 'first',
            'long': 'first'
        }).round(2)
        
        accident_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                   for col in accident_features.columns]
        accident_features = accident_features.reset_index()
        
        df = df.merge(accident_features, on='Num_Acc', how='left')
        
        df['nb_vehicules_cat'] = pd.cut(df['nb_vehicules'], 
                                       bins=[0, 1, 2, 5, float('inf')], 
                                       labels=['1', '2', '3-5', '6+'])
        
        df['nb_victimes_cat'] = pd.cut(df['nb_victimes'], 
                                      bins=[0, 1, 3, 5, float('inf')], 
                                      labels=['1', '2-3', '4-5', '6+'])
        
        df['age_mean'].fillna(df['age_mean'].median(), inplace=True)
        df['meteo_encoded'].fillna(1, inplace=True)  # 1 = 'Normal'
        df['lum_encoded'].fillna(1, inplace=True)    # 1 = 'Plein jour'
        
        self.df_ml = df
        print(f"✅ Dataset ML préparé: {len(df)} accidents")
        return df

    def define_features_target(self):
        """
        Définition des features et de la target
        """
        # Variables catégorielles
        # Remplacer les catégories par des valeurs numériques
        self.df_ml['nb_vehicules'] = self.df_ml['nb_vehicules'].astype(int)
        self.df_ml['nb_victimes'] = self.df_ml['nb_victimes'].astype(int)
        
        # Supprimer les colonnes catégorielles problématiques
        if 'nb_vehicules_cat' in self.df_ml.columns:
            self.df_ml.drop('nb_vehicules_cat', axis=1, inplace=True)
        if 'nb_victimes_cat' in self.df_ml.columns:
            self.df_ml.drop('nb_victimes_cat', axis=1, inplace=True)
            
        # Aucune caractéristique catégorielle restante
        categorical_features = []
        
        # Variables numériques
        numerical_features = []
        for col in ['heure', 'mois', 'jour_semaine', 'weekend', 'nb_vehicules', 
                   'nb_victimes', 'age_mean', 'conditions_difficiles']:
            if col in self.df_ml.columns:
                numerical_features.append(col)
        
        target = 'accident_mortel'
        
        print(f"Features catégorielles ({len(categorical_features)}): {categorical_features}")
        print(f"Features numériques ({len(numerical_features)}): {numerical_features}")
        
        return categorical_features, numerical_features, target

    def create_preprocessor(self, categorical_features, numerical_features):
        """
        Création du preprocesseur pour les données
        """
        # Preprocesseur pour variables numériques
        numerical_transformer = StandardScaler()
        
        # Preprocesseur pour variables catégorielles
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        # Combinaison
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor

    def split_data(self, categorical_features, numerical_features, target):
        """
        Division des données en train/test
        """
        # Préparation des features
        feature_cols = categorical_features + numerical_features
        X = self.df_ml[feature_cols]
        y = self.df_ml[target]
        
        # Split stratifié d'abord
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Suppression des valeurs manquantes après le split
        train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        
        X_train = X_train[train_mask].reset_index(drop=True)
        y_train = y_train[train_mask].reset_index(drop=True)
        X_test = X_test[test_mask].reset_index(drop=True)
        y_test = y_test[test_mask].reset_index(drop=True)
        
        # Vérification des tailles
        assert len(X_train) == len(y_train), f"X_train et y_train ont des tailles différentes: {len(X_train)} vs {len(y_train)}"
        assert len(X_test) == len(y_test), f"X_test et y_test ont des tailles différentes: {len(X_test)} vs {len(y_test)}"
        print(f"✅ Vérification des tailles : X_train={len(X_train)}, y_train={len(y_train)}, X_test={len(X_test)}, y_test={len(y_test)}")
        
        print(f"📊 Données d'entraînement: {len(X_train)} ({y_train.mean():.3f} positifs)")
        print(f"📊 Données de test: {len(X_test)} ({y_test.mean():.3f} positifs)")
        
        return X_train, X_test, y_train, y_test

    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """
        Gestion du déséquilibre des classes
        """
        print(f"⚖️  Gestion du déséquilibre avec {method}...")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=42)
        else:
            return X_train, y_train
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"   Avant: {len(X_train)} échantillons ({y_train.mean():.3f} positifs)")
        print(f"   Après: {len(X_resampled)} échantillons ({y_resampled.mean():.3f} positifs)")
        
        return X_resampled, y_resampled

    def define_models(self):
        """
        Définition des modèles à tester avec paramètres optimisés
        """
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'model__C': np.logspace(-3, 3, 7),  # Plage logarithmique
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'model__n_estimators': [50, 100, 150],  # Réduction des valeurs
                    'model__max_depth': [None, 10, 15],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'model__n_estimators': [50, 100],  # Réduction des valeurs
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__max_depth': [3, 4],
                    'model__subsample': [0.8]
                }
            },
            'LinearSVM': {
                'model': SGDClassifier(loss='log_loss', random_state=42, n_jobs=-1),
                'params': {
                    'model__alpha': [0.0001, 0.001, 0.01],
                    'model__penalty': ['l2', 'l1'],
                    'model__max_iter': [1000]
                }
            }
        }
        
        return models

    def train_models(self, X_train, X_test, y_train, y_test, preprocessor):
        """
        Entraînement et évaluation des modèles
        """
        print("🤖 Entraînement des modèles...")
        
        models_config = self.define_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, config in models_config.items():
            print(f"\n--- {name} ---")
            
            # Création du pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', config['model'])
            ])
            
            # Grid Search avec validation croisée
            print("🔍 Recherche aléatoire des meilleurs hyperparamètres...")
            # Utilisation de RandomizedSearchCV pour accélérer la recherche
            randomized_search = RandomizedSearchCV(
                pipeline,
                config['params'],
                n_iter=15,  # Réduction du nombre d'itérations
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1,  # Activation des logs pour suivre la progression
                random_state=42
            )
            
            randomized_search.fit(X_train, y_train)
            
            # Meilleur modèle
            best_model = randomized_search.best_estimator_
            
            # Prédictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Métriques
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_proba)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Stockage des résultats
            self.models[name] = best_model
            self.results[name] = {
                'best_params': randomized_search.best_params_,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'test_accuracy': test_accuracy,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba,
                'cv_score': randomized_search.best_score_,
                'true_y_test': y_test  # Stocker les vrais labels
            }
            
            print(f"✅ Meilleurs params: {randomized_search.best_params_}")
            print(f"📊 Train F1: {train_f1:.3f}")
            print(f"📊 Test F1: {test_f1:.3f}")
            print(f"📊 Test AUC: {test_auc:.3f}")
            print(f"📊 CV Score: {randomized_search.best_score_:.3f}")

    def evaluate_models(self, y_test):
        """
        Évaluation comparative des modèles
        """
        print("\n🏆 COMPARAISON DES MODÈLES")
        print("\n" + "="*60)
        
        # Tableau de comparaison
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('test_f1', ascending=False)
        
        print(results_df[['test_f1', 'test_auc', 'test_accuracy', 'cv_score']].round(3))
        
        # Identification du meilleur modèle
        best_model_name = results_df.index[0]
        self.best_model = self.models[best_model_name]
        
        print(f"\n🥇 Meilleur modèle: {best_model_name}")
        print(f"   F1-Score: {results_df.loc[best_model_name, 'test_f1']:.3f}")
        print(f"   AUC: {results_df.loc[best_model_name, 'test_auc']:.3f}")
        
        return best_model_name

    def plot_model_comparison(self):
        """
        Graphiques de comparaison des modèles
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Comparaison des métriques
        metrics_df = pd.DataFrame(self.results).T[['test_f1', 'test_auc', 'test_accuracy']]
        
        ax1 = axes[0, 0]
        metrics_df.plot(kind='bar', ax=ax1)
        ax1.set_title('Comparaison des Métriques par Modèle')
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Courbes ROC
        ax2 = axes[0, 1]
        # Utilisation des vrais labels y_test pour les courbes ROC
        true_y_test = None
        for result in self.results.values():
            if 'true_y_test' in result:
                true_y_test = result['true_y_test']
                break
        
        if true_y_test is None:
            print("⚠️  Impossible de trouver les vrais labels y_test pour les courbes ROC")
            return
            
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(true_y_test, result['y_test_proba'])
            ax2.plot(fpr, tpr, label=f"{name} (AUC={result['test_auc']:.3f})")
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_title('Courbes ROC')
        ax2.set_xlabel('Taux de Faux Positifs')
        ax2.set_ylabel('Taux de Vrais Positifs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Matrices de confusion du meilleur modèle
        best_model_name = list(self.results.keys())[0]  # Assume le premier est le meilleur
        y_test_pred = self.results[best_model_name]['y_test_pred']
        y_test = self.results[best_model_name]['true_y_test']  # Utilisation des vrais labels
        
        ax3 = axes[1, 0]
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax3, cmap='Blues')
        ax3.set_title(f'Matrice de Confusion - {best_model_name}')
        ax3.set_xlabel('Prédictions')
        ax3.set_ylabel('Réalité')
        
        # 4. Feature importance (si Random Forest)
        ax4 = axes[1, 1]
        if 'RandomForest' in self.models:
            model = self.models['RandomForest']
            # Récupération des noms de features après preprocessing
            feature_names = (model.named_steps['preprocessor']
                           .get_feature_names_out())
            
            importances = model.named_steps['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15
            
            ax4.barh(range(len(indices)), importances[indices])
            ax4.set_yticks(range(len(indices)))
            ax4.set_yticklabels([feature_names[i] for i in indices])
            ax4.set_title('Importance des Features (Random Forest)')
            ax4.set_xlabel('Importance')
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nnon disponible', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
        
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/ml_plot_model_comparison.png")
        plt.show()

    def analyze_feature_importance(self):
        """
        Analyse détaillée de l'importance des features
        """
        if 'RandomForest' not in self.models:
            print("⚠️  Feature importance disponible uniquement pour Random Forest")
            return
        
        model = self.models['RandomForest']
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = model.named_steps['model'].feature_importances_
        
        # Création d'un DataFrame pour l'analyse
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 TOP 20 FEATURES LES PLUS IMPORTANTES")
        print("="*50)
        for i, row in feature_df.head(20).iterrows():
            print(f"{row['feature']:.<40} {row['importance']:.4f}")
        
        # Graphique détaillé
        plt.figure(figsize=(12, 10))
        top_features = feature_df.head(20)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Features les Plus Importantes (Random Forest)')
        plt.gca().invert_yaxis()
        
        # Coloration des barres
        for i, bar in enumerate(bars):
            if i < 5:  # Top 5 en rouge
                bar.set_color(self.colors['danger'])
            elif i < 10:  # Top 6-10 en orange
                bar.set_color(self.colors['secondary'])
            else:  # Autres en bleu
                bar.set_color(self.colors['primary'])
        
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/ml_feature_importance.png")
        plt.show()
        
        return feature_df

    def detailed_error_analysis(self, X_test, y_test):
        """
        Analyse détaillée des erreurs du meilleur modèle
        """
        if self.best_model is None:
            print("⚠️  Aucun modèle entraîné")
            return
        
        print("\n🔍 ANALYSE DES ERREURS")
        print("="*40)
        
        # Prédictions du meilleur modèle
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Identification des erreurs
        false_positives = (y_test == 0) & (y_pred == 1)
        false_negatives = (y_test == 1) & (y_pred == 0)
        
        print(f"Faux positifs: {false_positives.sum()}")
        print(f"Faux négatifs: {false_negatives.sum()}")
        
        # Analyse des seuils
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred_thresh)
            precision = (y_pred_thresh & y_test).sum() / y_pred_thresh.sum() if y_pred_thresh.sum() > 0 else 0
            recall = (y_pred_thresh & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
            
            threshold_metrics.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
        
        threshold_df = pd.DataFrame(threshold_metrics)
        
        # Graphique des seuils
        plt.figure(figsize=(12, 6))
        plt.plot(threshold_df['threshold'], threshold_df['f1'], 'o-', label='F1-Score')
        plt.plot(threshold_df['threshold'], threshold_df['precision'], 's-', label='Precision')
        plt.plot(threshold_df['threshold'], threshold_df['recall'], '^-', label='Recall')
        plt.xlabel('Seuil de Classification')
        plt.ylabel('Score')
        plt.title('Impact du Seuil de Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/ml_error_analysis.png")
        plt.show()
        
        # Seuil optimal
        optimal_threshold = threshold_df.loc[threshold_df['f1'].idxmax(), 'threshold']
        print(f"\n🎯 Seuil optimal: {optimal_threshold:.2f}")
        
        return threshold_df

    def save_best_model(self, filepath='best_accident_model.pkl'):
        """
        Sauvegarde du meilleur modèle
        """
        if self.best_model is None:
            print("⚠️  Aucun modèle à sauvegarder")
            return
        
        # Sauvegarde du modèle
        joblib.dump(self.best_model, filepath)
        
        # Sauvegarde des métadonnées
        metadata = {
            'model_type': type(self.best_model.named_steps['model']).__name__,
            'features': list(self.best_model.named_steps['preprocessor'].get_feature_names_out()),
            'performance': self.results,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Modèle sauvegardé: {filepath}")
        print(f"✅ Métadonnées sauvegardées: {metadata_path}")

    def generate_ml_report(self):
        """
        Génération du rapport de modélisation
        """
        print("\n" + "="*70)
        print("                    RAPPORT DE MODÉLISATION ML")
        print("="*70)
        
        # Résumé des modèles testés
        print(f"\n📊 MODÈLES TESTÉS: {len(self.models)}")
        results_df = pd.DataFrame(self.results).T.sort_values('test_f1', ascending=False)
        
        print("\n🏆 CLASSEMENT DES MODÈLES (par F1-Score):")
        for i, (model_name, row) in enumerate(results_df.iterrows(), 1):
            print(f"   {i}. {model_name:.<20} F1: {row['test_f1']:.3f} | AUC: {row['test_auc']:.3f}")
        
        # Analyse du meilleur modèle
        best_model_name = results_df.index[0]
        best_results = results_df.iloc[0]
        
        print(f"\n🥇 MEILLEUR MODÈLE: {best_model_name}")
        print(f"   • F1-Score: {best_results['test_f1']:.3f}")
        print(f"   • AUC-ROC: {best_results['test_auc']:.3f}")
        print(f"   • Accuracy: {best_results['test_accuracy']:.3f}")
        print(f"   • CV Score: {best_results['cv_score']:.3f}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if best_results['test_f1'] > 0.7:
            print("   ✅ Performance satisfaisante pour la mise en production")
        elif best_results['test_f1'] > 0.5:
            print("   ⚠️  Performance modérée - amélioration souhaitable")
        else:
            print("   ❌ Performance insuffisante - révision nécessaire")
        
        if best_results['test_auc'] > 0.8:
            print("   ✅ Excellente capacité de discrimination")
        elif best_results['test_auc'] > 0.7:
            print("   ⚠️  Bonne capacité de discrimination")
        else:
            print("   ❌ Capacité de discrimination limitée")
        
        print(f"\n🎯 PROCHAINES ÉTAPES:")
        print("   1. Ajuster le seuil de classification selon les besoins métier")
        print("   2. Collecter plus de données si performance insuffisante")
        print("   3. Ingénierie de features supplémentaires")
        print("   4. Test sur données de validation externe")
        
        print("="*70)

    def run_full_pipeline(self, balance_method='smote'):
        """
        Pipeline complet de modélisation
        """
        print("🚀 LANCEMENT DU PIPELINE ML COMPLET")
        print("="*50)
        
        # 1. Préparation des données
        df = self.prepare_features()
        categorical_features, numerical_features, target = self.define_features_target()
        
        # 2. Split des données
        X_train, X_test, y_train, y_test = self.split_data(categorical_features, numerical_features, target)
        
        # 3. Préprocesseur
        preprocessor = self.create_preprocessor(categorical_features, numerical_features)
        
        # 4. Gestion du déséquilibre (optionnel)
        if balance_method:
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train, balance_method)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # 5. Entraînement des modèles
        self.train_models(X_train_balanced, X_test, y_train_balanced, y_test, preprocessor)
        
        # 6. Évaluation
        best_model_name = self.evaluate_models(y_test)
        
        # 7. Visualisations
        self.plot_model_comparison()
        
        # 8. Analyse des features
        feature_importance_df = self.analyze_feature_importance()
        
        # 9. Analyse des erreurs
        threshold_analysis = self.detailed_error_analysis(X_test, y_test)
        
        # 10. Sauvegarde
        self.save_best_model()
        
        # 11. Rapport final
        self.generate_ml_report()
        
        print("\n🎉 PIPELINE ML TERMINÉ AVEC SUCCÈS !")
        
        return {
            'best_model': self.best_model,
            'best_model_name': best_model_name,
            'results': self.results,
            'feature_importance': feature_importance_df,
            'threshold_analysis': threshold_analysis
        }

# Utilisation
if __name__ == "__main__":
    # Chargement des données
    print("📂 Chargement des données...")
    df_usagers = pd.read_csv('data/accidents_clean.csv')
    df_accidents = pd.read_csv('data/accidents_by_event.csv')
    
    # Initialisation du modeler
    modeler = AccidentMLModeler(df_usagers, df_accidents)
    
    # Lancement du pipeline complet
    results = modeler.run_full_pipeline(balance_method='smote')
    
    print(f"\n✅ Meilleur modèle: {results['best_model_name']}")
    print("📁 Fichiers générés:")
    print("   - best_accident_model.pkl")
    print("   - best_accident_model_metadata.json")