import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class AccidentExplorer:
    """
    Classe pour l'analyse exploratoire des données d'accidents
    """
    
    def __init__(self, df_usagers, df_accidents):
        self.df_usagers = df_usagers
        self.df_accidents = df_accidents
        
        # Couleurs pour les graphiques
        self.colors = {
            'mortel': '#d62728',
            'non_mortel': '#2ca02c',
            'primary': '#1f77b4'
        }

    def overview_statistics(self):
        """
        Statistiques générales sur le dataset
        """
        print("=== STATISTIQUES GÉNÉRALES ===\n")
        
        print(f"Période d'analyse: {self.df_usagers['annee'].min()} - {self.df_usagers['annee'].max()}")
        print(f"Nombre total d'usagers: {len(self.df_usagers):,}")
        print(f"Nombre total d'accidents: {len(self.df_accidents):,}")
        
        # Répartition par gravité
        gravite_counts = self.df_usagers['gravite_lib'].value_counts()
        gravite_pct = (gravite_counts / len(self.df_usagers) * 100).round(2)
        
        print("\n--- Répartition par gravité ---")
        for gravite, count in gravite_counts.items():
            pct = gravite_pct[gravite]
            print(f"{gravite}: {count:,} ({pct}%)")
        
        # Taux de mortalité par accident
        mortalite_stats = self.df_accidents.agg({
            'nb_morts': ['sum', 'mean'],
            'nb_victimes': 'mean'
        }).round(3)
        
        print("\n--- Statistiques de mortalité ---")
        print(f"Accidents mortels: {self.df_accidents['accident_mortel'].sum():,} "
              f"({self.df_accidents['accident_mortel'].mean()*100:.1f}%)")
        print(f"Total décès: {mortalite_stats['nb_morts']['sum']:,}")
        print(f"Moyenne décès/accident mortel: {mortalite_stats['nb_morts']['mean']:.2f}")
        print(f"Moyenne victimes/accident: {mortalite_stats['nb_victimes']['mean']:.2f}")

    def temporal_analysis(self):
        """
        Analyse temporelle des accidents
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Évolution par année
        yearly_stats = self.df_accidents.groupby('annee').agg({
            'accident_mortel': ['sum', 'count', 'mean']
        }).round(3)
        yearly_stats.columns = ['accidents_mortels', 'total_accidents', 'taux_mortalite']
        
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        ax1.bar(yearly_stats.index, yearly_stats['total_accidents'], 
                alpha=0.7, color=self.colors['primary'], label='Total accidents')
        ax1_twin.plot(yearly_stats.index, yearly_stats['taux_mortalite']*100, 
                     color=self.colors['mortel'], marker='o', linewidth=3, label='Taux mortalité (%)')
        
        ax1.set_title('Évolution Annuelle des Accidents')
        ax1.set_xlabel('Année')
        ax1.set_ylabel('Nombre d\'accidents', color=self.colors['primary'])
        ax1_twin.set_ylabel('Taux de mortalité (%)', color=self.colors['mortel'])
        
        # 2. Répartition par heure
        if 'heure' in self.df_usagers.columns:
            # Créer une copie pour éviter SettingWithCopyWarning
            temp_df = self.df_usagers.copy()
            
            # Convertir l'heure en heure numérique si nécessaire
            if temp_df['heure'].dtype == 'object':
                temp_df['hour'] = temp_df['heure'].apply(lambda x: x.hour if isinstance(x, pd.Timestamp) else x)
            else:
                # Si c'est déjà numérique, diviser par 100 pour obtenir l'heure
                temp_df['hour'] = temp_df['heure'] // 100
            
            # Filtrer les valeurs nulles
            temp_df = temp_df.dropna(subset=['hour'])
            
            hourly_data = temp_df.groupby(['hour', 'accident_mortel']).size().unstack(fill_value=0)
            hourly_pct = hourly_data.div(hourly_data.sum(axis=1), axis=0) * 100
            
            ax2 = axes[0, 1]
            ax2.plot(hourly_pct.index, hourly_pct[0], label='Non mortel',
                    color=self.colors['non_mortel'], linewidth=2)
            ax2.plot(hourly_pct.index, hourly_pct[1], label='Mortel',
                    color=self.colors['mortel'], linewidth=2)
            ax2.set_xticks(range(0, 24, 2))
            ax2.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])
            ax2.set_title('Répartition des Accidents par Heure')
            ax2.set_xlabel('Heure de la journée')
            ax2.set_ylabel('Pourcentage d\'accidents')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Répartition par jour de la semaine
        if 'jour_semaine' in self.df_usagers.columns:
            days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            daily_data = self.df_usagers.groupby(['jour_semaine', 'accident_mortel']).size().unstack(fill_value=0)
            
            ax3 = axes[1, 0]
            x = np.arange(len(days))
            width = 0.35
            
            ax3.bar(x - width/2, daily_data[0], width, label='Non mortel', 
                   color=self.colors['non_mortel'], alpha=0.8)
            ax3.bar(x + width/2, daily_data[1], width, label='Mortel', 
                   color=self.colors['mortel'], alpha=0.8)
            
            ax3.set_title('Accidents par Jour de la Semaine')
            ax3.set_xlabel('Jour')
            ax3.set_ylabel('Nombre d\'accidents')
            ax3.set_xticks(x)
            ax3.set_xticklabels(days, rotation=45)
            ax3.legend()
        
        # 4. Répartition par mois
        if 'mois' in self.df_usagers.columns:
            mois_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                         'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
            monthly_data = self.df_usagers.groupby('mois')['accident_mortel'].agg(['count', 'sum', 'mean'])
            
            ax4 = axes[1, 1]
            ax4_twin = ax4.twinx()
            
            ax4.bar(monthly_data.index, monthly_data['count'], 
                   alpha=0.7, color=self.colors['primary'])
            ax4_twin.plot(monthly_data.index, monthly_data['mean']*100, 
                         color=self.colors['mortel'], marker='s', linewidth=2)
            
            ax4.set_title('Saisonnalité des Accidents')
            ax4.set_xlabel('Mois')
            ax4.set_ylabel('Nombre d\'accidents', color=self.colors['primary'])
            ax4_twin.set_ylabel('Taux de mortalité (%)', color=self.colors['mortel'])
            ax4.set_xticks(range(1, 13))
            ax4.set_xticklabels(mois_names)
        
        plt.tight_layout()
        plt.show()

    def conditions_analysis(self):
        """
        Analyse des conditions météorologiques et de luminosité
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Conditions météorologiques
        if 'meteo_lib' in self.df_usagers.columns:
            meteo_data = self.df_usagers.groupby('meteo_lib')['accident_mortel'].agg(['count', 'sum', 'mean']).round(3)
            meteo_data = meteo_data.sort_values('mean', ascending=False)
            
            ax1 = axes[0, 0]
            bars = ax1.bar(range(len(meteo_data)), meteo_data['mean']*100, 
                          color=[self.colors['mortel'] if x > meteo_data['mean'].mean() 
                                else self.colors['non_mortel'] for x in meteo_data['mean']])
            ax1.set_title('Taux de Mortalité par Condition Météo')
            ax1.set_xlabel('Conditions météorologiques')
            ax1.set_ylabel('Taux de mortalité (%)')
            ax1.set_xticks(range(len(meteo_data)))
            ax1.set_xticklabels(meteo_data.index, rotation=45, ha='right')
            
            # Ajout des valeurs sur les barres
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}%', ha='center', va='bottom')
        
        # 2. Conditions de luminosité
        if 'lumiere_lib' in self.df_usagers.columns:
            lumiere_data = self.df_usagers.groupby('lumiere_lib')['accident_mortel'].agg(['count', 'sum', 'mean']).round(3)
            lumiere_data = lumiere_data.sort_values('mean', ascending=False)
            
            ax2 = axes[0, 1]
            bars = ax2.bar(range(len(lumiere_data)), lumiere_data['mean']*100,
                          color=[self.colors['mortel'] if x > lumiere_data['mean'].mean() 
                                else self.colors['non_mortel'] for x in lumiere_data['mean']])
            ax2.set_title('Taux de Mortalité par Condition de Luminosité')
            ax2.set_xlabel('Conditions de luminosité')
            ax2.set_ylabel('Taux de mortalité (%)')
            ax2.set_xticks(range(len(lumiere_data)))
            ax2.set_xticklabels(lumiere_data.index, rotation=45, ha='right')
        
        # 3. Croisement météo x luminosité
        if 'meteo_lib' in self.df_usagers.columns and 'lumiere_lib' in self.df_usagers.columns:
            cross_data = self.df_usagers.groupby(['meteo_lib', 'lumiere_lib'])['accident_mortel'].mean().unstack(fill_value=0)
            
            ax3 = axes[1, 0]
            im = ax3.imshow(cross_data.values, cmap='Reds', aspect='auto')
            ax3.set_title('Taux de Mortalité: Météo × Luminosité')
            ax3.set_xticks(range(len(cross_data.columns)))
            ax3.set_xticklabels(cross_data.columns, rotation=45, ha='right')
            ax3.set_yticks(range(len(cross_data.index)))
            ax3.set_yticklabels(cross_data.index)
            
            # Ajout des valeurs dans les cellules
            for i in range(len(cross_data.index)):
                for j in range(len(cross_data.columns)):
                    value = cross_data.iloc[i, j]
                    ax3.text(j, i, f'{value:.3f}', ha='center', va='center',
                            color='white' if value > 0.02 else 'black')
            
            plt.colorbar(im, ax=ax3, label='Taux de mortalité')
        
        # 4. Distribution des âges
        if 'age' in self.df_usagers.columns:
            ax4 = axes[1, 1]
            
            # Histogramme par gravité
            ages_mortel = self.df_usagers[self.df_usagers['accident_mortel'] == 1]['age'].dropna()
            ages_non_mortel = self.df_usagers[self.df_usagers['accident_mortel'] == 0]['age'].dropna()
            
            ax4.hist(ages_non_mortel, bins=30, alpha=0.7, density=True, 
                    label='Non mortel', color=self.colors['non_mortel'])
            ax4.hist(ages_mortel, bins=30, alpha=0.7, density=True, 
                    label='Mortel', color=self.colors['mortel'])
            
            ax4.set_title('Distribution des Âges par Gravité')
            ax4.set_xlabel('Âge')
            ax4.set_ylabel('Densité')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def geographical_analysis(self):
        """
        Analyse géographique des accidents
        """
        if 'lat' in self.df_usagers.columns and 'long' in self.df_usagers.columns:
            # Nettoyage des coordonnées
            geo_data = self.df_usagers.dropna(subset=['lat', 'long'])
            geo_data = geo_data[(geo_data['lat'] >= 41) & (geo_data['lat'] <= 52) & 
                               (geo_data['long'] >= -5) & (geo_data['long'] <= 10)]
            
            if len(geo_data) > 0:
                # Carte de base centrée sur la France
                m = folium.Map(location=[46.5, 2.5], zoom_start=6)
                
                # Ajout des points d'accidents mortels
                accidents_mortels = geo_data[geo_data['accident_mortel'] == 1]
                
                if len(accidents_mortels) > 0:
                    # Échantillonnage pour éviter la surcharge
                    sample_size = min(1000, len(accidents_mortels))
                    accidents_sample = accidents_mortels.sample(n=sample_size)
                    
                    for idx, row in accidents_sample.iterrows():
                        folium.CircleMarker(
                            location=[row['lat'], row['long']],
                            radius=3,
                            popup=f"Accident mortel - {row.get('date', 'Date inconnue')}",
                            color='red',
                            fill=True,
                            fillOpacity=0.6
                        ).add_to(m)
                
                # Heatmap des accidents
                heat_data = [[row['lat'], row['long']] for idx, row in geo_data.iterrows()]
                HeatMap(heat_data, radius=10, blur=15).add_to(m)
                
                # Sauvegarde de la carte
                m.save('accidents_map.html')
                print("Carte sauvegardée: accidents_map.html")

    def age_analysis(self):
        """
        Analyse détaillée par âge
        """
        if 'tranche_age' in self.df_usagers.columns:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. Taux de mortalité par tranche d'âge
            age_stats = self.df_usagers.groupby('tranche_age')['accident_mortel'].agg(['count', 'sum', 'mean']).round(3)
            
            ax1 = axes[0]
            bars = ax1.bar(range(len(age_stats)), age_stats['mean']*100,
                          color=[self.colors['mortel'] if x > age_stats['mean'].mean() 
                                else self.colors['non_mortel'] for x in age_stats['mean']])
            ax1.set_title('Taux de Mortalité par Tranche d\'Âge')
            ax1.set_xlabel('Tranche d\'âge')
            ax1.set_ylabel('Taux de mortalité (%)')
            ax1.set_xticks(range(len(age_stats)))
            ax1.set_xticklabels(age_stats.index)
            
            # Ajout des effectifs sur les barres
            for i, (bar, count) in enumerate(zip(bars, age_stats['count'])):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}%\n(n={count})', ha='center', va='bottom')
            
            # 2. Distribution des âges avec courbe de tendance
            if 'age' in self.df_usagers.columns:
                ax2 = axes[1]
                
                # Calcul du taux de mortalité par âge
                age_mortality = self.df_usagers.groupby('age')['accident_mortel'].agg(['count', 'mean']).reset_index()
                age_mortality = age_mortality[age_mortality['count'] >= 10]  # Minimum 10 cas
                
                # Lissage avec une moyenne mobile
                window = 5
                age_mortality['mean_smooth'] = age_mortality['mean'].rolling(window=window, center=True).mean()
                
                ax2.scatter(age_mortality['age'], age_mortality['mean']*100, 
                           alpha=0.3, color=self.colors['primary'], s=20)
                ax2.plot(age_mortality['age'], age_mortality['mean_smooth']*100, 
                        color=self.colors['mortel'], linewidth=3, label='Tendance (moyenne mobile)')
                
                ax2.set_title('Taux de Mortalité par Âge (détaillé)')
                ax2.set_xlabel('Âge')
                ax2.set_ylabel('Taux de mortalité (%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

    def correlation_analysis(self):
        """
        Analyse des corrélations entre variables
        """
        # Sélection des variables numériques pertinentes
        numeric_vars = []
        for col in ['age', 'heure', 'mois', 'jour_semaine', 'weekend', 'conditions_difficiles', 'accident_mortel']:
            if col in self.df_usagers.columns:
                numeric_vars.append(col)
        
        if len(numeric_vars) > 2:
            corr_data = self.df_usagers[numeric_vars].corr()
            
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_data, dtype=bool))
            
            sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Matrice de Corrélation des Variables')
            plt.tight_layout()
            plt.show()

    def interactive_dashboard(self):
        """
        Création d'un dashboard interactif avec Plotly
        """
        # 1. Évolution temporelle interactive
        if 'date' in self.df_usagers.columns:
            # Filtrer les dates invalides
            valid_dates = self.df_usagers.dropna(subset=['date'])
            # S'assurer que la colonne date est au format datetime
            valid_dates['date'] = pd.to_datetime(valid_dates['date'], errors='coerce')
            valid_dates = valid_dates.dropna(subset=['date'])
            
            monthly_evolution = valid_dates.groupby([
                valid_dates['date'].dt.to_period('M'), 'accident_mortel'
            ]).size().unstack(fill_value=0).reset_index()
            monthly_evolution['date'] = monthly_evolution['date'].astype(str)
            
            fig1 = px.line(monthly_evolution, x='date', y=[0, 1],
                          title='Évolution Mensuelle des Accidents',
                          labels={'value': 'Nombre d\'accidents', 'variable': 'Type'})
            fig1.update_layout(height=500)
            fig1.show()
        
        # 2. Répartition par conditions avec Plotly
        if 'meteo_lib' in self.df_usagers.columns:
            meteo_stats = self.df_usagers.groupby('meteo_lib')['accident_mortel'].agg(['count', 'mean']).reset_index()
            
            fig2 = px.bar(meteo_stats, x='meteo_lib', y='mean',
                         title='Taux de Mortalité par Condition Météorologique',
                         labels={'mean': 'Taux de mortalité', 'meteo_lib': 'Conditions météo'})
            fig2.update_layout(height=500, xaxis_tickangle=-45)
            fig2.show()

    def generate_summary_report(self):
        """
        Génère un rapport de synthèse
        """
        print("\n" + "="*60)
        print("           RAPPORT DE SYNTHÈSE - ANALYSE EXPLORATOIRE")
        print("\n" + "="*60)
        
        # Statistiques clés
        total_accidents = len(self.df_accidents)
        accidents_mortels = self.df_accidents['accident_mortel'].sum()
        taux_mortalite = accidents_mortels / total_accidents * 100
        
        print(f"\n📊 CHIFFRES CLÉS")
        print(f"   • Total accidents: {total_accidents:,}")
        print(f"   • Accidents mortels: {accidents_mortels:,} ({taux_mortalite:.2f}%)")
        print(f"   • Total décès: {self.df_accidents['nb_morts'].sum():,}")
        
        # Facteurs de risque principaux
        print(f"\n⚠️  FACTEURS DE RISQUE IDENTIFIÉS")
        
        # Analyse par heure si disponible
        if 'heure' in self.df_usagers.columns:
            heure_risque = self.df_usagers.groupby('heure')['accident_mortel'].mean()
            heure_max = heure_risque.idxmax()
            risque_max = heure_risque.max() * 100
            print(f"   • Heure la plus dangereuse: {heure_max}h ({risque_max:.2f}% de mortalité)")
        
        # Analyse météo si disponible
        if 'meteo_lib' in self.df_usagers.columns:
            meteo_risque = self.df_usagers.groupby('meteo_lib')['accident_mortel'].mean()
            meteo_max = meteo_risque.idxmax()
            risque_meteo = meteo_risque.max() * 100
            print(f"   • Condition météo la plus dangereuse: {meteo_max} ({risque_meteo:.2f}%)")
        
        # Analyse âge si disponible
        if 'tranche_age' in self.df_usagers.columns:
            age_risque = self.df_usagers.groupby('tranche_age')['accident_mortel'].mean()
            age_max = age_risque.idxmax()
            risque_age = age_risque.max() * 100
            print(f"   • Tranche d'âge la plus à risque: {age_max} ({risque_age:.2f}%)")
        
        print(f"\n🎯 RECOMMANDATIONS POUR LE MODÈLE ML")
        print(f"   • Variables temporelles (heure, jour) semblent discriminantes")
        print(f"   • Conditions météo/luminosité à intégrer prioritairement")
        print(f"   • Âge des usagers variable importante")
        print(f"   • Attention au déséquilibre des classes ({taux_mortalite:.1f}% positifs)")
        
        print("\n" + "="*60)

    def run_full_analysis(self):
        """
        Lance l'analyse exploratoire complète
        """
        print("🚀 Lancement de l'analyse exploratoire complète...\n")
        
        self.overview_statistics()
        print("\n" + "-"*50 + "\n")
        
        print("📈 Analyse temporelle...")
        self.temporal_analysis()
        
        print("🌤️  Analyse des conditions...")
        self.conditions_analysis()
        
        print("📍 Analyse géographique...")
        self.geographical_analysis()
        
        print("👥 Analyse par âge...")
        self.age_analysis()
        
        print("🔗 Analyse des corrélations...")
        self.correlation_analysis()
        
        print("📊 Dashboard interactif...")
        self.interactive_dashboard()
        
        self.generate_summary_report()
        
        print("\n✅ Analyse exploratoire terminée !")

# Utilisation
if __name__ == "__main__":
    # Chargement des données
    df_usagers = pd.read_csv('data/accidents_clean.csv')
    df_accidents = pd.read_csv('data/accidents_by_event.csv')
    
    # Lancement de l'analyse
    explorer = AccidentExplorer(df_usagers, df_accidents)
    explorer.run_full_analysis()