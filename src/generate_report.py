import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import textwrap

def generate_pdf_report(df_usagers, df_accidents, ml_results, output_path="rapport_accidents.pdf"):
    # Configuration du document avec des marges plus larges
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Création de styles personnalisés avec polices standard
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        alignment=1,
        spaceAfter=14,
        textColor=colors.darkblue
    )
    
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.darkblue
    )
    
    subsection_style = ParagraphStyle(
        'Subsection',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.darkblue
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        spaceAfter=6,
        leading=14
    )
    
    # Fonction pour formater les paragraphes avec indentation
    def format_paragraph(text, style=body_style):
        wrapped = textwrap.fill(text, width=100)
        return Paragraph(wrapped.replace('\n', '<br/>'), style)
    
    # Page de titre
    story.append(Paragraph("Rapport d'Analyse des Accidents Routiers", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Projet Machine Learning", section_style))
    story.append(Paragraph(f"Date: {pd.Timestamp.now().strftime('%d/%m/%Y')}", body_style))
    story.append(PageBreak())
    
    # Section 1: Question étudiée
    story.append(Paragraph("1. Question étudiée", section_style))
    story.append(format_paragraph(
        "L'objectif principal de cette étude est d'identifier les facteurs les plus significativement "
        "liés aux accidents mortels sur les routes françaises. Nous cherchons à comprendre quelles "
        "conditions (météo, heure, type de route, etc.) et quelles caractéristiques des usagers (âge, "
        "type de véhicule) contribuent le plus au risque d'accidents mortels."
    ))
    
    # Section 2: Données
    story.append(Paragraph("2. Source des données et constitution", section_style))
    
    story.append(Paragraph("Sources des données :", subsection_style))
    data_sources = [
        f"• Base de données des accidents corporels de la circulation (data.gouv.fr)",
        f"• Période couverte: {df_usagers['annee'].min()} à {df_usagers['annee'].max()}",
        f"• {len(df_usagers):,} usagers impliqués dans {len(df_accidents):,} accidents",
        f"• Taux de mortalité: {df_accidents['accident_mortel'].mean()*100:.2f}% des accidents"
    ]
    story.append(Paragraph("<br/>".join(data_sources), body_style))
    
    story.append(Paragraph("Méthodologie de collecte :", subsection_style))
    story.append(format_paragraph(
        "Les données ont été téléchargées depuis le portail open data du gouvernement français. "
        "Après téléchargement, nous avons effectué un processus de nettoyage et de fusion des "
        "différentes tables (caractéristiques, lieux, véhicules, usagers) pour constituer un "
        "jeu de données unifié prêt pour l'analyse."
    ))
    
    # Section 3: Outils
    story.append(Paragraph("3. Outils utilisés", section_style))
    
    tools_table_data = [
        ['Catégorie', 'Outils', 'Utilisation'],
        ['Traitement données', 'Pandas, NumPy', 'Nettoyage, transformation et fusion des données'],
        ['Analyse', 'Matplotlib, Seaborn', 'Visualisation et exploration des données'],
        ['ML', 'Scikit-learn, Imbalanced-learn', 'Modélisation prédictive et gestion du déséquilibre'],
        ['Rapport', 'ReportLab', 'Génération de ce document PDF']
    ]
    
    tools_table = Table(tools_table_data)
    tools_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(tools_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Section 4: Analyse
    story.append(Paragraph("4. Analyse des données et résultats", section_style))
    
    story.append(Paragraph("Principaux facteurs de risque :", subsection_style))
    risk_factors = [
        "• **Heure de la journée** : Pic de risque entre 3h et 5h du matin",
        "• **Conditions météo** : Risque accru par temps de pluie ou de brouillard",
        "• **Âge des conducteurs** : Jeunes (18-24 ans) et seniors (65+) plus à risque",
        "• **Type de route** : Routes départementales plus dangereuses que les autoroutes"
    ]
    story.append(Paragraph("<br/>".join(risk_factors), body_style))
    story.append(Spacer(1, 0.2*inch))
    
    if ml_results:
        story.append(Paragraph("Résultats de modélisation :", subsection_style))
        model_table_data = [['Modèle', 'F1-Score', 'AUC-ROC']]
        for model, metrics in ml_results['results'].items():
            model_table_data.append([
                model, 
                f"{metrics['test_f1']:.3f}", 
                f"{metrics['test_auc']:.3f}"
            ])
        
        # Tri par F1 décroissant
        model_table_data = [model_table_data[0]] + sorted(
            model_table_data[1:], 
            key=lambda x: float(x[1]), 
            reverse=True
        )
        
        model_table = Table(model_table_data)
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        story.append(model_table)
        story.append(Spacer(1, 0.1*inch))
        
        best_model = ml_results['best_model_name']
        story.append(Paragraph(
            f"<b>Meilleur modèle</b>: {best_model} (F1-Score: {ml_results['results'][best_model]['test_f1']:.3f})",
            body_style
        ))
        story.append(Spacer(1, 0.2*inch))
    
    # Conclusion
    story.append(Paragraph("Conclusion", section_style))
    story.append(format_paragraph(
        "Cette analyse a permis d'identifier plusieurs facteurs clés contribuant aux accidents mortels "
        "sur les routes françaises. Bien que les performances des modèles puissent être améliorées, "
        "les résultats fournissent des pistes d'action concrètes pour la prévention routière, notamment "
        "en ciblant les périodes et conditions à haut risque."
    ))
    
    # Génération du PDF
    doc.build(story)

if __name__ == "__main__":
    print("Ce script est conçu pour être appelé depuis main.py")
    print("Veuillez exécuter le pipeline principal avec: python main.py --step all")