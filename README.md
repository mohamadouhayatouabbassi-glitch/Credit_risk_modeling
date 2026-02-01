# 🏦 Modélisation du Risque de Crédit Bancaire

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

## 📋 Table des Matières
- [Vue d'ensemble](#vue-densemble)
- [Objectifs métier](#objectifs-métier)
- [Dataset](#dataset)
- [Méthodologie](#méthodologie)
- [Modèles utilisés](#modèles-utilisés)
- [Résultats clés](#résultats-clés)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Recommandations stratégiques](#recommandations-stratégiques)
- [Technologies](#technologies)

---

## 🎯 Vue d'ensemble

Ce projet propose une solution complète de **modélisation prédictive du risque de défaut de crédit** visant à optimiser les décisions d'octroi de prêts et à minimiser les pertes financières. En exploitant des techniques avancées de Machine Learning, notre modèle permet d'identifier avec précision les profils à risque et d'améliorer la résilience du portefeuille de crédit.

### Impact Business
- 💰 **Réduction des pertes attendues** : Optimisation des décisions de crédit pour limiter l'exposition au risque (~5,9M€)
- 📊 **Amélioration de la prise de décision** : Modèle prédictif basé sur des données quantifiables
- 🎯 **Segmentation du risque** : Identification précise des profils à risque élevé par grade de prêt
- ⚡ **Automatisation** : Processus d'évaluation du crédit plus rapide et objectif

---

## 🎯 Objectifs métier

1. **Prédire le risque de défaut** : Développer un modèle capable d'identifier les emprunteurs susceptibles de faire défaut
2. **Optimiser le seuil de décision** : Ajuster le seuil de classification pour maximiser la détection des défauts tout en maintenant un taux d'approbation acceptable
3. **Minimiser les pertes financières** : Réduire l'exposition au risque en identifiant proactivement les profils à risque
4. **Fournir des insights actionnables** : Identifier les facteurs clés de risque pour guider les politiques de crédit

---

## 📊 Dataset

### Source des données
- **Fichier** : `credit_risk_dataset.csv`
- **Volume** : 32 581 prêts
- **Caractéristiques** : 13 variables explicatives
- **Variable cible** : `loan_status` (0 = Non-défaut, 1 = Défaut)

### Variables clés

| Catégorie | Variables | Description |
|-----------|-----------|-------------|
| **👤 Emprunteur** | `person_age`, `person_income`, `person_emp_length`, `person_home_ownership` | Profil démographique et situation financière |
| **💳 Prêt** | `loan_amnt`, `loan_int_rate`, `loan_grade`, `loan_intent` | Caractéristiques du prêt demandé |
| **📈 Métriques** | `loan_percent_income` | **Ratio critique** : Montant du prêt / Revenu |
| **📜 Historique** | `cb_person_default_on_file`, `cb_person_cred_hist_length` | Antécédents de crédit |

---

## 🔬 Méthodologie

Notre approche suit un processus structuré en 6 phases :

### 1. 📥 Collecte et chargement des données
- Import du dataset de crédit (32 581 observations)
- Analyse de la structure et des types de données

### 2. 🔍 Analyse exploratoire (EDA)
- Analyse de la distribution de la variable cible
- Étude des relations entre variables explicatives et défaut
- Analyse par grade de prêt (A, B, C, D, E, F, G)
- Identification des patterns et corrélations

### 3. 🧹 Prétraitement des données
- **Gestion des valeurs manquantes** : Imputation stratégique
- **Détection des outliers** : Utilisation d'IsolationForest
- **Traitement des outliers** : Capping pour préserver l'information
- **Encodage** : Transformation des variables catégorielles

### 4. ⚙️ Feature Engineering
- Encodage des variables catégorielles (`loan_grade`, `home_ownership`, `loan_intent`)
- Création de variables dérivées si nécessaire
- Normalisation et standardisation des features numériques

### 5. 🤖 Entraînement des modèles
- **Baseline** : Régression Logistique (simple et multi-features)
- **Modèles avancés** : XGBoost, Gradient Boosting
- **Validation croisée** : Évaluation robuste des performances
- **Optimisation des hyperparamètres** : Grid Search / Random Search

### 6. 📊 Évaluation et optimisation
- Métriques de performance : Accuracy, ROC-AUC, Precision, Recall, F1-Score
- Matrices de confusion détaillées
- **Optimisation du seuil de décision** : Tests de seuils (0.35, 0.40, 0.50)
- Calcul de l'impact financier (pertes attendues)

---

## 🤖 Modèles utilisés

| Modèle | Type | Usage |
|--------|------|-------|
| **Régression Logistique** | Classification binaire | Modèle de référence (baseline) |
| **XGBoost / HistGradientBoosting** | Ensemble learning | Modèle principal pour prédictions |
| **Gradient Boosting Ensemble** | Ensemble learning | Optimisation des performances |
| **IsolationForest** | Détection d'anomalies | Identification des outliers |

### Métriques d'évaluation
- ✅ **Accuracy** : Taux de prédictions correctes
- 📈 **ROC-AUC Score** : Capacité de discrimination du modèle
- 🎯 **Precision / Recall / F1-Score** : Performance par classe
- 💰 **Impact financier** : Estimation des pertes en fonction des prédictions

---

## 🏆 Résultats clés

### 🔑 Découverte majeure : `loan_percent_income`

**Le ratio prêt/revenu (`loan_percent_income`) est le prédicteur le plus puissant du défaut de crédit.**

- Les emprunteurs en défaut présentent systématiquement des ratios prêt/revenu **significativement plus élevés**
- Cette tendance est observée **dans tous les grades de prêt** (A à G)
- Les grades A, B et C montrent les différences de risque les plus marquées

### 📊 Optimisation du seuil de décision

| Seuil | Impact | Recommandation |
|-------|--------|----------------|
| **0.50** | Standard (par défaut) | ❌ Taux de détection sous-optimal |
| **0.40** | Équilibré | ✅ Bon compromis détection/approbation |
| **0.35** | Conservateur | ✅ Maximise la détection des défauts |

**Recommandation** : Abaisser le seuil de 0.50 à **0.35-0.40** pour :
- ⬆️ Augmenter le taux de détection des défauts (Recall)
- 🛡️ Mieux capturer les profils à risque élevé
- 💵 Réduire l'exposition financière du portefeuille

### 🎯 Segmentation par grade de prêt

Chaque grade de prêt présente des patterns de défaut distincts, permettant une stratégie de risque différenciée :
- **Grades A-B** : Risque faible, critères stricts sur le ratio prêt/revenu
- **Grades C-D** : Risque moyen, évaluation approfondie nécessaire
- **Grades E-G** : Risque élevé, critères restrictifs recommandés

---

## 🚀 Installation

### Prérequis
```bash
Python 3.8+
pip
```

### Installation des dépendances
```bash
# Cloner le repository
git clone https://github.com/mohamadouhayatouabbassi-glitch/Credit_risk_modeling.git
cd Credit_risk_modeling

# Installer les packages requis
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Packages principaux
- `pandas` : Manipulation des données
- `numpy` : Calculs numériques
- `scikit-learn` : Modèles de Machine Learning
- `xgboost` : Gradient Boosting optimisé
- `matplotlib` / `seaborn` : Visualisations
- `jupyter` : Environnement de développement

---

## 💻 Utilisation

### Exécution du notebook
```bash
jupyter notebook credit_risk_modeling.ipynb
```

### Structure du code
1. **Importation des bibliothèques**
2. **Chargement des données** : `credit_risk_dataset.csv`
3. **Analyse exploratoire** : Visualisations et statistiques
4. **Prétraitement** : Nettoyage et transformation
5. **Modélisation** : Entraînement des modèles
6. **Évaluation** : Métriques et validation
7. **Optimisation** : Tuning des hyperparamètres et du seuil

### Workflow typique
```python
# 1. Charger les données
data = pd.read_csv('credit_risk_dataset.csv')

# 2. Prétraiter
X, y = preprocess_data(data)

# 3. Entraîner le modèle
model = train_model(X, y)

# 4. Prédire
predictions = model.predict(X_test)

# 5. Évaluer
evaluate_model(y_test, predictions)
```

---

## 💡 Recommandations stratégiques

### 🎯 Actions prioritaires

1. **Ajustement du seuil de décision**
   - Mettre en place un seuil de 0.35-0.40 au lieu de 0.50
   - Accepter un taux de rejet marginal pour minimiser les pertes
   
2. **Focus sur le ratio prêt/revenu**
   - Établir des seuils de `loan_percent_income` par grade de prêt
   - Renforcer les critères d'évaluation sur cette métrique clé
   
3. **Segmentation des politiques de crédit**
   - Appliquer des critères différenciés selon le grade de prêt
   - Grades A-C : Focus sur la détection fine des risques
   - Grades D-G : Critères plus restrictifs

4. **Monitoring continu**
   - Suivre l'évolution des performances du modèle
   - Réentraîner régulièrement sur les nouvelles données
   - Ajuster les seuils en fonction des objectifs business

### 📈 Bénéfices attendus
- ✅ Réduction significative des pertes financières
- ✅ Amélioration de la qualité du portefeuille de crédit
- ✅ Processus de décision plus objectif et data-driven
- ✅ Meilleure gestion du risque crédit

---

## 🛠 Technologies

| Catégorie | Technologies |
|-----------|--------------|
| **Langage** | Python 3.8+ |
| **Data Science** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualisation** | Matplotlib, Seaborn |
| **Environnement** | Jupyter Notebook, Google Colab |
| **Version Control** | Git, GitHub |

---

## 📝 Licence

Ce projet est développé dans un cadre académique et professionnel.

---

## 👤 Auteur

**Abbassi Mohamadou Hayatou**
- Email: abbassi.mohamadouhayatou@uit.ac.ma
- GitHub: [@mohamadouhayatouabbassi-glitch](https://github.com/mohamadouhayatouabbassi-glitch)

---

## 🙏 Remerciements

Projet réalisé dans le cadre d'une analyse de risque de crédit bancaire, démontrant l'application pratique du Machine Learning dans le secteur financier.

---

*Dernière mise à jour : Février 2026*
