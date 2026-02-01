# 🏦 Modélisation du Risque de Crédit Bancaire

## 📋 Contexte du Projet

**Projet personnel réalisé en totale autonomie** dans le cadre du développement de compétences en Data Science appliquée à la finance. Ce projet utilise un jeu de données publiques téléchargé depuis Kaggle pour construire des modèles prédictifs de risque de crédit.

> **Note:** Il s'agit d'un projet académique et personnel. Les données proviennent de Kaggle et ne représentent aucune institution financière réelle.

---

## 🎯 Objectifs Métier

### Problématique Business
Les institutions financières font face à un défi majeur : **identifier les emprunteurs susceptibles de faire défaut sur leurs prêts**. Un mauvais scoring peut entraîner :
- Des pertes financières importantes dues aux défauts de paiement
- Un manque à gagner si trop de bons clients sont rejetés
- Une dégradation de la qualité du portefeuille de prêts

### Objectifs du Projet
1. **Prédire le risque de défaut** : Développer un modèle capable d'estimer la probabilité qu'un emprunteur ne rembourse pas son prêt
2. **Optimiser les décisions d'octroi** : Fournir un outil d'aide à la décision pour équilibrer acceptation et risque
3. **Quantifier les pertes attendues** : Calculer l'exposition financière du portefeuille
4. **Identifier les facteurs de risque clés** : Comprendre quelles variables influencent le plus le risque de défaut

---

## 📊 Données

### Source
- **Origine** : Dataset public "Credit Risk Dataset" sur Kaggle
- **Taille** : 32 581 observations
- **Variables** : 12 caractéristiques + 1 variable cible

### Variables Clés

| Catégorie | Variables | Description |
|-----------|-----------|-------------|
| **Profil Emprunteur** | `person_age`, `person_income`, `person_emp_length`, `person_home_ownership` | Informations démographiques et situation professionnelle |
| **Caractéristiques Prêt** | `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income` | Détails du prêt demandé |
| **Historique Crédit** | `cb_person_default_on_file`, `cb_person_cred_hist_length` | Comportement de crédit passé |
| **Cible** | `loan_status` | 0 = Pas de défaut, 1 = Défaut |

### Distribution de la Cible
- **Classe minoritaire (défauts)** : Déséquilibre de classes nécessitant des techniques d'équilibrage
- **Stratégie adoptée** : Undersampling pour équilibrer l'entraînement

---

## 🔬 Méthodologie

### 1. Préparation des Données
- **Gestion des valeurs manquantes** :
  - `loan_int_rate` : Imputation par la moyenne
  - `person_emp_length` : Imputation par la médiane
- **Détection et traitement des outliers** :
  - Méthode IQR (Interquartile Range)
  - Isolation Forest pour les valeurs aberrantes multivariées
  - Limitation de l'âge maximum à 85 ans
- **Feature Engineering** :
  - Encodage one-hot des variables catégorielles
  - Standardisation des variables numériques (StandardScaler)

### 2. Analyse Exploratoire (EDA)
- Analyse des distributions et corrélations
- Visualisations des relations clés (montant vs statut, revenu vs âge)
- Identification des patterns de défaut par grade de prêt
- **Insight majeur** : Forte corrélation entre `loan_percent_income` et le risque de défaut

### 3. Modélisation

#### Modèles Développés
| Modèle | Approche | Caractéristiques |
|--------|----------|------------------|
| **Régression Logistique** | Baseline classique | Interprétabilité maximale, scaling requis |
| **XGBoost** | Gradient Boosting optimisé | Hautes performances, gestion automatique des features |
| **Gradient Boosted Tree** | Histogram-based | Rapidité, robustesse |

#### Stratégie de Validation
- **Split train/test** : 70/30 avec stratification
- **Cross-validation** : 5 folds pour robustesse
- **Métriques d'évaluation** :
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC
  - Matrice de confusion
  - Courbes de calibration
  - Expected Loss (perte attendue)

### 4. Optimisation
- **Équilibrage des classes** : Undersampling du dataset
- **Optimisation du seuil de décision** : Ajustement du threshold (0.5 → 0.35-0.40)
- **Feature importance** : Identification des variables les plus prédictives

---

## 📈 Résultats

### Performance des Modèles

#### Régression Logistique (Baseline)
```
Recall (Défaut)      : 58%  → Détecte 58% des défauts réels
Recall (Non-Défaut)  : 95%  → Excellente identification des bons clients
Precision (Défaut)   : 76%  → 3 prédictions sur 4 sont correctes
Perte Attendue       : 6,1 M$
```

#### XGBoost (Dataset Équilibré) ⭐
```
Performance          : Significativement améliorée
Perte Attendue       : 6,0 M$
Généralisation       : Meilleure robustesse
Feature Importance   : Identification précise des facteurs clés
```

### Impact Financier du Portefeuille
- **Perte Attendue Totale** : ~30,3 M$ (sur l'ensemble du portefeuille test)
- **Réduction potentielle** : Optimisation du seuil permet de réduire les pertes de 5-10%

---

## 💡 Insights Métier

### 1. Facteurs de Risque Critiques
**Top 3 des variables prédictives** :
1. 🥇 **`loan_percent_income`** : Pourcentage du revenu alloué au prêt
   - *Insight* : Les emprunteurs consacrant >50% de leur revenu au prêt présentent un risque significativement plus élevé
2. 🥈 **`loan_int_rate`** : Taux d'intérêt du prêt
   - *Insight* : Les taux élevés corrèlent avec des profils plus risqués
3. 🥉 **`person_income`** : Revenu de l'emprunteur
   - *Insight* : Le niveau de revenu est un indicateur fort de capacité de remboursement

### 2. Recommandations Stratégiques

#### ⚠️ Amélioration du Taux de Détection
**Problème** : Le modèle actuel ne détecte que 58% des défauts (42% de faux négatifs)

**Solution recommandée** :
- Abaisser le seuil de décision de 0.5 à 0.35-0.40
- **Impact** : Amélioration du recall sur les défauts, réduction des pertes attendues
- **Trade-off** : Légère augmentation des rejets de bons clients (à quantifier selon l'appétit au risque)

#### 📊 Stratégie d'Acceptation par Grade
| Stratégie | Taux d'Acceptation | Impact sur le Risque |
|-----------|-------------------|---------------------|
| Conservative | 75% | Risque portefeuille fortement réduit |
| Équilibrée | 85% | Bad rate acceptable, volume préservé |
| Agressive | >90% | Risque élevé, volume maximisé |

**Recommandation** : Viser 85% d'acceptation pour équilibrer croissance et qualité

#### 🎯 Politiques de Crédit Suggérées
1. **Ratio dette/revenu** : Établir une limite stricte (ex: loan_percent_income < 40%)
2. **Scoring par grade** : Renforcer les critères pour les grades F et G
3. **Historique de crédit** : Poids significatif pour `cb_person_default_on_file`

### 3. Impact du Rééquilibrage des Classes
L'utilisation d'un dataset équilibré (undersampling) a **drastiquement amélioré** la détection des défauts, validant l'importance de traiter le déséquilibre de classes dans les problèmes de crédit.

---

## 🛠️ Technologies Utilisées

- **Langage** : Python 3.x
- **Environnement** : Jupyter Notebook
- **Librairies principales** :
  - `pandas`, `numpy` : Manipulation de données
  - `scikit-learn` : Preprocessing, modèles ML, métriques
  - `xgboost` : Modèle de gradient boosting optimisé
  - `matplotlib`, `seaborn` : Visualisations
  - `scipy` : Statistiques

---

## 📁 Structure du Projet

```
Credit_risk_modeling/
│
├── credit_risk_modeling_in_Python.ipynb    # Notebook principal avec analyse complète
└── README.md                                # Documentation du projet
```

---

## 🚀 Utilisation

### Prérequis
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy jupyter
```

### Exécution
```bash
jupyter notebook credit_risk_modeling_in_Python.ipynb
```

---

## 📌 Livrables

1. ✅ **Modèles prédictifs** : Régression Logistique, XGBoost, GBT
2. ✅ **Analyse exploratoire complète** : EDA avec visualisations
3. ✅ **Feature importance** : Identification des facteurs de risque clés
4. ✅ **Recommandations métier** : Stratégies d'acceptation et politiques de crédit
5. ✅ **Quantification financière** : Calcul des pertes attendues par stratégie

---

## 🎓 Compétences Démontrées

- ✔️ **Data Cleaning & Preprocessing** : Gestion des valeurs manquantes, outliers, encoding
- ✔️ **Exploratory Data Analysis** : Visualisations, corrélations, insights
- ✔️ **Feature Engineering** : Création et sélection de variables
- ✔️ **Machine Learning** : Régression, ensemble methods, gradient boosting
- ✔️ **Class Imbalance** : Techniques d'équilibrage (undersampling)
- ✔️ **Model Evaluation** : Métriques multiples, cross-validation, courbes ROC
- ✔️ **Business Acumen** : Traduction des résultats techniques en recommandations métier
- ✔️ **Financial Modeling** : Calcul de pertes attendues, optimisation coût/bénéfice

---

## 📝 Conclusion

Ce projet démontre une approche end-to-end de résolution d'un problème de crédit risk modeling, depuis la compréhension des données jusqu'aux recommandations business actionnables. Les modèles développés permettent d'optimiser les décisions d'octroi de crédit en équilibrant acceptation et risque, avec un **potentiel de réduction des pertes de plusieurs millions de dollars**.

**Points forts** :
- Méthodologie rigoureuse et complète
- Gestion efficace du déséquilibre de classes
- Recommandations business concrètes et quantifiées
- Interprétabilité et explicabilité des modèles

**Axes d'amélioration futurs** :
- Intégration de données temporelles (évolution du comportement)
- Tests de modèles plus avancés (Neural Networks, stacking)
- Développement d'un pipeline de production (MLOps)
- Analyse de sensibilité et stress testing

---

## 👤 Auteur

**Projet Personnel & Autonome**  
Réalisé dans le cadre du développement de compétences en Data Science appliquée à la finance.

---

## 📄 Licence & Données

- **Code** : Projet personnel éducatif
- **Données** : Dataset public Kaggle - [Credit Risk Dataset](https://www.kaggle.com/)
- **Usage** : À des fins d'apprentissage et de démonstration uniquement

---

*Dernière mise à jour : Février 2026*
