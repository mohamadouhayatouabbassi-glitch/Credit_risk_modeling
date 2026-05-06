# Credit Risk Scoring — API de décision (FastAPI)

API de scoring qui estime la **probabilité de défaut** et retourne une **décision d’octroi** (*ACCEPT / REJECT*) à partir de données emprunteur/prêt.  
Objectif : fournir un **service exploitable en production** pour automatiser (ou assister) la décision de crédit.

## Démo en ligne (Render)
- **API** : https://credit-risk-modeling-2w8b.onrender.com/
- **Healthcheck** : https://credit-risk-modeling-2w8b.onrender.com/health
- **Docs (Swagger)** : https://credit-risk-modeling-2w8b.onrender.com/docs

## Cas d’usage métier
- Pré-qualification des demandes de crédit (réduction du risque & meilleure cohérence de décision)
- Aide à la décision pour les analystes risque (score + seuil)
- Simulation simple “what-if” (impact du taux, revenu, montant, ancienneté emploi)

## Endpoints principaux
- `GET /` : infos service + liens utiles
- `GET /health` : état du service et chargement du modèle
- `POST /predict` : score + décision

### Exemple d’appel (POST /predict)
```bash
curl -X POST "https://credit-risk-modeling-2w8b.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"loan_int_rate":12.5,"person_emp_length":3,"person_income":55000,"loan_amnt":12000}'
```

## Stack technique
- Python / FastAPI / Uvicorn
- scikit-learn + joblib (serving du modèle)
- Docker
- Déploiement : Render

## Structure du repo (simplifiée)
- `api/` : application FastAPI (routes, schémas)
- `src/` : configuration + logique modèle/portefeuille
- `artifacts/` : artefacts modèle (selon ton organisation)

## Auteur
Mohamadou Hayatou Abbassi

*Dernière mise à jour : Février 2026*
