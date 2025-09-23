---
layout: page
title: "Prévision de demande — retail"
permalink: /projets/forecast/
tags: [time-series, forecasting, mlops, python]
---

## TL;DR
MAPE 12.7% (−31% vs baseline naïve) sur 3 pays et 18 mois, modèle LightGBM + features calendrier/météo. Démo interactive et code reproductible ci‑dessous.

<p>
  <a href="https://github.com/ton-user/demand-forecast" class="btn">Code</a>
  <a href="https://huggingface.co/spaces/ton-id/forecast" class="btn">Démo</a>
  <a href="#repro" class="btn">Reproduire</a>
</p>

## Problème
- Objectif: prévoir la demande J+1 à J+28 pour ajuster la supply‑chain.
- Contraintes: forte saisonnalité, promos irrégulières, données manquantes.

## Données
- Sources: ventes historiques (anonymisées), météo (Open‑Meteo), calendrier scolaire FR.
- Taille: 4.2 M lignes, 3 pays, granularité: produit‑jour.
- Qualité: 1.3% valeurs manquantes; correction par imputation saisonnière.
- Éthique/RGPD: anonymisation ID produit/client, pas de données personnelles.

## Méthode
- Features: lags [7, 14, 28], moyennes mobiles, indicateurs jours fériés, embeddings vacances.
- Modèles testés: Prophet, XGBoost, LightGBM (gagnant), baseline naïve (last week).
- Validation: série temporelle (sliding windows), split temporel 70/30, seed fixé.
- Suivi: MLflow pour paramètres/métriques; versionnement données via DVC (échantillon public).

## Résultats
- Global: MAPE 12.7% (baseline 18.4%), RMSE 221 (−24%).
- Par segment:
  - FR: MAPE 11.9% | DE: 13.4% | ES: 12.8%
- Robustesse: drift faible Q2→Q3, performance stable sur produits “long tail”.

| Modèle      | MAPE ↓ | RMSE ↓ | Temps entraînement |
|-------------|--------|--------|--------------------|
| Baseline LW | 18.4%  | 292    | —                  |
| Prophet     | 16.9%  | 271    | 7 min              |
| XGBoost     | 13.2%  | 233    | 9 min              |
| LightGBM    | 12.7%  | 221    | 6 min              |

![Courbe prévs vs réels](/assets/img/forecast_preview.png)

## Limites
- Promotions non étiquetées → pics sous‑estimés.
- Ruptures structurelles post‑lancement produit.

## Pistes suivantes
- Features promos (scraping prospectus), modèle hiérarchique, quantile loss (P50/P90).

## Ce que j’ai appris
- Les features calendrier expliquent ~60% du gain vs baseline.
- Le choix de fenêtre de validation change 1–2 pts de MAPE.

## Liens
- Code: https://github.com/ton-user/demand-forecast  
- Démo: https://huggingface.co/spaces/ton-id/forecast  
- Notebook: /notebooks/forecast_report.html

## Reproductibilité
- Environnement: Python 3.11
- Données: échantillon 50k lignes fourni dans `data/sample.csv`
- Commandes:
```bash
git clone https://github.com/ton-user/demand-forecast
cd demand-forecast
pip install -r requirements.txt
python src/train.py --config configs/lightgbm.yaml --seed 42
