# Rapport d'Audit : Outil IA QuantAI

## 1. Résumé Exécutif
L'outil QuantAI est une plateforme de trading quantitatif intégrant des agents LLM pour l'analyse technique et visuelle. Bien que l'architecture technique soit robuste et modulaire pour la génération de signaux de trading (momentum, value, qualité), l'audit révèle une **absence totale de la fonctionnalité principale demandée** : la recommandation de dépôts GitHub pertinents pour des points financiers. Le système actuel est exclusivement tourné vers l'analyse de tickers boursiers.

## 2. Diagnostic des Forces
*   **Architecture Modulaire & Scalable :** Utilisation pertinente de LangGraph pour diviser l'analyse technique entre des agents spécialisés (Indicateurs, Patterns, Tendances).
*   **Gestion du Risque Intégrée :** Le `RiskEngine` synchrone (`execution/risk.py`) impose des garde-fous essentiels (VaR, Drawdown, sizing) avant toute exécution, ce qui démontre une maturité dans la conception financière.
*   **Pipeline de Données Propre :** Le wrapper `OpenBBClient` centralise efficacement les flux de données (OHLCV, News, Macro) et assure une normalisation nécessaire à l'analyse quantitative.
*   **Richesse des Signaux :** La combinaison de facteurs quantitatifs classiques et d'analyses qualitatives via LLM (agent_bias) est une approche hybride puissante.

## 3. Diagnostic des Faiblesses (Fond et Exécution)
*   **Fonctionnalité Hors-Sujet (Majeur) :** Le produit actuel ne répond pas au besoin de "recherche de dépôts GitHub". Il n'y a aucune intégration avec l'API GitHub ou des moteurs de recherche type Tavily/Serper.
*   **Biais de Persona (HFT vs Daily) :** Tous les agents LLM (`signals/agents/quantagent/`) sont promptés comme des analystes "High-Frequency Trading" alors que le pipeline injecte des données journalières ("Daily").
*   **Incohérence des Règles Métier :** L'agent de décision interdit les positions "HOLD", ce qui est une contrainte de HFT absurde pour un investisseur se basant sur des bougies quotidiennes. Cela force l'IA à prendre des décisions binaires risquées.
*   **Absence de Déduplication & Filtrage :** Aucun mécanisme n'est prévu pour dédupliquer des résultats ou assurer la diversité des sources (ex: plusieurs news redondantes ne sont pas filtrées avant injection).
*   **Spécialisation Limitée :** Très orienté "Trading", le système ignore les autres pans de la finance mentionnés dans l'audit (Audit, Forecasting, Reporting, Portfolio Management global).

## 4. Bugs et Limites Observés
1.  **Crash sur Requêtes Ambiguës :** Une requête comme "portfolio management" au lieu d'un ticker (ex: AAPL) provoque un crash de l'API OpenBB (`Expecting value: line 1 column 1 (char 0)`) sans gestion d'erreur gracieuse au niveau du pipeline.
2.  **Limite de Contexte Visuel :** Les agents `Pattern` et `Trend` se basent sur des captures d'écran générées statiquement (`static_util.py`). Si la génération échoue, l'agent tombe en mode "dégradé" sans en avertir l'utilisateur.
3.  **Gestion des Dépendances :** Plusieurs dépendances critiques (ex: `flask`, `yfinance`) sont nécessaires pour les agents mais absentes des dépendances racines, ce qui complique l'installation et l'audit.

## 5. Recommandations Priorisées
1.  **Implémenter la Couche GitHub (P0) :** Créer un agent `GitHubSearchAgent` utilisant l'API Search de GitHub. Cet agent doit extraire des mots-clés financiers de la requête utilisateur pour trouver des dépôts pertinents.
2.  **Harmoniser les Horizons Temporels (P1) :** Aligner les prompts des agents avec la fréquence réelle des données (retirer les mentions "HFT" et autoriser le "HOLD/NEUTRAL").
3.  **Ajouter un Agent de Synthèse & Filtrage (P1) :** Développer un agent chargé de classer les dépôts GitHub par pertinence (stars/forks/activité) et d'éliminer les doublons.
4.  **Gestion d'Erreurs Robuste (P2) :** Intercepter les erreurs de ticker et rediriger les requêtes de langage naturel vers un module de compréhension d'intention (NLU).

## 6. Exemples de Requêtes Financial & Comportement
| Requête (Point Financier) | Résultat Actuel | Problème Observé | Résultat Attendu après Amélioration |
|---|---|---|---|
| "AAPL" | Signal de trading (LONG/SHORT) | Correct pour le trading, mais aucun dépôt GitHub retourné. | Signal + Liste de repos (ex: `apple/swift`, `q-trading/aapl-backtest`). |
| "Analyse de risque VaR" | Erreur JSON (Ticker non trouvé) | Manque de robustesse face à une requête non-ticker. | Top 3 repos GitHub sur le calcul de VaR/ES en Python. |
| "Reporting Audit" | Erreur JSON | Hors sujet. | Repos pertinents pour l'automatisation d'audit financier. |
| "" (Requête vide) | Crash pipeline | Aucune validation en amont. | Message d'erreur demandant de préciser un point financier ou un ticker. |

---
**Conclusion de l'Audit :** L'outil possède une excellente base technique pour l'analyse quantitative, mais échoue totalement sur sa mission de curation de ressources GitHub. Une refonte de la couche "Input Interpretation" et l'ajout d'outils de recherche externe sont impératifs.
