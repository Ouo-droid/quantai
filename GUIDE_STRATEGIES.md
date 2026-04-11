# Guide de Stratégies Trading - QuantAI

Ce guide explique le fonctionnement de l'application QuantAI et comment intégrer et optimiser vos stratégies de trading en vous basant sur les recherches académiques récentes, notamment l'étude de Quantopian "All that Glitters Is Not Gold" (2016).

## 1. Architecture du Système

QuantAI est structuré en plusieurs couches modulaires :

- **Data Layer (`data/`)** : Utilise OpenBB pour récupérer des données OHLCV, fondamentales et macroéconomiques.
- **Signal Layer (`signals/`)** : Calcule des facteurs quantitatifs (Momentum, Value, Quality, Volatility) et des moments d'ordre supérieur (Skewness, Kurtosis).
- **Agent Layer (`signals/agents/`)** : Intègre des agents LLM (QuantAgent) et des modèles de Machine Learning (QuantMuse) pour enrichir les signaux.
- **Risk Engine (`execution/risk.py`)** : Filtre les ordres en fonction de contraintes de risque strictes (VaR, Drawdown, Concentration).
- **Decision Layer (`execution/decision_agent.py`)** : Utilise un LLM (Claude) pour synthétiser le `SignalVector` en une décision de trading.

## 2. Intégration des Stratégies (Approche Quantopian 2016)

L'étude de Quantopian a révélé des facteurs clés pour prédire la performance "Out-of-Sample" (OOS). Voici comment ils sont intégrés dans QuantAI :

### Moments d'Ordre Supérieur
Ne vous limitez pas au ratio de Sharpe. QuantAI calcule désormais :
- **Skewness** : Pour identifier l'asymétrie des rendements.
- **Kurtosis** : Pour mesurer le risque de queue (fat tails).
- **Tail Ratio** : Pour comparer les gains extrêmes aux pertes extrêmes.

### Machine Learning non-linéaire
Utilisez le module `QuantMuse` qui combine XGBoost et RandomForest. L'étude montre que les classifieurs non-linéaires prédisent mieux la performance OOS que les modèles linéaires simples.

### Limitation du Surapprentissage (Overfitting)
- **Nombre de Backtests** : L'étude montre que plus on effectue de backtests, plus l'écart entre In-Sample et OOS est grand.
- **Pénalisation du Sharpe** : Il est recommandé de pénaliser le ratio de Sharpe par le nombre de tests effectués.

## 3. Comment ajouter une nouvelle stratégie ?

1. **Créer un facteur** : Ajoutez une classe héritant de `BaseFactor` dans `signals/factors/`.
2. **Implémenter `compute()`** : Définissez la logique de calcul de votre signal à partir du DataFrame de prix.
3. **Enregistrer dans l'agrégateur** : Ajoutez votre facteur dans `SignalAggregator.__init__` dans `signals/aggregator.py`.
4. **Valider** : Utilisez `compute_with_stats()` pour vérifier le Information Coefficient (IC) et la t-statistique de votre facteur.

## 4. Quelle stratégie est la plus optimale ?

Selon les conclusions de l'étude :
- **Le Sharpe de la dernière année** : C'est la mesure la plus prédictive de la performance future à court terme.
- **Approche Multi-Facteurs** : La combinaison de facteurs décorrélés (Momentum + Value + Moments d'ordre supérieur) via un modèle ML est plus robuste qu'un facteur unique.
- **Gestion du Risque** : Le hedging et le contrôle strict de la volatilité sont des prédicteurs forts de succès OOS.

## 5. Recommandations pour le Backtesting

- **Indépendance des données** : Testez toujours sur un dataset indépendant ou une période très longue.
- **Approche Scientifique** : Utilisez une approche data-driven (ML) plutôt que de simples heuristiques financières.
- **Validation Walk-Forward** : Évitez le "look-ahead bias" en utilisant des fenêtres de validation glissantes (implémenté dans `QuantMuseAdapter`).
