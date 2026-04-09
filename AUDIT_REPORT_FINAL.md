# Rapport d'Audit : Système QuantAI (Analyse de Dépôts GitHub)

## 1. Résumé Exécutif
L'audit a porté sur la capacité de l'outil QuantAI à analyser des points financiers et à retourner des dépôts GitHub pertinents. Le constat majeur est une **absence totale de cette fonctionnalité** dans le codebase actuel. Le système est structuré comme un pipeline de trading quantitatif (OpenBB → Signaux → Decision Agent), mais il manque la couche de recherche externe nécessaire pour satisfaire le besoin d'audit et de recommandation de ressources GitHub.

## 2. Diagnostic des Forces
*   **Architecture Modulaire et Extensible :** La séparation claire entre la couche de données (`OpenBBClient`), la couche de signaux (`SignalAggregator`) et la couche de décision (`DecisionAgent`) facilite l'ajout de nouveaux modules.
*   **Infrastructure de Diagnostic Robuste :** La présence de `scripts/diagnose.py` permet de vérifier rapidement l'intégrité du système, bien que plusieurs tests échouent actuellement par manque d'API.
*   **Moteur de Risque (Risk Engine) :** La validation synchrone des ordres (`RiskEngine`) est un atout sérieux pour toute application financière, garantissant une robustesse face aux décisions absurdes des LLM.

## 3. Diagnostic des Faiblesses
*   **Lacune Fonctionnelle Critique :** Aucun module ne permet d'interroger GitHub ou un moteur de recherche pour l'associer à des points financiers.
*   **Incohérence Temporelle (Mismatch HFT vs Daily) :** Les agents LLM (notamment dans le submodule `QuantAgent`) sont promptés pour du trading haute fréquence (HFT) alors que les données sont journalières (Daily). Cela fausse l'analyse et le "justification" du signal.
*   **Horizon de Spécialisation Réduit :** L'outil est hyper-spécialisé dans le trading technique. Il ignore les sous-domaines comme l'audit, le reporting ESG, ou le risk management global.
*   **Fragilité de l'Environnement :** La CI échoue actuellement car des dépendances requises par les tests du submodule (ex: `flask`) ne sont pas incluses dans le `pyproject.toml` de la racine, créant un environnement de test instable.

## 4. Bugs et Limites Observés
1.  **Interdiction du "HOLD" :** Le `DecisionMaker` de `QuantAgent` interdit explicitement la décision "HOLD", ce qui est une erreur grave pour des investissements basés sur des données quotidiennes ou pour de l'analyse long-terme.
2.  **Gestion d'Erreur API :** Le système plante violemment (`ModuleNotFoundError` ou `AuthenticationError`) au lieu de proposer un mode dégradé gracieux lors de l'absence de clés API.
3.  **Absence de Déduplication :** Si une recherche était implémentée, rien n'est prévu pour filtrer les dépôts redondants ou peu qualitatifs (ex: repos vides, forks sans valeur ajoutée).

## 5. Recommandations Priorisées
1.  **Intégration GitHub Search API (P0) :** Créer un `GitHubAdapter` capable d'extraire des dépôts en fonction des mots-clés financiers identifiés par les agents.
2.  **Agent de Filtrage et Classement (P1) :** Implémenter un agent LLM dédié à l'évaluation de la qualité des dépôts trouvés (nombre de stars, fraîcheur du dernier commit, pertinence de la documentation).
3.  **Harmonisation des Prompts (P1) :** Réécrire les prompts des agents pour supprimer la mention "HFT" et s'adapter à l'analyse financière multi-horizon.
4.  **Module de Diversité (P2) :** Ajouter une logique de déduplication pour assurer une variété de dépôts (ex: un dépôt de backtest, un dépôt de données, un dépôt de reporting).

## 6. Exemples de Requêtes et Comportements (Audit)

| Requête Financière | Résultats GitHub (Actuels) | Problème Observé |
|:--- |:--- |:--- |
| "Analyse de risque VaR" | Aucun (Erreur API) | Le système tente une analyse technique sur le texte "Analyse de risque VaR" au lieu de chercher des ressources. |
| "Algorithme momentum" | Aucun (Signal Brut) | Retourne un z-score de momentum quantitatif sans lien vers des implémentations de référence. |
| "Audit de portefeuille" | Aucun (Hors-sujet) | L'agent de décision n'a aucune connaissance du concept d'audit et reste bloqué sur LONG/SHORT. |

---
**Conclusion de l'Audit :** L'outil possède une base technique saine pour la finance quantitative mais échoue totalement sur sa mission de recommandation GitHub. Une refonte de la couche "Agent Cluster" pour inclure des outils de recherche externe est indispensable.
