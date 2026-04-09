# Rapport d'Audit : Outil IA QuantAI

## 1. Résumé Exécutif
L'outil QuantAI est une plateforme de trading quantitatif intégrant des agents LLM pour l'analyse visuelle et technique. Bien que l'architecture soit modulaire et robuste pour l'analyse de signaux (momentum, value, qualité), la fonctionnalité principale demandée pour cet audit — **la recommandation de dépôts GitHub pertinents pour des points financiers** — est totalement absente du codebase actuel.

## 2. Diagnostic des Forces
*   **Architecture Modulaire :** Séparation claire entre la donnée (OpenBB), les signaux (facteurs quantitatifs) et la décision (Claude).
*   **Gestion du Risque :** Présence d'un `RiskEngine` synchrone imposant des limites de VaR, drawdown et concentration, ce qui est crucial pour une application financière.
*   **Analyse Multi-Agent :** L'utilisation de LangGraph pour diviser l'analyse entre indicateurs, patterns visuels et tendances est une approche moderne et scalable.
*   **Robustesse des Signaux :** Le pipeline gère correctement les données manquantes ou insuffisantes (marquage en `data_quality` faible).

## 3. Diagnostic des Faiblesses et Limites
*   **Fonctionnalité manquante (Majeur) :** Aucun module de recherche ou de recommandation de dépôts GitHub n'est implémenté. L'outil ne peut pas remplir sa mission de "retourner plusieurs repos GitHub pertinents".
*   **Incohérence des Prompts :** Les agents sont promptés pour du "High-Frequency Trading" (HFT) alors qu'ils opèrent sur des données journalières (Daily) fournies par OpenBB. Cela induit un biais d'analyse temporel.
*   **Spécialisation Limitée :** Les agents actuels sont très orientés "Trading Technique". Les sous-domaines comme l'audit, le reporting financier ou le risk management global ne sont pas couverts par les agents spécialisés.
*   **Absence de Déduplication :** Le système ne possède aucune logique pour filtrer les résultats redondants ou assurer la diversité des sources externes (news/repos).

## 4. Bugs Observés
1.  **Dépendances manquantes :** Le fichier `pyproject.toml` liste des dépendances (ex: `httpx`) qui ne sont pas toujours présentes dans l'environnement de base, provoquant des erreurs à l'exécution.
2.  **Mismatch HFT vs Daily :** L'agent `DecisionMaker` interdit les positions "HOLD" (stratégie HFT), ce qui est absurde pour une analyse basée sur des bougies quotidiennes.

## 5. Recommandations Concrètes (Priorisées)
1.  **Implémenter un Search Tool :** Intégrer un outil comme Tavily ou Serper, ou utiliser l'API GitHub Search pour trouver des repos basés sur les points financiers identifiés.
2.  **Créer un Agent de Filtrage/Classement :** Développer un agent chargé de dédupliquer les dépôts trouvés et de les classer par pertinence (stars, activité, adéquation financière).
3.  **Harmoniser les horizons temporels :** Aligner les prompts des agents LLM avec la fréquence réelle des données injectées.
4.  **Diversifier les Agents :** Ajouter des agents spécialisés dans l'audit et le reporting pour couvrir l'ensemble du spectre financier.

## 6. Exemples de Requêtes Financial & Comportement
| Requête (Point Financier) | Résultat Attendu | Comportement Actuel | Problème |
|---|---|---|---|
| "Analyse de risque VaR" | Repos sur la VaR/ES | Analyse technique AAPL | Hors sujet (pas de recherche de repos) |
| "Algorithme de trading momentum" | Bibliothèques de backtest | Signal quantitatif brut | Pas de lien avec l'écosystème GitHub |
| "" (Requête vide) | Message d'erreur robuste | Erreur `ConnectionError` | Manque de gestion d'erreur au niveau API |

---
**Note d'Audit :** Le produit nécessite une mise à jour majeure pour inclure la couche de recherche externe et de recommandation de ressources pour être conforme aux attentes du marché.
