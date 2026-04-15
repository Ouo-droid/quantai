# 🔍 RAPPORT D'AUDIT - QUANTAI

## 🎯 RÉSUMÉ
L'application **QuantAI** est une plateforme robuste d'analyse financière IA utilisant la stack OpenBB, LangGraph et Claude. Elle offre une base solide pour le trading quantitatif, mais présente des lacunes critiques en termes de **sécurité** et de **scalabilité** pour un usage multi-utilisateur.

---

## 📋 CHECKLIST 5 POINTS

### 1. DONNÉES 💾 — **MOYEN**
- **Sources :** Utilise **OpenBB Platform** (yfinance, FRED, Alpaca), ce qui est fiable et standard.
- **Qualité :** Normalisation basique présente (`_normalize_ohlcv`), mais peu de gestion robuste des gaps ou doublons complexes.
- **Historique :** Le client OpenBB demande par défaut **1 an** d'historique. Bien que le pipeline demande parfois depuis 2022, la contrainte de **3 ans minimum** n'est pas garantie par défaut dans le code core.

### 2. PRÉDICTIONS 🧠 — **BON**
- **Méthodologie :** Utilisation de facteurs académiques (Jegadeesh & Titman) et d'agents LLM (Claude) pour la décision.
- **Validation :** Présence de notebooks de recherche (`research/01_momentum_factor.ipynb`) testant les signaux sur S&P 500 (2015-2025).
- **Réalisme :** Intégration d'un **Risk Engine** complet (VaR 95%, Max Drawdown, Position Sizing) qui filtre les prédictions trop risquées.
- **Points faibles :** Les frais de transaction et le slippage ne semblent pas explicitement modélisés dans l'agrégateur de signaux.

### 3. SÉCURITÉ 🔒 — **CRITIQUE**
- **Gestion des clés :** Stockage correct dans `.env`, mais le module `web_interface.py` permet de **modifier les variables d'environnement globales** via une API publique (`/api/update-api-key`).
- **Authentification :** **ABSENTE**. L'interface web (Flask) n'a aucune protection (pas de login/mot de passe). N'importe qui accédant au port 5000 peut contrôler l'application et utiliser les crédentiels LLM.
- **Traçabilité :** Le `RiskEngine` logue ses décisions, mais il n'y a pas d'audit log des actions utilisateurs sur l'interface web.

### 4. PERFORMANCES ⚡ — **INSUFFISANT**
- **Vitesse :** L'application tourne en mode `debug=True` sur Flask. Le chargement dépend fortement de la latence des API LLM (Claude/Anthropic), ce qui peut dépasser les 3s.
- **Scalabilité :** Le système est conçu comme un **outil local monocompte**. Il ne peut PAS supporter **100 utilisateurs simultanés** car il partage les mêmes variables d'environnement globales pour les clés API.
- **Mobile :** Interface web basique (Bootstrap), fonctionnelle mais pas optimisée pour du 60fps mobile.

---

## 🛠 RECOMMANDATIONS PRIORITAIRES
1. **Ajouter une couche d'authentification** (ex: Flask-Login ou OAuth) avant toute exposition réseau.
2. **Isoler les sessions utilisateurs** : Ne pas stocker les clés API dans `os.environ` de manière globale si l'app devient multi-utilisateur.
3. **Augmenter l'historique par défaut** à 756 jours de bourse (~3 ans) dans `OpenBBClient.ohlcv`.
4. **Déployer via un serveur WSGI** (Gunicorn/Uvicorn) au lieu du serveur de développement Flask.
