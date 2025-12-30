# Wazuh → Dataset ML (Streamlit)

Application Streamlit pour interroger Wazuh Manager et Wazuh Indexer (OpenSearch), visualiser les alertes et générer un dataset CSV prêt pour le Machine Learning.

## Fonctionnalités
- Connexion sécurisée au Wazuh Manager (JWT) et à l'Indexer (Basic Auth).
- Filtrage par plage de dates, agents, techniques MITRE, rule.id, niveau/level et option "Atomic only" (rule.mitre.id présent).
- Tableau filtrable et cartes KPI (alertes, agents, top MITRE, top rule).
- Graphiques Plotly interactifs : top MITRE, volume temporel, répartition par agent, top descriptions de règle.
- Export CSV (dossier `./exports/` + téléchargement direct) avec pagination pour de gros volumes.
- Cache Streamlit pour les chargements répétés.
- Entraînement ML intégré : les alertes chargées peuvent être utilisées pour entraîner et télécharger un modèle PKL/Joblib de classification de sévérité ; si la colonne `severity` est absente, elle est dérivée automatiquement depuis `rule.level` (0-3 low, 4-6 medium, 7-10 high, 11-15 critical).

## Prérequis
- Ubuntu Server avec Python 3.9+.
- Accès réseau au Wazuh Manager (`https://192.168.100.11:55000` par défaut) et au Wazuh Indexer (`https://192.168.100.11:9200` par défaut).

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Variables d'environnement
Aucun mot de passe en clair dans le code. Renseignez :
```bash
export WAZUH_MANAGER_USER="<manager-user>"
export WAZUH_MANAGER_PASS="<manager-pass>"
export WAZUH_INDEXER_USER="<indexer-user>"
export WAZUH_INDEXER_PASS="<indexer-pass>"
```

Vous pouvez aussi saisir les mots de passe dans l'interface Streamlit (champs masqués) ou via `st.secrets` si vous déployez l'app.

## Exécution
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
L'application sera accessible via l'adresse IP du serveur Ubuntu (ex. `http://192.168.100.10:8501`).

## Entraînement du modèle de sévérité
- Chargez les données via l'indexer puis cliquez sur **Entraîner modèle severity** (bouton dans la sidebar).
- La sévérité est auto-ajoutée depuis `rule.level` si absente.
- Les modèles Logistic Regression, Random Forest et XGBoost (si installé) sont comparés ; le meilleur est sauvegardé dans `./models/best_severity_model_<timestamp>.joblib` et téléchargeable dans l'UI. Les métriques sont enregistrées dans `./models/severity_metrics_<timestamp>.json`.

## Notebook ML : classification de sévérité
- Fichier : `notebooks/wazuh_ml_severity_classification.py`
- Placez votre dataset exporté dans `data/wazuh_logs.csv` (mêmes colonnes que l'export CSV).
- Le notebook entraîne et compare Logistic Regression, Random Forest et XGBoost, génère des métriques, et sauvegarde le meilleur modèle (`models/best_severity_model.joblib`) ainsi que les métriques (`models/severity_metrics.json`).

## Utilisation
1. Dans la barre latérale, saisissez les URLs/ports, utilisateurs et (optionnellement) les mots de passe.
2. Ajustez les filtres (dates, agents, MITRE, rules, niveau, Atomic only).
3. Cliquez sur **Tester connexion** pour valider les accès Wazuh Manager / Indexer.
4. Cliquez sur **Charger données** pour récupérer jusqu'à 10k alertes filtrées.
5. Visualisez les KPI, graphiques et le tableau.
6. Cliquez sur **Exporter CSV** pour générer un fichier complet (pagination) dans `./exports/` et le télécharger.

## Notes
- Le pattern d'index par défaut est `wazuh-alerts-4.x-*`.
- La vérification TLS est désactivée par défaut (lab) mais peut être activée via la case "Vérifier TLS".
- Les champs absents sont gérés et laissés vides dans le tableau/export.
