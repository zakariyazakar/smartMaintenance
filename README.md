# Smart Maintenance

Système intelligent d’inspection et de maintenance prédictive pour équipements électriques.

Projet réalisé dans le cadre d’un stage PFA au Groupe OCP à Khouribga, au sein du Service Électricité et Télécommunication.

## Présentation

Cette application réunit deux approches pour aider à surveiller les équipements :
- L’inspection visuelle à partir d’images.
- Le diagnostic de défauts de roulements à partir de signaux vibratoires.

Les résultats sont regroupés dans une interface de supervision développée avec Streamlit.

## Fonctionnement

### Inspection visuelle
1. L’utilisateur fournit une image.
2. YOLOv8 détecte et localise les composants électriques.
3. Des classifieurs spécialisés analysent les composants pris en charge pour identifier leurs défauts visibles.
4. Les résultats sont affichés dans l’interface.

### Analyse vibratoire
1. L’utilisateur fournit un signal vibratoire.
2. Le programme extrait les caractéristiques du signal.
3. Un modèle Random Forest classe l’état du roulement : normal, défaut de bille, de bague intérieure ou de bague extérieure.
4. Les résultats sont affichés dans la plateforme.

### Suivi
Un historique permet de conserver les alertes produites par l’application.

## Technologies

- Python
- Streamlit
- Ultralytics YOLOv8
- Scikit-learn et Random Forest
- Traitement d’images et de signaux

## Données utilisées pour les modèles

- InsPLAD : détection des composants électriques et classification de défauts visuels.
- CWRU : signaux vibratoires de roulements sains et défectueux.

## Organisation du dépôt

| Élément | Rôle |
| --- | --- |
| app.py | Interface principale |
| src/vision.py | Inspection visuelle |
| src/vibration.py | Analyse vibratoire |
| src/history.py | Gestion de l’historique |
| models/vision/ | Détecteur et classifieurs entraînés |
| models/vibration/ | Modèle de diagnostic vibratoire |
| data/ | Données produites pendant l’utilisation |

Les modèles entraînés sont inclus dans le dépôt. L’historique local des alertes est exclu du suivi Git.

## Portée du projet

Il s’agit d’un prototype académique. Le module vibratoire fournit une classification de l’état du roulement à partir du signal analysé.

## Auteur

Zakariya Zakar

GitHub : https://github.com/zakariyazakar
