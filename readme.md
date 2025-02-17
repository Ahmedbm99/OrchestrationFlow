# Orchestration de Flux avec Apache Airflow

Ce projet vise à orchestrer des flux de travail (workflows) en utilisant Apache Airflow. Airflow est une plateforme open-source pour la programmation, la planification et la surveillance des workflows.
## Table des Matières

- [Présentation du Projet](#présentation-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du Projet](#structure-du-projet)
- [Licence](#licence)

## Présentation du Projet

Ce projet utilise Apache Airflow pour automatiser et orchestrer des flux de travail complexes. Les workflows sont définis comme des Directed Acyclic Graphs (DAGs) dans Airflow, permettant une gestion flexible et évolutive des tâches.

## Fonctionnalités

- **Orchestration de tâches** : Définissez et orchestrez des tâches complexes avec des dépendances.
- **Planification** : Planifiez l'exécution des tâches à des intervalles spécifiques.
- **Surveillance** : Surveillez l'état des tâches en temps réel via l'interface web d'Airflow.
- **Extensibilité** : Ajoutez des opérateurs personnalisés pour des besoins spécifiques.

## Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :

- Python 3.13.2 ou supérieur
- pip (gestionnaire de paquets Python)
- Apache Airflow
- Kubectl et kublet
- Minikube
- Docker 

## Installation

1. Clonez ce dépôt sur votre machine locale :

   ```bash
   git clone https://github.com/votre-utilisateur/votre-projet.git
   cd votre-projet

2. Installez les dépendances nécessaires :

    pip install -r requirements.txt

3. Démarrez le serveur web d'Airflow :

    airflow webserver --port 8080
    or 
    kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow

## Utilisation

1.  Accédez à l'interface web d'Airflow à l'adresse http://localhost:8080.

2.  Connectez-vous avec les identifiants et password admin,admin.

3.  Vous pouvez maintenant voir, planifier et surveiller vos DAGs.

## Structure du projet
Model_Weather_Flood_Prediction_/
├── dags/                  # Dossier contenant les DAGs
│   |── weather_data_processing.py         #  DAG d'extraction , transformer et sauvgarder les data météorologique.
|   |__ satellite_data_processing.py       #  DAG d'extraction , transformer et sauvgarder les data de detection inondation.
├── plugins/               # Dossier pour les plugins personnalisés
├── logs/                  # Logs générés par Airflow
├── airflow.cfg            # Fichier de configuration d'Airflow
├── requirements.txt       # Dépendances Python
└── README.md              # Ce fichier


## Licence

Ce projet a été développé dans le cadre d'un projet de fin d'année par **Ahmed Ben Makhlouf** et **Saida Dammak**.  
Il est destiné à des fins éducatives et démonstratives uniquement.  

**Le code est restrictif et protégé.** Toute utilisation, reproduction ou modification du code source est strictement interdite sans une autorisation préalable. Pour toute demande d'utilisation, veuillez contacter **Monsieur Yessine Boujelben** afin d'obtenir une autorisation écrite et explicite.