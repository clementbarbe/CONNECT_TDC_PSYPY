# Temporal Judgement Task

Tâche de jugement de délai temporel entre une action et un stimulus visuel.  
Conçue pour l'IRMf et le comportemental — CENIR, Institut du Cerveau (ICM).

## Principe

1. Une ampoule éteinte apparaît à l'écran
2. **Condition ACTIVE** (barre verte) : le participant appuie pour l'allumer
3. **Condition PASSIVE** (barre rouge) : un opérateur appuie sur le doigt du participant 
4. Après un délai variable (200–700 ms), l'ampoule s'allume
5. Le participant estime le délai perçu (100 à 800 ms, 8 boutons)

## Modes

| Mode | Essais | Feedback | Description |
|------|--------|----------|-------------|
| `training` | 12 | Oui | Entraînement, conditions actives uniquement |
| `base` | 72 + 24 | Non | Baseline → validation crise → post-crise |
| `block` | 24 | Non | Repos → validation crise → bloc court |

## Lancement

Prérequis

    Python 3.10+
    PsychoPy 2025.1.1
    PyQt6

## Données

Les résultats sont sauvegardés dans data/temporal_judgement/ :

    *_{run_type}_{timestamp}.csv — fichier final complet
    *_incremental.csv — backup trial par trial (protection anti-crash)

## Hardware (optionnel)

    Port parallèle : triggers TTL synchronisés
    Eyetracker : messages événementiels EyeLink

Si le matériel est absent, des substituts silencieux prennent le relais automatiquement.
Auteur

Clément BARBE — CENIR, Institut du Cerveau (ICM), Paris