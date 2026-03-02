# Stimulation Électrique — Somatotopie Digitale & Tâche de Prédiction

## Vue d'ensemble

Protocole de stimulation électrique digitale en IRMf pour l'étude de la somatotopie corticale et des mécanismes de prédiction sensorielle.

---

## Architecture

project/
├── tasks/
│   └── stimulation_electrique.py    # Tâche principale
├── gui/
│   └── tabs/
│       └── stimulation_electrique_tab.py  # Onglet GUI (PyQt6)
├── utils/
│   ├── base_task.py                 # Classe mère
│   ├── hardware_manager.py          # Gestion port parallèle + eye-tracker
│   ├── logger.py                    # Logging
│   └── task_factory.py             # Factory (create_task)
├── data/
│   └── stimulation_electrique/      # Données sauvegardées (CSV)
└── README.md

---

## Protocole

### Phase 1 — Finger Mapping

Cartographie somatotopique par block design ON/OFF.

| Paramètre              | Valeur par défaut |
|-------------------------|-------------------|
| Blocs ON                | 20                |
| Blocs OFF               | 20                |
| Durée bloc ON           | 10 s              |
| Durée bloc OFF          | 10 s ± 5 s       |
| Doigts stimulés         | D1, D2, D3, D4   |
| Stims par doigt par bloc| 5                 |
| Total stims par bloc ON | 20                |
| ISI (inter-stim)        | 500 ms            |
| Durée impulsion         | 1 ms              |
| Séquence                | Pseudo-aléatoire  |
| Durée estimée           | ~6 min 40 s       |

**Contraintes de séquence :**
- Pas deux stimulations consécutives sur le même doigt
- Chaque doigt stimulé exactement 5 fois par bloc ON

---

### Phase 2 — Tâche de Prédiction

4 conditions expérimentales, chacune avec la même structure de blocs ON/OFF que le mapping.

| Condition | Code | Doigts | Séquence      | D4              |
|-----------|------|--------|---------------|-----------------|
| FP        | 50   | 4      | Prédictible   | Stimulé         |
| TP        | 51   | 3      | Prédictible   | Omission        |
| FR        | 52   | 4      | Aléatoire     | Stimulé         |
| TR        | 53   | 3      | Aléatoire     | Omission        |

**Séquence prédictible** : D1 → D2 → D3 → D4 (cyclique)

**Omission** : Le slot temporel de D4 est préservé (même ISI), aucune stimulation n'est envoyée. Cela permet de mesurer la réponse cérébrale à l'absence d'un stimulus attendu.

**Pause entre conditions** : 3 min (configurable)

**Durée estimée** : ~35 min (4 conditions × ~7 min + 3 pauses × 3 min)

## Hardware

### Port Parallèle

Chaque doigt est mappé sur un pin dédié du port données (adresse `0x378`) :

| Doigt | Pin  | Code hex | Code décimal |
|-------|------|----------|--------------|
| D1    | D0   | `0x01`   | 1            |
| D2    | D1   | `0x02`   | 2            |
| D3    | D2   | `0x04`   | 4            |
| D4    | D3   | `0x08`   | 8            |

**Principe** : Front montant (1ms) sur le pin → déclenche le stimulateur électrique.

### Codes TTL (Event Markers)

| Événement          | Code |
|--------------------|------|
| Start experiment   | 255  |
| End experiment     | 254  |
| Rest start/end     | 200/201 |
| Mapping start/end  | 210/211 |
| Prediction start/end| 220/221 |
| Block ON start/end | 100/101 |
| Block OFF start/end| 110/111 |
| Stim D1            | 11   |
| Stim D2            | 12   |
| Stim D3            | 13   |
| Stim D4            | 14   |
| Stim omission      | 15   |
| Condition FP       | 50   |
| Condition TP       | 51   |
| Condition FR       | 52   |
| Condition TR       | 53   |
| Pause start/end    | 120/121 |

---

## Installation

### Dépendances

pip install psychopy PyQt6

Drivers port parallèle (Windows)

pip install pyparallel
