# connectelec.py
"""
Stimulation Électrique — File-Driven Pre-Computed Timeline
===========================================================
7T laminar fMRI — UN run par lancement.

FORMATS DE DESIGN (TSV avec header) :
──────────────────────────────────────
SOMATOTOPIE  → design/somatotopie/somatotopie.tsv
    colonnes : onset   duration   finger
    ex :       13.500  2.5        D1

PREDICTION   → design/prediction/runN/prediction.tsv
    colonnes : onset  duration  type  condition  finger  is_omission
    ex :       10.000 3.000     consigne  FP      NA     0
               14.000 3.000     stim      FP      D4     0
               93.764 0.000     stim      TR      D1     1

    type = "consigne"  →  affichage image + texte (finger = NA)
    type = "stim" + is_omission=0  →  stimulation électrique
    type = "stim" + is_omission=1  →  omission (duration=0, pas de trigger)

    Images attendues dans : design/prediction/images/
        FP.png   FR.png   TP.png   TR.png

DURÉES HARDCODÉES :
───────────────────
Chaque run a une durée totale fixe (incluant un padding post-stim).
La timeline se termine exactement à cette durée, indépendamment du
dernier événement de stimulation.

ARCHITECTURE :
──────────────
1.  Les timings sont lus depuis les fichiers de design.
2.  Pour chaque consigne :
      a. visual_consigne_on  à onset
      b. visual_consigne_off à onset + duration (retour fixation)
3.  Pour chaque événement stimulé :
      a. finger_select (pin doigt) à onset − 250 ms
      b. Train de triggers 64 espacés de burst_interval_ms
4.  Pour chaque omission : un marqueur est enregistré, aucun trigger.
5.  La timeline complète est pré-calculée avant le trigger IRM.

Fichiers produits :
    *_planned.csv       → timeline planifiée
    *_incremental.csv   → écriture progressive pendant l'exécution
    *_<timestamp>.csv   → fichier final propre
"""

from __future__ import annotations

import csv
import gc
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from psychopy import core, visual
from utils.base_task import BaseTask

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

FINGER_PIN_MAP: Dict[str, int] = {
    "D1": 2, "D2": 4, "D3": 8, "D4": 16, "D5": 32,
}

STIM_TRIGGER: int = 64
DEFAULT_BURST_INTERVAL_MS: float = 75.0
FINGER_SWITCH_LEAD_MS: float = 250.0        # fixe
TR_S: float = 2.0                            # fixe

# ── DURÉES HARDCODÉES (secondes) ──────────────────────────────────────
RUN_DURATIONS_S: Dict[str, float] = {
    "somatotopy":   16 * 60,    # 16 min
    "prediction_1": 772.0 + 10.0,      # 8 min
    "prediction_2": 786.0 + 10.0,
    "prediction_3": 794.0 + 10.0,
}

_ACTION_PRIORITY: Dict[str, int] = {
    "visual_consigne_off": -1,   # nettoyage avant tout autre visuel
    "visual_fixation":      0,
    "visual_consigne_on":   0,
    "finger_select":        1,
    "marker":               2,
    "stim_burst":           3,
    "stim_omit":            3,
}

DESIGN_PATHS: Dict[str, str] = {
    "somatotopy":   os.path.join("design", "somatotopie"),
    "prediction_1": os.path.join("design", "prediction", "run1"),
    "prediction_2": os.path.join("design", "prediction", "run2"),
    "prediction_3": os.path.join("design", "prediction", "run3"),
}

SOMATOTOPY_TSV_NAME: str = "somatotopie.tsv"
PREDICTION_TSV_NAME: str = "prediction.tsv"

# ── Consignes visuelles (prediction uniquement) ─────────────────────────
CONSIGNE_TEXTS: Dict[str, str] = {
    "FP": (
        "Faites attention à la stimulation de chaque doigt et prédisez\n"
        "quand l'index sera stimulé selon le rythme temporel."
    ),
    "TP": (
        "Faites attention à la stimulation de chaque doigt et prédisez\n"
        "quand l'index sera stimulé, même si aucune stimulation "
        "n'est délivrée."
    ),
    "FR": (
        "Faites attention à la stimulation de chaque doigt,\n"
        "mais n'essayez pas de prédire un motif rythmique ou temporel."
    ),
    "TR": (
        "Faites attention à la stimulation de chaque doigt,\n"
        "mais n'essayez pas de prédire un motif rythmique ou temporel."
    ),
}
CONSIGNE_IMAGES_DIR: str = os.path.join("images")
_CONSIGNE_IMG_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".bmp")

# ── Layout consigne (unités norm : −1 → +1) ─────────────────────────────
CONSIGNE_IMG_POS: tuple    = (0.0, 0.25)
CONSIGNE_IMG_SIZE: tuple   = (0.50, 0.50)
CONSIGNE_TXT_POS: tuple    = (0.0, -0.15)
CONSIGNE_TXT_HEIGHT: float = 0.055


# ═════════════════════════════════════════════════════════════════════════════

class ConnectElec(BaseTask):
    """
    One run per instantiation.
    Timeline entièrement construite depuis les fichiers de design.
    """

    # ─────────────────────────────────────────────────────────────────────
    #  INIT
    # ─────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        win: visual.Window,
        nom: str,
        session: str = "01",
        mode: str = "fmri",
        run_type: str = "somatotopy",
        run_number: int = 1,
        burst_interval_ms: float = DEFAULT_BURST_INTERVAL_MS,
        enregistrer: bool = True,
        eyetracker_actif: bool = False,
        parport_actif: bool = True,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="Stimulation_Electrique",
            folder_name="stimulation_electrique",
            eyetracker_actif=eyetracker_actif,
            parport_actif=parport_actif,
            enregistrer=enregistrer,
            et_prefix="SE",
        )

        # ── identifiers ──────────────────────────────────────────────────
        self.mode: str       = mode.lower()
        self.run_type: str   = run_type.lower()
        if self.run_type == "mapping":
            self.run_type = "somatotopy"
        self.run_number: int = run_number

        # ── burst timing ─────────────────────────────────────────────────
        self.burst_interval_ms: float    = burst_interval_ms
        self.burst_interval_s: float     = burst_interval_ms / 1000.0
        self.finger_switch_lead_s: float = FINGER_SWITCH_LEAD_MS / 1000.0

        # ── run duration (hardcoded) ─────────────────────────────────────
        dur_key = (
            "somatotopy" if self.run_type == "somatotopy"
            else f"prediction_{self.run_number}"
        )
        self.run_duration_s: float = RUN_DURATIONS_S.get(dur_key, 360.0)

        # ── hardware map ─────────────────────────────────────────────────
        self.finger_pin_map: Dict[str, int] = dict(FINGER_PIN_MAP)

        # ── runtime state ─────────────────────────────────────────────────
        self.global_records: List[Dict[str, Any]] = []
        self.timeline: List[Dict[str, Any]] = []

        # ── consigne visuals (prediction only) ────────────────────────────
        self._consigne_images: Dict[str, visual.ImageStim] = {}
        self._consigne_texts:  Dict[str, visual.TextStim]  = {}

        # ── init chain ────────────────────────────────────────────────────
        self._detect_display_scaling()
        self._measure_frame_rate()
        self._setup_key_mapping()
        self._init_incremental_file(
            suffix=f"_{self.run_type}_run{self.run_number:02d}"
        )

        # ── LOAD FILES & BUILD TIMELINE ──────────────────────────────────
        self._design_dir: str = self._resolve_design_dir()
        if self.run_type != "somatotopy":
            self._load_consigne_visuals()
        self._stim_events: List[Dict[str, Any]] = self._load_stim_events()
        self._build_full_timeline()
        self._save_planned_timeline()

        n_vol = int(self.run_duration_s / TR_S)
        self.logger.ok(
            f"ConnectElec ready | {self.run_type} "
            f"run {self.run_number:02d} | "
            f"{len(self.timeline)} events | "
            f"run duration = {self.run_duration_s:.0f} s "
            f"({self.run_duration_s / 60:.1f} min) | "
            f"{n_vol} volumes (TR={TR_S}s)"
        )

    # =====================================================================
    #  INIT HELPERS
    # =====================================================================

    def _detect_display_scaling(self) -> None:
        self.pixel_scale = 2.0 if self.win.size[1] > 1200 else 1.0

    def _measure_frame_rate(self) -> None:
        measured = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, threshold=1
        )
        self.frame_rate: float = measured if measured else 60.0
        self.frame_duration_s: float = 1.0 / self.frame_rate
        self.logger.log(
            f"Frame rate: {self.frame_rate:.1f} Hz "
            f"({self.frame_duration_s * 1000:.2f} ms/frame)"
        )

    def _setup_key_mapping(self) -> None:
        if self.mode == "fmri":
            self.key_trigger  = "t"
            self.key_continue = "b"
        else:
            self.key_trigger  = "t"
            self.key_continue = "space"

    # =====================================================================
    #  CONSIGNE VISUAL LOADING
    # =====================================================================

    def _load_consigne_visuals(self) -> None:
        """
        Charge les images et crée les TextStim pour chaque condition
        de consigne (FP, FR, TP, TR).

        Images cherchées dans :
            {root_dir}/design/prediction/images/<COND>.{png,jpg,…}

        Textes définis dans CONSIGNE_TEXTS (module-level).
        """
        images_dir = os.path.join(self.root_dir, CONSIGNE_IMAGES_DIR)

        for cond, label in CONSIGNE_TEXTS.items():
            # ── Image ────────────────────────────────────────────────
            img_path: Optional[str] = None
            for ext in _CONSIGNE_IMG_EXTENSIONS:
                candidate = os.path.join(images_dir, f"{cond}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is not None:
                try:
                    self._consigne_images[cond] = visual.ImageStim(
                        self.win,
                        image=img_path,
                        pos=CONSIGNE_IMG_POS,
                        size=CONSIGNE_IMG_SIZE,
                        units="norm",
                    )
                except Exception as exc:
                    self.logger.warn(
                        f"Image {cond} non chargée : {exc}"
                    )
            else:
                self.logger.warn(
                    f"Image consigne manquante : "
                    f"{images_dir}/{cond}.*"
                )

            # ── Texte ────────────────────────────────────────────────
            self._consigne_texts[cond] = visual.TextStim(
                self.win,
                text=label,
                pos=CONSIGNE_TXT_POS,
                height=CONSIGNE_TXT_HEIGHT,
                color="white",
                units="norm",
                wrapWidth=1.6,
            )

        loaded_imgs = sorted(self._consigne_images.keys())
        self.logger.log(
            f"Consigne visuals : {len(self._consigne_texts)} textes, "
            f"{len(self._consigne_images)} images {loaded_imgs}"
        )

    # =====================================================================
    #  DESIGN FILE LOADING
    # =====================================================================

    def _resolve_design_dir(self) -> str:
        if self.run_type == "somatotopy":
            key = "somatotopy"
        else:
            key = f"prediction_{self.run_number}"

        rel = DESIGN_PATHS.get(key)
        if rel is None:
            raise FileNotFoundError(
                f"Configuration de run inconnue : {key}"
            )

        full = os.path.join(self.root_dir, rel)
        if not os.path.isdir(full):
            raise FileNotFoundError(
                f"Dossier de design introuvable : {full}"
            )

        self.logger.log(f"Design dir : {full}")
        return full

    # ── Dispatcher ───────────────────────────────────────────────────────

    def _load_stim_events(self) -> List[Dict[str, Any]]:
        if self.run_type == "somatotopy":
            events = self._load_somatotopy_tsv()
        else:
            events = self._load_prediction_tsv()

        if not events:
            raise ValueError(
                f"Aucun événement dans {self._design_dir}"
            )

        events.sort(key=lambda e: e["onset_s"])

        # ── Validation : aucun événement ne dépasse la durée du run ──
        last_end = max(
            e["onset_s"] + e["duration_s"] for e in events
        )
        if last_end > self.run_duration_s:
            self.logger.warn(
                f"Dernier événement se termine à {last_end:.1f} s "
                f"mais le run dure {self.run_duration_s:.1f} s — "
                f"les événements tardifs seront quand même exécutés."
            )

        return events

    # ── Somatotopie : somatotopie.tsv (3 colonnes TSV) ──────────────────

    def _load_somatotopy_tsv(self) -> List[Dict[str, Any]]:
        """
        Format TSV avec header :
            onset   duration   finger
            13.500  2.5        D1
        """
        fpath = os.path.join(self._design_dir, SOMATOTOPY_TSV_NAME)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Fichier {SOMATOTOPY_TSV_NAME} introuvable : {fpath}"
            )

        events: List[Dict[str, Any]] = []
        fingers_seen: set = set()

        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            if reader.fieldnames is None:
                raise ValueError(f"Fichier vide : {fpath}")

            required = {"onset", "duration", "finger"}
            actual = set(reader.fieldnames)
            missing = required - actual
            if missing:
                raise ValueError(
                    f"Colonnes manquantes dans {fpath} : {missing}\n"
                    f"Colonnes trouvées : {actual}"
                )

            for row_num, row in enumerate(reader, start=2):
                try:
                    onset_s    = float(row["onset"])
                    duration_s = float(row["duration"])
                    finger     = row["finger"].strip()
                except (ValueError, KeyError) as exc:
                    self.logger.warn(
                        f"{SOMATOTOPY_TSV_NAME}:{row_num} — "
                        f"parse error : {exc}"
                    )
                    continue

                if finger not in FINGER_PIN_MAP:
                    self.logger.warn(
                        f"{SOMATOTOPY_TSV_NAME}:{row_num} — "
                        f"doigt inconnu '{finger}', ignoré"
                    )
                    continue

                fingers_seen.add(finger)
                events.append({
                    "finger":        finger,
                    "onset_s":       onset_s,
                    "duration_s":    duration_s,
                    "condition":     "somatotopy",
                    "event_type":    "stim",
                    "is_omission":   False,
                    "is_consigne":   False,
                })

        self.logger.log(
            f"Somatotopy : {len(events)} événements "
            f"({', '.join(sorted(fingers_seen))})"
        )
        return events

    # ── Prediction : prediction.tsv (6 colonnes TSV) ────────────────────

    def _load_prediction_tsv(self) -> List[Dict[str, Any]]:
        """
        Format TSV avec header :
            onset  duration  type  condition  finger  is_omission

        type = "consigne"  → affichage image + texte (finger = NA)
        type = "stim" + is_omission=0  → stimulation électrique
        type = "stim" + is_omission=1  → omission (duration=0)
        """
        fpath = os.path.join(self._design_dir, PREDICTION_TSV_NAME)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Fichier {PREDICTION_TSV_NAME} introuvable : {fpath}"
            )

        events: List[Dict[str, Any]] = []
        conditions_seen: set = set()
        fingers_seen: set = set()

        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            if reader.fieldnames is None:
                raise ValueError(f"Fichier vide : {fpath}")

            required = {
                "onset", "duration", "type",
                "condition", "finger", "is_omission",
            }
            actual = set(reader.fieldnames)
            missing = required - actual
            if missing:
                raise ValueError(
                    f"Colonnes manquantes dans {fpath} : {missing}\n"
                    f"Colonnes trouvées : {actual}"
                )

            for row_num, row in enumerate(reader, start=2):
                try:
                    onset_s     = float(row["onset"])
                    duration_s  = float(row["duration"])
                    event_type  = row["type"].strip().lower()
                    condition   = row["condition"].strip()
                    finger_raw  = row["finger"].strip()
                    is_omission = int(row["is_omission"]) == 1
                except (ValueError, KeyError) as exc:
                    self.logger.warn(
                        f"{PREDICTION_TSV_NAME}:{row_num} — "
                        f"parse error : {exc}"
                    )
                    continue

                is_consigne = event_type == "consigne"
                is_stim     = event_type == "stim" and not is_omission

                # Finger : "NA" ou vide accepté pour consignes
                finger = finger_raw if finger_raw.upper() != "NA" else ""

                # Doigt obligatoire pour stim / omission
                if not is_consigne:
                    if finger not in FINGER_PIN_MAP:
                        self.logger.warn(
                            f"{PREDICTION_TSV_NAME}:{row_num} — "
                            f"doigt inconnu '{finger_raw}', ignoré"
                        )
                        continue

                conditions_seen.add(condition)
                if finger in FINGER_PIN_MAP:
                    fingers_seen.add(finger)

                events.append({
                    "finger":      finger,
                    "onset_s":     onset_s,
                    "duration_s":  duration_s,
                    "condition":   condition,
                    "event_type":  event_type,
                    "is_omission": is_omission,
                    "is_consigne": is_consigne,
                    "is_stim":     is_stim,
                })

        n_stim    = sum(1 for e in events if e["is_stim"])
        n_omit    = sum(1 for e in events if e["is_omission"])
        n_consign = sum(1 for e in events if e["is_consigne"])
        self.logger.log(
            f"Prediction : {len(events)} événements "
            f"({n_stim} stimulés, {n_omit} omissions, "
            f"{n_consign} consignes) | "
            f"conditions : {sorted(conditions_seen)} | "
            f"doigts : {sorted(fingers_seen)}"
        )
        return events

    # =====================================================================
    #  TIMELINE CONSTRUCTION
    # =====================================================================

    def _add_event(self, onset_s: float, action: str, **kw: Any) -> None:
        evt: Dict[str, Any] = {
            "onset_s":   round(onset_s, 6),
            "action":    action,
            "_priority": _ACTION_PRIORITY.get(action, 9),
        }
        evt.update(kw)
        self.timeline.append(evt)

    def _build_full_timeline(self) -> None:
        """
        Construit la timeline complète.

        Pour chaque CONSIGNE :
          1. visual_consigne_on   à onset
          2. visual_consigne_off  à onset + duration  (→ retour fixation)

        Pour chaque STIM (is_omission=0) :
          1. finger_select   à onset − 250 ms
          2. stim_burst × N  à onset + i × burst_interval

        Pour chaque OMISSION (is_omission=1) :
          1. stim_omit (marqueur seul)

        La timeline se termine à self.run_duration_s (hardcodé).
        """
        self.timeline.clear()

        # ── Run start + fixation ──
        self._add_event(0.0, "marker", label="run_start",
                        run_type=self.run_type,
                        run_number=self.run_number)
        self._add_event(0.0, "visual_fixation", label="fixation_start")

        # ── Événements ──
        for ei, stim in enumerate(self._stim_events):
            onset       = stim["onset_s"]
            duration    = stim["duration_s"]
            finger      = stim["finger"]
            condition   = stim.get("condition", "")
            is_consigne = stim.get("is_consigne", False)
            is_stim     = stim.get("is_stim", False)
            is_omit     = stim.get("is_omission", False)

            if is_consigne:
                # ── Consigne visuelle ON ──
                self._add_event(
                    onset, "visual_consigne_on",
                    label="consigne_on",
                    condition=condition,
                    stim_event_idx=ei,
                    duration_s=duration,
                )
                # ── Consigne visuelle OFF → retour fixation ──
                self._add_event(
                    onset + duration, "visual_consigne_off",
                    label="consigne_off",
                    condition=condition,
                    stim_event_idx=ei,
                )

            elif is_stim and duration > 0:
                pin = self.finger_pin_map[finger]

                # ── Sélection du doigt ──
                sel_t = max(0.0, onset - self.finger_switch_lead_s)
                self._add_event(
                    sel_t, "finger_select",
                    label="finger_select",
                    finger=finger,
                    pin_code=pin,
                    condition=condition,
                    stim_event_idx=ei,
                )

                # ── Train de bursts ──
                n_bursts = max(
                    1,
                    int(duration / self.burst_interval_s + 1e-9) + 1,
                )
                for bi in range(n_bursts):
                    burst_t = onset + bi * self.burst_interval_s
                    if bi > 0 and burst_t > onset + duration + 1e-6:
                        break
                    self._add_event(
                        burst_t, "stim_burst",
                        label="stim_burst",
                        finger=finger,
                        pin_code=STIM_TRIGGER,
                        condition=condition,
                        stim_event_idx=ei,
                        burst_index=bi,
                        n_bursts=n_bursts,
                        is_omission=False,
                    )

            elif is_omit:
                # ── Omission ──
                self._add_event(
                    onset, "stim_omit",
                    label="stim_omit",
                    finger=finger,
                    pin_code=0,
                    condition=condition,
                    stim_event_idx=ei,
                    is_omission=True,
                    duration_s=0.0,
                )

            else:
                self.logger.warn(
                    f"Événement {ei} ignoré (type inconnu)"
                )

        # ── Run end : durée hardcodée ──
        self._add_event(
            self.run_duration_s, "visual_fixation",
            label="final_fixation",
        )
        self._add_event(
            self.run_duration_s, "marker",
            label="run_end",
            run_duration_s=self.run_duration_s,
        )

        # ── Tri stable : onset → priorité ──
        self.timeline.sort(key=lambda e: (e["onset_s"], e["_priority"]))

        for i, evt in enumerate(self.timeline):
            evt["event_index"] = i

        # ── Résumé ──
        n_sel    = sum(1 for e in self.timeline
                       if e["action"] == "finger_select")
        n_burst  = sum(1 for e in self.timeline
                       if e["action"] == "stim_burst")
        n_omit   = sum(1 for e in self.timeline
                       if e["action"] == "stim_omit")
        n_con_on = sum(1 for e in self.timeline
                       if e["action"] == "visual_consigne_on")

        last_stim = max(
            (e["onset_s"] + e.get("duration_s", 0)
             for e in self._stim_events),
            default=0.0,
        )
        padding = self.run_duration_s - last_stim

        self.logger.log(
            f"Timeline : {len(self.timeline)} events "
            f"({n_sel} select, {n_burst} burst, {n_omit} omit, "
            f"{n_con_on} consignes) | "
            f"last event ends at {last_stim:.1f} s | "
            f"padding = {padding:.1f} s | "
            f"run_end = {self.run_duration_s:.1f} s"
        )

    # ── Sauvegarde planned ───────────────────────────────────────────────

    def _save_planned_timeline(self) -> None:
        if not self.enregistrer or not self.timeline:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            f"{self.nom}_{self.task_name}"
            f"_{self.run_type}_run{self.run_number:02d}"
            f"_{ts}_planned.csv"
        )
        path = os.path.join(self.data_dir, fname)
        try:
            all_keys = sorted(
                set().union(*(e.keys() for e in self.timeline))
                - {"_priority"}
            )
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=all_keys, extrasaction="ignore"
                )
                writer.writeheader()
                for evt in self.timeline:
                    writer.writerow(
                        {k: v for k, v in evt.items() if k != "_priority"}
                    )
            self.logger.ok(f"Planned timeline : {path}")
        except Exception as exc:
            self.logger.err(f"Sauvegarde planned échouée : {exc}")

    # =====================================================================
    #  EXECUTION ENGINE
    # =====================================================================

    def _wait_until(
        self, target_s: float, high_precision: bool = False
    ) -> None:
        remaining = target_s - self.task_clock.getTime()
        if remaining <= 0:
            return
        if high_precision:
            if remaining > 0.003:
                core.wait(remaining - 0.002, hogCPUperiod=0.0)
            while self.task_clock.getTime() < target_s:
                pass
        else:
            core.wait(remaining, hogCPUperiod=0.0)

    # ─────────────────────────────────────────────────────────────────────
    #  CONSIGNE DRAWING HELPER
    # ─────────────────────────────────────────────────────────────────────

    def _draw_consigne(self, condition: str) -> None:
        """
        Dessine l'image de la condition (au-dessus) puis le texte
        descriptif (en-dessous), et flip.

        Si l'image est manquante seul le texte est affiché.
        Si la condition est inconnue un fallback texte brut est utilisé.
        """
        if condition in self._consigne_images:
            self._consigne_images[condition].draw()

        if condition in self._consigne_texts:
            self._consigne_texts[condition].draw()
        else:
            fallback = visual.TextStim(
                self.win,
                text=condition,
                pos=CONSIGNE_TXT_POS,
                height=CONSIGNE_TXT_HEIGHT,
                color="white",
                units="norm",
            )
            fallback.draw()

        self.win.flip()

    # ─────────────────────────────────────────────────────────────────────

    def _dispatch_event(self, event: Dict[str, Any]) -> float:
        action = event["action"]

        if action == "visual_fixation":
            self.fixation.draw()
            self.win.flip()
            return self.task_clock.getTime()

        if action == "visual_consigne_on":
            t = self.task_clock.getTime()
            self._draw_consigne(event.get("condition", ""))
            return t

        if action == "visual_consigne_off":
            self.fixation.draw()
            self.win.flip()
            return self.task_clock.getTime()

        if action == "finger_select":
            t = self.task_clock.getTime()
            self.ParPort.send_trigger(event["pin_code"])
            return t

        if action == "stim_burst":
            t = self.task_clock.getTime()
            self.ParPort.send_trigger(STIM_TRIGGER)
            return t

        if action == "stim_omit":
            return self.task_clock.getTime()

        return self.task_clock.getTime()

    def _build_record(
        self, event: Dict[str, Any], actual_t: float
    ) -> Dict[str, Any]:
        err_ms = (actual_t - event["onset_s"]) * 1000.0
        return {
            "participant":         self.nom,
            "session":             self.session,
            "run_type":            self.run_type,
            "run_number":          self.run_number,
            "event_index":         event.get("event_index", ""),
            "action":              event["action"],
            "label":               event.get("label", ""),
            "onset_planned_s":     event["onset_s"],
            "onset_actual_s":      round(actual_t, 6),
            "scheduling_error_ms": round(err_ms, 3),
            "condition":           event.get("condition", ""),
            "finger":              event.get("finger", ""),
            "pin_code":            event.get("pin_code", ""),
            "is_omission":         event.get("is_omission", ""),
            "stim_event_idx":      event.get("stim_event_idx", ""),
            "burst_index":         event.get("burst_index", ""),
            "n_bursts":            event.get("n_bursts", ""),
        }

    def _execute_timeline(self) -> None:
        n_events = len(self.timeline)
        self.logger.log(f"Exécution : {n_events} événements …")

        gc.disable()

        try:
            for i, event in enumerate(self.timeline):

                is_burst = event["action"] == "stim_burst"

                if not is_burst:
                    self.should_quit()
                elif event.get("burst_index", 0) == 0:
                    self.should_quit()

                self._wait_until(
                    event["onset_s"],
                    high_precision=(
                        is_burst or event["action"] == "finger_select"
                    ),
                )

                actual_t = self._dispatch_event(event)

                rec = self._build_record(event, actual_t)
                self.global_records.append(rec)

                if not is_burst or event.get("burst_index", 0) == 0:
                    self.save_trial_incremental(rec)

                if self.eyetracker_actif:
                    lbl = event.get("label", event["action"])
                    self.EyeTracker.send_message(
                        f"R{self.run_number:02d}_"
                        f"E{i:04d}_{lbl.upper()}"
                    )

                # ── Logging conditionnel ──
                if is_burst and abs(rec["scheduling_error_ms"]) > 1.0:
                    self.logger.warn(
                        f"TIMING E{i} "
                        f"{event.get('finger', '?')} "
                        f"B{event.get('burst_index', '?')}: "
                        f"{rec['scheduling_error_ms']:+.2f} ms"
                    )

                if event["action"] == "stim_omit":
                    self.logger.log(
                        f"  Omission {event.get('finger', '?')} "
                        f"({event.get('condition', '?')}) "
                        f"[t={actual_t:.2f} s]"
                    )

                if event["action"] == "visual_consigne_on":
                    self.logger.log(
                        f"  Consigne ON : {event.get('condition', '?')} "
                        f"[t={actual_t:.2f} s, "
                        f"dur={event.get('duration_s', '?')} s]"
                    )

                if event["action"] == "visual_consigne_off":
                    self.logger.log(
                        f"  Consigne OFF : "
                        f"{event.get('condition', '?')} "
                        f"[t={actual_t:.2f} s]"
                    )

                if event.get("label") == "run_end":
                    self.logger.log(
                        f"  run_end [t={actual_t:.2f} s / "
                        f"planned {event['onset_s']:.1f} s]"
                    )

        finally:
            gc.enable()
            gc.collect()

        self.logger.ok("Exécution de la timeline terminée.")

        bursts = [
            r for r in self.global_records
            if r["action"] == "stim_burst"
            and r["scheduling_error_ms"] != ""
        ]
        if bursts:
            errors = [abs(r["scheduling_error_ms"]) for r in bursts]
            mean_e = sum(errors) / len(errors)
            max_e  = max(errors)
            n_over_1 = sum(1 for e in errors if e > 1.0)
            n_over_2 = sum(1 for e in errors if e > 2.0)
            self.logger.log(
                f"Timing résumé : {len(bursts)} bursts | "
                f"mean |err| = {mean_e:.3f} ms | "
                f"max |err| = {max_e:.3f} ms | "
                f">1 ms : {n_over_1} | >2 ms : {n_over_2}"
            )

    # =====================================================================
    #  MAIN ENTRY
    # =====================================================================

    def run(self) -> None:
        finished = False

        try:
            self._show_instructions()
            self.wait_for_trigger()
            self._execute_timeline()

            finished = True
            self.logger.ok(
                f"Run {self.run_number:02d} ({self.run_type}) terminé."
            )

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Interruption manuelle.")

        except Exception as exc:
            self.logger.err(f"CRITICAL : {exc}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            if self.eyetracker_actif:
                self.EyeTracker.stop_recording()
                self.EyeTracker.send_message("END_EXP")
                self.EyeTracker.close_and_transfer_data(self.data_dir)

            saved_path = self.save_data(
                data_list=self.global_records,
                filename_suffix=(
                    f"_{self.run_type}_run{self.run_number:02d}"
                ),
            )

            if saved_path and os.path.exists(saved_path):
                try:
                    from tasks.qc.qc_connectelec import qc_connectelec
                    qc_connectelec(saved_path)
                except ImportError:
                    self.logger.warn("Module QC non trouvé (non bloquant)")
                except Exception as qc_exc:
                    self.logger.warn(f"QC échoué (non bloquant) : {qc_exc}")

            if finished:
                self.show_instructions(
                    f"Run {self.run_number:02d} terminé.\nMerci !"
                )
                core.wait(3.0)

    # ─────────────────────────────────────────────────────────────────────
    #  INSTRUCTIONS
    # ─────────────────────────────────────────────────────────────────────

    def _show_instructions(self) -> None:
        n_vol = int(self.run_duration_s / TR_S)
        if self.run_type == "somatotopy":
            txt = (
                f"SOMATOTOPIE — Run {self.run_number:02d}\n"
                f"Durée : {self.run_duration_s:.0f} s "
                f"({n_vol} volumes)\n\n"
                "Faites attention au bout des doigts de la main droite.\n"
                "Maintenez votre regard sur la croix de fixation.\n\n"
                "En attente du scanner …"
            )
        else:
            txt = (
                f"PRÉDICTION — Run {self.run_number:02d}\n"
                f"Durée : {self.run_duration_s:.0f} s "
                f"({n_vol} volumes)\n\n"
                "Faites attention au bout des doigts de la main droite.\n"
                "Maintenez votre regard sur la croix de fixation.\n\n"
                "En attente du scanner …"
            )
        self.show_instructions(txt)

    # =====================================================================
    #  STATIC HELPERS (utilisés par le GUI)
    # =====================================================================

    @staticmethod
    def get_design_dir(
        root_dir: str, run_type: str, run_number: int = 1
    ) -> Optional[str]:
        run_type = run_type.lower()
        if run_type in ("somatotopy", "mapping"):
            key = "somatotopy"
        else:
            key = f"prediction_{run_number}"

        rel = DESIGN_PATHS.get(key)
        if rel is None:
            return None
        return os.path.join(root_dir, rel)

    @staticmethod
    def get_run_duration_s(
        run_type: str, run_number: int = 1
    ) -> float:
        """Retourne la durée hardcodée du run."""
        run_type = run_type.lower()
        if run_type in ("somatotopy", "mapping"):
            key = "somatotopy"
        else:
            key = f"prediction_{run_number}"
        return RUN_DURATIONS_S.get(key, 360.0)

    @staticmethod
    def compute_run_info(
        design_dir: str,
        run_type: str = "somatotopy",
        run_number: int = 1,
        tr_s: float = TR_S,
    ) -> Optional[Dict[str, Any]]:
        """
        Calcule les infos depuis les fichiers design + durée hardcodée.
        """
        if not design_dir or not os.path.isdir(design_dir):
            return None

        run_type = run_type.lower()

        if run_type in ("somatotopy", "mapping"):
            file_info = ConnectElec._compute_somatotopy_info(design_dir)
        else:
            file_info = ConnectElec._compute_prediction_info(design_dir)

        if file_info is None:
            return None

        dur_key = (
            "somatotopy" if run_type in ("somatotopy", "mapping")
            else f"prediction_{run_number}"
        )
        run_dur = RUN_DURATIONS_S.get(dur_key, 360.0)
        n_vol   = int(run_dur / tr_s)

        last_end = file_info.pop("last_stim_end_s", 0.0)
        padding  = run_dur - last_end

        file_info.update({
            "run_duration_s":   run_dur,
            "n_volumes":        n_vol,
            "last_stim_end_s":  round(last_end, 1),
            "padding_s":        round(padding, 1),
        })

        return file_info

    @staticmethod
    def _compute_somatotopy_info(
        design_dir: str,
    ) -> Optional[Dict[str, Any]]:
        import csv as _csv

        fpath = os.path.join(design_dir, SOMATOTOPY_TSV_NAME)
        if not os.path.exists(fpath):
            return None

        max_end: float = 0.0
        n_events: int  = 0
        fingers: set   = set()

        with open(fpath, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                return None
            for row in reader:
                try:
                    onset = float(row["onset"])
                    dur   = float(row["duration"])
                    fing  = row["finger"].strip()
                except (ValueError, KeyError):
                    continue
                end = onset + dur
                if end > max_end:
                    max_end = end
                n_events += 1
                fingers.add(fing)

        if n_events == 0:
            return None

        return {
            "last_stim_end_s": max_end,
            "n_events":        n_events,
            "fingers":         sorted(fingers),
            "n_stimulated":    n_events,
            "n_omissions":     0,
            "n_consignes":     0,
            "conditions":      ["somatotopy"],
        }

    @staticmethod
    def _compute_prediction_info(
        design_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Lit prediction.tsv (colonnes : onset duration type condition
        finger is_omission) et calcule les stats.
        """
        import csv as _csv

        fpath = os.path.join(design_dir, PREDICTION_TSV_NAME)
        if not os.path.exists(fpath):
            return None

        max_end: float    = 0.0
        n_events: int     = 0
        n_stim: int       = 0
        n_omit: int       = 0
        n_consign: int    = 0
        fingers: set      = set()
        conditions: set   = set()

        with open(fpath, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                return None

            for row in reader:
                try:
                    onset      = float(row["onset"])
                    dur        = float(row["duration"])
                    event_type = row["type"].strip().lower()
                    is_om      = int(row["is_omission"]) == 1
                    finger_raw = row["finger"].strip()
                    cond       = row["condition"].strip()
                except (ValueError, KeyError):
                    continue

                n_events += 1
                end = onset + dur
                if end > max_end:
                    max_end = end

                finger = (
                    finger_raw
                    if finger_raw.upper() != "NA"
                    else ""
                )
                if finger in FINGER_PIN_MAP:
                    fingers.add(finger)
                conditions.add(cond)

                if event_type == "consigne":
                    n_consign += 1
                elif is_om:
                    n_omit += 1
                else:
                    n_stim += 1

        if n_events == 0:
            return None

        return {
            "last_stim_end_s": max_end,
            "n_events":        n_events,
            "fingers":         sorted(fingers),
            "n_stimulated":    n_stim,
            "n_omissions":     n_omit,
            "n_consignes":     n_consign,
            "conditions":      sorted(conditions),
        }

    @staticmethod
    def test_finger(
        parport: Any,
        finger_idx: int,
        delay_s: float = 0.2,
    ) -> None:
        """
        Test manuel :
            1. send_trigger(pin)     ← sélection
            2. sleep(200 ms)
            3. send_trigger(64)      ← stimulation
            4. send_trigger(0)       ← reset
        """
        pin_map = {1: 2, 2: 4, 3: 8, 4: 16, 5: 32}
        pin = pin_map.get(finger_idx)
        if pin is None:
            return

        import time
        parport.send_trigger(pin)
        time.sleep(delay_s)
        parport.send_trigger(STIM_TRIGGER)
        parport.send_trigger(0)