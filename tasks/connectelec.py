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
               60.000  25.0       controle

PREDICTION   → design/prediction/runN/prediction.tsv
    colonnes : onset  duration  type  condition  finger  is_omission
    type = "consigne"  → affichage image condition + texte (FP/FR/TP/TR)
    type = "stim" + is_omission=0  → stimulation électrique
    type = "stim" + is_omission=1  → omission
    type = "controle"  → sous-tâche boutons main gauche

TERMINOLOGIE :
  • CONSIGNE  = instruction prédictive (FP, FR, TP, TR) — image + texte
  • CONTROLE  = sous-tâche boutons (B1–B4) main gauche

SORTIES :
  1. CSV complet : tous les événements bruts (burst start/end, markers, boutons…)
  2. TSV events  : miroir du design d'entrée avec timings réels mesurés
     – stimulations : onset/duration effectifs
     – omissions    : finger = Dx_omitted, duration = 0.5 s
     – consignes    : onset/duration effectifs
     – contrôles    : onset/duration effectifs
"""

from __future__ import annotations

import csv
import gc
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from psychopy import core, event as psychopy_event, visual
from utils.base_task import BaseTask

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

FINGER_PIN_MAP: Dict[str, int] = {
    "D1": 2, "D2": 4, "D3": 8, "D4": 16, "D5": 32,
}

STIM_TRIGGER: int = 64
DEFAULT_BURST_INTERVAL_MS: float = 75.0
FINGER_SWITCH_LEAD_MS: float = 250.0
TR_S: float = 2.0

BACKGROUND_COLOR: list = [0, 0, 0]  # mid-grey PsychoPy (-1..+1)

RUN_DURATIONS_S: Dict[str, float] = {
    "somatotopy":   960,
    "prediction_1": 750,
    "prediction_2": 750,
    "prediction_3": 750
}

_ACTION_PRIORITY: Dict[str, int] = {
    "visual_consigne_off": -1,
    "visual_controle_off": -1,
    "visual_fixation":      0,
    "visual_consigne_on":   0,
    "visual_controle_on":   0,
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

# ── Consignes visuelles (prediction : FP, FR, TP, TR) ───────────────────
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

CONSIGNE_IMG_POS: tuple    = (0.0, 0.0)
CONSIGNE_IMG_SIZE: tuple   = (1.0/1.6, 1.0)
CONSIGNE_TXT_POS: tuple    = (0.0, -0.15)
CONSIGNE_TXT_HEIGHT: float = 0.055

# ── Sous-tâche CONTRÔLE (boutons B1–B4) ─────────────────────────────────
CONTROLE_INSTR_TEXT: str = "Appuyer sur le bouton affiché à l'écran"
CONTROLE_INSTR_DURATION_S: float = 5.0
CONTROLE_BUTTON_IMAGE_NAMES: tuple = ("B1.png", "B2.png", "B3.png", "B4.png")
CONTROLE_BUTTON_DISPLAY_S: float  = 1.0
CONTROLE_BUTTON_IMG_POS: tuple    = (0.0, 0.0)
CONTROLE_BUTTON_IMG_SIZE: tuple   = (1/1.6,1.0)
CONTROLE_INSTR_POS: tuple         = (0.0, 0.0)
CONTROLE_INSTR_HEIGHT: float      = 0.07

# ── Sous-tâche CONTRÔLE : mapping bouton-box fMRI ───────────────────────
BUTTON_RESPONSE_KEYS: tuple = (
    "b", "y", "g", "r",
    "1", "2", "3", "4",
    "num_1", "num_2", "num_3", "num_4",
)

BUTTON_CORRECT_MAP: Dict[str, tuple] = {
    "B1.png": ("b", "1", "num_1"),
    "B2.png": ("y", "2", "num_2"),
    "B3.png": ("g", "3", "num_3"),
    "B4.png": ("r", "4", "num_4"),
}

# ── Reverse mapping touche → numéro bouton ──────────────────────────────
_KEY_TO_BUTTON: Dict[str, str] = {}
for _img_name, _accepted_keys in BUTTON_CORRECT_MAP.items():
    _btn_num = _img_name.replace("B", "").replace(".png", "")
    for _k in _accepted_keys:
        _KEY_TO_BUTTON[_k] = _btn_num

# Quit keys vérifiés dans la boucle contrôle
_QUIT_KEYS: tuple = ("escape", "q")

# ── Durée fictive pour omissions dans le TSV events ─────────────────────
OMISSION_DURATION_S: float = 0.5


# ═════════════════════════════════════════════════════════════════════════════

class ConnectElec(BaseTask):

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

        # ── Fond gris permanent ──────────────────────────────────────────
        self.win.color = BACKGROUND_COLOR
        self.win.colorSpace = "rgb"
        self.win.flip()
        self.win.flip()

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

        # ── run duration ─────────────────────────────────────────────────
        dur_key = (
            "somatotopy" if self.run_type == "somatotopy"
            else f"prediction_{self.run_number}"
        )
        self.run_duration_s: float = RUN_DURATIONS_S.get(dur_key, 360.0)

        # ── hardware ─────────────────────────────────────────────────────
        self.finger_pin_map: Dict[str, int] = dict(FINGER_PIN_MAP)

        # ── runtime state ─────────────────────────────────────────────────
        self.global_records: List[Dict[str, Any]] = []
        self.timeline: List[Dict[str, Any]] = []

        # ── visual caches ─────────────────────────────────────────────────
        self._consigne_images: Dict[str, visual.ImageStim] = {}
        self._consigne_texts:  Dict[str, visual.TextStim]  = {}
        self._controle_stims:  List[Any] = []
        self._controle_instr:  Optional[visual.TextStim] = None

        # ── grey background rect (drawn behind everything) ───────────────
        self._bg_rect = visual.Rect(
            self.win,
            width=2.0, height=2.0,
            pos=(0, 0),
            fillColor=[0.0, 0.0, 0.0],
            lineColor=[0.0, 0.0, 0.0],
            units="norm",
        )

        # ── init chain ────────────────────────────────────────────────────
        self._detect_display_scaling()
        self._measure_frame_rate()
        self._setup_key_mapping()
        self._init_incremental_file(
            suffix=f"_{self.run_type}_run{self.run_number:02d}"
        )

        self._design_dir: str = self._resolve_design_dir()
        self._load_controle_images()
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

    def _draw_bg(self) -> None:
        """Dessine le fond gris (appelé avant tout autre stimulus)."""
        self._bg_rect.draw()

    # =====================================================================
    #  CONTROLE (boutons B1–B4) IMAGE LOADING
    # =====================================================================

    def _load_controle_images(self) -> None:
        images_dir = os.path.join(self.root_dir, CONSIGNE_IMAGES_DIR)
        self._controle_stims = []
        n_loaded = 0

        for img_name in CONTROLE_BUTTON_IMAGE_NAMES:
            img_path = os.path.join(images_dir, img_name)
            loaded = False

            if os.path.exists(img_path):
                try:
                    stim = visual.ImageStim(
                        self.win,
                        image=img_path,
                        pos=CONTROLE_BUTTON_IMG_POS,
                        size=CONTROLE_BUTTON_IMG_SIZE,
                        units="norm",
                    )
                    self._controle_stims.append(stim)
                    n_loaded += 1
                    loaded = True
                    self.logger.log(f"  ✓ Loaded controle image: {img_path}")
                except Exception as exc:
                    self.logger.warn(f"Image controle {img_name} non chargée : {exc}")

            if not loaded:
                self.logger.warn(f"Image controle manquante : {img_path} — fallback texte")
                fallback = visual.TextStim(
                    self.win,
                    text=img_name.replace(".png", ""),
                    pos=CONTROLE_BUTTON_IMG_POS,
                    height=0.20,
                    color="white", units="norm", bold=True,
                )
                self._controle_stims.append(fallback)

        self._controle_instr = visual.TextStim(
            self.win,
            text=CONTROLE_INSTR_TEXT,
            pos=CONTROLE_INSTR_POS,
            height=CONTROLE_INSTR_HEIGHT,
            color="white", units="norm", wrapWidth=1.6,
        )

        self.logger.log(
            f"Controle sub-task : {n_loaded}/{len(CONTROLE_BUTTON_IMAGE_NAMES)} images"
        )

    # =====================================================================
    #  CONSIGNE VISUAL LOADING (prediction : FP, FR, TP, TR)
    # =====================================================================

    def _load_consigne_visuals(self) -> None:
        images_dir = os.path.join(self.root_dir, CONSIGNE_IMAGES_DIR)

        for cond, label in CONSIGNE_TEXTS.items():
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
                    self.logger.warn(f"Image {cond} non chargée : {exc}")
            else:
                self.logger.warn(f"Image consigne manquante : {images_dir}/{cond}.*")

            self._consigne_texts[cond] = visual.TextStim(
                self.win,
                text=label,
                pos=CONSIGNE_TXT_POS,
                height=CONSIGNE_TXT_HEIGHT,
                color="white", units="norm", wrapWidth=1.6,
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
            raise FileNotFoundError(f"Configuration de run inconnue : {key}")

        full = os.path.join(self.root_dir, rel)
        if not os.path.isdir(full):
            raise FileNotFoundError(f"Dossier de design introuvable : {full}")

        self.logger.log(f"Design dir : {full}")
        return full

    def _load_stim_events(self) -> List[Dict[str, Any]]:
        if self.run_type == "somatotopy":
            events = self._load_somatotopy_tsv()
        else:
            events = self._load_prediction_tsv()

        if not events:
            raise ValueError(f"Aucun événement dans {self._design_dir}")

        events.sort(key=lambda e: e["onset_s"])

        last_end = max(e["onset_s"] + e["duration_s"] for e in events)
        if last_end > self.run_duration_s:
            self.logger.warn(
                f"Dernier événement se termine à {last_end:.1f} s "
                f"mais le run dure {self.run_duration_s:.1f} s"
            )

        return events

    # ── Somatotopie ──────────────────────────────────────────────────────

    def _load_somatotopy_tsv(self) -> List[Dict[str, Any]]:
        fpath = os.path.join(self._design_dir, SOMATOTOPY_TSV_NAME)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Fichier {SOMATOTOPY_TSV_NAME} introuvable : {fpath}")

        events: List[Dict[str, Any]] = []
        fingers_seen: set = set()
        n_controles: int = 0

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
                    self.logger.warn(f"{SOMATOTOPY_TSV_NAME}:{row_num} — parse error : {exc}")
                    continue

                if finger.lower() == "controle":
                    events.append({
                        "finger": "", "onset_s": onset_s, "duration_s": duration_s,
                        "condition": "controle", "event_type": "controle",
                        "is_omission": False, "is_consigne": False,
                        "is_controle": True, "is_stim": False,
                    })
                    n_controles += 1
                    continue

                if finger not in FINGER_PIN_MAP:
                    self.logger.warn(
                        f"{SOMATOTOPY_TSV_NAME}:{row_num} — doigt inconnu '{finger}', ignoré"
                    )
                    continue

                fingers_seen.add(finger)
                events.append({
                    "finger": finger, "onset_s": onset_s, "duration_s": duration_s,
                    "condition": "somatotopy", "event_type": "stim",
                    "is_omission": False, "is_consigne": False,
                    "is_controle": False, "is_stim": True,
                })

        self.logger.log(
            f"Somatotopy : {len(events)} événements "
            f"({', '.join(sorted(fingers_seen))})"
            + (f" | {n_controles} contrôles" if n_controles else "")
        )
        return events

    # ── Prediction ───────────────────────────────────────────────────────

    def _load_prediction_tsv(self) -> List[Dict[str, Any]]:
        fpath = os.path.join(self._design_dir, PREDICTION_TSV_NAME)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Fichier {PREDICTION_TSV_NAME} introuvable : {fpath}")

        events: List[Dict[str, Any]] = []
        conditions_seen: set = set()
        fingers_seen: set = set()

        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            if reader.fieldnames is None:
                raise ValueError(f"Fichier vide : {fpath}")

            required = {"onset", "duration", "type", "condition", "finger", "is_omission"}
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
                    self.logger.warn(f"{PREDICTION_TSV_NAME}:{row_num} — parse error : {exc}")
                    continue

                is_consigne = event_type == "consigne"
                is_controle = event_type == "controle"
                is_stim     = event_type == "stim" and not is_omission

                finger = finger_raw if finger_raw.upper() != "NA" else ""

                if not is_consigne and not is_controle:
                    if finger not in FINGER_PIN_MAP:
                        self.logger.warn(
                            f"{PREDICTION_TSV_NAME}:{row_num} — doigt inconnu '{finger_raw}', ignoré"
                        )
                        continue

                conditions_seen.add(condition)
                if finger in FINGER_PIN_MAP:
                    fingers_seen.add(finger)

                events.append({
                    "finger": finger, "onset_s": onset_s, "duration_s": duration_s,
                    "condition": condition, "event_type": event_type,
                    "is_omission": is_omission, "is_consigne": is_consigne,
                    "is_controle": is_controle, "is_stim": is_stim,
                })

        n_stim    = sum(1 for e in events if e["is_stim"])
        n_omit    = sum(1 for e in events if e["is_omission"])
        n_consign = sum(1 for e in events if e["is_consigne"])
        n_ctrl    = sum(1 for e in events if e["is_controle"])
        self.logger.log(
            f"Prediction : {len(events)} événements "
            f"({n_stim} stim, {n_omit} omissions, "
            f"{n_consign} consignes, {n_ctrl} contrôles) | "
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
        self.timeline.clear()

        self._add_event(0.0, "marker", label="run_start",
                        run_type=self.run_type, run_number=self.run_number)
        self._add_event(0.0, "visual_fixation", label="fixation_start")

        for ei, stim in enumerate(self._stim_events):
            onset       = stim["onset_s"]
            duration    = stim["duration_s"]
            finger      = stim["finger"]
            condition   = stim.get("condition", "")
            is_consigne = stim.get("is_consigne", False)
            is_controle = stim.get("is_controle", False)
            is_omit     = stim.get("is_omission", False)
            is_stim     = stim.get("is_stim", False)

            if is_consigne:
                self._add_event(
                    onset, "visual_consigne_on",
                    label="consigne_on", condition=condition,
                    stim_event_idx=ei, duration_s=duration,
                )
                self._add_event(
                    onset + duration, "visual_consigne_off",
                    label="consigne_off", condition=condition,
                    stim_event_idx=ei,
                )

            elif is_controle:
                self._add_event(
                    onset, "visual_controle_on",
                    label="controle_on", condition=condition,
                    stim_event_idx=ei, duration_s=duration,
                )
                self._add_event(
                    onset + duration, "visual_controle_off",
                    label="controle_off", condition=condition,
                    stim_event_idx=ei,
                )

            elif is_stim and duration > 0:
                pin = self.finger_pin_map[finger]
                sel_t = max(0.0, onset - self.finger_switch_lead_s)
                self._add_event(
                    sel_t, "finger_select",
                    label="finger_select", finger=finger, pin_code=pin,
                    condition=condition, stim_event_idx=ei,
                )
                n_bursts = max(1, int(duration / self.burst_interval_s + 1e-9) + 1)
                for bi in range(n_bursts):
                    burst_t = onset + bi * self.burst_interval_s
                    if bi > 0 and burst_t > onset + duration + 1e-6:
                        break
                    self._add_event(
                        burst_t, "stim_burst",
                        label="stim_burst", finger=finger,
                        pin_code=STIM_TRIGGER, condition=condition,
                        stim_event_idx=ei, burst_index=bi,
                        n_bursts=n_bursts, is_omission=False,
                    )

            elif is_omit:
                self._add_event(
                    onset, "stim_omit",
                    label="stim_omit", finger=finger, pin_code=0,
                    condition=condition, stim_event_idx=ei,
                    is_omission=True, duration_s=0.0,
                )
            else:
                self.logger.warn(f"Événement {ei} ignoré")

        self._add_event(self.run_duration_s, "visual_fixation", label="final_fixation")
        self._add_event(self.run_duration_s, "marker", label="run_end",
                        run_duration_s=self.run_duration_s)

        self.timeline.sort(key=lambda e: (e["onset_s"], e["_priority"]))
        for i, evt in enumerate(self.timeline):
            evt["event_index"] = i

        n_sel    = sum(1 for e in self.timeline if e["action"] == "finger_select")
        n_burst  = sum(1 for e in self.timeline if e["action"] == "stim_burst")
        n_omit   = sum(1 for e in self.timeline if e["action"] == "stim_omit")
        n_con_on = sum(1 for e in self.timeline if e["action"] == "visual_consigne_on")
        n_ctr_on = sum(1 for e in self.timeline if e["action"] == "visual_controle_on")

        last_stim = max(
            (e["onset_s"] + e.get("duration_s", 0) for e in self._stim_events),
            default=0.0,
        )
        padding = self.run_duration_s - last_stim

        self.logger.log(
            f"Timeline : {len(self.timeline)} events "
            f"({n_sel} select, {n_burst} burst, {n_omit} omit, "
            f"{n_con_on} consignes, {n_ctr_on} contrôles) | "
            f"last event ends at {last_stim:.1f} s | "
            f"padding = {padding:.1f} s | "
            f"run_end = {self.run_duration_s:.1f} s"
        )

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
                set().union(*(e.keys() for e in self.timeline)) - {"_priority"}
            )
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                for evt in self.timeline:
                    writer.writerow({k: v for k, v in evt.items() if k != "_priority"})
            self.logger.ok(f"Planned timeline : {path}")
        except Exception as exc:
            self.logger.err(f"Sauvegarde planned échouée : {exc}")

    # =====================================================================
    #  BIDS-LIKE EVENTS TSV (timings réels)
    # =====================================================================

    def _save_bids_events(self) -> Optional[str]:
        """
        Sauvegarde un TSV simplifié miroir du design d'entrée,
        mais avec les timings réellement mesurés.

        Somatotopie → colonnes : onset  duration  finger
        Prédiction  → colonnes : onset  duration  type  condition  finger  is_omission

        • Stimulations : onset/duration réels (burst_start → burst_end)
        • Omissions    : finger = « Dx_omitted », duration = OMISSION_DURATION_S
        • Consignes    : onset/duration réels (on → off)
        • Contrôles    : onset/duration réels (on → off)
        """
        if not self.enregistrer or not self.global_records:
            return None

        # ── Index des records par stim_event_idx ──────────────────────
        by_idx: Dict[int, List[Dict[str, Any]]] = {}
        for rec in self.global_records:
            idx = rec.get("stim_event_idx")
            if idx == "" or idx is None:
                continue
            by_idx.setdefault(idx, []).append(rec)

        rows: List[Dict[str, Any]] = []

        for ei, stim in enumerate(self._stim_events):
            recs    = by_idx.get(ei, [])
            onset   = stim["onset_s"]       # fallback : planifié
            dur     = stim["duration_s"]     # fallback : planifié
            finger  = stim.get("finger", "")
            cond    = stim.get("condition", "")
            etype   = stim.get("event_type", "")
            is_omit = stim.get("is_omission", False)

            # ── Stimulation effective ────────────────────────────────
            if stim.get("is_stim") and not is_omit:
                starts = [r for r in recs if r.get("label") == "burst_start"]
                ends   = [r for r in recs if r.get("label") == "burst_end"]
                if starts:
                    onset = starts[0]["onset_actual_s"]
                if starts and ends:
                    measured = ends[0]["onset_actual_s"] - starts[0]["onset_actual_s"]
                    dur = measured if measured > 0 else stim["duration_s"]

            # ── Omission ─────────────────────────────────────────────
            elif is_omit:
                omits = [r for r in recs if r["action"] == "stim_omit"]
                if omits:
                    onset = omits[0]["onset_actual_s"]
                dur    = OMISSION_DURATION_S
                finger = f"{finger}_omitted" if finger else "omitted"

            # ── Consigne ─────────────────────────────────────────────
            elif stim.get("is_consigne"):
                ons  = [r for r in recs if r["action"] == "visual_consigne_on"]
                offs = [r for r in recs if r["action"] == "visual_consigne_off"]
                if ons:
                    onset = ons[0]["onset_actual_s"]
                if ons and offs:
                    dur = offs[0]["onset_actual_s"] - ons[0]["onset_actual_s"]

            # ── Contrôle ─────────────────────────────────────────────
            elif stim.get("is_controle"):
                ons  = [r for r in recs if r["action"] == "visual_controle_on"]
                offs = [r for r in recs if r["action"] == "visual_controle_off"]
                if ons:
                    onset = ons[0]["onset_actual_s"]
                if ons and offs:
                    dur = offs[0]["onset_actual_s"] - ons[0]["onset_actual_s"]

            # ── Construire la ligne ──────────────────────────────────
            if self.run_type == "somatotopy":
                rows.append({
                    "onset":    round(onset, 3),
                    "duration": round(dur, 3),
                    "finger":   "controle" if stim.get("is_controle") else finger,
                })
            else:  # prediction
                rows.append({
                    "onset":       round(onset, 3),
                    "duration":    round(dur, 3),
                    "type":        etype,
                    "condition":   cond,
                    "finger":      finger if finger else "NA",
                    "is_omission": int(is_omit),
                })

        if not rows:
            self.logger.warn("Aucun événement à écrire dans le TSV events.")
            return None

        # ── Écriture TSV ─────────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            f"{self.nom}_{self.task_name}"
            f"_{self.run_type}_run{self.run_number:02d}"
            f"_{ts}_events.tsv"
        )
        path = os.path.join(self.data_dir, fname)

        fieldnames = list(rows[0].keys())
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames,
                    delimiter="\t", extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
            self.logger.ok(
                f"Events TSV : {path} ({len(rows)} événements)"
            )
            return path
        except Exception as exc:
            self.logger.err(f"Events TSV — échec : {exc}")
            return None

    # =====================================================================
    #  EXECUTION ENGINE
    # =====================================================================

    def _wait_until(self, target_s: float, high_precision: bool = False) -> None:
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
    #  CONSIGNE DRAWING (prediction : FP, FR, TP, TR)
    # ─────────────────────────────────────────────────────────────────────

    def _draw_consigne(self, condition: str) -> None:
        self._draw_bg()
        if condition in self._consigne_images:
            self._consigne_images[condition].draw()
        self.win.flip()

    # ─────────────────────────────────────────────────────────────────────
    #  FIXATION DRAWING (avec fond gris)
    # ─────────────────────────────────────────────────────────────────────

    def _draw_fixation(self) -> None:
        self._draw_bg()
        self.fixation.draw()
        self.win.flip()

    # ─────────────────────────────────────────────────────────────────────
    #  SOUS-TÂCHE CONTRÔLE (boutons B1–B4)
    # ─────────────────────────────────────────────────────────────────────

    def _process_keys_in_controle(
        self,
        condition: str,
        stim_event_idx: Any,
        current_image: str,
        image_onset: float,
    ) -> None:
        """
        Lecture directe via self.kb (psychtoolbox backend).
        Aucun keyList filter → on récupère TOUT, puis on trie nous-mêmes.
        """
        raw = self.kb.getKeys(waitRelease=False, clear=True)

        if not raw:
            return

        # Debug : toujours loguer ce qui arrive
        self.logger.log(
            f"    ⌨ RAW keys: {[(k.name, round(k.rt, 3)) for k in raw]}"
        )

        for k in raw:
            # ── Quit ─────────────────────────────────────────────
            if k.name in ('escape', 'q', 'a'):
                self.should_quit(force_quit=True)
                return

            # ── Match bouton ─────────────────────────────────────
            button_num = _KEY_TO_BUTTON.get(k.name)
            if button_num is None:
                self.logger.log(f"    ⌨ ignored key: '{k.name}'")
                continue

            # ── Correctness ──────────────────────────────────────
            expected_num = current_image.replace("B", "").replace(".png", "")
            is_correct = int(button_num == expected_num) if current_image != "instruction" else ""
            rt = round(k.rt - image_onset, 6)

            rec: Dict[str, Any] = {
                "participant":         self.nom,
                "session":             self.session,
                "run_type":            self.run_type,
                "run_number":          self.run_number,
                "event_index":         "",
                "action":              "button_response",
                "label":               "controle_button_press",
                "onset_planned_s":     "",
                "onset_actual_s":      round(k.rt, 6),
                "scheduling_error_ms": "",
                "condition":           condition,
                "finger":              "",
                "pin_code":            "",
                "is_omission":         "",
                "stim_event_idx":      stim_event_idx,
                "burst_index":         "",
                "n_bursts":            "",
                "button_pressed":      button_num,
                "button_image":        current_image,
                "button_correct":      is_correct,
                "button_rt_s":         rt,
            }
            self.global_records.append(rec)
            self.save_trial_incremental(rec)

            self.logger.log(
                f"    ✓ Button {button_num} "
                f"(key='{k.name}', image={current_image}, "
                f"correct={is_correct}, RT={rt:.3f} s)"
            )

    def _run_controle_subtask(self, event: Dict[str, Any]) -> float:
        onset_planned: float = event["onset_s"]
        duration_s: float    = event.get("duration_s", 0.0)
        condition: str       = event.get("condition", "")
        stim_idx: Any        = event.get("stim_event_idx", "")

        subtask_start: float = self.task_clock.getTime()
        subtask_end: float   = onset_planned + duration_s

        self.logger.log(
            f"  ▶ Contrôle sub-task START "
            f"[t={subtask_start:.2f} s, dur={duration_s:.1f} s]"
        )

        n_records_before = len(self.global_records)
        self.flush_keyboard()

        # ── Phase 1 : instruction (5 s max) ─────────────────────────
        instr_end = min(
            subtask_start + CONTROLE_INSTR_DURATION_S,
            subtask_end,
        )

        while self.task_clock.getTime() < instr_end:
            self._draw_bg()
            self._controle_instr.draw()
            self.win.flip()
            self._process_keys_in_controle(
                condition=condition, stim_event_idx=stim_idx,
                current_image="instruction", image_onset=subtask_start,
            )

        self.logger.log(
            f"    Instruction done [t={self.task_clock.getTime():.2f} s], "
            f"starting button cycle …"
        )

        # ── Phase 2 : cycle B1 → B4 ─────────────────────────────────
        img_idx = 0
        n_images = len(CONTROLE_BUTTON_IMAGE_NAMES)

        while self.task_clock.getTime() < subtask_end:
            img_name: str    = CONTROLE_BUTTON_IMAGE_NAMES[img_idx]
            img_onset: float = self.task_clock.getTime()
            img_end: float   = min(
                img_onset + CONTROLE_BUTTON_DISPLAY_S,
                subtask_end,
            )

            while self.task_clock.getTime() < img_end:
                self._draw_bg()
                self._controle_stims[img_idx].draw()
                self.win.flip()
                self._process_keys_in_controle(
                    condition=condition, stim_event_idx=stim_idx,
                    current_image=img_name, image_onset=img_onset,
                )

            img_idx = (img_idx + 1) % n_images

        n_responses = sum(
            1 for r in self.global_records[n_records_before:]
            if r.get("action") == "button_response"
        )
        self.logger.log(
            f"  ■ Contrôle sub-task END "
            f"[t={self.task_clock.getTime():.2f} s, "
            f"{n_responses} réponses]"
        )

        return subtask_start

    # ─────────────────────────────────────────────────────────────────────
    #  DISPATCH
    # ─────────────────────────────────────────────────────────────────────

    def _dispatch_event(self, event: Dict[str, Any]) -> float:
        action = event["action"]

        if action == "visual_fixation":
            self._draw_fixation()
            return self.task_clock.getTime()

        if action == "visual_consigne_on":
            t = self.task_clock.getTime()
            self._draw_consigne(event.get("condition", ""))
            return t

        if action == "visual_consigne_off":
            self._draw_fixation()
            return self.task_clock.getTime()

        if action == "visual_controle_on":
            return self._run_controle_subtask(event)

        if action == "visual_controle_off":
            self._draw_fixation()
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

    def _build_record(self, event: Dict[str, Any], actual_t: float) -> Dict[str, Any]:
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
            "button_pressed":      "",
            "button_image":        "",
            "button_correct":      "",
            "button_rt_s":         "",
        }

    def _execute_timeline(self) -> None:
        n_events = len(self.timeline)
        self.logger.log(f"Exécution : {n_events} événements …")

        gc.disable()

        try:
            for i, event in enumerate(self.timeline):

                is_burst = event["action"] == "stim_burst"

                # Quit check : seulement au début de chaque train
                if not is_burst:
                    self.should_quit()
                elif event.get("burst_index", 0) == 0:
                    self.should_quit()

                self._wait_until(
                    event["onset_s"],
                    high_precision=(is_burst or event["action"] == "finger_select"),
                )

                actual_t = self._dispatch_event(event)

                # ── Bursts : ne loguer que début et fin du train ─────
                if is_burst:
                    bi = event.get("burst_index", 0)
                    nb = event.get("n_bursts", 1)

                    if bi == 0 or bi == nb - 1:
                        rec = self._build_record(event, actual_t)
                        rec["label"] = "burst_start" if bi == 0 else "burst_end"
                        self.global_records.append(rec)
                        self.save_trial_incremental(rec)

                        if self.eyetracker_actif:
                            tag = "START" if bi == 0 else "END"
                            self.EyeTracker.send_message(
                                f"R{self.run_number:02d}_E{i:04d}_BURST_{tag}"
                            )

                        err_ms = (actual_t - event["onset_s"]) * 1000.0
                        if abs(err_ms) > 1.0:
                            self.logger.warn(
                                f"TIMING E{i} {event.get('finger', '?')} "
                                f"B{bi}: {err_ms:+.2f} ms"
                            )
                    # bursts intermédiaires : trigger envoyé, rien d'autre
                    continue

                # ── Tous les autres événements ───────────────────────
                rec = self._build_record(event, actual_t)
                self.global_records.append(rec)
                self.save_trial_incremental(rec)

                if self.eyetracker_actif:
                    lbl = event.get("label", event["action"])
                    self.EyeTracker.send_message(
                        f"R{self.run_number:02d}_E{i:04d}_{lbl.upper()}"
                    )

                if event["action"] == "stim_omit":
                    self.logger.log(
                        f"  Omission {event.get('finger', '?')} "
                        f"({event.get('condition', '?')}) [t={actual_t:.2f} s]"
                    )

                if event["action"] == "visual_consigne_on":
                    self.logger.log(
                        f"  Consigne ON : {event.get('condition', '?')} "
                        f"[t={actual_t:.2f} s, dur={event.get('duration_s', '?')} s]"
                    )

                if event["action"] == "visual_consigne_off":
                    self.logger.log(
                        f"  Consigne OFF : {event.get('condition', '?')} [t={actual_t:.2f} s]"
                    )

                if event["action"] == "visual_controle_on":
                    self.logger.log(
                        f"  Contrôle ON [t={actual_t:.2f} s, "
                        f"dur={event.get('duration_s', '?')} s]"
                    )

                if event["action"] == "visual_controle_off":
                    self.logger.log(f"  Contrôle OFF [t={actual_t:.2f} s]")

                if event.get("label") == "run_end":
                    self.logger.log(
                        f"  run_end [t={actual_t:.2f} s / planned {event['onset_s']:.1f} s]"
                    )

        finally:
            gc.enable()
            gc.collect()

        self.logger.ok("Exécution de la timeline terminée.")

        # ── Résumé timing (burst_start + burst_end seulement) ────────
        burst_recs = [
            r for r in self.global_records
            if r["action"] == "stim_burst" and r["scheduling_error_ms"] != ""
        ]
        if burst_recs:
            errors = [abs(r["scheduling_error_ms"]) for r in burst_recs]
            mean_e = sum(errors) / len(errors)
            max_e  = max(errors)
            self.logger.log(
                f"Timing résumé : {len(burst_recs)} burst start/end | "
                f"mean |err| = {mean_e:.3f} ms | max |err| = {max_e:.3f} ms"
            )

        # ── Résumé boutons ───────────────────────────────────────────
        btn_records = [r for r in self.global_records if r.get("action") == "button_response"]
        if btn_records:
            n_correct = sum(1 for r in btn_records if r.get("button_correct") == 1)
            rts = [
                r["button_rt_s"] for r in btn_records
                if isinstance(r.get("button_rt_s"), (int, float))
                and r.get("button_image", "") != "instruction"
            ]
            mean_rt = (sum(rts) / len(rts)) if rts else float("nan")
            self.logger.log(
                f"Boutons résumé : {len(btn_records)} réponses | "
                f"{n_correct} correctes | mean RT = {mean_rt:.3f} s"
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
            self.logger.ok(f"Run {self.run_number:02d} ({self.run_type}) terminé.")

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

            # ── SORTIE 1 : CSV complet (tous les événements bruts) ───
            saved_path = self.save_data(
                data_list=self.global_records,
                filename_suffix=f"_{self.run_type}_run{self.run_number:02d}",
            )

            # ── SORTIE 2 : TSV événements effectifs (BIDS-like) ─────
            self._save_bids_events()

            # ── QC optionnel ─────────────────────────────────────────
            if saved_path and os.path.exists(saved_path):
                try:
                    from tasks.qc.qc_connectelec import qc_connectelec
                    qc_connectelec(saved_path)
                except ImportError:
                    self.logger.warn("Module QC non trouvé (non bloquant)")
                except Exception as qc_exc:
                    self.logger.warn(f"QC échoué (non bloquant) : {qc_exc}")

            if finished:
                self.show_instructions(f"Run {self.run_number:02d} terminé.\nMerci !")
                core.wait(3.0)

    # ─────────────────────────────────────────────────────────────────────
    #  INSTRUCTIONS
    # ─────────────────────────────────────────────────────────────────────

    def _show_instructions(self) -> None:
        n_vol = int(self.run_duration_s / TR_S)
        if self.run_type == "somatotopy":
            txt = (
                f"SOMATOTOPIE — Run {self.run_number:02d}\n"
                f"Durée : {self.run_duration_s:.0f} s ({n_vol} volumes)\n\n"
                "Faites attention au bout des doigts de la main droite.\n"
                "Maintenez votre regard sur la croix de fixation.\n\n"
                "En attente du scanner …"
            )
        else:
            txt = (
                f"PRÉDICTION — Run {self.run_number:02d}\n"
                "Faites attention au bout des doigts de la main droite\n"
                "dans certaines conditions essayer de predire la stimulation\n"
                "de votre pouce selon le rythme temporel\n"
                "Maintenez votre regard sur la croix de fixation.\n\n"
            )
        self.show_instructions(txt)

    # =====================================================================
    #  STATIC HELPERS
    # =====================================================================

    @staticmethod
    def get_design_dir(root_dir: str, run_type: str, run_number: int = 1) -> Optional[str]:
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
    def get_run_duration_s(run_type: str, run_number: int = 1) -> float:
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
    def _compute_somatotopy_info(design_dir: str) -> Optional[Dict[str, Any]]:
        import csv as _csv

        fpath = os.path.join(design_dir, SOMATOTOPY_TSV_NAME)
        if not os.path.exists(fpath):
            return None

        max_end: float   = 0.0
        n_events: int    = 0
        n_stim: int      = 0
        n_controles: int = 0
        fingers: set     = set()

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
                if fing.lower() == "controle":
                    n_controles += 1
                else:
                    n_stim += 1
                    fingers.add(fing)

        if n_events == 0:
            return None

        return {
            "last_stim_end_s": max_end,
            "n_events":        n_events,
            "fingers":         sorted(fingers),
            "n_stimulated":    n_stim,
            "n_omissions":     0,
            "n_consignes":     0,
            "n_controles":     n_controles,
            "conditions":      ["somatotopy"],
        }

    @staticmethod
    def _compute_prediction_info(design_dir: str) -> Optional[Dict[str, Any]]:
        import csv as _csv

        fpath = os.path.join(design_dir, PREDICTION_TSV_NAME)
        if not os.path.exists(fpath):
            return None

        max_end: float    = 0.0
        n_events: int     = 0
        n_stim: int       = 0
        n_omit: int       = 0
        n_consign: int    = 0
        n_ctrl: int       = 0
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
                finger = finger_raw if finger_raw.upper() != "NA" else ""
                if finger in FINGER_PIN_MAP:
                    fingers.add(finger)
                conditions.add(cond)
                if event_type == "consigne":
                    n_consign += 1
                elif event_type == "controle":
                    n_ctrl += 1
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
            "n_controles":     n_ctrl,
            "conditions":      sorted(conditions),
        }

    @staticmethod
    def test_finger(parport: Any, finger_idx: int, delay_s: float = 0.2) -> None:
        pin_map = {1: 2, 2: 4, 3: 8, 4: 16, 5: 32}
        pin = pin_map.get(finger_idx)
        if pin is None:
            return
        import time
        parport.send_trigger(pin)
        time.sleep(delay_s)
        parport.send_trigger(STIM_TRIGGER)
        parport.send_trigger(0)