# connectelec.py — version corrigée
"""
Stimulation Électrique — Somatotopie Digitale & Tâche de Prédiction
====================================================================
7T laminar fMRI — UN run par lancement.

ARCHITECTURE : PRE-COMPUTED TIMELINE
─────────────────────────────────────
1. À l'initialisation, TOUS les événements sont pré-calculés dans une
   timeline unique, triée par onset.
2. Après le trigger IRM, le moteur d'exécution parcourt la timeline :
     • attente de l'onset prévu (spin-wait haute précision pour les stims)
     • exécution de l'action (flip, port parallèle, ou simple marqueur)
     • enregistrement du temps réel
3. Zéro calcul de séquence ou de jitter pendant l'acquisition.

CORRECTION CRITIQUE : les événements visuels (win.flip) sont planifiés
UN FRAME AVANT les événements de stimulation pour éviter que le flip
bloquant ne retarde les pulses.

Fichiers produits :
    *_planned.csv       → timeline planifiée (avant exécution)
    *_incremental.csv   → écriture événement par événement pendant l'exécution
    *_<timestamp>.csv   → fichier final propre (planned + actual)
"""

from __future__ import annotations

import csv
import gc
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from psychopy import core, visual
from utils.base_task import BaseTask

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

FINGER_PIN_MAP: Dict[str, int] = {
    "D1": 2, "D2": 4, "D3": 8, "D4": 16,
}

FINGERS_4: List[str]              = ["D1", "D2", "D3", "D4"]
PREDICTABLE_ORDER: List[str]      = ["D1", "D2", "D3", "D4"]
PREDICTION_CONDITIONS: List[str]  = ["FP", "TP", "FR", "TR"]
OMISSION_FINGER: str              = "D4"

# Priorité de tri quand plusieurs événements partagent le même onset
_ACTION_PRIORITY: Dict[str, int] = {
    "visual_fixation":     0,   # flip d'abord
    "visual_instruction":  0,
    "marker":              1,   # puis marqueurs
    "stim_deliver":        2,   # puis stims (ne devrait plus arriver
    "stim_omit":           2,   # au même onset qu'un flip)
}


# ═════════════════════════════════════════════════════════════════════════════

class ConnectElec(BaseTask):
    """
    One run per instantiation.
    run_type = 'mapping' or 'prediction'
    run_number = integer assigned from the GUI.

    Toute la séquence est pré-calculée dans self.timeline avant le trigger.
    """

    def __init__(
        self,
        win: visual.Window,
        nom: str,
        session: str = "01",
        mode: str = "fmri",
        run_type: str = "mapping",
        run_number: int = 1,
        # ── mapping ──
        n_mapping_blocks: int = 20,
        mapping_off_jitter: float = 0.0,
        # ── prediction ──
        n_reps_per_condition: int = 5,
        # ── stimulation timing ──
        stims_per_finger: int = 5,
        stim_interval_ms: float = 500.0,
        # ── block timing ──
        block_on_duration: float = 10.0,
        block_off_duration: float = 10.0,
        instruction_duration: float = 5.0,
        instruction_jitter: float = 1.0,
        # ── pauses ──
        initial_baseline: float = 10.0,
        # ── misc ──
        prediction_conditions: Optional[List[str]] = None,
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
        self.run_number: int = run_number

        # ── mapping ───────────────────────────────────────────────────────
        self.n_mapping_blocks: int     = n_mapping_blocks
        self.mapping_off_jitter: float = mapping_off_jitter

        # ── prediction ────────────────────────────────────────────────────
        self.n_reps_per_condition: int = n_reps_per_condition
        self.n_blocks_per_run: int = (
            len(PREDICTION_CONDITIONS) * n_reps_per_condition
        )
        self.prediction_conditions: List[str] = (
            prediction_conditions or list(PREDICTION_CONDITIONS)
        )

        # ── stim timing ──────────────────────────────────────────────────
        self.stims_per_finger: int  = stims_per_finger
        self.stim_interval_s: float = stim_interval_ms / 1000.0
        self.n_stims_per_block: int = len(FINGERS_4) * stims_per_finger

        # ── block timing ─────────────────────────────────────────────────
        self.block_on_duration: float    = block_on_duration
        self.block_off_duration: float   = block_off_duration
        self.instruction_duration: float = instruction_duration
        self.instruction_jitter: float   = instruction_jitter
        self.initial_baseline: float     = initial_baseline

        # ── hardware ──────────────────────────────────────────────────────
        self.finger_pin_map: Dict[str, int] = dict(FINGER_PIN_MAP)

        # ── runtime state ─────────────────────────────────────────────────
        self.global_records: List[Dict[str, Any]] = []

        # ═══ PRE-COMPUTED TIMELINE ═══
        self.timeline: List[Dict[str, Any]] = []

        # ── init chain ────────────────────────────────────────────────────
        self._detect_display_scaling()
        self._measure_frame_rate()
        self._setup_key_mapping()
        self._setup_visual_stimuli()
        self._init_incremental_file(
            suffix=f"_{self.run_type}_run{self.run_number:02d}"
        )

        # ── BUILD THE ENTIRE TIMELINE ────────────────────────────────────
        self._build_full_timeline()
        self._save_planned_timeline()

        self.logger.ok(
            f"ConnectElec ready | {self.run_type} "
            f"run {self.run_number:02d} | "
            f"{self.frame_rate:.1f} Hz | "
            f"frame = {self.frame_duration_s * 1000:.1f} ms | "
            f"{len(self.timeline)} events pre-computed | "
            f"~{self.timeline[-1]['onset_s']:.1f} s"
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
        self.frame_rate = measured if measured else 60.0
        self.frame_duration_s  = 1.0 / self.frame_rate
        self.frame_tolerance_s = 0.75 / self.frame_rate
        self.logger.log(
            f"Frame rate: {self.frame_rate:.1f} Hz → "
            f"{self.frame_duration_s * 1000:.2f} ms/frame"
        )

    def _setup_key_mapping(self) -> None:
        if self.mode == "fmri":
            self.key_trigger  = "t"
            self.key_continue = "b"
        else:
            self.key_trigger  = "t"
            self.key_continue = "space"

    # =====================================================================
    #  VISUAL STIMULI
    # =====================================================================

    def _setup_visual_stimuli(self) -> None:
        self.cue_stim = visual.TextStim(
            self.win, text="", height=0.06, color="white",
            pos=(0.0, 0.25), wrapWidth=1.6, font="Arial", bold=True,
        )
        self.condition_cues: Dict[str, str] = {
            "FP": (
                "Faites attention à la stimulation de chaque doigt et "
                "prédisez\nquand l'index sera stimulé selon le rythme "
                "temporel."
            ),
            "TP": (
                "Faites attention à la stimulation de chaque doigt et "
                "prédisez\nquand l'index sera stimulé, même si aucune "
                "stimulation n'est délivrée."
            ),
            "FR": (
                "Faites attention à la stimulation de chaque doigt,\n"
                "mais n'essayez pas de prédire un motif rythmique ou "
                "temporel."
            ),
            "TR": (
                "Faites attention à la stimulation de chaque doigt,\n"
                "mais n'essayez pas de prédire un motif rythmique ou "
                "temporel."
            ),
        }
        self._setup_condition_images()

    def _setup_condition_images(self) -> None:
        from pathlib import Path
        self.condition_image_paths = {
            "FP": "image/fp.png", "TP": "image/fp.png",
            "FR": "image/fr.png", "TR": "image/fr.png",
        }
        self.condition_images_stim = {}
        for cond, path in self.condition_image_paths.items():
            if Path(path).exists():
                self.condition_images_stim[cond] = visual.ImageStim(
                    self.win, image=path,
                    size=(0.5 * 0.7, 0.7 * 0.7), pos=(0, -0.5),
                )

    # =====================================================================
    #  SEQUENCE GENERATION (appelé au build, jamais pendant le run)
    # =====================================================================

    @staticmethod
    def _pseudo_random_no_repeat(
        items: List[str], reps: int, max_attempts: int = 500
    ) -> List[str]:
        pool = [it for it in items for _ in range(reps)]
        for _ in range(max_attempts):
            seq: List[str] = []
            bag = pool[:]
            random.shuffle(bag)
            ok = True
            while bag:
                candidates = (
                    [x for x in bag if x != seq[-1]] if seq else bag[:]
                )
                if not candidates:
                    ok = False
                    break
                chosen = random.choice(candidates)
                seq.append(chosen)
                bag.remove(chosen)
            if ok and len(seq) == len(pool):
                return seq
        random.shuffle(pool)
        return pool

    def _build_predictable_seq(self) -> List[str]:
        total = len(FINGERS_4) * self.stims_per_finger
        return [
            PREDICTABLE_ORDER[i % len(PREDICTABLE_ORDER)]
            for i in range(total)
        ]

    def _build_random_seq(self) -> List[str]:
        return self._pseudo_random_no_repeat(FINGERS_4, self.stims_per_finger)

    def _build_block_stim_list(
        self, condition: str
    ) -> List[Dict[str, Any]]:
        if condition in ("mapping", "FR"):
            raw = self._build_random_seq()
            return [{"finger": f, "is_omission": False} for f in raw]
        if condition == "FP":
            raw = self._build_predictable_seq()
            return [{"finger": f, "is_omission": False} for f in raw]
        if condition == "TP":
            raw = self._build_predictable_seq()
            return [
                {"finger": f, "is_omission": f == OMISSION_FINGER}
                for f in raw
            ]
        if condition == "TR":
            raw = self._build_random_seq()
            return [
                {"finger": f, "is_omission": f == OMISSION_FINGER}
                for f in raw
            ]
        self.logger.err(f"Unknown condition '{condition}'")
        return []

    def _build_run_block_order(self) -> List[str]:
        return self._pseudo_random_no_repeat(
            self.prediction_conditions, self.n_reps_per_condition
        )

    # ═════════════════════════════════════════════════════════════════════
    #  TIMELINE CONSTRUCTION
    # ═════════════════════════════════════════════════════════════════════

    def _add_event(
        self, onset_s: float, action: str, **kwargs: Any
    ) -> None:
        """Ajoute un événement à la timeline pré-calculée."""
        event: Dict[str, Any] = {
            "onset_s":   round(onset_s, 6),
            "action":    action,
            "_priority": _ACTION_PRIORITY.get(action, 9),
        }
        event.update(kwargs)
        self.timeline.append(event)

    def _build_full_timeline(self) -> None:
        """
        Construit la timeline complète AVANT le trigger IRM.

        RÈGLE CRITIQUE : tout événement visual (win.flip) est planifié
        au minimum 2 frames avant le prochain événement de stimulation,
        pour que le flip bloquant soit terminé bien avant le pulse.
        """
        self.timeline.clear()

        if self.run_type == "mapping":
            self._build_mapping_timeline()
        else:
            self._build_prediction_timeline()

        # Tri stable : onset, puis priorité
        self.timeline.sort(key=lambda e: (e["onset_s"], e["_priority"]))

        # Numérotation séquentielle
        for i, evt in enumerate(self.timeline):
            evt["event_index"] = i

        # Validation : aucun flip ne doit être au même onset qu'un stim
        self._validate_no_flip_stim_collision()

        n_stim = sum(
            1 for e in self.timeline
            if e["action"] in ("stim_deliver", "stim_omit")
        )
        n_vis = sum(
            1 for e in self.timeline
            if e["action"].startswith("visual_")
        )
        total_dur = self.timeline[-1]["onset_s"] if self.timeline else 0

        self.logger.log(
            f"Timeline built: {len(self.timeline)} events "
            f"({n_stim} stim, {n_vis} visual, "
            f"{len(self.timeline) - n_stim - n_vis} markers) | "
            f"~{total_dur:.1f} s ({total_dur / 60:.1f} min)"
        )

    def _validate_no_flip_stim_collision(self) -> None:
        """
        Vérifie qu'aucun événement visuel (flip bloquant) n'est planifié
        au même onset qu'une stimulation. Si c'est le cas, c'est un bug
        de construction de la timeline.
        """
        visual_onsets = set()
        stim_onsets = set()

        for evt in self.timeline:
            if evt["action"].startswith("visual_"):
                visual_onsets.add(evt["onset_s"])
            elif evt["action"] in ("stim_deliver", "stim_omit"):
                stim_onsets.add(evt["onset_s"])

        collisions = visual_onsets & stim_onsets
        if collisions:
            self.logger.err(
                f"TIMELINE BUG: {len(collisions)} flip/stim collisions "
                f"detected! First at t={min(collisions):.3f} s. "
                f"Flips will delay stim pulses."
            )
        else:
            self.logger.ok(
                "Timeline validated: no flip/stim collisions."
            )

    # ── Mapping ──────────────────────────────────────────────────────────

    def _build_mapping_timeline(self) -> None:
        t = 0.0
        n = self.n_mapping_blocks

        # Marge de sécurité : 2 frames avant la première stim
        flip_lead_s = 2.0 * self.frame_duration_s

        self._add_event(t, "marker", label="run_start",
                        run_type="mapping", n_blocks=n)

        # ── Baseline : fixation ──
        self._add_event(t, "visual_fixation", label="baseline_start")
        t += self.initial_baseline

        for b in range(1, n + 1):
            stim_seq = self._build_block_stim_list("mapping")
            block_on_start = t

            # ── ON : fixation AVANT les stims ──
            # Le flip est planifié 2 frames AVANT le premier pulse
            self._add_event(
                block_on_start - flip_lead_s,
                "visual_fixation",
                label="block_on_start",
                block_index=b,
                condition="mapping",
                n_stims=len(stim_seq),
            )

            self._add_event(
                block_on_start,
                "marker",
                label="block_on_stim_start",
                block_index=b,
                condition="mapping",
            )

            # ── Train de stimulations ──
            for si, info in enumerate(stim_seq):
                stim_t = block_on_start + si * self.stim_interval_s
                self._add_event(
                    stim_t,
                    "stim_omit" if info["is_omission"] else "stim_deliver",
                    finger=info["finger"],
                    pin_code=(
                        0 if info["is_omission"]
                        else self.finger_pin_map[info["finger"]]
                    ),
                    is_omission=info["is_omission"],
                    condition="mapping",
                    block_index=b,
                    stim_index_in_block=si,
                )

            t = block_on_start + self.block_on_duration

            self._add_event(
                t, "marker", label="block_on_end",
                block_index=b, condition="mapping",
            )

            # ── OFF : fixation ──
            off_jitter = (
                random.uniform(
                    -self.mapping_off_jitter,
                    self.mapping_off_jitter,
                )
                if self.mapping_off_jitter > 0 else 0.0
            )
            off_dur = max(1.0, self.block_off_duration + off_jitter)

            # Le flip pour l'OFF n'a pas besoin de lead car
            # aucune stim ne suit immédiatement dans le même bloc
            self._add_event(
                t, "visual_fixation",
                label="block_off_start",
                block_index=b,
                off_duration_s=round(off_dur, 3),
            )
            t += off_dur

            self._add_event(
                t, "marker", label="block_off_end",
                block_index=b,
            )

        self._add_event(t, "marker", label="run_end")

    # ── Prediction ───────────────────────────────────────────────────────

    def _build_prediction_timeline(self) -> None:
        t = 0.0
        block_order = self._build_run_block_order()
        n = len(block_order)

        flip_lead_s = 2.0 * self.frame_duration_s

        self._add_event(
            t, "marker", label="run_start",
            run_type="prediction", n_blocks=n,
            block_order=str(block_order),
        )

        # ── Baseline ──
        self._add_event(t, "visual_fixation", label="baseline_start")
        t += self.initial_baseline

        for b_idx, cond in enumerate(block_order, start=1):

            # ── Instruction (durée jittée, pré-calculée) ──
            jitter = random.uniform(
                -self.instruction_jitter, self.instruction_jitter
            )
            instr_dur = max(2.0, self.instruction_duration + jitter)

            self._add_event(
                t, "visual_instruction",
                label="instruction_start",
                block_index=b_idx, condition=cond,
                instruction_text=self.condition_cues.get(cond, cond),
                instruction_duration_s=round(instr_dur, 3),
            )
            t += instr_dur

            self._add_event(
                t, "marker",
                label="instruction_end",
                block_index=b_idx, condition=cond,
            )

            # ── ON : fixation 2 frames AVANT le premier pulse ──
            block_on_start = t

            self._add_event(
                block_on_start - flip_lead_s,
                "visual_fixation",
                label="block_on_start",
                block_index=b_idx,
                condition=cond,
                n_stims=len(self._build_block_stim_list(cond)),
            )

            # Note : on re-build stim_seq ici car _build_block_stim_list
            # est pseudo-random ; on veut la séquence qui sera réellement
            # utilisée, pas une autre.
            stim_seq = self._build_block_stim_list(cond)

            self._add_event(
                block_on_start, "marker",
                label="block_on_stim_start",
                block_index=b_idx, condition=cond,
            )

            for si, info in enumerate(stim_seq):
                stim_t = block_on_start + si * self.stim_interval_s
                self._add_event(
                    stim_t,
                    "stim_omit" if info["is_omission"] else "stim_deliver",
                    finger=info["finger"],
                    pin_code=(
                        0 if info["is_omission"]
                        else self.finger_pin_map[info["finger"]]
                    ),
                    is_omission=info["is_omission"],
                    condition=cond,
                    block_index=b_idx,
                    stim_index_in_block=si,
                )

            t = block_on_start + self.block_on_duration

            self._add_event(
                t, "marker", label="block_on_end",
                block_index=b_idx, condition=cond,
            )

            # ── OFF ──
            self._add_event(
                t, "visual_fixation",
                label="block_off_start",
                block_index=b_idx, condition=cond,
                off_duration_s=round(self.block_off_duration, 3),
            )
            t += self.block_off_duration

            self._add_event(
                t, "marker", label="block_off_end",
                block_index=b_idx, condition=cond,
            )

        self._add_event(t, "marker", label="run_end")

    # ── Sauvegarde planned ───────────────────────────────────────────────

    def _save_planned_timeline(self) -> None:
        if not self.enregistrer or not self.timeline:
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self.task_name.replace(' ', '')
        fname = (
            f"{self.nom}_{safe_name}"
            f"_{self.run_type}_run{self.run_number:02d}"
            f"_{timestamp}_planned.csv"
        )
        path = os.path.join(self.data_dir, fname)
        try:
            all_keys = sorted(
                set().union(*(e.keys() for e in self.timeline))
                - {"_priority"}
            )
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, fieldnames=all_keys, extrasaction='ignore',
                )
                writer.writeheader()
                for evt in self.timeline:
                    writer.writerow(
                        {k: v for k, v in evt.items() if k != "_priority"}
                    )
            self.logger.ok(f"Planned timeline saved: {path}")
        except Exception as e:
            self.logger.err(f"Failed to save planned timeline: {e}")

    # ═════════════════════════════════════════════════════════════════════
    #  TIMELINE EXECUTION — Moteur temps-réel minimal
    # ═════════════════════════════════════════════════════════════════════

    def _wait_until(
        self, target_s: float, high_precision: bool = False
    ) -> None:
        """
        Attend jusqu'à target_s sur task_clock.

        high_precision=True (stims) :
            Sleep pour le gros, puis spin-wait pur les 2 dernières ms.
            Précision sub-milliseconde, CPU burst très court.

        high_precision=False (visuels, marqueurs) :
            core.wait standard (sleep), suffisant.
        """
        remaining = target_s - self.task_clock.getTime()
        if remaining <= 0:
            return

        if high_precision:
            # Sleep standard pour le gros de l'attente
            if remaining > 0.003:
                core.wait(remaining - 0.002, hogCPUperiod=0.0)
            # Spin-wait pour les dernières ~2 ms
            while self.task_clock.getTime() < target_s:
                pass
        else:
            core.wait(remaining, hogCPUperiod=0.0)

    def _dispatch_event(self, event: Dict[str, Any]) -> float:
        """
        Exécute l'action d'un événement. Retourne le temps réel.
        """
        action = event["action"]

        if action == "visual_fixation":
            self.fixation.draw()
            self.win.flip()
            return self.task_clock.getTime()

        if action == "visual_instruction":
            self.cue_stim.text = event.get("instruction_text", "")
            self.cue_stim.draw()
            img = self.condition_images_stim.get(event.get("condition"))
            if img:
                img.draw()
            self.fixation.draw()
            self.win.flip()
            return self.task_clock.getTime()

        if action == "stim_deliver":
            self.ParPort.send_trigger(event["pin_code"])
            return self.task_clock.getTime()

        if action == "stim_omit":
            return self.task_clock.getTime()

        if action == "marker":
            return self.task_clock.getTime()

        return self.task_clock.getTime()

    def _build_execution_record(
        self, event: Dict[str, Any], actual_time_s: float
    ) -> Dict[str, Any]:
        """Construit l'enregistrement pour UN événement exécuté."""
        error_ms = (actual_time_s - event["onset_s"]) * 1000.0
        return {
            "participant":         self.nom,
            "session":             self.session,
            "run_type":            self.run_type,
            "run_number":          self.run_number,
            "event_index":         event.get("event_index", ""),
            "action":              event["action"],
            "label":               event.get("label", ""),
            "onset_planned_s":     event["onset_s"],
            "onset_actual_s":      round(actual_time_s, 6),
            "scheduling_error_ms": round(error_ms, 3),
            "block_index":         event.get("block_index", ""),
            "condition":           event.get("condition", ""),
            "n_stims":             event.get("n_stims", ""),
            "stim_index_in_block": event.get("stim_index_in_block", ""),
            "finger":              event.get("finger", ""),
            "pin_code":            event.get("pin_code", ""),
            "is_omission":         event.get("is_omission", ""),
        }

    def _execute_timeline(self) -> None:
        """
        Boucle principale : parcourt la timeline pré-calculée.
        Seule boucle active pendant l'acquisition IRM.
        """
        n_events = len(self.timeline)
        self.logger.log(f"Executing timeline: {n_events} events …")

        # ══ GC désactivé pour tout le run ══
        gc.disable()

        try:
            for i, event in enumerate(self.timeline):

                # ── Quit check sur événements non-critiques ──
                is_stim = event["action"] in ("stim_deliver", "stim_omit")
                if not is_stim:
                    self.should_quit()
                elif event.get("stim_index_in_block", 0) == 0:
                    # Quit check à S0 de chaque bloc (avant le spin-wait)
                    self.should_quit()

                # ── Attente de l'onset ──
                self._wait_until(
                    event["onset_s"], high_precision=is_stim
                )

                # ── Exécution ──
                actual_t = self._dispatch_event(event)

                # ── Enregistrement ──
                record = self._build_execution_record(event, actual_t)
                self.global_records.append(record)
                self.save_trial_incremental(record)

                # ── Eyetracker ──
                if self.eyetracker_actif:
                    label = event.get("label", event["action"])
                    self.EyeTracker.send_message(
                        f"R{self.run_number:02d}_"
                        f"E{i:04d}_"
                        f"{label.upper()}"
                    )

                # ── Alerte timing (stim uniquement, seuil > 1 ms) ──
                if is_stim:
                    err_ms = record["scheduling_error_ms"]
                    if abs(err_ms) > 1.0:
                        self.logger.warn(
                            f"TIMING E{i} "
                            f"B{event.get('block_index', '?')} "
                            f"S{event.get('stim_index_in_block', '?')} "
                            f"({event.get('finger', '?')}): "
                            f"{err_ms:+.2f} ms"
                        )

                # ── Log de progression ──
                if event.get("label") == "block_on_end":
                    b = event.get("block_index", "?")
                    c = event.get("condition", "?")
                    self.logger.log(
                        f"  Block {b} ({c}) ON done  "
                        f"[t={actual_t:.2f} s]"
                    )

        finally:
            gc.enable()
            gc.collect()

        self.logger.ok("Timeline execution complete.")

        # ── Résumé timing ──
        stim_records = [
            r for r in self.global_records
            if r["action"] in ("stim_deliver", "stim_omit")
            and r["scheduling_error_ms"] != ""
        ]
        if stim_records:
            errors = [abs(r["scheduling_error_ms"]) for r in stim_records]
            mean_err = sum(errors) / len(errors)
            max_err  = max(errors)
            n_over_05 = sum(1 for e in errors if e > 0.5)
            n_over_1  = sum(1 for e in errors if e > 1.0)
            n_over_2  = sum(1 for e in errors if e > 2.0)
            self.logger.log(
                f"Timing summary: {len(stim_records)} stim events | "
                f"mean |err| = {mean_err:.3f} ms | "
                f"max |err| = {max_err:.3f} ms | "
                f">0.5 ms: {n_over_05} | "
                f">1 ms: {n_over_1} | >2 ms: {n_over_2}"
            )

    # ═════════════════════════════════════════════════════════════════════
    #  MAIN ENTRY
    # ═════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        finished = False

        try:
            self._show_instructions()
            self.wait_for_trigger()

            self._execute_timeline()

            finished = True
            self.logger.ok(
                f"Run {self.run_number:02d} ({self.run_type}) done."
            )

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Interruption manuelle.")

        except Exception as exc:
            self.logger.err(f"CRITICAL: {exc}")
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
                    self.logger.warn("QC module not found (non bloquant)")
                except Exception as qc_exc:
                    self.logger.warn(
                        f"QC échoué (non bloquant) : {qc_exc}"
                    )

            if finished:
                self.show_instructions(
                    f"Run {self.run_number:02d} terminé.\nMerci !"
                )
                core.wait(3.0)

    # ─────────────────────────────────────────────────────────────────────
    #  INSTRUCTIONS
    # ─────────────────────────────────────────────────────────────────────

    def _show_instructions(self) -> None:
        if self.run_type == "mapping":
            txt = (
                f"CARTOGRAPHIE — Run {self.run_number:02d}\n\n"
                "Faites attention au bout des doigts de la main droite\n"
                "pendant la phase de stimulation.\n\n"
                "Maintenez votre regard sur la croix de fixation.\n\n"
                "En attente du scanner …"
            )
        else:
            txt = (
                f"TÂCHE DE PRÉDICTION — Run {self.run_number:02d}\n\n"
                "Faites attention au bout des doigts de la main droite\n"
                "pendant la phase de stimulation.\n\n"
                "Des instructions spécifiques s'afficheront avant "
                "chaque bloc.\n"
                "Maintenez votre regard sur la croix de fixation.\n\n"
                "En attente du scanner …"
            )
        self.show_instructions(txt)