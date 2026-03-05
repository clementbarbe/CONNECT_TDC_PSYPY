# connectelec.py
"""
Stimulation Électrique — Somatotopie Digitale & Tâche de Prédiction
====================================================================
7T laminar fMRI — UN run par lancement.

Mapping run :
    N blocs ON/OFF, pseudo-random D1–D4

Prediction run :
    20 blocs (5 reps × 4 conditions, pseudo-randomisés)
    Bloc = Instruction (5 s ± 1 s) → ON (10 s) → OFF (10 s)

Hardware — front montant sur port parallèle :
    D1 → 0x02   D2 → 0x04   D3 → 0x08   D4 → 0x10
    Le stimulateur gère le pulse en interne.
"""

from __future__ import annotations

import gc
import random
from typing import Any, Dict, List, Optional

from psychopy import core, visual
from utils.base_task import BaseTask
import os

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

FINGER_PIN_MAP: Dict[str, int] = {
    "D1": 0x02,
    "D2": 0x04,
    "D3": 0x08,
    "D4": 0x10,
}

FINGERS_4: List[str]          = ["D1", "D2", "D3", "D4"]
PREDICTABLE_ORDER: List[str]  = ["D1", "D2", "D3", "D4"]
PREDICTION_CONDITIONS: List[str] = ["FP", "TP", "FR", "TR"]
OMISSION_FINGER: str          = "D4"

# ═════════════════════════════════════════════════════════════════════════════

class ConnectElec(BaseTask):
    """
    One run per instantiation.
    run_type = 'mapping' or 'prediction'
    run_number = integer assigned from the GUI.
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
        self.current_trial_idx: int = 0
        self.current_phase: str     = self.run_type

        # ── init chain ────────────────────────────────────────────────────
        self._detect_display_scaling()
        self._measure_frame_rate()
        self._define_ttl_codes()
        self._setup_key_mapping()
        self._setup_visual_stimuli()
        self._init_incremental_file(
            suffix=f"_{self.run_type}_run{self.run_number:02d}"
        )
        self._validate_timing()

        self.logger.ok(
            f"ConnectElec ready | {self.run_type} "
            f"run {self.run_number:02d} | "
            f"{self.frame_rate:.1f} Hz | "
            f"{self.n_stims_per_block} stim/blk @ "
            f"{self.stim_interval_s * 1000:.0f} ms ISI"
        )

    # ─────────────────────────────────────────────────────────────────────
    # INIT HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _detect_display_scaling(self) -> None:
        self.pixel_scale = 2.0 if self.win.size[1] > 1200 else 1.0

    def _measure_frame_rate(self) -> None:
        measured = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, threshold=1
        )
        self.frame_rate = measured if measured else 60.0
        self.frame_duration_s  = 1.0 / self.frame_rate
        self.frame_tolerance_s = 0.75 / self.frame_rate

    def _validate_timing(self) -> None:
        expected_on = self.n_stims_per_block * self.stim_interval_s
        if abs(expected_on - self.block_on_duration) > 0.1:
            self.logger.warn(
                f"TIMING: {self.n_stims_per_block} stim × "
                f"{self.stim_interval_s * 1000:.0f} ms = "
                f"{expected_on:.1f} s ≠ "
                f"block_on {self.block_on_duration:.1f} s"
            )

        if self.run_type == "mapping":
            n = self.n_mapping_blocks
            dur = self.initial_baseline + n * (
                self.block_on_duration + self.block_off_duration
            )
        else:
            n = self.n_blocks_per_run
            dur = self.initial_baseline + n * (
                self.instruction_duration
                + self.block_on_duration
                + self.block_off_duration
            )

        self.logger.log(
            f"Run {self.run_number:02d} ({self.run_type}) | "
            f"{n} blocs | ~{dur:.0f} s ({dur / 60:.1f} min)"
        )

    # ─────────────────────────────────────────────────────────────────────
    # TTL CODES
    # ─────────────────────────────────────────────────────────────────────

    def _define_ttl_codes(self) -> None:
        self.codes: Dict[str, int] = {
            "start_exp":         255,
            "end_exp":           254,
            "rest_start":        200,
            "rest_end":          201,
            "run_start":         230,
            "run_end":           231,
            "instruction_start": 120,
            "instruction_end":   121,
            "block_on_start":    100,
            "block_on_end":      101,
            "block_off_start":   110,
            "block_off_end":     111,
            "stim_D1":            11,
            "stim_D2":            12,
            "stim_D3":            13,
            "stim_D4":            14,
            "stim_omission":      15,
            "condition_FP":       50,
            "condition_TP":       51,
            "condition_FR":       52,
            "condition_TR":       53,
            "condition_mapping":  54,
        }

    # ─────────────────────────────────────────────────────────────────────
    # KEYS
    # ─────────────────────────────────────────────────────────────────────

    def _setup_key_mapping(self) -> None:
        if self.mode == "fmri":
            self.key_trigger  = "t"
            self.key_continue = "b"
        else:
            self.key_trigger  = "t"
            self.key_continue = "space"

    # ─────────────────────────────────────────────────────────────────────
    # VISUAL
    # ─────────────────────────────────────────────────────────────────────

    def _setup_visual_stimuli(self) -> None:
        self.cue_stim = visual.TextStim(
            self.win,
            text="",
            height=0.06,  # Taille légèrement réduite pour les phrases longues
            color="white",
            pos=(0.0, 0.25),
            wrapWidth=1.6,  # Plus large pour éviter de couper les mots
            font="Arial",
            bold=True,
        )
        
        self.condition_cues: Dict[str, str] = {
            "FP": (
                "Faites attention à la stimulation de chaque doigt et prédisez\n"
                "quand l'index sera stimulé selon le rythme temporel."
            ),
            "TP": (
                "Faites attention à la stimulation de chaque doigt et prédisez\n"
                "quand l'index sera stimulé, même si aucune stimulation n'est délivrée."
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

        self._setup_condition_images()

    def _setup_condition_images(self) -> None:
        from pathlib import Path

        self.condition_image_paths = {
            "FP": "image/fp.png",
            "TP": "image/fp.png",
            "FR": "image/fr.png",
            "TR": "image/fr.png",
        }

        self.condition_images_stim = {}

        for cond, path in self.condition_image_paths.items():
            if Path(path).exists():
                self.condition_images_stim[cond] = visual.ImageStim(
                    self.win,
                    image=path,
                    size=(0.5*0.7, 0.7*0.7),
                    pos=(0, -0.5)
                )

    # ═════════════════════════════════════════════════════════════════════
    # EVENT LOGGING
    # ═════════════════════════════════════════════════════════════════════

    def log_trial_event(self, event_type: str, **kwargs: Any) -> None:
        t = self.task_clock.getTime()
        if self.eyetracker_actif:
            self.EyeTracker.send_message(
                f"R{self.run_number:02d}_"
                f"{self.current_phase.upper()}_"
                f"S{self.current_trial_idx:03d}_"
                f"{event_type.upper()}"
            )
        entry: Dict[str, Any] = {
            "participant": self.nom,
            "session":     self.session,
            "run_type":    self.run_type,
            "run_number":  self.run_number,
            "phase":       self.current_phase,
            "stim_index":  self.current_trial_idx,
            "time_s":      round(t, 6),
            "event_type":  event_type,
        }
        entry.update(kwargs)
        self.global_records.append(entry)

    # ═════════════════════════════════════════════════════════════════════
    # ELECTRICAL STIMULATION — FRONT MONTANT
    # ═════════════════════════════════════════════════════════════════════

    def _send_stim_pulse(
        self, finger: str, is_omission: bool = False
    ) -> Dict[str, Any]:
        """
        Front montant sur le port parallèle.
        Le stimulateur détecte le front et gère le pulse.
        """
        t_now = self.task_clock.getTime()

        if is_omission:
            return {
                "finger":      finger,
                "is_omission": True,
                "pin_code":    0,
                "time_s":      round(t_now, 6),
            }

        pin_code = self.finger_pin_map.get(finger, 0)
        ttl_code = self.codes.get(f"stim_{finger}", 0)

        # ── Front montant → reset immédiat ──
        try:
            port = getattr(self.ParPort, "port", None)
            if port is not None:
                port.setData(pin_code)       # front montant
                t_sent = self.task_clock.getTime()
                port.setData(0x00)           # reset
            else:
                self.ParPort.send_trigger(pin_code)
                t_sent = self.task_clock.getTime()
        except Exception:
            self.ParPort.send_trigger(pin_code)
            t_sent = self.task_clock.getTime()

        return {
            "finger":      finger,
            "is_omission": False,
            "pin_code":    pin_code,
            "time_s":      round(t_sent, 6),
        }

    # ═════════════════════════════════════════════════════════════════════
    # SEQUENCE GENERATION
    # ═════════════════════════════════════════════════════════════════════

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
    # BLOCK EXECUTORS
    # ═════════════════════════════════════════════════════════════════════

    def _run_instruction_cue(
        self, condition: str, block_index: int
    ) -> float:
        self.should_quit()

        jitter   = random.uniform(
            -self.instruction_jitter, self.instruction_jitter
        )
        duration = max(2.0, self.instruction_duration + jitter)

        cond_ttl = self.codes.get(f"condition_{condition}", 0)

        self.log_trial_event(
            "instruction_start",
            condition=condition,
            block_index=block_index,
            duration_planned_s=round(duration, 3),
        )

        self.cue_stim.text = self.condition_cues.get(condition, condition)
        self.cue_stim.draw()
        img = self.condition_images_stim.get(condition)
        if img:
            img.draw()
        self.fixation.draw()
        self.win.flip()
        core.wait(duration)

        self.log_trial_event(
            "instruction_end",
            condition=condition,
            block_index=block_index,
        )
        return duration

    def _run_on_block(
        self, block_index: int, total_blocks: int, condition: str
    ) -> List[Dict[str, Any]]:
        self.should_quit()

        stim_seq = self._build_block_stim_list(condition)
        n_stims  = len(stim_seq)

        self.log_trial_event(
            "block_on_start",
            condition=condition,
            block_index=block_index,
            n_stims=n_stims,
        )

        self.fixation.draw()
        self.win.flip()

        # ══ CRITICAL TIMING ══
        gc.disable()
        t_start = self.task_clock.getTime()
        records: List[Dict[str, Any]] = []

        for si, info in enumerate(stim_seq):
            self.current_trial_idx = si

            # ── chaque stim espacée de 500 ms ──
            t_target = t_start + si * self.stim_interval_s

            remaining = t_target - self.task_clock.getTime()
            if remaining > 0:
                core.wait(remaining, hogCPUperiod=0.001)

            rec = self._send_stim_pulse(
                finger=info["finger"],
                is_omission=info["is_omission"],
            )
            rec.update(
                condition=condition,
                block_index=block_index,
                stim_index_in_block=si,
                target_time_s=round(t_target, 6),
            )
            records.append(rec)

            sched_err_ms = (rec["time_s"] - t_target) * 1000

            self.log_trial_event(
                "stim_omission" if info["is_omission"] else "stim_delivered",
                finger=info["finger"],
                is_omission=info["is_omission"],
                condition=condition,
                target_time_s=round(t_target, 6),
                actual_time_s=rec["time_s"],
                scheduling_error_ms=round(sched_err_ms, 3),
            )

            self.save_trial_incremental({
                "participant":         self.nom,
                "session":             self.session,
                "run_type":            self.run_type,
                "run_number":          self.run_number,
                "phase":               self.current_phase,
                "condition":           condition,
                "block_index":         block_index,
                "block_type":          "ON",
                "stim_index":          si,
                "finger":              info["finger"],
                "is_omission":         info["is_omission"],
                "pin_code":            rec.get("pin_code", 0),
                "time_s":              rec["time_s"],
                "target_time_s":       round(t_target, 6),
                "scheduling_error_ms": round(sched_err_ms, 3),
            })

            if abs(sched_err_ms) > 2.0:
                self.logger.warn(
                    f"TIMING B{block_index} S{si} "
                    f"({info['finger']}): {sched_err_ms:+.2f} ms"
                )

        # ── attendre la fin réelle du bloc ON (10 s) ──
        t_block_end_target = t_start + self.block_on_duration
        remaining_block = t_block_end_target - self.task_clock.getTime()
        if remaining_block > 0:
            core.wait(remaining_block, hogCPUperiod=0.001)

        gc.enable()
        gc.collect()
        # ══ END CRITICAL ══

        t_end = self.task_clock.getTime()
        dur   = t_end - t_start
        n_omit  = sum(1 for r in records if r["is_omission"])
        n_deliv = len(records) - n_omit

        self.log_trial_event(
            "block_on_end",
            condition=condition,
            block_index=block_index,
            actual_duration_s=round(dur, 4),
            n_delivered=n_deliv,
            n_omissions=n_omit,
        )

        self.logger.log(
            f"  ON  B{block_index:>2}/{total_blocks:<2}  {condition:<7}  "
            f"{dur:.3f} s | {n_deliv} stim  {n_omit} omit"
        )
        return records
    
    def _run_off_block(
        self,
        block_index: int,
        duration: Optional[float] = None,
        jitter: float = 0.0,
    ) -> float:
        self.should_quit()

        dur = duration if duration is not None else self.block_off_duration
        j   = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
        dur = max(1.0, dur + j)

        self.log_trial_event(
            "block_off_start",
            block_index=block_index,
            duration_planned_s=round(dur, 3),
        )

        self.fixation.draw()
        self.win.flip()
        core.wait(dur)

        self.log_trial_event("block_off_end", block_index=block_index)
        return dur

    # ═════════════════════════════════════════════════════════════════════
    # RUN EXECUTORS
    # ═════════════════════════════════════════════════════════════════════

    def _run_mapping(self) -> None:
        self.current_phase = "mapping"
        n = self.n_mapping_blocks

        self.logger.log(f"{'=' * 50}")
        self.logger.log(
            f"  MAPPING — Run {self.run_number:02d} — {n} blocs"
        )
        self.logger.log(f"{'=' * 50}")

        self.log_trial_event("run_start", n_blocks=n)

        for b in range(1, n + 1):
            self._run_on_block(
                block_index=b,
                total_blocks=n,
                condition="mapping",
            )
            self._run_off_block(
                block_index=b,
                duration=self.block_off_duration,
                jitter=0.0,
            )

        self.log_trial_event("run_end")
        self.logger.ok(f"Mapping Run {self.run_number:02d} complete.")

    def _run_prediction(self) -> None:
        self.current_phase = "prediction"
        block_order = self._build_run_block_order()
        n = len(block_order)

        self.logger.log(f"{'=' * 50}")
        self.logger.log(
            f"  PREDICTION — Run {self.run_number:02d} — {n} blocs"
        )
        self.logger.log(f"  Order: {block_order}")
        self.logger.log(f"{'=' * 50}")

        self.log_trial_event(
            "run_start",
            block_order=str(block_order),
            n_blocks=n,
        )

        for b_idx, cond in enumerate(block_order, start=1):
            self.current_phase = f"prediction_{cond}"

            self._run_instruction_cue(
                condition=cond,
                block_index=b_idx,
            )
            self._run_on_block(
                block_index=b_idx,
                total_blocks=n,
                condition=cond,
            )
            self._run_off_block(
                block_index=b_idx,
                duration=self.block_off_duration,
                jitter=0.0,
            )

        self.log_trial_event("run_end")
        self.logger.ok(f"Prediction Run {self.run_number:02d} complete.")

    # ═════════════════════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        finished = False

        try:
            self._show_instructions()
            self.wait_for_trigger()

            # baseline
            if self.initial_baseline > 0:
                self.show_resting_state(
                    duration_s=self.initial_baseline,
                    code_start_key="rest_start",
                    code_end_key="rest_end",
                )

            # single run
            if self.run_type == "mapping":
                self._run_mapping()
            else:
                self._run_prediction()

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
                    self.logger.warn(f"QC échoué (non bloquant) : {qc_exc}")

            if finished:
                self.show_instructions(
                    f"Run {self.run_number:02d} terminé.\nMerci !"
                )
                core.wait(3.0)

    # ─────────────────────────────────────────────────────────────────────
    # INSTRUCTIONS
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
                "Des instructions spécifiques s'afficheront avant chaque bloc.\n"
                "Maintenez votre regard sur la croix de fixation.\n\n"
                "En attente du scanner …"
            )
        self.show_instructions(txt)
