# stimulation_electrique.py
"""
Stimulation Électrique — Somatotopie Digitale & Tâche de Prédiction
====================================================================
Phase 1 : Finger Mapping  — blocs ON/OFF, stimulation pseudo-aléatoire D1–D4
Phase 2 : Prediction Task — 4 conditions (FP, TP, FR, TR)

Stimulation déclenchée par front montant sur pins du port parallèle.
"""

import random
import gc
import os
from psychopy import visual, core
from utils.base_task import BaseTask


# Mapping doigts → pin data bits du port parallèle
FINGER_PIN_MAP = {
    'D1': 0x01,   # Pin D0
    'D2': 0x02,   # Pin D1
    'D3': 0x04,   # Pin D2
    'D4': 0x08,   # Pin D3
}

FINGERS_4 = ['D1', 'D2', 'D3', 'D4']
PREDICTABLE_ORDER = ['D1', 'D2', 'D3', 'D4']


class ConnectElec(BaseTask):
    """
    Tâche de stimulation électrique digitale en IRMf.
    """

    def __init__(self, win, nom, session='01', mode='fmri', run_type='full',
                 n_blocks_on=20,
                 stims_per_finger=5,
                 stim_interval_ms=500,
                 stim_duration_ms=1,
                 block_off_duration=10.0,
                 block_off_jitter=5.0,
                 pause_between_phases=180.0,
                 pause_between_blocks=180.0,
                 prediction_block_order=None,
                 enregistrer=True, eyetracker_actif=False, parport_actif=True,
                 **kwargs):

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="Stimulation Electrique",
            folder_name="stimulation_electrique",
            eyetracker_actif=eyetracker_actif,
            parport_actif=parport_actif,
            enregistrer=enregistrer,
            et_prefix='SE'
        )

        self.mode = mode.lower()
        self.run_type = run_type.lower()

        self.n_blocks_on = n_blocks_on
        self.stims_per_finger = stims_per_finger
        self.stim_interval_s = stim_interval_ms / 1000.0
        self.stim_duration_s = stim_duration_ms / 1000.0
        self.block_off_duration = block_off_duration
        self.block_off_jitter = block_off_jitter
        self.pause_between_phases = pause_between_phases
        self.pause_between_blocks = pause_between_blocks

        self.prediction_block_order = prediction_block_order or ['FP', 'TP', 'FR', 'TR']
        self.finger_pin_map = dict(FINGER_PIN_MAP)

        # Durée d'un bloc ON = n_doigts × stims_per_finger × ISI
        self.n_stims_per_block = len(FINGERS_4) * self.stims_per_finger
        self.block_on_duration = self.n_stims_per_block * self.stim_interval_s

        self.global_records = []
        self.current_trial_idx = 0
        self.current_phase = 'setup'

        self._detect_display_scaling()
        self._measure_frame_rate()
        self._define_ttl_codes()
        self._setup_key_mapping()
        self._setup_task_stimuli()
        self._init_incremental_file(suffix=f"_{self.run_type}")

        self.logger.ok(
            f"StimulationElectrique init | Mode: {self.run_type} | "
            f"Frame Rate: {self.frame_rate:.2f} Hz | "
            f"Blocks ON: {self.n_blocks_on} | "
            f"Stims/finger/block: {self.stims_per_finger} | "
            f"ISI: {self.stim_interval_s*1000:.0f}ms | "
            f"Block ON: {self.block_on_duration:.1f}s"
        )

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def _detect_display_scaling(self):
        if self.win.size[1] > 1200:
            self.pixel_scale = 2.0
            self.logger.log(f"High-res display ({self.win.size}). Scale: x2.0")
        else:
            self.pixel_scale = 1.0
            self.logger.log(f"Standard display ({self.win.size}). Scale: x1.0")

    def _measure_frame_rate(self):
        self.logger.log("Measuring frame rate...")
        self.frame_rate = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, threshold=1
        )
        if self.frame_rate is None:
            self.frame_rate = 60.0
            self.logger.warn("Frame rate not detected, defaulting to 60.0 Hz")
        else:
            self.logger.ok(f"Frame rate: {self.frame_rate:.2f} Hz")

        self.frame_duration_s = 1.0 / self.frame_rate
        self.frame_tolerance_s = 0.75 / self.frame_rate
        self.logger.log(f"Frame tolerance: {self.frame_tolerance_s * 1000:.2f} ms")

    def _define_ttl_codes(self):
        self.codes = {
            'start_exp': 255,
            'end_exp': 254,
            'rest_start': 200,
            'rest_end': 201,
            'mapping_start': 210,
            'mapping_end': 211,
            'prediction_start': 220,
            'prediction_end': 221,
            'block_on_start': 100,
            'block_on_end': 101,
            'block_off_start': 110,
            'block_off_end': 111,
            'stim_D1': 11,
            'stim_D2': 12,
            'stim_D3': 13,
            'stim_D4': 14,
            'stim_omission': 15,
            'condition_FP': 50,
            'condition_TP': 51,
            'condition_FR': 52,
            'condition_TR': 53,
            'pause_start': 120,
            'pause_end': 121,
        }

    def _setup_key_mapping(self):
        if self.mode == 'fmri':
            self.key_trigger = 't'
            self.key_continue = 'b'
        else:
            self.key_trigger = 't'
            self.key_continue = 'space'

    def _setup_task_stimuli(self):
        # Tâche passive : uniquement la croix de fixation (déjà dans BaseTask)
        self.logger.log("Stimuli loaded (fixation cross only — passive task).")

    # =========================================================================
    # LOGGING
    # =========================================================================

    def log_trial_event(self, event_type, **kwargs):
        current_time = self.task_clock.getTime()

        if self.eyetracker_actif:
            self.EyeTracker.send_message(
                f"PHASE_{self.current_phase.upper()}_"
                f"STIM_{self.current_trial_idx:03d}_{event_type.upper()}"
            )

        entry = {
            'participant': self.nom,
            'session': self.session,
            'phase': self.current_phase,
            'stim_index': self.current_trial_idx,
            'time_s': round(current_time, 6),
            'event_type': event_type
        }
        entry.update(kwargs)
        self.global_records.append(entry)

    # =========================================================================
    # STIMULATION ÉLECTRIQUE — CONTRÔLE HARDWARE
    # =========================================================================

    def send_stim_pulse(self, finger, is_omission=False):
        """
        Envoie une stimulation électrique via front montant sur le port parallèle.
        Busy-wait pour garantir la durée de 1ms.
        """
        t_before = self.task_clock.getTime()

        if is_omission:
            # Slot temporel conservé, pas de stimulation
            self.ParPort.send_trigger(self.codes['stim_omission'])
            t_after = self.task_clock.getTime()
            return {
                'finger': finger,
                'is_omission': True,
                'pin_code': 0,
                'time_s': round(t_before, 6),
                'timing_error_ms': 0.0,
            }

        pin_code = self.finger_pin_map.get(finger, 0)
        ttl_code = self.codes.get(f'stim_{finger}', 0)

        # Front montant
        try:
            if hasattr(self.ParPort, 'port') and self.ParPort.port is not None:
                self.ParPort.port.setData(pin_code)
                t0 = core.getTime()
                while core.getTime() < t0 + self.stim_duration_s:
                    pass
                self.ParPort.port.setData(0)
            else:
                self.ParPort.send_trigger(pin_code, duration=self.stim_duration_s)
        except Exception:
            self.ParPort.send_trigger(pin_code, duration=self.stim_duration_s)

        t_after = self.task_clock.getTime()

        # TTL event marker après le pulse
        self.ParPort.send_trigger(ttl_code)

        return {
            'finger': finger,
            'is_omission': False,
            'pin_code': pin_code,
            'time_s': round(t_after, 6),
            'timing_error_ms': round((t_after - t_before) * 1000 - self.stim_duration_s * 1000, 3),
        }

    # =========================================================================
    # GÉNÉRATION DE SÉQUENCES
    # =========================================================================

    def build_pseudo_random_sequence(self, fingers, stims_per_finger):
        """
        Séquence pseudo-aléatoire : pas de répétition consécutive,
        chaque doigt exactement stims_per_finger fois.
        """
        pool = []
        for f in fingers:
            pool.extend([f] * stims_per_finger)

        for _attempt in range(100):
            sequence = []
            remaining = pool[:]
            random.shuffle(remaining)
            success = True

            while remaining:
                if sequence:
                    candidates = [f for f in remaining if f != sequence[-1]]
                else:
                    candidates = remaining[:]

                if not candidates:
                    success = False
                    break

                chosen = random.choice(candidates)
                sequence.append(chosen)
                remaining.remove(chosen)

            if success and len(sequence) == len(pool):
                return sequence

        self.logger.warn("Pseudo-random constraints failed. Using simple shuffle.")
        random.shuffle(pool)
        return pool

    def build_predictable_sequence(self, fingers, stims_per_finger):
        """Séquence cyclique fixe : D1,D2,D3,D4,D1,D2,..."""
        total = len(fingers) * stims_per_finger
        return [fingers[i % len(fingers)] for i in range(total)]

    def build_block_sequence(self, condition):
        """
        Construit la séquence de stim pour un bloc ON selon la condition.

        Returns:
            list[dict]: [{finger, is_omission}, ...]
        """
        if condition == 'mapping':
            raw = self.build_pseudo_random_sequence(FINGERS_4, self.stims_per_finger)
            return [{'finger': f, 'is_omission': False} for f in raw]

        elif condition == 'FP':
            raw = self.build_predictable_sequence(PREDICTABLE_ORDER, self.stims_per_finger)
            return [{'finger': f, 'is_omission': False} for f in raw]

        elif condition == 'TP':
            raw = self.build_predictable_sequence(PREDICTABLE_ORDER, self.stims_per_finger)
            return [{'finger': f, 'is_omission': (f == 'D4')} for f in raw]

        elif condition == 'FR':
            raw = self.build_pseudo_random_sequence(FINGERS_4, self.stims_per_finger)
            return [{'finger': f, 'is_omission': False} for f in raw]

        elif condition == 'TR':
            raw = self.build_pseudo_random_sequence(FINGERS_4, self.stims_per_finger)
            return [{'finger': f, 'is_omission': (f == 'D4')} for f in raw]

        else:
            self.logger.err(f"Unknown condition: {condition}")
            return []

    # =========================================================================
    # CORE TASK LOGIC
    # =========================================================================

    def run_on_block(self, block_index, total_blocks, condition):
        """
        Exécute un bloc ON complet avec timing sub-milliseconde.
        """
        self.should_quit()

        stim_sequence = self.build_block_sequence(condition)
        n_stims = len(stim_sequence)

        self.log_trial_event(
            'block_on_start', condition=condition,
            block_index=block_index, n_stims=n_stims
        )
        self.ParPort.send_trigger(self.codes['block_on_start'])

        self.fixation.draw()
        self.win.flip()

        # =================================================================
        # CRITICAL TIMING: DISABLE GC
        # =================================================================
        gc.disable()

        t_block_start = self.task_clock.getTime()
        block_records = []

        for stim_idx, stim_info in enumerate(stim_sequence):
            self.current_trial_idx = stim_idx

            # Timing cible pour cette stimulation
            t_target = t_block_start + (stim_idx * self.stim_interval_s)

            # Busy-wait jusqu'au moment précis
            while self.task_clock.getTime() < (t_target - 0.0001):
                pass

            # Stimulation
            stim_record = self.send_stim_pulse(
                finger=stim_info['finger'],
                is_omission=stim_info['is_omission']
            )

            stim_record.update({
                'condition': condition,
                'block_index': block_index,
                'stim_index_in_block': stim_idx,
                'target_time_s': round(t_target, 6),
            })
            block_records.append(stim_record)

            self.log_trial_event(
                'stim_omission' if stim_info['is_omission'] else 'stim_delivered',
                finger=stim_info['finger'],
                is_omission=stim_info['is_omission'],
                target_time_s=round(t_target, 6),
                actual_time_s=stim_record['time_s'],
                timing_error_ms=stim_record['timing_error_ms']
            )

            # Sauvegarde incrémentale
            trial_summary = {
                'participant': self.nom,
                'session': self.session,
                'phase': self.current_phase,
                'condition': condition,
                'block_index': block_index,
                'block_type': 'ON',
                'stim_index': stim_idx,
                'finger': stim_info['finger'],
                'is_omission': stim_info['is_omission'],
                'pin_code': stim_record.get('pin_code', 0),
                'time_s': stim_record['time_s'],
                'target_time_s': round(t_target, 6),
                'timing_error_ms': stim_record['timing_error_ms'],
            }
            self.save_trial_incremental(trial_summary)

            # Warning si timing dérape
            t_now = self.task_clock.getTime()
            error_ms = (t_now - t_target) * 1000
            if abs(error_ms) > 2.0:
                self.logger.warn(
                    f"TIMING WARNING Block {block_index}, Stim {stim_idx} "
                    f"({stim_info['finger']}): error={error_ms:.2f}ms"
                )

        # =================================================================
        # CRITICAL TIMING END: RE-ENABLE GC
        # =================================================================
        gc.enable()
        gc.collect()

        t_block_end = self.task_clock.getTime()
        actual_duration = t_block_end - t_block_start

        self.ParPort.send_trigger(self.codes['block_on_end'])
        self.log_trial_event(
            'block_on_end', condition=condition,
            block_index=block_index,
            actual_duration_s=round(actual_duration, 4),
            n_stims_delivered=sum(1 for r in block_records if not r['is_omission']),
            n_omissions=sum(1 for r in block_records if r['is_omission'])
        )

        omissions = sum(1 for r in block_records if r['is_omission'])
        delivered = len(block_records) - omissions
        self.logger.log(
            f"Block ON {block_index:>2}/{total_blocks:<2} | "
            f"{condition:<7} | Duration: {actual_duration:.3f}s | "
            f"Stims: {delivered} delivered, {omissions} omissions"
        )

        return block_records

    def run_off_block(self, block_index):
        """
        Exécute un bloc OFF (repos) avec jitter ±5s.
        """
        self.should_quit()

        jitter = random.uniform(-self.block_off_jitter, self.block_off_jitter)
        duration = max(1.0, self.block_off_duration + jitter)

        self.log_trial_event(
            'block_off_start', block_index=block_index,
            planned_duration_s=round(duration, 3), jitter_s=round(jitter, 3)
        )
        self.ParPort.send_trigger(self.codes['block_off_start'])

        self.fixation.draw()
        self.win.flip()
        core.wait(duration)

        self.ParPort.send_trigger(self.codes['block_off_end'])
        self.log_trial_event('block_off_end', block_index=block_index)

        self.logger.log(
            f"Block OFF {block_index:>2} | "
            f"Duration: {duration:.1f}s (jitter: {jitter:+.1f}s)"
        )

        return duration

    def run_block_series(self, n_blocks, condition, series_name):
        """
        Alterne n blocs ON/OFF pour une condition donnée.
        """
        self.log_trial_event(
            'series_start', series_name=series_name,
            condition=condition, n_blocks=n_blocks
        )
        self.logger.log(f"--- Series Start: {series_name} ({n_blocks} ON blocks) ---")

        for block_idx in range(1, n_blocks + 1):
            self.run_on_block(block_idx, n_blocks, condition)

            if block_idx < n_blocks:
                self.run_off_block(block_idx)

        self.log_trial_event('series_end', series_name=series_name)
        self.logger.log(f"--- Series End: {series_name} ---")

    # =========================================================================
    # PHASE 1 : FINGER MAPPING
    # =========================================================================

    def run_finger_mapping(self):
        self.current_phase = 'finger_mapping'
        self.logger.log("=" * 50)
        self.logger.log("  PHASE 1 : FINGER MAPPING")
        self.logger.log("=" * 50)

        self.ParPort.send_trigger(self.codes['mapping_start'])
        self.log_trial_event('mapping_start', n_blocks=self.n_blocks_on)

        self.run_block_series(
            n_blocks=self.n_blocks_on,
            condition='mapping',
            series_name='FINGER_MAPPING'
        )

        self.ParPort.send_trigger(self.codes['mapping_end'])
        self.log_trial_event('mapping_end')
        self.logger.ok("Finger Mapping complete.")

    # =========================================================================
    # PHASE 2 : PREDICTION TASK
    # =========================================================================

    def show_condition_instruction(self, condition):
        instructions = {
            'FP': (
                "Condition : 4 doigts — Séquence prédictible\n\n"
                "Portez attention à la stimulation de chaque doigt et\n"
                "prédisez quand votre index sera stimulé\n"
                "en vous basant sur le rythme temporel.\n\n"
                "Fixez la croix. Appuyez pour continuer."
            ),
            'TP': (
                "Condition : 3 doigts — Séquence prédictible\n\n"
                "Portez attention à la stimulation de chaque doigt et\n"
                "prédisez quand l'index sera stimulé,\n"
                "même si aucune stimulation réelle n'a lieu.\n\n"
                "Fixez la croix. Appuyez pour continuer."
            ),
            'FR': (
                "Condition : 4 doigts — Séquence aléatoire\n\n"
                "Portez attention à la stimulation de chaque doigt.\n"
                "N'essayez pas de prédire un quelconque motif.\n\n"
                "Fixez la croix. Appuyez pour continuer."
            ),
            'TR': (
                "Condition : 3 doigts — Séquence aléatoire\n\n"
                "Portez attention à la stimulation de chaque doigt.\n"
                "N'essayez pas de prédire un quelconque motif.\n\n"
                "Fixez la croix. Appuyez pour continuer."
            ),
        }

        text = instructions.get(condition, "Préparez-vous. Appuyez pour continuer.")
        self.log_trial_event('instruction_shown', condition=condition)
        self.show_instructions(text)

    def show_pause_screen(self, duration_label):
        self.log_trial_event('pause_start')
        self.ParPort.send_trigger(self.codes['pause_start'])

        text = (
            f"Pause — {duration_label}\n\n"
            "L'expérimentateur vérifie que tout va bien.\n"
            "Appuyez pour continuer quand vous êtes prêt."
        )
        self.instr_stim.text = text
        self.instr_stim.draw()
        self.win.flip()

        core.wait(5.0)
        self.wait_keys(key_list=[self.key_continue])

        self.ParPort.send_trigger(self.codes['pause_end'])
        self.log_trial_event('pause_end')

    def run_prediction_task(self):
        self.current_phase = 'prediction'
        self.logger.log("=" * 50)
        self.logger.log("  PHASE 2 : PREDICTION TASK")
        self.logger.log("=" * 50)

        self.ParPort.send_trigger(self.codes['prediction_start'])
        self.log_trial_event(
            'prediction_start',
            conditions=self.prediction_block_order
        )

        n_conditions = len(self.prediction_block_order)

        for cond_idx, condition in enumerate(self.prediction_block_order, 1):
            self.current_phase = f'prediction_{condition}'

            # Instruction
            self.show_condition_instruction(condition)

            # TTL condition
            cond_code = self.codes.get(f'condition_{condition}', 0)
            self.ParPort.send_trigger(cond_code)

            # Blocs ON/OFF pour cette condition
            self.run_block_series(
                n_blocks=self.n_blocks_on,
                condition=condition,
                series_name=f'PREDICTION_{condition}'
            )

            self.logger.ok(
                f"Prediction condition {condition} complete "
                f"({cond_idx}/{n_conditions})"
            )

            # Pause entre conditions (sauf après la dernière)
            if cond_idx < n_conditions:
                minutes = self.pause_between_blocks / 60
                self.show_pause_screen(
                    duration_label=f"{minutes:.0f} min "
                                   f"({cond_idx}/{n_conditions - 1})"
                )

        self.ParPort.send_trigger(self.codes['prediction_end'])
        self.log_trial_event('prediction_end')
        self.logger.ok("Prediction Task complete.")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        finished_naturally = False
        saved_path = None

        try:
            # Instructions
            if self.run_type == 'mapping_only':
                instructions = (
                    "FINGER MAPPING — Stimulation Électrique\n\n"
                    "Vous allez recevoir des stimulations électriques\n"
                    "sur les doigts de la main droite.\n\n"
                    "Restez immobile et fixez la croix.\n\n"
                    f"Nombre de blocs : {self.n_blocks_on}\n"
                    f"Stimulations par bloc : {self.n_stims_per_block}\n\n"
                    "Appuyez sur une touche pour continuer..."
                )
            elif self.run_type == 'prediction_only':
                instructions = (
                    "TÂCHE DE PRÉDICTION — Stimulation Électrique\n\n"
                    "Vous allez recevoir des stimulations électriques\n"
                    "selon différentes conditions.\n\n"
                    "Suivez les instructions avant chaque bloc.\n"
                    "Restez immobile et fixez la croix.\n\n"
                    f"Conditions : {', '.join(self.prediction_block_order)}\n\n"
                    "Appuyez sur une touche pour continuer..."
                )
            else:
                instructions = (
                    "PROTOCOLE COMPLET — Stimulation Électrique\n\n"
                    "Phase 1 : Finger Mapping\n"
                    "Phase 2 : Tâche de Prédiction\n\n"
                    "Restez immobile et fixez la croix.\n\n"
                    "Appuyez sur une touche pour continuer..."
                )

            self.show_instructions(instructions)
            self.wait_for_trigger()

            if self.run_type in ('full', 'mapping_only'):
                self.logger.log(
                    f"Starting: FINGER MAPPING ({self.n_blocks_on} blocks)"
                )
                self.show_resting_state(duration_s=10.0)
                self.run_finger_mapping()

                if self.run_type == 'full':
                    minutes = self.pause_between_phases / 60
                    self.show_pause_screen(
                        duration_label=f"{minutes:.0f} min (entre phases)"
                    )

            if self.run_type in ('full', 'prediction_only'):
                self.logger.log(
                    f"Starting: PREDICTION TASK "
                    f"({self.prediction_block_order})"
                )
                if self.run_type == 'prediction_only':
                    self.show_resting_state(duration_s=10.0)
                self.run_prediction_task()

            finished_naturally = True
            self.logger.ok("Task completed successfully.")

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Manual interruption.")

        except Exception as e:
            self.logger.err(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            self.logger.log("Final save...")

            if self.eyetracker_actif:
                self.EyeTracker.stop_recording()
                self.EyeTracker.send_message("END_EXP")
                self.EyeTracker.close_and_transfer_data(self.data_dir)

            saved_path = self.save_data(
                data_list=self.global_records,
                filename_suffix=f"_{self.run_type}"
            )

            if finished_naturally:
                end_msg = "Fin de la session.\nMerci pour votre participation."
                self.show_instructions(end_msg)
                core.wait(3.0)