# connect_elec_tab.py
"""
PyQt6 control panel for ConnectElec — 7T somatotopy.
Lance UN run à la fois (mapping ou prediction) avec un numéro de run.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QFrame,
    QMessageBox,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _h_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _label(text: str) -> QLabel:
    return QLabel(text)


def _spin_int(value: int, lo: int, hi: int, suffix: str = "") -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    sb.setMinimumWidth(80)
    return sb


def _spin_float(
    value: float, lo: float, hi: float,
    step: float = 0.1, decimals: int = 1, suffix: str = ""
) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    if suffix:
        sb.setSuffix(suffix)
    sb.setMinimumWidth(80)
    return sb


# ═════════════════════════════════════════════════════════════════════════════

class ConnectElecTab(QWidget):

    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self._init_ui()

    # ─────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(10)
        self.setLayout(root)

        root.addWidget(self._build_timing_group())
        root.addWidget(self._build_mapping_group())
        root.addWidget(self._build_prediction_group())

        # ── Estimation ──
        self.lbl_estimate = QLabel("")
        self.lbl_estimate.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.lbl_estimate.setWordWrap(True)
        root.addWidget(self.lbl_estimate)

        root.addStretch()
        self._update_time_estimate()

    # ── Stimulation Timing (commun) ──────────────────────────────────────

    def _build_timing_group(self) -> QGroupBox:
        grp = QGroupBox("⚡ Paramètres de Stimulation")
        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        row = 0

        grid.addWidget(_label("ISI (ms) :"), row, 0)
        self.spin_isi = _spin_float(500.0, 100.0, 2000.0, 50.0, 0, " ms")
        grid.addWidget(self.spin_isi, row, 1)
        row += 1

        grid.addWidget(_label("Stims / doigt / bloc :"), row, 0)
        self.spin_stims_per_finger = _spin_int(5, 1, 20)
        self.spin_stims_per_finger.valueChanged.connect(
            self._update_time_estimate
        )
        grid.addWidget(self.spin_stims_per_finger, row, 1)
        grid.addWidget(_label("(× 4 doigts = stims / bloc)"), row, 2)
        row += 1

        grid.addWidget(_label("Durée bloc ON (s) :"), row, 0)
        self.spin_on_dur = _spin_float(10.0, 1.0, 60.0, 1.0, 1, " s")
        self.spin_on_dur.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_on_dur, row, 1)
        row += 1

        grid.addWidget(_label("Durée bloc OFF (s) :"), row, 0)
        self.spin_off_dur = _spin_float(10.0, 1.0, 60.0, 1.0, 1, " s")
        self.spin_off_dur.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_off_dur, row, 1)
        row += 1

        grid.addWidget(_label("Baseline initiale (s) :"), row, 0)
        self.spin_baseline = _spin_float(10.0, 0.0, 60.0, 1.0, 1, " s")
        grid.addWidget(self.spin_baseline, row, 1)
        row += 1

        info = QLabel(
            "ℹ  Front montant uniquement — "
            "le stimulateur gère le pulse."
        )
        info.setStyleSheet("color: gray; font-style: italic;")
        grid.addWidget(info, row, 0, 1, 3)

        grp.setLayout(grid)

        self.spin_isi.valueChanged.connect(self._update_time_estimate)
        self.spin_off_dur.valueChanged.connect(self._update_time_estimate)
        self.spin_baseline.valueChanged.connect(self._update_time_estimate)
        return grp

    # ── Finger Mapping ───────────────────────────────────────────────────

    def _build_mapping_group(self) -> QGroupBox:
        grp = QGroupBox("🖐 Finger Mapping")
        layout = QVBoxLayout()

        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        row = 0

        grid.addWidget(_label("Run n° :"), row, 0)
        self.spin_mapping_run = _spin_int(1, 1, 20)
        grid.addWidget(self.spin_mapping_run, row, 1)
        row += 1

        grid.addWidget(_label("Blocs ON :"), row, 0)
        self.spin_mapping_blocks = _spin_int(20, 1, 60)
        self.spin_mapping_blocks.valueChanged.connect(
            self._update_time_estimate
        )
        grid.addWidget(self.spin_mapping_blocks, row, 1)
        row += 1

        grid.addWidget(_label("Jitter OFF (± s) :"), row, 0)
        self.spin_mapping_jitter = _spin_float(
            0.0, 0.0, 0.0, 0.0, 1, " s"
        )
        grid.addWidget(self.spin_mapping_jitter, row, 1)

        layout.addLayout(grid)

        btn = QPushButton("🖐  Lancer Mapping")
        btn.setMinimumHeight(42)
        btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        btn.clicked.connect(self.run_mapping)
        layout.addWidget(btn)

        grp.setLayout(layout)
        return grp

    # ── Prediction ───────────────────────────────────────────────────────

    def _build_prediction_group(self) -> QGroupBox:
        grp = QGroupBox("🧠 Prediction Task")
        layout = QVBoxLayout()

        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        row = 0

        grid.addWidget(_label("Run n° :"), row, 0)
        self.spin_pred_run = _spin_int(1, 1, 20)
        grid.addWidget(self.spin_pred_run, row, 1)
        row += 1

        grid.addWidget(_label("Répétitions / condition :"), row, 0)
        self.spin_n_reps = _spin_int(5, 1, 15)
        self.spin_n_reps.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_n_reps, row, 1)
        grid.addWidget(_label("(× 4 cond. = blocs / run)"), row, 2)
        row += 1

        grid.addWidget(_h_separator(), row, 0, 1, 3)
        row += 1

        grid.addWidget(_label("Instruction cue (s) :"), row, 0)
        self.spin_instr_dur = _spin_float(5.0, 1.0, 15.0, 0.5, 1, " s")
        self.spin_instr_dur.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_instr_dur, row, 1)
        row += 1

        grid.addWidget(_label("Instruction jitter (± s) :"), row, 0)
        self.spin_instr_jitter = _spin_float(
            1.0, 0.0, 5.0, 0.5, 1, " s"
        )
        grid.addWidget(self.spin_instr_jitter, row, 1)

        layout.addLayout(grid)

        btn = QPushButton("🧠  Lancer Prediction")
        btn.setMinimumHeight(42)
        btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        btn.clicked.connect(self.run_prediction)
        layout.addWidget(btn)

        grp.setLayout(layout)
        return grp

    # ─────────────────────────────────────────────────────────────────────
    # TIME ESTIMATION
    # ─────────────────────────────────────────────────────────────────────

    def _update_time_estimate(self):
        on_s  = self.spin_on_dur.value()
        off_s = self.spin_off_dur.value()
        instr = self.spin_instr_dur.value()
        bl_s  = self.spin_baseline.value()

        n_map = self.spin_mapping_blocks.value()
        map_s = bl_s + n_map * (on_s + off_s)

        n_reps = self.spin_n_reps.value()
        n_blk  = 4 * n_reps
        pred_s = bl_s + n_blk * (instr + on_s + off_s)

        spf    = self.spin_stims_per_finger.value()
        n_stim = 4 * spf

        self.lbl_estimate.setText(
            f"📊  Mapping run: ~{map_s / 60:.1f} min ({n_map} blocs)  │  "
            f"Prediction run: ~{pred_s / 60:.1f} min ({n_blk} blocs)  │  "
            f"{n_stim} stim/bloc"
        )

        # ── Synchronisation stricte ON duration ──
        isi_s = self.spin_isi.value() / 1000.0
        spf   = self.spin_stims_per_finger.value()
        expected_on = 4 * spf * isi_s

        if abs(self.spin_on_dur.value() - expected_on) > 0.001:
            self.spin_on_dur.blockSignals(True)
            self.spin_on_dur.setValue(expected_on)
            self.spin_on_dur.blockSignals(False)

    # ─────────────────────────────────────────────────────────────────────
    # PARAMETER ASSEMBLY
    # ─────────────────────────────────────────────────────────────────────

    def _get_common_params(self) -> dict:
        return {
            "tache":                "ConnectElec",
            "stim_interval_ms":     self.spin_isi.value(),
            "stims_per_finger":     self.spin_stims_per_finger.value(),
            "block_on_duration":    self.spin_on_dur.value(),
            "block_off_duration":   self.spin_off_dur.value(),
            "initial_baseline":     self.spin_baseline.value(),
            "instruction_duration": self.spin_instr_dur.value(),
            "instruction_jitter":   self.spin_instr_jitter.value(),
            "n_reps_per_condition": self.spin_n_reps.value(),
            "n_mapping_blocks":     self.spin_mapping_blocks.value(),
            "mapping_off_jitter":   self.spin_mapping_jitter.value(),
        }

    def _confirm_launch(self, run_type: str, run_num: int) -> bool:
        on_s  = self.spin_on_dur.value()
        off_s = self.spin_off_dur.value()
        instr = self.spin_instr_dur.value()
        bl_s  = self.spin_baseline.value()

        if run_type == "mapping":
            n   = self.spin_mapping_blocks.value()
            dur = bl_s + n * (on_s + off_s)
            label = f"Finger Mapping — Run {run_num:02d} — {n} blocs"
        else:
            n_blk = 4 * self.spin_n_reps.value()
            dur   = bl_s + n_blk * (instr + on_s + off_s)
            label = f"Prediction — Run {run_num:02d} — {n_blk} blocs"

        reply = QMessageBox.question(
            self,
            "Confirmer le lancement",
            f"{label}\n"
            f"Durée estimée : {dur / 60:.1f} min\n\n"
            f"Lancer ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    # ─────────────────────────────────────────────────────────────────────
    # LAUNCH
    # ─────────────────────────────────────────────────────────────────────

    def run_mapping(self):
        run_num = self.spin_mapping_run.value()
        if not self._confirm_launch("mapping", run_num):
            return
        params = self._get_common_params()
        params["run_type"] = "mapping"
        params["run_number"] = run_num
        self.parent_menu.run_experiment(params)

    def run_prediction(self):
        run_num = self.spin_pred_run.value()
        if not self._confirm_launch("prediction", run_num):
            return
        params = self._get_common_params()
        params["run_type"] = "prediction"
        params["run_number"] = run_num
        self.parent_menu.run_experiment(params)