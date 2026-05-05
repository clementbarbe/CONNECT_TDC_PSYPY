# tabs_electdc.py
"""
PyQt6 control panel for ConnectElec — 7T somatotopy / prediction.

SOMATOTOPIE :  design/somatotopie/somatotopie.tsv   (onset  duration  finger)
PREDICTION  :  design/prediction/runN/prediction.tsv (onset  duration  condition  finger  is_stimulated  is_omission)
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QDoubleSpinBox, QPushButton, QFrame, QMessageBox,
)


# ─────────────────────────────────────────────────────────────────────────────

def _h_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


# ═════════════════════════════════════════════════════════════════════════════

class ConnectElecTab(QWidget):

    RUN_CONFIGS = [
        # (label,          icône, run_type,    run_number, couleur)
        ("SOMATOTOPY",     "🖐", "somatotopy",  1, "#4CAF50"),
        ("PREDICTION 1",   "🧠", "prediction",   1, "#2196F3"),
        ("PREDICTION 2",   "🧠", "prediction",   2, "#2196F3"),
        ("PREDICTION 3",   "🧠", "prediction",   3, "#2196F3"),
    ]

    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self._parport = None
        self._root_dir = self._find_root_dir()
        self._init_ui()

    @staticmethod
    def _find_root_dir() -> str:
        d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            if os.path.isdir(os.path.join(d, "design")):
                return d
            d = os.path.dirname(d)
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ─────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        root = QVBoxLayout()
        root.setSpacing(12)
        self.setLayout(root)

        root.addWidget(self._build_stim_params_group())
        root.addWidget(self._build_test_group())
        root.addWidget(self._build_runs_group())
        root.addStretch()

    # ── Paramètres ───────────────────────────────────────────────────────

    def _build_stim_params_group(self) -> QGroupBox:
        grp = QGroupBox("⚡ Paramètres de Stimulation")
        layout = QGridLayout()

        layout.addWidget(QLabel("Durée des bursts (ms) :"), 0, 0)
        self.spin_burst = QDoubleSpinBox()
        self.spin_burst.setRange(10.0, 500.0)
        self.spin_burst.setValue(75.0)
        self.spin_burst.setSingleStep(5.0)
        self.spin_burst.setDecimals(0)
        self.spin_burst.setSuffix(" ms")
        self.spin_burst.setMinimumWidth(100)
        layout.addWidget(self.spin_burst, 0, 1)

        info = QLabel(
            "ℹ  Trigger 64 envoyé toutes les X ms pendant la durée "
            "de chaque stimulation."
        )
        info.setStyleSheet("color: gray; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info, 1, 0, 1, 3)

        layout.addWidget(QLabel("Délai sélection doigt :"), 2, 0)
        lbl_lead = QLabel("250 ms (fixe)")
        lbl_lead.setStyleSheet("color: gray;")
        layout.addWidget(lbl_lead, 2, 1)

        layout.addWidget(QLabel("TR :"), 3, 0)
        lbl_tr = QLabel("2.0 s (fixe)")
        lbl_tr.setStyleSheet("color: gray;")
        layout.addWidget(lbl_tr, 3, 1)

        layout.setColumnStretch(2, 1)
        grp.setLayout(layout)
        return grp

    # ── Test ─────────────────────────────────────────────────────────────

    def _build_test_group(self) -> QGroupBox:
        grp = QGroupBox("🔧 Test Stimulation Manuelle")
        layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        for i in range(1, 6):
            btn = QPushButton(f"D{i}")
            btn.setMinimumWidth(55)
            btn.setMinimumHeight(36)
            btn.setStyleSheet(
                "QPushButton { font-weight: bold; border-radius: 4px; "
                "padding: 4px; }"
                "QPushButton:hover { background-color: #e0e0e0; }"
            )
            btn.clicked.connect(
                lambda checked, idx=i: self._test_finger(idx)
            )
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        self.lbl_test_status = QLabel(
            "Prêt — cliquez sur un doigt pour tester"
        )
        self.lbl_test_status.setStyleSheet(
            "color: gray; font-style: italic; padding: 2px;"
        )
        layout.addWidget(self.lbl_test_status)

        info = QLabel(
            "Séquence : trigger pin → 200 ms → trigger 64 → trigger 0"
        )
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)

        grp.setLayout(layout)
        return grp

    # ── Runs ─────────────────────────────────────────────────────────────

    def _build_runs_group(self) -> QGroupBox:
        grp = QGroupBox("🚀 Lancer un Run")
        layout = QVBoxLayout()

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)

        self.run_info_labels: list[QLabel] = []

        for row, (label, icon, rtype, rnum, color) in enumerate(
            self.RUN_CONFIGS
        ):
            btn = QPushButton(f"{icon}  {label}")
            btn.setMinimumHeight(42)
            btn.setMinimumWidth(200)
            hover = "#45a049" if color == "#4CAF50" else "#1976D2"
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color}; color: white; "
                f"font-weight: bold; border-radius: 4px; "
                f"padding: 6px 16px; }}"
                f"QPushButton:hover {{ background-color: {hover}; }}"
            )
            btn.clicked.connect(
                lambda checked, rt=rtype, rn=rnum: self._launch_run(rt, rn)
            )
            grid.addWidget(btn, row, 0)

            info_lbl = QLabel("")
            info_lbl.setMinimumWidth(380)
            info_lbl.setWordWrap(True)
            self.run_info_labels.append(info_lbl)
            grid.addWidget(info_lbl, row, 1)

        layout.addLayout(grid)
        layout.addWidget(_h_separator())

        btn_refresh = QPushButton("🔄  Actualiser les fichiers design")
        btn_refresh.setStyleSheet("QPushButton { padding: 4px 12px; }")
        btn_refresh.clicked.connect(self._update_all_run_info)
        layout.addWidget(btn_refresh)

        grp.setLayout(layout)
        self._update_all_run_info()
        return grp

    # ─────────────────────────────────────────────────────────────────────
    # INFO RUNS
    # ─────────────────────────────────────────────────────────────────────

    def _update_all_run_info(self) -> None:
        try:
            from tasks.connectelec import ConnectElec
        except ImportError:
            for lbl in self.run_info_labels:
                lbl.setText("⚠ Import ConnectElec échoué")
                lbl.setStyleSheet("color: orange;")
            return

        for i, (_, _, rtype, rnum, _) in enumerate(self.RUN_CONFIGS):
            ddir = ConnectElec.get_design_dir(
                self._root_dir, rtype, rnum
            )
            if ddir is None:
                self.run_info_labels[i].setText("❌ Configuration inconnue")
                self.run_info_labels[i].setStyleSheet("color: red;")
                continue

            info = ConnectElec.compute_run_info(
                ddir, run_type=rtype, run_number=rnum,
            )
            if info is None:
                if rtype == "somatotopy":
                    expected = f"somatotopie.tsv"
                else:
                    expected = f"prediction.tsv"
                self.run_info_labels[i].setText(
                    f"❌ {expected} manquant\n    {ddir}"
                )
                self.run_info_labels[i].setStyleSheet("color: red;")
            else:
                dur = info["run_duration_s"]
                parts = [
                    f"✅  {dur:.0f} s ({dur / 60:.1f} min)",
                    f"{info['n_volumes']} vol",
                    f"{info['n_events']} events",
                ]

                if info.get("n_omissions", 0) > 0:
                    parts.append(
                        f"{info['n_stimulated']} stim / "
                        f"{info['n_omissions']} omit"
                    )

                if (info.get("conditions")
                        and info["conditions"] != ["somatotopy"]):
                    parts.append(
                        f"cond: {', '.join(info['conditions'])}"
                    )

                parts.append(', '.join(info["fingers"]))

                padding = info.get("padding_s", 0)
                parts.append(f"pad: {padding:.0f} s")

                self.run_info_labels[i].setText("  │  ".join(parts))
                self.run_info_labels[i].setStyleSheet(
                    "color: #2e7d32; font-weight: bold;"
                )

    # ─────────────────────────────────────────────────────────────────────
    # TEST
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_parport(self):
        if self._parport is None:
            try:
                from utils.hardware_manager import setup_hardware
                self._parport, _ = setup_hardware(parport_actif=True)
            except Exception:
                from utils.hardware_manager import SafeDummyParPort
                self._parport = SafeDummyParPort()
        return self._parport

    def _test_finger(self, finger_idx: int) -> None:
        self.lbl_test_status.setText(f"⏳ Test D{finger_idx} en cours…")
        self.lbl_test_status.setStyleSheet("color: #1565C0;")
        self.lbl_test_status.repaint()

        try:
            pp = self._ensure_parport()
            from tasks.connectelec import ConnectElec
            ConnectElec.test_finger(pp, finger_idx)

            from utils.hardware_manager import SafeDummyParPort
            if isinstance(pp, SafeDummyParPort):
                self.lbl_test_status.setText(
                    f"D{finger_idx} testé (⚠ simulation — "
                    f"port parallèle non connecté)"
                )
                self.lbl_test_status.setStyleSheet("color: #E65100;")
            else:
                self.lbl_test_status.setText(
                    f"✅ D{finger_idx} testé avec succès"
                )
                self.lbl_test_status.setStyleSheet("color: #2e7d32;")

        except Exception as exc:
            self.lbl_test_status.setText(f"❌ Erreur : {exc}")
            self.lbl_test_status.setStyleSheet("color: red;")

    # ─────────────────────────────────────────────────────────────────────
    # LAUNCH
    # ─────────────────────────────────────────────────────────────────────

    def _launch_run(self, run_type: str, run_number: int) -> None:
        try:
            from tasks.connectelec import ConnectElec
        except ImportError as exc:
            QMessageBox.critical(
                self, "Erreur d'import",
                f"Impossible d'importer ConnectElec :\n{exc}"
            )
            return

        ddir = ConnectElec.get_design_dir(
            self._root_dir, run_type, run_number
        )
        info = ConnectElec.compute_run_info(
            ddir, run_type=run_type, run_number=run_number,
        ) if ddir else None

        if info is None:
            if run_type == "somatotopy":
                expected = "somatotopie.tsv"
            else:
                expected = "prediction.tsv"
            QMessageBox.critical(
                self,
                "Fichiers de design manquants",
                f"Impossible de trouver {expected} pour\n"
                f"{run_type.upper()} run {run_number}.\n\n"
                f"Répertoire attendu :\n{ddir}",
            )
            return

        dur     = info["run_duration_s"]
        details = (
            f"{run_type.upper()} — Run {run_number:02d}\n\n"
            f"Durée totale :     {dur:.0f} s ({dur / 60:.1f} min)\n"
            f"Volumes :          {info['n_volumes']}  (TR = 2.0 s)\n"
            f"Événements :       {info['n_events']}\n"
            f"Doigts :           {', '.join(info['fingers'])}\n"
            f"Dernier stim fin : {info['last_stim_end_s']:.1f} s\n"
            f"Padding post-stim: {info['padding_s']:.1f} s\n"
        )
        if info.get("n_omissions", 0) > 0:
            details += (
                f"Stimulés :         {info['n_stimulated']}\n"
                f"Omissions :        {info['n_omissions']}\n"
            )
        if (info.get("conditions")
                and info["conditions"] != ["somatotopy"]):
            details += (
                f"Conditions :       {', '.join(info['conditions'])}\n"
            )
        details += (
            f"Burst interval :   {self.spin_burst.value():.0f} ms\n\n"
            f"Lancer ?"
        )

        reply = QMessageBox.question(
            self,
            "Confirmer le lancement",
            details,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        params = {
            "tache":             "ConnectElec",
            "run_type":          run_type,
            "run_number":        run_number,
            "burst_interval_ms": self.spin_burst.value(),
        }
        self.parent_menu.run_experiment(params)