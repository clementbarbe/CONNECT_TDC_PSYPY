from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QSpinBox, QPushButton)


class ConnectElecTab(QWidget):
    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # FINGER MAPPING
        mapping_group = QGroupBox("Finger Mapping")
        mapping_layout = QVBoxLayout()
        blocks_mapping_layout = QHBoxLayout()
        blocks_mapping_layout.addWidget(QLabel("Blocs ON :"))
        self.spin_mapping_blocks = QSpinBox()
        self.spin_mapping_blocks.setRange(1, 40)
        self.spin_mapping_blocks.setValue(20)
        blocks_mapping_layout.addWidget(self.spin_mapping_blocks)
        blocks_mapping_layout.addStretch()
        mapping_layout.addLayout(blocks_mapping_layout)

        stims_mapping_layout = QHBoxLayout()
        stims_mapping_layout.addWidget(QLabel("Stims / doigt / bloc :"))
        self.spin_mapping_stims = QSpinBox()
        self.spin_mapping_stims.setRange(1, 20)
        self.spin_mapping_stims.setValue(5)
        stims_mapping_layout.addWidget(self.spin_mapping_stims)
        stims_mapping_layout.addStretch()
        mapping_layout.addLayout(stims_mapping_layout)

        btn_run_mapping = QPushButton("Lancer Finger Mapping")
        btn_run_mapping.clicked.connect(self.run_mapping)
        mapping_layout.addWidget(btn_run_mapping)
        mapping_group.setLayout(mapping_layout)
        layout.addWidget(mapping_group)

        # PREDICTION TASK
        prediction_group = QGroupBox("Prediction Task")
        prediction_layout = QVBoxLayout()
        blocks_pred_layout = QHBoxLayout()
        blocks_pred_layout.addWidget(QLabel("Blocs ON :"))
        self.spin_pred_blocks = QSpinBox()
        self.spin_pred_blocks.setRange(1, 40)
        self.spin_pred_blocks.setValue(20)
        blocks_pred_layout.addWidget(self.spin_pred_blocks)
        blocks_pred_layout.addStretch()
        prediction_layout.addLayout(blocks_pred_layout)

        stims_pred_layout = QHBoxLayout()
        stims_pred_layout.addWidget(QLabel("Stims / doigt / bloc :"))
        self.spin_pred_stims = QSpinBox()
        self.spin_pred_stims.setRange(1, 20)
        self.spin_pred_stims.setValue(5)
        stims_pred_layout.addWidget(self.spin_pred_stims)
        stims_pred_layout.addStretch()
        prediction_layout.addLayout(stims_pred_layout)

        btn_run_prediction = QPushButton("Lancer Prediction Task")
        btn_run_prediction.clicked.connect(self.run_prediction)
        prediction_layout.addWidget(btn_run_prediction)
        prediction_group.setLayout(prediction_layout)
        layout.addWidget(prediction_group)

        # FULL RUN
        full_group = QGroupBox("Full Run (Mapping + Prediction)")
        full_layout = QVBoxLayout()
        blocks_full_layout = QHBoxLayout()
        blocks_full_layout.addWidget(QLabel("Blocs ON :"))
        self.spin_full_blocks = QSpinBox()
        self.spin_full_blocks.setRange(1, 40)
        self.spin_full_blocks.setValue(20)
        blocks_full_layout.addWidget(self.spin_full_blocks)
        blocks_full_layout.addStretch()
        full_layout.addLayout(blocks_full_layout)

        stims_full_layout = QHBoxLayout()
        stims_full_layout.addWidget(QLabel("Stims / doigt / bloc :"))
        self.spin_full_stims = QSpinBox()
        self.spin_full_stims.setRange(1, 20)
        self.spin_full_stims.setValue(5)
        stims_full_layout.addWidget(self.spin_full_stims)
        stims_full_layout.addStretch()
        full_layout.addLayout(stims_full_layout)

        btn_run_full = QPushButton("Lancer Full Run")
        btn_run_full.clicked.connect(self.run_full)
        full_layout.addWidget(btn_run_full)
        full_group.setLayout(full_layout)
        layout.addWidget(full_group)

        layout.addStretch()

    def get_common(self):
        return {
            'tache': 'ConnectElec',
            'stim_interval_ms': 500,
            'stim_duration_ms': 1,
            'block_off_duration': 10.0,
            'block_off_jitter': 5.0,
            'pause_between_phases': 180.0,
            'pause_between_blocks': 180.0,
            'prediction_block_order': ['FP', 'TP', 'FR', 'TR'],
            'n_blocks_on': 20,
            'stims_per_finger': 5,
        }

    def run_mapping(self):
        params = self.get_common()
        params.update({
            'run_type': 'mapping_only',
            'run_id': '00',
            'n_blocks_on': self.spin_mapping_blocks.value(),
            'stims_per_finger': self.spin_mapping_stims.value(),
        })
        self.parent_menu.run_experiment(params)

    def run_prediction(self):
        params = self.get_common()
        params.update({
            'run_type': 'prediction_only',
            'run_id': '01',
            'n_blocks_on': self.spin_pred_blocks.value(),
            'stims_per_finger': self.spin_pred_stims.value(),
        })
        self.parent_menu.run_experiment(params)

    def run_full(self):
        params = self.get_common()
        params.update({
            'run_type': 'full',
            'run_id': '00',
            'n_blocks_on': self.spin_full_blocks.value(),
            'stims_per_finger': self.spin_full_stims.value(),
        })
        self.parent_menu.run_experiment(params)