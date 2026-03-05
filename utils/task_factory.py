from tasks.connectelec import ConnectElec


def create_task(config, win):

    base_kwargs = {
        "win": win,
        "nom": config["nom"],
        "session": config["session"],
        "mode": config["mode"],
        "enregistrer": config["enregistrer"],
        "parport_actif": config["parport_actif"],
    }

    task_name = config["tache"]

    if task_name == "ConnectElec":

        return ConnectElec(
            **base_kwargs,

            # ── run selection ──
            run_type=config.get("run_type", "mapping"),
            run_number=config.get("run_number", 1),

            # ── mapping ──
            n_mapping_blocks=config.get("n_mapping_blocks", 20),
            mapping_off_jitter=config.get("mapping_off_jitter", 0.0),

            # ── prediction ──
            n_reps_per_condition=config.get("n_reps_per_condition", 5),

            # ── stimulation timing ──
            stims_per_finger=config.get("stims_per_finger", 5),
            stim_interval_ms=config.get("stim_interval_ms", 500.0),

            # ── block timing ──
            block_on_duration=config.get("block_on_duration", 10.0),
            block_off_duration=config.get("block_off_duration", 10.0),
            instruction_duration=config.get("instruction_duration", 5.0),
            instruction_jitter=config.get("instruction_jitter", 1.0),
            initial_baseline=config.get("initial_baseline", 10.0),
        )

    else:
        print("Tâche inconnue.")
        return None