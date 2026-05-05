# task_factory.py
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
            run_type=config.get("run_type", "somatotopy"),
            run_number=config.get("run_number", 1),
            burst_interval_ms=config.get("burst_interval_ms", 75.0),
        )

    else:
        print("Tâche inconnue.")
        return None