from tasks.connectelec import ConnectElec

def create_task(config, win):
    base_kwargs = {
        'win': win,
        'nom': config['nom'],
        'enregistrer': config['enregistrer'],
        'screenid': config['screenid'],
        'parport_actif': config['parport_actif'],
        'mode': config['mode'],
        'session': config['session'],
    }

    task_config = config['tache']

    if task_config == 'ConnectElec':
        return ConnectElec(
            **base_kwargs,  
            n_blocks_on=config['n_blocks_on'],
            stims_per_finger=config['stims_per_finger'],
            stim_interval_ms=config.get('stim_interval_ms', 500),
            stim_duration_ms=config.get('stim_duration_ms', 1),
            block_off_duration=config.get('block_off_duration', 10.0),
            block_off_jitter=config.get('block_off_jitter', 5.0),
            pause_between_phases=config.get('pause_between_phases', 180.0),
            pause_between_blocks=config.get('pause_between_blocks', 180.0),
            prediction_block_order=config.get('prediction_block_order', None),
            run_type=config['run_type']
        )

    else:
        print("Tâche inconnue.")
        return None