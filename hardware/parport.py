from psychopy import parallel, core

class DummyParPort:
    def __init__(self, *args, **kwargs):
        pass 

    def send_trigger(self, code, duration=0.03):
        pass 

    def reset(self):
        pass

# --- CLASSE PRINCIPALE (CONNEXION PHYSIQUE) ---
class ParPort:
    def __init__(self, address=0x378):
        """
        Initialise le port parallèle.
        """
        self.address = address
        self.port = None
        self.dummy_mode = False

        try:
            parallel.setPortAddress(address)
            self.port = parallel.ParallelPort(address)
            self.port.setData(0)
        except Exception as e:
            self.dummy_mode = True

    def send_trigger(self, code, duration=0.03):
        """
        Envoie un trigger (code) et remet à 0 après duration secondes.
        """
        if self.dummy_mode:
            return

        try:
            self.port.setData(int(code))
            core.wait(duration)
            self.port.setData(0)
        except Exception as e:
            print(f"Erreur envoi trigger {code}: {e}")

    def reset(self):
        """Force la remise à zéro des pins"""
        if not self.dummy_mode and self.port:
            self.port.setData(0)