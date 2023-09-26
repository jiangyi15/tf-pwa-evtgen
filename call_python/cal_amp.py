
from tf_pwa.config_loader import ConfigLoader


config = ConigLoader("config.yml")
config.set_params("a_params.json")

def do_weight(p4):
    amp = config.eval_amplitude(p4)
    return float(amp)
