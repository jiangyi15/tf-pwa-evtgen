from tf_pwa.config_loader import ConfigLoader
# import the model used
from tf_pwa.amp import interpolation

# create config
config = ConfigLoader("config.yml")
config.set_params("a_params.json")

# save model into SavedModel format
config.save_tensorflow_model("model2")
