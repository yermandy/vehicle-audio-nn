from src import *

uuid = "045_RX100+iPhone/0"

config = load_config_locally(uuid)

trn_datapool = DataPool(config.training_files, config)
val_datapool = DataPool(config.validation_files, config)
tst_datapool = DataPool(config.testing_files, config)

validate_and_save(uuid, val_datapool, "val", Part.WHOLE, "rvce")
validate_and_save(uuid, val_datapool, "val", Part.WHOLE, "mae")

validate_and_save(uuid, trn_datapool, "trn", Part.WHOLE, "rvce")
validate_and_save(uuid, trn_datapool, "trn", Part.WHOLE, "mae")

validate_and_save(uuid, tst_datapool, "tst", Part.WHOLE, "rvce")
validate_and_save(uuid, tst_datapool, "tst", Part.WHOLE, "mae")
