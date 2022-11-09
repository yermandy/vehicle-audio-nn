from src import *
import hydra


@hydra.main(version_base="1.2", config_path="config", config_name="default")
def run(config: Config):
    # print(config)
    print(os.getcwd())
    # exit()
    config = Config(config)
    # print(config)
    print(config)


run()
