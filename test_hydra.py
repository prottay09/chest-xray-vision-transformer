import hydra
from omegaconf import DictConfig, omegaconf

@hydra.main(config_name="config")
def train(
    cfg : DictConfig
    )->None:

    print(cfg.OUT_SIZE)

if __name__ == '__main__':
    train()
