import argparse
from senseflow.trainer.trainer_sd35_senseflow import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args, _ = parser.parse_known_args()
    trainer = Trainer(args.config_path, args.save_path)
    trainer.setup()
    trainer.train()

