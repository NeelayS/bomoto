import argparse

from bomoto.config import get_cfg
from bomoto.engine import Engine


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
args = parser.parse_args()

cfg = get_cfg(args.cfg)

engine = Engine(cfg)
engine.run()
