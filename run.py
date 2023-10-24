import argparse

from bomoto.engine import Engine


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
args = parser.parse_args()

engine = Engine(args.cfg)
engine.run()
