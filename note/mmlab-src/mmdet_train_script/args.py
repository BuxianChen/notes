import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--abc")

args = parser.parse_args()

print(args)