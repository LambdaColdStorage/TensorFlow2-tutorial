import argparse
import sys
import os
import downloader

def main():
  sys.path.append('.')

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--data_url",
                      help="URL for downloading data",
                      default="")
  parser.add_argument("--data_dir",
                      help="Path for data folder",
                      default="")

  args = parser.parse_args()

  args.data_dir = os.path.expanduser(args.data_dir)
  downloader.download(args.data_url, args.data_dir)

if __name__ == "__main__":
  main()