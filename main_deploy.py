import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print(parser)
    # We pass in a explicit notebook arg so that we can provide an ordered list
    # and produce an ordered PDF.
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    parser.add_argument("--pdf_filename", type=str, required=True)
    args = parser.parse_args()
    print(args.notebooks)
