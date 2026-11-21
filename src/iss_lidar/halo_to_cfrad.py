# Use xradar to convert HALO ascii files to CFRadial2

import argparse
import xradar as xd
from iss_lidar.tools import process_inputs


def create_parser():
    parser = argparse.ArgumentParser(description="Generate CFRadial2 output "
                                     "from the source HALO ascii files")
    parser.add_argument("destdir", help="directory to save VAD files to")
    parser.add_argument("sources", help="HALO ascii file(s) for input",
                        nargs='+')
    return parser.parse_args()


def main():
    args = create_parser()
    sources = process_inputs(args.sources)
    for source in sources:
        data = xd.io.open_datatree(source, engine="hpl")
        destfile = f"{args.destdir}/{source.split('/')[-1].replace('.hpl', '.nc')}"
        xd.io.to_cfradial2(data, destfile)


if __name__ == "__main__":
    main()
