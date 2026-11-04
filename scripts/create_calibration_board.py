#!/usr/bin/env python3
"""Create an explicit-size or grid-unit SVG checkerboard, never object geometry."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from calibration_core import board_svg
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--inner-rows',required=True,type=int);p.add_argument('--inner-cols',required=True,type=int);p.add_argument('--square-size-mm',type=float)
    a=p.parse_args();out=Path(a.output);out=out if out.is_absolute() else ROOT/out
    if out.exists():p.error('Choose a fresh output file')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:stream.write(board_svg({'inner_rows':a.inner_rows,'inner_cols':a.inner_cols,'square_size_mm':a.square_size_mm}))
    print(out)
