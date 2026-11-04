#!/usr/bin/env python3
"""Explicit opt-in transfer-onlyV2; GPU execution uses unchanged existing V1 bridge."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sam3d_cluster_bridge import reconstruct
from sam3d_transfer_v2 import VerifiedTransferTransport

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff',required=True,type=Path);parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args()
    result=reconstruct(args.handoff,args.destination,root=ROOT,transport_factory=VerifiedTransferTransport)
    print(json.dumps(result,indent=2))
