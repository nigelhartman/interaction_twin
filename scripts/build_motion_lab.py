#!/usr/bin/env python3
"""Publish read-only viewer derivatives of explicit real interval-motion outputs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_lab import normalize,sha


def build(inputs,root=ROOT,feedback=None,density=()):
    root=Path(root).resolve();base=root/'data/motion-lab-20261031';base.mkdir(parents=True,exist_ok=True)
    rows=[]
    results=[normalize(root,source,feedback)for source in inputs]
    if density:
        from motion_density_lab import normalize as normalize_density
        results.extend(normalize_density(root,source)for source in density)
    for result in results:
        directory=base/result['id'];directory.mkdir(exist_ok=True)
        output=directory/'viewer.json';raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
        if output.exists()and output.read_bytes()!=raw:raise ValueError('Preserve existing derivative; version adapter changes explicitly')
        if not output.exists():output.write_bytes(raw)
        rows.append({k:result[k]for k in ['id','case_id','component_id','label','status','comparison_counts']})
        rows[-1].update(viewer_url='/files/'+str(output.relative_to(root)),viewer_sha256=sha(output))
    index={'format':'interval-motion-lab-index','version':1,'runs':rows,'new_inference':False,
           'policy':'Separate interval branches; exact samples only; no interpolation or physical pose certification.'}
    (base/'index.json').write_text(json.dumps(index,indent=2)+'\n');return index


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',action='append',default=[])
    parser.add_argument('--density-experiment',action='append',default=[],help='Explicit controlled source-extension comparison')
    parser.add_argument('--feedback',help='Explicit separately saved source-bound capture coach report')
    args=parser.parse_args()
    if not args.experiment and not args.density_experiment:parser.error('Choose an actual saved experiment')
    result=build(args.experiment,feedback=args.feedback,density=args.density_experiment);print(json.dumps({'runs':len(result['runs']),'index_url':'/files/data/motion-lab-20261031/index.json'}))
