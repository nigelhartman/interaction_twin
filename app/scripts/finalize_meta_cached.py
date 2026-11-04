#!/usr/bin/env python3
"""Compile a preserved paid Meta response with conservative scope omissions.

No model, credential, budget, network or geometry operation is performed.
Original failed runs stay unchanged; the derived proposal and program are new.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import stage5 as P
from meta_semantics import normalize_proposal
from pipeline import confined, label_generated_meta_viewer


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def recover(run_path, suffix):
    run_path = confined(run_path)
    run = P.read(run_path)
    phases = {p['name']: p for p in run['phases']}
    discovery = confined(ROOT/phases['discover']['result']['discovery'])
    annotations = confined(ROOT/phases['discover']['result']['annotations'])
    segment = confined(ROOT/phases['segment']['result']['directory'])
    case = confined(ROOT/'data'/run['artifacts']['case'])
    attempt = phases['procedure_proposal']['attempt']
    original = case/f'flow-procedure-proposal-{attempt}'
    record = P.read(original/'metadata.json')
    receipt = P.read(original/'meta-request.json')
    if (record.get('provider') != 'meta_api' or record.get('model') != 'muse-spark-1.3'
            or record.get('manual_semantic_story_used') is not False
            or receipt.get('status') != 'complete_accounted'
            or receipt.get('model') != record['model']):
        raise ValueError('A genuine completed, accounted Meta proposal is required')
    for key, path in [('source_sha256', case/'source/video.mp4'), ('discovery_sha256', discovery),
                      ('annotation_sha256', annotations), ('frame_index_sha256', case/'frames/index.json'),
                      ('prompt_sha256', original/'prompt.txt'),
                      ('schema_sha256', ROOT/'schemas/interaction-proposal.schema.json')]:
        if record[key] != P.sha(path):
            raise ValueError('Saved semantic proposal inputs changed: '+key)
    for image in record['input_images']:
        if P.sha(confined(ROOT/image['path'])) != image['sha256']:
            raise ValueError('Original model input image changed')
    raw = P.read(original/'response.pending.json')
    normalized, trace = normalize_proposal(raw, P.read(case/'frames/index.json')['frames'], P.read(annotations)['tracks'])
    proposal_out = case/f'flow-procedure-proposal-{suffix}'
    proposal_out.mkdir(exist_ok=False)
    shutil.copyfile(original/'response.pending.json', proposal_out/'raw-model-response.json')
    shutil.copyfile(original/'prompt.txt', proposal_out/'prompt.txt')
    write(proposal_out/'normalization-trace.json', trace)
    derived = copy.deepcopy(record)
    derived.pop('error', None)
    derived.update(status='complete_validated_model_proposal', result=normalized,
                   fresh_model_call_made=False,
                   original_paid_response={'path': P.relative(ROOT, original/'response.pending.json'),
                                           'sha256': P.sha(original/'response.pending.json'),
                                           'receipt_sha256': P.sha(original/'meta-request.json')},
                   normalization={'trace': 'normalization-trace.json',
                                  'trace_sha256': P.sha(proposal_out/'normalization-trace.json'),
                                  'code': 'meta_semantics.py', 'code_sha256': P.sha(ROOT/'meta_semantics.py'),
                                  'removed_count': trace['removed_count'], 'new_semantic_claims_added': False})
    write(proposal_out/'proposal.json', derived)
    write(proposal_out/'metadata.json', {k:v for k,v in derived.items() if k != 'result'})
    spec = importlib.util.spec_from_file_location('preserved_stage5_compiler', ROOT/'scripts/stage5_compile.py')
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    output = case/f'flow-procedure-{suffix}'
    args = SimpleNamespace(case=case, discovery=discovery, annotations=annotations,
                           proposal=proposal_out/'proposal.json', segmentation=segment, out=output,
                           model_metadata=[confined(ROOT/x) for x in run['artifacts'].get('models', [])],
                           motion=[confined(ROOT/x) for x in run['artifacts'].get('motion', [])],
                           scene_viewer=next((confined(ROOT/x) for x in run['artifacts'].get('scene_viewers', [])), None),
                           dense_viewer=None)
    compiler.build(args)
    program = P.read(output/'interaction.json')
    P.validate_interaction(program)
    label_generated_meta_viewer(output/'index.html', program)
    report = {'status': 'compiled_validated_cached_Meta_response_with_conservative_omission',
              'original_run': P.relative(ROOT, run_path), 'original_run_sha256': P.sha(run_path),
              'original_failure_preserved': True, 'new_model_calls': 0, 'new_api_cost_usd': '0.000000',
              'output': P.relative(ROOT, output/'interaction.json'), 'program_sha256': P.sha(output/'interaction.json'),
              'removed_unsupported_point_claims': trace['removed_count'],
              'steps': len(program['steps']), 'assertions': len(program['assertions']), 'assets': len(program['assets']),
              'physical_accuracy_verified': False}
    write(output/'recovery.json', report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--suffix', default='scope-normalized-v1')
    args = parser.parse_args()
    import re
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,50}', args.suffix):
        parser.error('Invalid derived output suffix')
    recover(args.run, args.suffix)


if __name__ == '__main__':
    main()
