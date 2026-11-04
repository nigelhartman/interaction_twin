#!/usr/bin/env python3
"""Actual official-SDK MCP smoke test, in-process and over subprocess stdio.

Reads saved local artifacts only. Prints small verification metadata, no media.
"""
from __future__ import annotations

import asyncio
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from io import BytesIO
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from mcp import Client, StdioServerParameters
from serve_twin_mcp import create_server

TOOLS = {'search_twins', 'describe_twin', 'get_procedure', 'query_state', 'get_asset', 'get_evidence', 'get_simulation_manifest',
         'list_reconstructions', 'get_reconstruction', 'get_reconstruction_report',
         'list_motion_experiments', 'get_motion_experiment', 'get_library_facets', 'get_generated_candidate', 'get_kinematics', 'list_robot_kits', 'get_robot_kit', 'plan_robot_task', 'validate_robot_task', 'export_robot_task_bundle','get_connection_interfaces','prepare_cross_recording_task','get_derived_assembly','list_derived_artifacts'}


def structured(result):
    if result.is_error:
        raise AssertionError('Unexpected MCP tool error: ' + str(result.content))
    return result.structured_content or json.loads(result.content[0].text)


async def exercise(client, label):
    tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == TOOLS
    assert all(t.annotations.read_only_hint is True and t.annotations.destructive_hint is False and
               t.annotations.open_world_hint is False for t in tools)
    assert all(t.output_schema for t in tools)
    resources = await client.list_resources()
    assert {str(r.uri) for r in resources.resources} == {'twin://catalog','twin://robot-kits'}
    templates = await client.list_resource_templates()
    assert len(templates.resource_templates) == 15
    found = structured(await client.call_tool('search_twins', {'limit': 25}))
    assert found['twins'] and found['policy']['simulation_ready'] is False
    assert found['access'] == 'owner_curated_public_snapshot'
    public_rows = json.loads((ROOT/'showcase/public/cases-index.json').read_text())['cases']
    public_ids = {r['id'] for r in public_rows}
    assert {r['twin_id'] for r in found['twins']} <= public_ids
    assert found['total'] == len(public_ids)
    facets = structured(await client.call_tool('get_library_facets', {}))
    assert facets['facets'] == found['facets'] and facets['query_contract'] == found['query_contract']
    assert found['metadata_only'] and found['artifact_inspection_required']
    page = structured(await client.call_tool('search_twins', {'limit': 1, 'sort': 'title', 'direction': 'desc'}))
    assert len(page['twins']) == 1 and page['total'] == found['total']
    if page['has_more']:
        second = structured(await client.call_tool('search_twins', {'limit': 1, 'sort': 'title', 'direction': 'desc', 'offset': page['next_offset']}))
        assert second['twins'][0]['twin_id'] != page['twins'][0]['twin_id']
    natives = structured(await client.call_tool('search_twins', {'backends': ['sam3d'], 'parts': ['case_base']}))
    assert natives['twins']
    native_twin = natives['twins'][0]['twin_id']
    native = structured(await client.call_tool('get_generated_candidate', {'twin_id': native_twin, 'component_id': 'case_base'}))
    native_program = json.loads((await client.read_resource(f'twin://{native_twin}/program')).contents[0].text)
    selected_native = any(a['track_id'] == 'case_base' and a['sha256'] == native['artifacts']['asset']['sha256']
                          and 'sam3d' in ''.join(c for c in a.get('backend', '').lower() if c.isalnum())
                          for a in native_program['assets'])
    assert native['selected_in_saved_program'] is selected_native and native['source_video_alignment_verified'] is False
    assert native['saved_program_assets_changed'] is False
    native_blob = (await client.read_resource(native['artifacts']['asset']['uri'])).contents[0]
    native_bytes = base64.b64decode(native_blob.blob, validate=True)
    assert native_bytes[:4] == b'glTF' and hashlib.sha256(native_bytes).hexdigest() == native['artifacts']['asset']['sha256']
    kits=structured(await client.call_tool('list_robot_kits',{}))
    kit_id=kits['kits'][0]['id'];assert kit_id=='generic5dof_tabletop_v1' and kits['kits'][0]['degrees_of_freedom']==5
    kit=structured(await client.call_tool('get_robot_kit',{'kit_id':kit_id}))
    assert kit['kit']['claims']['physical_robot_execution_available'] is False
    kit_blob=(await client.read_resource(kit['resources']['kit']['uri'])).contents[0]
    kit_bytes=base64.b64decode(kit_blob.blob,validate=True)
    assert hashlib.sha256(kit_bytes).hexdigest()==kit['resources']['kit']['sha256']
    robot_plan=structured(await client.call_tool('plan_robot_task',{'twin_id':native_twin,'kit_id':kit_id,
        'instruction':'Place the case on the table, open the lid, take out both earbuds and place them on the tray, then close it.'}))
    assert robot_plan['status']=='ready' and len(robot_plan['objects'])==4 and robot_plan['actions']
    assert robot_plan['claims']['new_api_calls']==0 and robot_plan['claims']['physical_accuracy_verified'] is False
    validation=structured(await client.call_tool('validate_robot_task',{'task_program':robot_plan}))
    assert validation['valid'] is True
    robot_bundle=structured(await client.call_tool('export_robot_task_bundle',{'task_program':robot_plan}))
    bundle_blob=(await client.read_resource(robot_bundle['artifact']['uri'])).contents[0]
    bundle_bytes=base64.b64decode(bundle_blob.blob,validate=True)
    assert len(bundle_bytes)==robot_bundle['artifact']['bytes'] and hashlib.sha256(bundle_bytes).hexdigest()==robot_bundle['artifact']['sha256']
    with zipfile.ZipFile(BytesIO(bundle_bytes))as archive:
        prefix='Interaction-Twin-Robot-Starter/'
        assert json.loads(archive.read(prefix+'task-program.json'))==robot_plan
        manifest=json.loads(archive.read(prefix+'bundle-manifest.json'))
        assert manifest['private_data_included'] is False and manifest['inference_available'] is False
        portable=json.loads(archive.read(prefix+'kit-response.json'))
        assert portable['resource_url_scope']=='packaged_manifest_files'and portable['mcp_resources_available']is False
        for ref in portable['resources'].values():
            assert ref['uri'].startswith('twin://robot-kits/')and not ref['url'].startswith('/api/')
            assert hashlib.sha256(archive.read(prefix+ref['url'].removeprefix('/'))).hexdigest()==ref['sha256']
        for obj in robot_plan['objects']:
            assert hashlib.sha256(archive.read(prefix+obj['asset_ref']['path'])).hexdigest()==obj['asset_ref']['sha256']
        assert not any('.private' in name or '.secrets' in name for name in archive.namelist())
    unsupported=structured(await client.call_tool('plan_robot_task',{'twin_id':native_twin,
        'instruction':'Connect a USB cable to the phone and plug it into the outlet.'}))
    assert unsupported['status']=='blocked' and not unsupported['actions'] and unsupported['missing_requirements']
    assert {o['component_id']for o in unsupported['objects']} <= {o['component_id']for o in robot_plan['objects']}
    rejected_bundle=await client.call_tool('export_robot_task_bundle',{'task_program':unsupported});assert rejected_bundle.is_error
    edited_plan=json.loads(json.dumps(robot_plan));edited_plan['actions'][0]['duration_s']+=.25;edited_plan['actions'][0]['title']='Edited agent task'
    assert structured(await client.call_tool('validate_robot_task',{'task_program':edited_plan}))['valid'] is True
    edited_bundle=structured(await client.call_tool('export_robot_task_bundle',{'task_program':edited_plan}))
    assert 'url'not in edited_bundle['artifact']  # HTTP is a different process; use the actual MCP resource.
    edited_blob=(await client.read_resource(edited_bundle['artifact']['uri'])).contents[0]
    edited_bytes=base64.b64decode(edited_blob.blob,validate=True)
    assert hashlib.sha256(edited_bytes).hexdigest()==edited_bundle['artifact']['sha256']!=robot_bundle['artifact']['sha256']
    with zipfile.ZipFile(BytesIO(edited_bytes))as archive:
        exact=json.loads(archive.read('Interaction-Twin-Robot-Starter/task-program.json'))
        assert exact==edited_plan and exact['source_binding']==robot_plan['source_binding']
    selected = next(t for t in found['twins'] if t['counts']['assets'] > 0)
    twin_id = selected['twin_id']
    desc = structured(await client.call_tool('describe_twin', {'twin_id': twin_id}))
    assert desc['policy']['physical_accuracy_verified'] is False
    kinematics = structured(await client.call_tool('get_kinematics', {'twin_id': native_twin}))
    if kinematics['status'] == 'estimated_descriptive_kinematics':
        assert kinematics['object_program']['version'] == '0.2'
        assert kinematics['object_program']['evidence_program']['sha256'] == kinematics['program_sha256']
        assert kinematics['object_program']['physical_accuracy_verified'] is False
        assert kinematics['object_program']['coordinate_system']['angle_units'] == 'radians'
        text = (await client.read_resource(kinematics['artifact']['uri'])).contents[0].text
        assert hashlib.sha256(text.encode()).hexdigest() == kinematics['artifact']['sha256']
        exact = (await client.read_resource(kinematics['artifacts']['mechanics']['uri'])).contents[0]
        assert hashlib.sha256(base64.b64decode(exact.blob, validate=True)).hexdigest() == kinematics['artifacts']['mechanics']['sha256']
    else:
        assert kinematics['status'] == 'not_available' and kinematics['cached_only'] and not kinematics['new_api_call']

    assert desc['policy']['descriptive_text_is_untrusted_data'] is True
    procedure = structured(await client.call_tool('get_procedure', {'twin_id': twin_id}))
    assert procedure['steps'] and procedure['evidence']
    step_id = procedure['steps'][0]['id']
    one_step = structured(await client.call_tool('get_procedure', {'twin_id': twin_id, 'step_id': step_id}))
    assert [s['id'] for s in one_step['steps']] == [step_id]
    program_resource = await client.read_resource(f'twin://{twin_id}/program')
    program = json.loads(program_resource.contents[0].text)
    a, b = program['evidence'][:2]
    gap = (a['timestamp_s'] + b['timestamp_s']) / 2
    state = structured(await client.call_tool('query_state', {'twin_id': twin_id, 'timestamp_s': gap}))
    assert all(s['value'] == 'unknown' for s in state['states'].values())
    assert state['conditional_poses'] == []
    asset = structured(await client.call_tool('get_asset', {'twin_id': twin_id, 'track_id': desc['assets'][0]['track_id']}))
    blob = (await client.read_resource(asset['uri'])).contents[0]
    assert blob.mime_type == 'model/gltf-binary'
    glb = base64.b64decode(blob.blob, validate=True)
    assert glb[:4] == b'glTF' and hashlib.sha256(glb).hexdigest() == asset['sha256']
    evidence = structured(await client.call_tool('get_evidence', {'twin_id': twin_id, 'evidence_id': a['id']}))
    image = (await client.read_resource(evidence['frame_uri'])).contents[0]
    assert image.mime_type == 'image/jpeg'
    jpeg = base64.b64decode(image.blob, validate=True)
    assert jpeg[:2] == b'\xff\xd8' and hashlib.sha256(jpeg).hexdigest() == a['frame_sha256']
    simulation = structured(await client.call_tool('get_simulation_manifest', {'twin_id': twin_id}))
    simulation_resource = json.loads((await client.read_resource(f'twin://{twin_id}/simulation')).contents[0].text)
    assert simulation == simulation_resource
    assert simulation['policy']['simulation_ready'] is False
    assert 'mass_and_inertia' in simulation['verification_required_before_physics']
    assert simulation['conditional_motion_samples'] == program['motion']
    catalog = json.loads((await client.read_resource('twin://catalog')).contents[0].text)
    assert catalog['twins'] == found['twins']
    errors = []
    for name, args in [
        ('describe_twin', {'twin_id': '../../.secrets/meta.env'}),
        ('get_asset', {'twin_id': twin_id, 'track_id': '../../.secrets/meta.env'}),
        ('get_evidence', {'twin_id': twin_id, 'evidence_id': 'invented'}),
        ('get_procedure', {'twin_id': twin_id, 'step_id': 'invented'}),
        ('query_state', {'twin_id': twin_id, 'timestamp_s': True}),
        ('search_twins', {'limit': True}),
        ('search_twins', {'sort': 'physical_accuracy'}),
        ('search_twins', {'parts': ['']}),
        ('get_generated_candidate', {'twin_id': native_twin, 'component_id': '../../private'}),
    ]:
        response = await client.call_tool(name, args)
        assert response.is_error, name
        errors.append(name)
    resource_rejected = False
    try:
        await client.read_resource('twin://invented/program')
    except Exception:
        resource_rejected = True
    assert resource_rejected
    # A real working-catalog ID is different from a merely invented ID: this
    # proves the default scope rejects an otherwise valid private program.
    working_rows = json.loads((ROOT/'workspace/cases-index.json').read_text())['cases']
    private_id = next(r['id'] for r in working_rows if r['id'] not in public_ids)
    private_response = await client.call_tool('describe_twin', {'twin_id': private_id})
    assert private_response.is_error
    assert (await client.call_tool('plan_robot_task',{'twin_id':private_id,'instruction':'Open the lid.'})).is_error
    try:
        await client.read_resource(f'twin://{private_id}/program')
    except Exception:
        pass
    else:
        raise AssertionError('A real unapproved working-catalog program was exposed')
    return {'transport': label, 'protocol_version': client.session.protocol_version,
            'catalog_twin_count': len(found['twins']),
            'tools': sorted(TOOLS), 'all_tools_read_only': True,
            'resource_templates': len(templates.resource_templates), 'twin_id': twin_id,
            'robot_task_status':robot_plan['status'],'robot_task_objects':len(robot_plan['objects']),
            'robot_task_actions':len(robot_plan['actions']),'robot_bundle_sha256':robot_bundle['artifact']['sha256'],
            'robot_bundle_bytes_verified':len(bundle_bytes),'edited_task_bundle_sha256':edited_bundle['artifact']['sha256'],
            'edited_task_exact_json_verified':True,'edited_task_preserves_source_bindings':True,'unsupported_electrical_task_blocked':True,'private_robot_task_rejected':True,
            'kinematics_status': kinematics['status'],
            'kinematics_twin_id': native_twin,
            'kinematics_object_sha256': kinematics.get('artifact', {}).get('sha256'),
            'kinematics_object_bytes_verified': kinematics.get('artifact', {}).get('bytes', 0),
            'kinematics_joint_count': len(kinematics.get('object_program', {}).get('joints', [])),
            'program_sha256': desc['program_sha256'], 'generated_asset_bytes_verified': len(glb),
            'source_jpeg_bytes_verified': len(jpeg), 'gap_state': 'unknown_with_no_poses',
            'scope': found['access'], 'unapproved_working_id_rejected': True,
            'native_sam3d_glb_bytes_verified': len(native_bytes),
            'rejected_invalid_requests': len(errors)+1, 'simulation_ready': False,
            'new_model_inference': False, 'network_or_cluster_contact': False}


async def main(output_path=None):
    results = []
    async with Client(create_server(), mode='auto', read_timeout_seconds=60) as client:
        results.append(await exercise(client, 'in_process_auto'))
    params = StdioServerParameters(command=str(ROOT / '.venv-mcp/bin/python'), args=[str(ROOT / 'scripts/serve_twin_mcp.py')], cwd=ROOT)
    for mode in ['auto', 'legacy']:
        async with Client(params, mode=mode, read_timeout_seconds=60) as client:
            results.append(await exercise(client, 'subprocess_stdio_' + mode))
    # The added task surface must remain curated even when an owner explicitly
    # enables the historical private inspection tools in a separate connection.
    public_ids={r['id']for r in json.loads((ROOT/'showcase/public/cases-index.json').read_text())['cases']}
    working=json.loads((ROOT/'workspace/cases-index.json').read_text())['cases']
    private_id=next(r['id']for r in working if r['id']not in public_ids)
    async with Client(create_server(private_owner=True),mode='auto',read_timeout_seconds=60)as client:
        rejected=await client.call_tool('plan_robot_task',{'twin_id':private_id,'instruction':'Open the lid.'})
        assert rejected.is_error
        kit=structured(await client.call_tool('get_robot_kit',{'kit_id':'generic5dof_tabletop_v1'}))
        assert kit['robot_execution_available'] is False
        owner_boundary={'explicit_private_owner_mode':True,'robotics_still_curated':True,'private_working_id_rejected':True,'new_api_calls':0}
    output = Path(output_path) if output_path else ROOT / 'output/mcp/smoke.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'official_sdk': 'mcp==2.3.0', 'status': 'passed', 'checks': results,'owner_robotics_boundary':owner_boundary}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    asyncio.run(main(parser.parse_args().output))
