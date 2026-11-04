"""Bounded, source-pinned Newton CPU contact experiment for generated components.

This is an assumed-parameter release test, not recovered dynamics, a grasp test
or physical validation. Native appearance assets are never edited. Preparation
uses the existing CV environment; execution uses an isolated Newton runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/robotics-physics-20261104-v1'
FORMAT = 'interaction-twin-newton-contact-experiment'
SPECS = [
    ('flow-sam3d-default-e2e-20261102-v1--flow-procedure-1', 'earbud_b', 'Earbud', .032, -.085),
    ('flow-rigid-parts-wallet-20261031-v1--flow-procedure-1', 'gold_coin', 'Coin', .026, 0.),
    ('flow-capture-charger-20261103-v1--flow-procedure-1', 'adapter_early', 'USB-C adapter', .052, .085),
]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def ref(path, root=ROOT):
    path = Path(path).resolve(); return {'path': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}

def prepare(output_path=None):
    import numpy as np
    from scipy.spatial import ConvexHull
    from curated_twin_catalog import CuratedTwinCatalog
    from native_appearance import load_native_glb
    catalog = CuratedTwinCatalog(ROOT)
    objects = []
    for twin, component, label, dimension, x in SPECS:
        record, _ = catalog._record(twin)
        asset = next(c['asset'] for c in record['components'] if c['id'] == component)
        if asset.get('backend')!='Meta SAM 3D Objects':
            native_reviews=[r for r in record.get('sam3d_reviews',[]) if r.get('component_id')==component and r.get('backend')=='Meta SAM 3D Objects' and r.get('source_sha256')==record['source']['sha256']]
            if len(native_reviews)!=1:raise ValueError('An exact curated native SAM 3D review is required')
            asset=native_reviews[0]['native_glb']
        asset_ref = {k:asset[k] for k in ('path', 'sha256', 'bytes')}
        path = ROOT/asset_ref['path']
        if sha(path) != asset_ref['sha256'] or path.stat().st_size != asset_ref['bytes']:
            raise ValueError('Curated native geometry changed')
        mesh = load_native_glb(path)
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        center, scale = (lo+hi)/2, dimension/max(hi-lo)
        v = (mesh.vertices-center)*scale
        # Preserve the native mesh facing: native/Three Y-up to Newton Z-up.
        nv = np.column_stack((v[:,0], -v[:,2], v[:,1]))
        # Deterministic support samples, followed by an oriented convex hull.
        n = 512; i=np.arange(n); z=1-2*(i+.5)/n; theta=i*math.pi*(3-math.sqrt(5))
        dirs=np.column_stack((np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z))
        selected=set()
        for batch in np.array_split(dirs, 16): selected.update(np.argmax(nv@batch.T,axis=0).tolist())
        points=nv[sorted(selected)]; hull=ConvexHull(points)
        faces=hull.simplices.copy()
        for index, face in enumerate(faces):
            a,b,c=points[face]
            if np.dot(np.cross(b-a,c-a),hull.equations[index,:3])<0:faces[index]=face[[0,2,1]]
        # Report support loss on 512 directions. This is a fitting diagnostic.
        loss=[]
        for batch in np.array_split(dirs,16):loss.extend((np.max(nv@batch.T,0)-np.max(points@batch.T,0)).tolist())
        objects.append({'id':component,'label':label,'twin_id':twin,'asset_ref':asset_ref,
            'source_program_ref':record['interaction'], 'native_center':center.tolist(),
            'native_to_assumed_metre_scale':float(scale),'assumed_max_dimension_m':dimension,
            'initial_pose_xyz_xyzw':[x,0.,.145,0.,0.,0.,1.],
            'collision_proxy':{'kind':'deterministic_convex_support_hull','vertices':points.tolist(),
                'triangles':faces.tolist(),'native_vertex_count':len(mesh.vertices),
                'support_direction_count':n,'maximum_sampled_support_loss_m':max(loss),
                'concavities_preserved':False},
            'parameters':{'density_kg_m3':600.,'friction':.65,'restitution':.08,
                'origin':'authored_assumptions_not_measured_from_video'}})
    document={'format':FORMAT,'version':1,'objects':objects,
        'coordinates':{'engine_up':'Z','native_up':'Y','native_to_engine':'[x,-z,y]'},
        'assumptions':{'gravity_m_s2':[0,0,-9.81],'release_height_m':.145,
            'timestep_s':1/600,'duration_s':4.,'display_fps':60,'solver_iterations':8},
        'claims':{'real_physics_engine_execution':False,'metric_scale_measured':False,
            'material_parameters_measured':False,'physical_accuracy_verified':False,
            'robot_grasp_validated':False,'simulation_ready':False}}
    output_path=Path(output_path or OUT/'input.json')
    write(output_path,document)
    print(json.dumps({'input':ref(output_path),'objects':len(objects)}))

def execute(input_path, output_path):
    import numpy as np
    import warp as wp
    import newton
    source=Path(input_path); doc=json.loads(source.read_text())
    if doc.get('format')!=FORMAT or doc.get('version')!=1 or not 1<=len(doc['objects'])<=8:
        raise ValueError('Unknown or oversized physics input')
    wp.config.kernel_cache_dir=str(Path(output_path).parent/'warp-kernel-cache')
    wp.init()
    started=time.perf_counter()
    with wp.ScopedDevice('cpu'):
        builder=newton.ModelBuilder()
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=.65,restitution=.08))
        for obj in doc['objects']:
            points=np.asarray(obj['collision_proxy']['vertices'],dtype=np.float32)
            triangles=np.asarray(obj['collision_proxy']['triangles'],dtype=np.int32)
            if points.ndim!=2 or points.shape[1]!=3 or len(points)>1024 or not np.isfinite(points).all():
                raise ValueError('Invalid collision hull')
            if triangles.ndim!=2 or triangles.shape[1]!=3 or len(triangles)>4096 or triangles.min()<0 or triangles.max()>=len(points):
                raise ValueError('Invalid collision triangles')
            p=obj['initial_pose_xyz_xyzw']; body=builder.add_body(xform=wp.transform(p=p[:3],q=p[3:]),label=obj['id'])
            hull=newton.Mesh(points,triangles.reshape(-1),compute_inertia=True,is_solid=True)
            params=obj['parameters']
            builder.add_shape_convex_hull(body,mesh=hull,cfg=newton.ModelBuilder.ShapeConfig(
                density=params['density_kg_m3'],mu=params['friction'],restitution=params['restitution'],margin=.00005,gap=.0003),label=obj['id']+'_collision_hull')
        model=builder.finalize(device='cpu')
        solver=newton.solvers.SolverXPBD(model,iterations=8,enable_restitution=True,angular_damping=.08)
        a,b=model.state(),model.state(); control=model.control()
        collision=newton.CollisionPipeline(model); contacts=collision.contacts()
        frames=[]; dt=1/600; contact_steps=0; max_contacts=0
        for step in range(2401):
            collision.collide(a,contacts)
            count=int(contacts.rigid_contact_count.numpy()[0]);contact_steps+=int(count>0);max_contacts=max(max_contacts,count)
            if step%10==0:
                contact_points=[]
                if count:
                    shapes=contacts.rigid_contact_shape0.numpy()[:count]
                    local_points=contacts.rigid_contact_point0.numpy()[:count]
                    poses=a.body_q.numpy();shape_bodies=model.shape_body.numpy()
                    for shape,point in zip(shapes,local_points):
                        body=int(shape_bodies[shape])
                        world=point if body<0 else np.asarray(list(wp.transform_point(wp.transform(*poses[body]),wp.vec3(*point))))
                        contact_points.append(world.astype(float).tolist())
                frames.append({'time_s':step*dt,'poses_xyz_xyzw':a.body_q.numpy().astype(float).tolist(),
                    'velocities':a.body_qd.numpy().astype(float).tolist(),'contact_count':count,
                    'contact_points_xyz':contact_points})
            if step<2400:
                a.clear_forces();solver.step(a,b,control,contacts,dt);a,b=b,a
        positions=np.asarray([f['poses_xyz_xyzw'] for f in frames]);speeds=np.asarray(frames[-1]['velocities'])
        if not np.isfinite(positions).all():raise ValueError('Non-finite Newton output')
        # Full transformed hull support, rather than checking body-centre height.
        minima=[]
        for fi, frame in enumerate(frames):
            for oi,obj in enumerate(doc['objects']):
                pose=frame['poses_xyz_xyzw'][oi]
                q=wp.quat(*pose[3:]);pv=np.asarray(obj['collision_proxy']['vertices'])
                vv=np.asarray([list(wp.quat_rotate(q,wp.vec3(*v))) for v in pv])+pose[:3]
                minima.append(float(vv[:,2].min()))
        report={'format':FORMAT,'version':1,'input_sha256':sha(source),'engine':{
            'name':'Newton','version':newton.__version__,'warp_version':wp.__version__,'device':'cpu',
            'solver':'SolverXPBD','iterations':8,'timestep_s':dt,'steps':2400,'gravity_m_s2':[0,0,-9.81]},
            'execution':{'elapsed_s':time.perf_counter()-started,'contact_steps':contact_steps,
                'maximum_contact_count':max_contacts,'minimum_collision_vertex_height_m':min(minima),
                'maximum_final_linear_speed_m_s':float(np.linalg.norm(speeds[:,:3],axis=1).max()),
                'maximum_final_angular_speed_rad_s':float(np.linalg.norm(speeds[:,3:],axis=1).max()),
                'finite_output':True},'objects':doc['objects'],'frames':frames,
            'assumptions':doc['assumptions'],'claims':{**doc['claims'],'real_physics_engine_execution':True},
            'limitations':['Assumed size, density, friction and restitution; not estimated from the source footage.',
                'Convex collision hulls fill native concavities and are separate from the unchanged appearance mesh.',
                'This demonstrates release, gravity and tabletop contact only; no grasp, insertion, robot policy or physical validation.']}
        write(output_path,report)
        print(json.dumps({'output':str(output_path),'engine':report['engine'],'execution':report['execution']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,default=OUT/'input.json');run=sub.add_parser('run');run.add_argument('--input',type=Path,default=OUT/'input.json');run.add_argument('--output',type=Path,default=OUT/'replay.json')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    else:execute(args.input,args.output)
