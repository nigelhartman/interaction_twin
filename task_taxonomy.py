"""Conservative public collection tags derived only from typed saved claims.

A tag describes a proposed task family, never geometry/physics acceptance.
Labels/free prose are intentionally not classified into capabilities.
"""
TASK_CLASSES = {
    'hinged_assemblies': 'Hinged assemblies',
    'sliding_mechanisms': 'Sliding mechanisms',
    'removable_components': 'Removable components',
    'connectors_and_insertion': 'Connectors and insertion attempts',
    'flexible_or_uncertain': 'Flexible or uncertain mechanisms',
}


def derive_task_classes(interaction, *, object_program=None, mechanism_proposals=None):
    result=set()
    typed=[]
    if object_program:
        typed += [j.get('type') for j in object_program.get('joints',[])]
        typed += [j.get('type') for j in object_program.get('unresolved_connections',[])]
        typed += [j.get('joint_type') for j in object_program.get('removable_relationships',[])]
    if mechanism_proposals:
        typed += [j.get('joint_type') for j in mechanism_proposals.get('joints',[])]
    if 'revolute' in typed:result.add('hinged_assemblies')
    if 'prismatic' in typed:result.add('sliding_mechanisms')
    if 'removable' in typed or any(r.get('predicate')=='removable_from' for r in interaction.get('relationships',[])):
        result.add('removable_components')
    if 'unknown' in typed or any(t.get('material_behavior')=='deformable' for t in interaction.get('tracklets',[])):
        result.add('flexible_or_uncertain')
    # Only an explicit typed future-adapter annotation can make this tag. A
    # container, touching parts or a movement toward a hole does not prove it.
    tags=interaction.get('task_tags',[])
    if any(t in {'connector_insertion_attempt','connector_removal_attempt'} for t in tags):
        result.add('connectors_and_insertion')
    return sorted(result)
