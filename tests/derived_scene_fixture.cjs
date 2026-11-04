/* Test transport boundary only: run real lineage validation on saved documents. */
const fs=require('node:fs'),path=require('node:path'),C=require('../showcase/aligned-scene-contract.js'),ROOT=path.resolve(__dirname,'..');
const read=reference=>JSON.parse(fs.readFileSync(path.join(ROOT,reference.path)));
module.exports=function verifySavedDerivedContext(row){
 if(!row?.derived_assembly)return row;
 const d=row.derived_assembly.document,base=read(d.base_object_program_ref);
 C.validateDerivedAssembly(d,row.manifest,{sceneReference:row.reference,baseProgramReference:row.base_object_program||d.base_object_program_ref,baseProgram:base,baseSceneReference:d.baseline_scene_ref||base.constrained_scene,baseScene:read(d.baseline_scene_ref||base.constrained_scene),parentDerivedAssembly:d.parent_derived_assembly_ref?read(d.parent_derived_assembly_ref):null,refinementReport:read(d.refinement_report_ref),sourceAudit:read(d.source_alignment_audit_ref)});
 return row;
};
