"""Automatic geometric hypotheses are exact subsets, never accepted repairs."""
import unittest
from unittest.mock import patch
import numpy as np

import native_competing_slab_subset_v2 as H


def box(extents, count=18):
    vertices=[];faces=[]
    for axis in range(3):
        other=[i for i in range(3)if i!=axis]
        for side in [-1,1]:
            start=len(vertices)
            for a in np.linspace(-1,1,count):
                for b in np.linspace(-1,1,count):
                    point=np.zeros(3);point[axis]=side*extents[axis]/2
                    point[other[0]]=a*extents[other[0]]/2
                    point[other[1]]=b*extents[other[1]]/2;vertices.append(point)
            for i in range(count-1):
                for j in range(count-1):
                    a=start+i*count+j;b=a+count
                    faces.extend([[a,b,b+1],[a,b+1,a+1]])
    return np.asarray(vertices),np.asarray(faces)


def fixture():
    a,fa=box([.54,1.,.08]);b,fb=box([.54,.09,.96])
    return np.vstack([a,b]),np.vstack([fa,fb+len(a)])


def classify_with_fixture_pca_seeds(vertices,faces):
    # The production seed finder intentionally requires broad noisy density
    # peaks. Analytic zero-noise box faces have delta peaks, so isolate the
    # subsequent competing-surface classifier with automatically computed PCA
    # seeds, rather than weakening the production native seed eligibility gate.
    _,axes=np.linalg.eigh(np.cov(vertices.T))
    seeds=[{'id':f'planar_surface_{i}','selection':{'axis':axes[:,i].tolist()}}
           for i in range(3)]
    with patch.object(H,'surface_hypotheses',return_value=seeds):
        return H.hypotheses(vertices,faces)


class CompetingSlabTests(unittest.TestCase):
    def test_each_hypothesis_selects_only_original_faces_and_declares_missing_surfaces(self):
        v,f=fixture();rows=classify_with_fixture_pca_seeds(v,f)
        self.assertEqual(len(rows),2)
        for row in rows:
            ids=row['face_indices'];self.assertTrue(np.all(np.diff(ids)>0))
            self.assertGreater(len(ids),0);self.assertLess(len(ids),len(f))
            self.assertTrue(np.array_equal(f[ids],np.asarray([f[i]for i in ids])))
            self.assertTrue(row['geometry_displayshape_nonwatertight_possible'])
            self.assertFalse(row['physical_accuracy_verified'])
            self.assertFalse(row['selection']['manual_geometry_or_poses_used'])
            self.assertTrue(row['selection']['source_single_planar_rigid_object_premise_requires_visual_review'])

    def test_rigid_rotation_preserves_candidate_face_membership(self):
        v,f=fixture();before={tuple(r['face_indices'])for r in classify_with_fixture_pca_seeds(v,f)}
        axis=np.asarray([1.,2.,3.]);axis/=np.linalg.norm(axis);angle=.371
        skew=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        rotation=np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
        after={tuple(r['face_indices'])for r in classify_with_fixture_pca_seeds(v@rotation.T,f)}
        self.assertTrue(before==after,'Rigid rotation changed original face membership')

    def test_invalid_coordinates_fail_closed(self):
        v,f=fixture();v[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Finite original native vertices'):
            H.hypotheses(v,f)


if __name__=='__main__':unittest.main()
