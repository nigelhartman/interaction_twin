"""Small deterministic CPU mesh previews; geometry and colours are not invented."""
import numpy as np
from PIL import Image,ImageDraw

def render_mesh(mesh,yaw=0,elevation=.25,size=(500,500),background='#edf2f5'):
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);center=(vertices.min(0)+vertices.max(0))/2
    c,s=np.cos(yaw),np.sin(yaw);ce,se=np.cos(elevation),np.sin(elevation)
    rotation=np.array([[c,s,0],[se*s,-se*c,ce],[-ce*s,ce*c,se]])
    points=(vertices-center)@rotation.T
    scale=min(size)*.82/(2*np.linalg.norm(vertices-center,axis=1).max())
    projected=points[:,:2]*np.array([scale,-scale])+np.array(size)/2
    normals=np.asarray(mesh.face_normals)@rotation.T;light=np.array([-.3,.5,1.]);light/=np.linalg.norm(light)
    shade=.42+.58*np.clip(normals@light,0,1)
    colors=np.asarray(mesh.visual.vertex_colors)[:,:3][faces].mean(1)
    image=Image.new('RGB',size,background);draw=ImageDraw.Draw(image)
    for i in np.argsort(points[faces,2].mean(1)):
        color=tuple(np.clip(colors[i]*shade[i],0,255).astype(int));draw.polygon([tuple(v) for v in projected[faces[i]]],fill=color)
    return image
