// Exact saved OpenCV camera -> Three.js camera. All matrices are column-major.
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const matrix=m=>Array.isArray(m)&&m.length===3&&m.every(r=>Array.isArray(r)&&r.length===3&&r.every(finite));
export function sourceCameraMatrices(R,t,K,width,height,near=.001,far=10000){
 if(!matrix(R)||!matrix(K)||!Array.isArray(t)||t.length!==3||!t.every(finite)||![width,height,near,far].every(finite)||width<=0||height<=0||near<=0||far<=near||K[0][0]<=0||K[1][1]<=0)throw new Error('Invalid saved camera');
 const centre=[0,1,2].map(i=>-R[0][i]*t[0]-R[1][i]*t[1]-R[2][i]*t[2]);
 const world=[...R[0],0,...R[1].map(n=>-n),0,...R[2].map(n=>-n),0,...centre,1];
 const rows=[[2*K[0][0]/width,-2*K[0][1]/width,1-2*(K[0][2]+.5)/width,0],
             [0,2*K[1][1]/height,2*(K[1][2]+.5)/height-1,0],
             [0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]];
 return{world,projection:[0,1,2,3].flatMap(c=>rows.map(r=>r[c]))};
}
export function letterbox(width,height,imageWidth,imageHeight){
 if(![width,height,imageWidth,imageHeight].every(n=>finite(n)&&n>0))throw new Error('Invalid image rectangle');
 const scale=Math.min(width/imageWidth,height/imageHeight),w=imageWidth*scale,h=imageHeight*scale;
 return{x:(width-w)/2,y:(height-h)/2,width:w,height:h};
}
