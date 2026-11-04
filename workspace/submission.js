'use strict';
document.getElementById('copy-pitch').addEventListener('click',async()=>{
 const status=document.getElementById('copy-status');
 try{await navigator.clipboard.writeText(document.getElementById('pitch-text').textContent);status.textContent='Copied.';}
 catch{status.textContent='Select the pitch text above to copy it.';}
});

// Both players are opt-in. Keep a closed draft or another film from
// continuing its audio while the visitor inspects the page.
const reviewFilms=['review-film','walkthrough-film'].map(id=>document.getElementById(id)).filter(Boolean);
for(const film of reviewFilms){
 film.addEventListener('play',()=>{
  for(const other of reviewFilms)if(other!==film)other.pause();
 });
}
const walkthroughDisclosure=document.getElementById('walkthrough');
walkthroughDisclosure?.addEventListener('toggle',()=>{
 if(!walkthroughDisclosure.open)document.getElementById('walkthrough-film')?.pause();
});
