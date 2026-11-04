(function () {
  'use strict';
  const collection = window.TwinTrailerCollection;
  const grid = document.getElementById('trailer-grid');
  const dialog = document.getElementById('trailer-dialog');
  const video = document.getElementById('collection-video');
  const error = document.getElementById('video-error');
  let selected = null, opener = null;
  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text) element.textContent = text;
    return element;
  }
  grid.replaceChildren();
  for (const film of collection.FILMS) {
    const card=node('article','trailer-card');card.dataset.format=film.format;
    const button=node('button');button.type='button';button.setAttribute('aria-label','Watch '+film.genre+': '+film.title);button.setAttribute('aria-haspopup','dialog');
    const thumb=node('div','trailer-thumb'),image=node('img');image.src=collection.media(film.id).poster;image.alt='';image.loading='lazy';image.width=640;image.height=400;
    thumb.append(image,node('span','trailer-number',film.number),node('span','trailer-play','▶'));
    const copy=node('div','trailer-card-copy'),topline=node('div','trailer-card-topline');
    topline.append(node('span','',film.genre),node('span','',film.duration_s+'s · '+(film.format==='portrait'?'9:16':film.format==='square'?'1:1':'16:9')));
    const footer=node('div','trailer-card-footer');footer.append(node('span','','Watch film'),node('span','','↗'));
    copy.append(topline,node('h2','',film.title),node('p','',film.description),footer);button.append(thumb,copy);card.append(button);grid.append(card);
    button.addEventListener('click',()=>open(film.id,button));
  }
  function updateURL(id) {const url=new URL(location.href);if(id)url.searchParams.set('cut',id);else url.searchParams.delete('cut');history.replaceState(null,'',url);}
  function select(id) {
    const film=collection.checkedFilm(id),paths=collection.media(id);selected=id;
    video.pause();video.removeAttribute('src');video.replaceChildren();error.hidden=true;
    dialog.dataset.format=film.format;
    document.getElementById('trailer-number').textContent=film.number+' / 06';
    for(const [key,value]of [['trailer-genre',film.genre],['trailer-title',film.title],['trailer-description',film.description],['trailer-audience',film.audience]])document.getElementById(key).textContent=value;
    const arc=document.getElementById('trailer-arc');arc.replaceChildren(...film.arc.map(text=>node('li','',text)));
    video.poster=paths.poster;video.src=paths.video;
    const track=node('track');track.kind='captions';track.srclang='en';track.label='English';track.src=paths.captions;video.append(track);video.load();
    document.getElementById('download-trailer').href=paths.video;
    document.getElementById('trailer-transcript').href=paths.transcript;
    updateURL(id);
  }
  function open(id,button) {opener=button||opener;select(id);if(!dialog.open)dialog.showModal();document.getElementById('close-trailer').focus();}
  video.addEventListener('error',()=>{error.hidden=false;});
  document.getElementById('close-trailer').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',event=>{if(event.target!==dialog)return;const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();});
  dialog.addEventListener('close',()=>{video.pause();video.removeAttribute('src');video.replaceChildren();video.load();selected=null;updateURL(null);opener?.focus();});
  document.getElementById('previous-trailer').addEventListener('click',()=>select(collection.adjacent(selected,-1).id));
  document.getElementById('next-trailer').addEventListener('click',()=>select(collection.adjacent(selected,1).id));
  for(const button of document.querySelectorAll('[data-format]'))if(button.tagName==='BUTTON')button.addEventListener('click',()=>{
    const films=collection.filtered(button.dataset.format),ids=new Set(films.map(f=>f.id));
    for(const item of document.querySelectorAll('.trailer-filters button'))item.setAttribute('aria-pressed',String(item===button));
    for(const [index,card]of [...grid.children].entries())card.hidden=!ids.has(collection.FILMS[index].id);
    document.getElementById('film-count').textContent=films.length+' '+(films.length===1?'film':'films')+' · '+(button.dataset.format==='all'?'every perspective':button.textContent.trim());
  });
  window.addEventListener('pagehide',()=>video.pause(),{once:true});
  const requested=new URL(location.href).searchParams.get('cut');if(collection.FILMS.some(f=>f.id===requested))open(requested);
})();
