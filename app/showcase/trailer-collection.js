/* Six editorial perspectives on the same saved project. No inference or uploads. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TwinTrailerCollection = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const BASE = 'media/trailer-collection-20261104-v1/';
  const FILMS = [
    {id:'origin', duration_s:43, number:'01', stem:'01-origin', title:'From a moment to a twin.', genre:'The origin story', format:'landscape', audience:'A first introduction', description:'A real demonstration becomes separate parts, connected motion and something another builder can use.', arc:['Watch the interaction','Understand its parts','Keep how it works']},
    {id:'blueprint', duration_s:34, number:'02', stem:'02-blueprint', title:'More than a mesh.', genre:'The technical blueprint', format:'landscape', audience:'Builders and technical judges', description:'Go inside the object program: shape, connections, actions and the evidence behind them.', arc:['Look beneath the surface','Connect shape and meaning','Make it reusable']},
    {id:'mission', duration_s:39, number:'03', stem:'03-mission', title:'Give the robot a task.', genre:'The robot mission', format:'landscape', audience:'Robotics and the use case', description:'Start with a goal. Follow a complete robot rehearsal, then discover the demonstration that made the twin possible.', arc:['Set the goal','Follow the whole task','Reveal the source']},
    {id:'proof', duration_s:30, number:'04', stem:'04-proof', title:'Reality. Replay. Reuse.', genre:'The continuous comparison', format:'landscape', audience:'Seeing the result directly', description:'The real interaction and its connected reconstruction share the screen. Follow what the twin keeps from the original.', arc:['See both worlds','Follow connected motion','Reuse the description']},
    {id:'pocket-worlds', duration_s:26, number:'05', stem:'05-pocket-worlds', title:'Small objects. Bigger possibilities.', genre:'The vertical social cut', format:'portrait', audience:'A quick mobile introduction', description:'Everyday objects become a growing library of parts, connections and actions. A compact story made for the phone.', arc:['Everyday objects','Reusable interactions','A library for builders']},
    {id:'agent', duration_s:38, number:'06', stem:'06-agent', title:'An agent needs more than shape.', genre:'The agent story', format:'square', audience:'Agents, MCP and reuse', description:'An illustrated agent workflow connects a task to the library, a descriptive program and an inspectable robot rehearsal.', arc:['Ask for an interaction','Find the object program','Inspect the task']}
  ];
  const FORMATS = new Set(['all','landscape','portrait','square']);
  function checkedFilm(id) {
    const film = FILMS.find(item => item.id === id);
    if (!film) throw new Error('Unknown trailer');
    return film;
  }
  function media(id) {
    const film = checkedFilm(id);
    return {video:BASE+film.stem+'.mp4',poster:BASE+film.stem+'-poster.jpg',captions:BASE+film.stem+'.vtt',transcript:BASE+film.stem+'-transcript.txt'};
  }
  function filtered(format) {
    if (!FORMATS.has(format)) throw new Error('Unknown film format');
    return FILMS.filter(film => format === 'all' || film.format === format);
  }
  function adjacent(id, direction) {
    if (direction !== -1 && direction !== 1) throw new Error('Invalid film direction');
    checkedFilm(id);
    return FILMS[(FILMS.findIndex(film=>film.id===id)+direction+FILMS.length)%FILMS.length];
  }
  return {FILMS,media,filtered,adjacent,checkedFilm};
});
