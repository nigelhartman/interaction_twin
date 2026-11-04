"""Private, hash-bound soundtrack imports. No third-party downloader or API.

Licence metadata is an owner's acknowledgement, not a legal certification.
Only declared audio IDs reach the renderer; filenames/URLs never become commands.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
import re
import secrets
import subprocess
from urllib.parse import urlsplit
from agent_memory import private_directory,write_immutable,digest

ROOT=Path(__file__).resolve().parent
MAX_UPLOAD=25*1024*1024
MAX_AUDIO=180*1024*1024
MAX_DURATION=900
ALLOWED_FORMATS={'wav','mp3','mov','flac','ogg'}
ORIGINAL='output/trailer-20261102/v4/original-instrumental-only.wav'
ID=re.compile(r'(?:music|audio)_[a-f0-9]{64}\Z')


def ref(path,root):
    if path.resolve()!=path or not path.is_relative_to(root)or any(p.is_symlink()for p in [path,*path.parents])or not path.is_file():raise ValueError('Ordinary confined audio file required')
    if not 0<path.stat().st_size<=MAX_AUDIO:raise ValueError('Bounded audio bytes required')
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return {'path':path.relative_to(root).as_posix(),'sha256':h.hexdigest(),'bytes':path.stat().st_size}


def _mov_self_contained(path):
    """Reject external QuickTime/MP4 data-reference entries before any decoder."""
    if path.stat().st_size>MAX_UPLOAD:raise ValueError('Bounded source container required')
    raw=path.read_bytes()
    def boxes(start,end,depth=0):
        if depth>8:raise ValueError('Bounded audio container nesting required')
        result=[];at=start
        while at<end:
            if end-at<8:raise ValueError('Incomplete audio container box')
            size,kind=struct.unpack_from('>I4s',raw,at);header=8
            if size==1:
                if end-at<16:raise ValueError('Incomplete large audio box')
                size=struct.unpack_from('>Q',raw,at+8)[0];header=16
            if size==0:size=end-at
            if size<header or at+size>end:raise ValueError('Invalid audio container box')
            result.append((kind,at+header,at+size));at+=size
            if len(result)>10000:raise ValueError('Bounded audio box count required')
        return result
    def walk(start,end,depth=0):
        for kind,body,limit in boxes(start,end,depth):
            if kind in {b'moov',b'trak',b'mdia',b'minf',b'dinf'}:walk(body,limit,depth+1)
            elif kind==b'dref':
                if limit-body<8:raise ValueError('Invalid audio data references')
                count=struct.unpack_from('>I',raw,body+4)[0];entries=boxes(body+8,limit,depth+1)
                if count!=len(entries)or not 1<=count<=32:raise ValueError('Bounded internal audio data references required')
                for typ,payload,last in entries:
                    if typ!=b'url 'or last-payload<4 or struct.unpack_from('>I',raw,payload)[0]&0xffffff!=1:
                        raise ValueError('External audio data references are not accepted')
    walk(0,len(raw))


def sniff_audio(path):
    """Actual bytes determine a fixed self-contained demuxer, never an extension."""
    if path.is_symlink()or not path.is_file()or not 0<path.stat().st_size<=MAX_AUDIO:raise ValueError('Bounded ordinary audio required')
    with path.open('rb')as stream:head=stream.read(64)
    if head[:4]==b'RIFF'and head[8:12]==b'WAVE':return 'wav'
    if head.startswith(b'fLaC'):return 'flac'
    if head.startswith(b'OggS'):return 'ogg'
    if head.startswith(b'ID3')or(len(head)>3 and head[0]==255 and head[1]&0xe0==0xe0 and head[1]&6):return 'mp3'
    if len(head)>=12 and head[4:8]==b'ftyp':_mov_self_contained(path);return 'mov'
    raise ValueError('Choose a self-contained WAV, MP3, M4A, FLAC or OGG file; playlists are not accepted')


def _input(path,fmt):
    # Format forcing blocks playlist masquerading. MOV's explicit data-reference
    # switches plus our dref parser forbid files outside this one container.
    return ['-protocol_whitelist','file','-format_whitelist',fmt,'-f',fmt]+(['-enable_drefs','0','-use_absolute_path','0']if fmt=='mov'else[])+['-i',str(path)]


def probe_audio(path):
    fmt=sniff_audio(path)
    result=subprocess.run(['ffprobe','-v','error',*_input(path,fmt),'-show_format','-show_streams','-of','json'],capture_output=True,timeout=15,check=True)
    if len(result.stdout)>64000:raise ValueError('Audio metadata exceeds its bound')
    value=json.loads(result.stdout);streams=value.get('streams',[])
    if len(streams)!=1 or streams[0].get('codec_type')!='audio':raise ValueError('Choose an audio-only soundtrack')
    actual=set(value.get('format',{}).get('format_name','').split(','))
    if fmt not in actual:raise ValueError('Audio demuxer differs from the inspected format')
    duration=float(value.get('format',{}).get('duration',0));stream=streams[0]
    if not math.isfinite(duration)or not .1<=duration<=MAX_DURATION or stream.get('channels')not in {1,2}:raise ValueError('Choose mono/stereo audio between0.1seconds and15minutes')
    return {'duration_s':duration,'codec':str(stream.get('codec_name','unknown'))[:40],'channels':stream['channels']}


def normalize_audio(path,directory):
    """Decode a forced safe format into bounded, owned, metadata-free PCM WAV."""
    fmt=sniff_audio(path);source_audio=probe_audio(path)
    fd,name=tempfile.mkstemp(prefix='.decode-',suffix='.wav',dir=directory);os.close(fd);temporary=Path(name)
    try:
        command=['ffmpeg','-nostdin','-v','error','-y',*_input(path,fmt),'-map','0:a:0','-vn','-sn','-dn','-map_metadata','-1',
            '-t',str(MAX_DURATION+.05),'-ar','48000','-ac','2','-c:a','pcm_s16le','-threads','2','-fflags','+bitexact','-flags:a','+bitexact',
            '-fs',str(MAX_AUDIO),'-f','wav',str(temporary)]
        subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120,check=True)
        if not 0<temporary.stat().st_size<=MAX_AUDIO:raise ValueError('Decoded audio exceeds its bound')
        audio=probe_audio(temporary)
        if sniff_audio(temporary)!='wav'or audio['codec']!='pcm_s16le'or audio['channels']!=2 or abs(audio['duration_s']-source_audio['duration_s'])>.25:
            raise ValueError('Decoded audio does not match its complete source duration')
        with temporary.open('rb')as stream:os.fsync(stream.fileno())
        hasher=hashlib.sha256()
        with temporary.open('rb')as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):hasher.update(block)
        token=hasher.hexdigest();output=directory/(token+'.wav')
        try:os.link(temporary,output)
        except FileExistsError:
            if output.is_symlink()or output.stat().st_size!=temporary.stat().st_size:raise ValueError('Decoded audio identity conflict')
            existing=hashlib.sha256()
            with output.open('rb')as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):existing.update(block)
            if existing.hexdigest()!=token:raise ValueError('Decoded audio identity conflict')
        output.chmod(0o600)
        return output,audio,{'format':'interaction-twin-self-contained-audio-ingest','version':1,'status':'decoded_owned_pcm_wav',
            'input_container':fmt,'output_codec':'pcm_s16le','sample_rate_hz':48000,'channels':2,'maximum_duration_s':MAX_DURATION,
            'external_references_allowed':False,'metadata_stripped':True,'source_audio':source_audio}
    finally:
        if temporary.exists():temporary.unlink()


class MusicStore:
    def __init__(self,root=ROOT):
        self.root=Path(root).resolve();self.folder=private_directory(self.root,'marketing-music')
        self.records=self.folder/'records';self.records.mkdir(exist_ok=True,mode=0o700)
        self.uploads=self.folder/'uploads';self.uploads.mkdir(exist_ok=True,mode=0o700)
        self.decoded=self.folder/'decoded';self.decoded.mkdir(exist_ok=True,mode=0o700)
        if self.records.is_symlink()or self.uploads.is_symlink()or self.decoded.is_symlink():raise ValueError('Ordinary private music directories required')

    def checked(self,item):
        path=self.root/item['path']
        if path.resolve()!=path or not path.is_relative_to(self.root)or any(p.is_symlink()for p in [path,*path.parents])or not path.is_file():raise ValueError('Soundtrack unavailable')
        if not 0<path.stat().st_size<=MAX_AUDIO or ref(path,self.root)!=item:raise ValueError('Soundtrack bytes changed')
        return path

    def upload(self,data,filename):
        if not isinstance(data,bytes)or not 0<len(data)<=MAX_UPLOAD:raise ValueError('Choose an audio file up to25MB')
        if not isinstance(filename,str)or len(filename)>200 or '/'in filename or '\\'in filename or '\x00'in filename:raise ValueError('Choose a simple audio filename')
        suffix=Path(filename).suffix.lower()
        if suffix not in {'.wav','.mp3','.m4a','.flac','.ogg'}:raise ValueError('Supported audio: WAV, MP3, M4A, FLAC or OGG')
        token=hashlib.sha256(data).hexdigest();path=self.uploads/(token+suffix)
        if path.is_symlink():raise ValueError('Ordinary audio file required')
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Audio identity conflict')
        else:
            with path.open('xb')as f:f.write(data)
            path.chmod(0o600)
        source=ref(path,self.root)
        try:
            decoded,audio,normalization=normalize_audio(path,self.decoded)
            if ref(path,self.root)!=source:raise ValueError('Source bytes changed during audio ingestion')
        except (OSError,ValueError,subprocess.SubprocessError,json.JSONDecodeError):raise ValueError('Audio could not be validated; no soundtrack was registered')from None
        record={'file':ref(decoded,self.root),'input_file':source,'normalization':normalization,'audio':audio,'imported_utc':datetime.now(timezone.utc).isoformat(),'original_filename':filename}
        identifier=write_immutable(self.records,'audio',record)
        return {'upload_id':identifier,**audio,'bytes':len(data),'decoded_bytes':record['file']['bytes'],'normalization':normalization,'registered_for_rendering':False}

    def _record(self,identifier,prefix='music'):
        if not isinstance(identifier,str)or not ID.fullmatch(identifier)or not identifier.startswith(prefix+'_'):raise ValueError('Unknown soundtrack ID')
        path=self.records/(identifier+'.json')
        if path.resolve()!=path or any(p.is_symlink()for p in [path,*path.parents])or not path.is_file()or not 0<path.stat().st_size<32000:raise ValueError('Soundtrack metadata unavailable')
        value=json.loads(path.read_text())
        if not isinstance(value,dict)or identifier!=prefix+'_'+digest(value):raise ValueError('Immutable soundtrack metadata changed')
        self.checked(value['file'])
        if prefix=='music'and(value.get('license_acknowledged')is not True or value.get('provider')not in {'pixabay','owner_original'}):raise ValueError('Acknowledged soundtrack provenance required')
        # Legacy immutable records remain readable only if their exact original
        # container also passes the same self-contained format gate. New records
        # always point at the owned decoded WAV and retain original input refs.
        sniff_audio(self.root/value['file']['path'])
        return value

    def register(self,body):
        allowed={'upload_id','title','contributor','provider','source_url','license_acknowledged','content_id','certificate_note'}
        if not isinstance(body,dict)or set(body)-allowed or not {'upload_id','title','contributor','provider','license_acknowledged'}<=set(body):raise ValueError('Soundtrack metadata fields required')
        if body['license_acknowledged']is not True:raise ValueError('Acknowledge your soundtrack rights before using it')
        for name in ['title','contributor']:
            if not isinstance(body[name],str)or not 1<=len(body[name].strip())<=200:raise ValueError('Bounded title and contributor required')
        provider=body['provider'];url=body.get('source_url','');parsed=urlsplit(url)
        if provider=='pixabay':
            if parsed.scheme!='https'or parsed.hostname!='pixabay.com'or parsed.netloc!='pixabay.com'or not parsed.path.startswith('/music/')or parsed.query or parsed.fragment:raise ValueError('Use the original HTTPS Pixabay music track page')
        elif provider=='owner_original':
            if url:raise ValueError('Original music does not need an external source URL')
        else:raise ValueError('Choose Pixabay or owner-original music')
        content_id=body.get('content_id','unknown');note=body.get('certificate_note','')
        if content_id not in {'unknown','registered','not_marked'}or not isinstance(note,str)or len(note)>1000:raise ValueError('Invalid soundtrack rights note')
        upload=self._record(body['upload_id'],'audio')
        record={'file':upload['file'],'audio':upload['audio'],**{k:upload[k]for k in ['input_file','normalization']if k in upload},'title':body['title'].strip(),'contributor':body['contributor'].strip(),'provider':provider,'source_url':url,'license_acknowledged':True,'license_url':'https://pixabay.com/service/license-summary/'if provider=='pixabay'else None,'content_id':content_id,'certificate_note':note,'imported_utc':upload['imported_utc'],'claim_free_guaranteed':False,'rights_origin':'owner_acknowledgement_not_legal_verification'}
        identifier=write_immutable(self.records,'music',record);return self.describe(identifier)

    def describe(self,identifier):
        if identifier=='project_original':
            path=self.root/ORIGINAL
            item=ref(path,self.root);self.checked(item)
            return {'id':identifier,'title':'Cobalt original instrumental','contributor':'Interaction Twin','provider':'owner_original','source_url':'','license_acknowledged':True,'content_id':'not_marked','rights_origin':'original_procedural_project_composition','claim_free_guaranteed':False,'file':item,'audio':probe_audio(path),'preview_url':'/api/admin/marketing/music/project_original/audio'}
        value=self._record(identifier);return {'id':identifier,**value,'preview_url':'/api/admin/marketing/music/'+identifier+'/audio'}

    def list(self):
        rows=[]
        if (self.root/ORIGINAL).is_file():rows.append(self.describe('project_original'))
        for path in sorted(self.records.glob('music_*.json'))[:500]:
            try:rows.append(self.describe(path.stem))
            except (OSError,ValueError,KeyError):continue
        return {'music':rows,'external_music_search_url':'https://pixabay.com/music/search/technology%20ambient/','automatic_pixabay_music_api_available':False,'new_api_calls':0}

    def file(self,identifier):return self.checked(self.describe(identifier)['file'])
