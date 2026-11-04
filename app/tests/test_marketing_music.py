import io
import json
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave
import marketing_music as M


def wav():
    out=io.BytesIO()
    with wave.open(out,'wb')as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*16000)
    return out.getvalue()


class MusicTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.store=M.MusicStore(self.root)

    def upload(self):
        return self.store.upload(wav(),'track.wav')

    def metadata(self,upload):return {'upload_id':upload['upload_id'],'title':'Minimal pulse','contributor':'Example musician','provider':'pixabay','source_url':'https://pixabay.com/music/corporate-example-123/','license_acknowledged':True,'content_id':'unknown','certificate_note':''}

    def test_actual_audio_probe_and_registration_preserve_bytes_and_rights_origin(self):
        upload=self.store.upload(wav(),'track.wav');self.assertAlmostEqual(upload['duration_s'],1.)
        self.assertFalse(upload['registered_for_rendering']);record=self.store.register(self.metadata(upload))
        self.assertEqual((self.root/record['input_file']['path']).read_bytes(),wav());self.assertEqual(record['normalization']['status'],'decoded_owned_pcm_wav')
        self.assertEqual(M.sniff_audio(self.store.file(record['id'])),'wav');self.assertEqual(record['audio']['channels'],2);self.assertFalse(record['claim_free_guaranteed'])
        self.assertEqual(record['rights_origin'],'owner_acknowledgement_not_legal_verification')
        self.assertEqual(len(self.store.list()['music']),1)

    def test_missing_license_acknowledgement_cannot_enter_renderer_inventory(self):
        body=self.metadata(self.upload());body['license_acknowledged']=False
        with self.assertRaisesRegex(ValueError,'Acknowledge'):self.store.register(body)
        self.assertEqual(self.store.list()['music'],[])

    def test_source_url_cannot_be_an_arbitrary_fetch_or_credential_url(self):
        body=self.metadata(self.upload())
        for url in ['file:///etc/passwd','https://attacker.example/music/test','https://pixabay.com.attacker.example/music/a','https://user@pixabay.com/music/a','https://pixabay.com/music/a?secret=x']:
            with self.subTest(url=url),self.assertRaises(ValueError):self.store.register({**body,'source_url':url})

    def test_soundtrack_mutation_or_foreign_id_is_refused(self):
        record=self.store.register(self.metadata(self.upload()));path=self.store.file(record['id']);path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):self.store.file(record['id'])
        with self.assertRaises(ValueError):self.store.file('../../.secrets/meta.env')

    def test_filename_and_type_limits_precede_probe(self):
        with patch.object(M,'probe_audio')as probe:
            for name in ['../../secret.wav','private/key.wav','bad.mp4','x\\file.mp3']:
                with self.subTest(name=name),self.assertRaises(ValueError):self.store.upload(b'audio',name)
            probe.assert_not_called()

    def test_invalid_audio_is_not_registered_and_metadata_remains_small(self):
        with self.assertRaisesRegex(ValueError,'could not be validated'):self.store.upload(b'not audio','track.wav')
        self.assertEqual(self.store.list()['music'],[])
        body=self.metadata(self.upload());body['certificate_note']='x'*1001
        with self.assertRaises(ValueError):self.store.register(body)

    def test_symlink_is_rejected_before_source_read(self):
        target=self.root/'outside.wav';target.write_bytes(wav());link=self.root/'link.wav';link.symlink_to(target)
        with patch.object(Path,'read_bytes',side_effect=AssertionError('Must not read symlink target')),self.assertRaisesRegex(ValueError,'Ordinary confined'):M.ref(link,self.root)

    def test_metadata_digest_is_checked_for_audio_and_registered_music(self):
        upload=self.upload();music=self.store.register(self.metadata(upload))
        for identifier,prefix in [(upload['upload_id'],'audio'),(music['id'],'music')]:
            with self.subTest(prefix=prefix):
                path=self.store.records/(identifier+'.json');original=path.read_bytes();record=json.loads(original)
                record['title']='Changed rights metadata'
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'Immutable soundtrack metadata changed'):self.store._record(identifier,prefix)
                path.write_bytes(original)

    def test_playlists_and_unknown_bytes_are_rejected_before_any_subprocess(self):
        for data in [b'#EXTM3U\nhttps://example.invalid/audio.mp3\n',b'ffconcat version 1.0\nfile /private/audio.wav\n',b'[playlist]\nFile1=file:///private/audio.wav\n']:
            with self.subTest(data=data),patch.object(M.subprocess,'run',side_effect=AssertionError('No decoder may inspect playlists'))as run:
                with self.assertRaisesRegex(ValueError,'could not be validated'):self.store.upload(data,'track.wav')
                run.assert_not_called()
        self.assertFalse(list(self.store.records.iterdir()));self.assertFalse(list(self.store.decoded.iterdir()))

    def test_external_mov_reference_is_rejected_before_decoder(self):
        def box(kind,body):return struct.pack('>I4s',8+len(body),kind)+body
        external=box(b'url ',b'\0\0\0\0file:///private/other.wav\0')
        dref=box(b'dref',b'\0\0\0\0'+struct.pack('>I',1)+external)
        nested=dref
        for kind in [b'dinf',b'minf',b'mdia',b'trak',b'moov']:nested=box(kind,nested)
        data=box(b'ftyp',b'M4A \0\0\0\0M4A ')+nested
        with patch.object(M.subprocess,'run',side_effect=AssertionError('External references must precede probing'))as run:
            with self.assertRaisesRegex(ValueError,'could not be validated'):self.store.upload(data,'track.m4a')
            run.assert_not_called()
        self.assertFalse(list(self.store.records.iterdir()))

    def test_actual_supported_formats_decode_into_owned_complete_pcm(self):
        source=self.root/'synthetic.wav';source.write_bytes(wav())
        for suffix,codec in [('mp3','libmp3lame'),('m4a','aac'),('flac','flac'),('ogg','libopus')]:
            with self.subTest(format=suffix):
                encoded=self.root/('synthetic.'+suffix)
                subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-c:a',codec,str(encoded)],check=True,capture_output=True,timeout=15)
                original=encoded.read_bytes();upload=self.store.upload(original,'track.'+suffix)
                record=self.store.register(self.metadata(upload));output=self.store.file(record['id'])
                self.assertEqual((self.root/record['input_file']['path']).read_bytes(),original)
                self.assertEqual(record['input_file']['sha256'],hashlib.sha256(original).hexdigest())
                self.assertEqual(M.sniff_audio(output),'wav');self.assertEqual(record['audio']['codec'],'pcm_s16le')
                self.assertEqual(record['audio']['channels'],2);self.assertAlmostEqual(record['audio']['duration_s'],1.,delta=.1)
                self.assertFalse(record['normalization']['external_references_allowed'])
                self.assertTrue(record['normalization']['metadata_stripped'])
                self.assertEqual(output.name,record['file']['sha256']+'.wav')
        self.assertFalse(list(self.store.decoded.glob('.decode-*')))

    def test_decode_timeout_cleans_temporary_and_never_registers(self):
        original_run=M.subprocess.run
        def run(command,**kwargs):
            if command[0]=='ffmpeg':raise subprocess.TimeoutExpired(command,120)
            return original_run(command,**kwargs)
        with patch.object(M.subprocess,'run',side_effect=run),self.assertRaisesRegex(ValueError,'could not be validated'):self.upload()
        self.assertFalse(list(self.store.decoded.iterdir()));self.assertFalse(list(self.store.records.iterdir()))

    def test_legacy_hash_correct_playlist_and_unacknowledged_rights_are_rejected(self):
        playlist=self.store.uploads/'legacy.wav';playlist.write_bytes(b'#EXTM3U\n/private/other.wav\n')
        value={'file':M.ref(playlist,self.root),'license_acknowledged':True,'provider':'pixabay'}
        identifier=M.write_immutable(self.store.records,'music',value)
        with patch.object(M.subprocess,'run',side_effect=AssertionError('No playlist probe')),self.assertRaisesRegex(ValueError,'playlists'):self.store.describe(identifier)
        upload=self.upload();value=self.store._record(upload['upload_id'],'audio')
        identifier=M.write_immutable(self.store.records,'music',{**value,'license_acknowledged':False,'provider':'pixabay'})
        with self.assertRaisesRegex(ValueError,'Acknowledged soundtrack'):self.store.describe(identifier)

    def test_identical_decode_reuses_exact_normalized_bytes(self):
        first=self.store._record(self.upload()['upload_id'],'audio');second=self.store._record(self.upload()['upload_id'],'audio')
        self.assertEqual(first['file'],second['file']);self.assertEqual(first['input_file'],second['input_file'])
        self.assertEqual(len(list(self.store.decoded.iterdir())),1)


if __name__=='__main__':unittest.main()
