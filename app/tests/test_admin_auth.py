"""Real scrypt, opaque sessions, one-use owner setup and mutation boundaries."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import admin_auth as A

PASSWORD='test-only owner passphrase 2026'


class AdminAuthTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.now=1000.;self.auth=A.AdminAuth(self.root,clock=lambda:self.now)

    def session(self):
        status,token=self.auth.status();return status,f'{A.COOKIE}={token}'

    def login(self):
        A.configure_password(self.root,PASSWORD);status,cookie=self.session()
        result,token=self.auth.login(PASSWORD,cookie,status['csrf_token'])
        return result,f'{A.COOKIE}={token}'

    def test_no_default_owner_and_real_scrypt_hash_with_private_permissions(self):
        self.assertFalse(self.auth.configured());self.assertFalse(self.auth.authenticated(None))
        A.configure_password(self.root,PASSWORD)
        path=self.root/'.secrets/admin-auth.json';saved=json.loads(path.read_text())
        self.assertEqual(saved['algorithm'],'scrypt');self.assertEqual(saved['n'],32768)
        self.assertNotIn(PASSWORD,path.read_text());self.assertEqual(path.stat().st_mode&0o777,0o600)
        self.assertTrue(self.auth.configured())
        with self.assertRaises(A.AuthError):A.configure_password(self.root,'short')

    def test_login_rotates_cookie_and_csrf_and_owner_mutations_require_both(self):
        A.configure_password(self.root,PASSWORD);before,cookie=self.session()
        status,token=self.auth.login(PASSWORD,cookie,before['csrf_token']);owner=f'{A.COOKIE}={token}'
        self.assertTrue(status['authenticated']);self.assertNotEqual(status['csrf_token'],before['csrf_token'])
        self.assertFalse(self.auth.authenticated(cookie));self.assertTrue(self.auth.authenticated(owner))
        for passed,csrf in [(None,status['csrf_token']),(owner,None),(owner,'wrong'),(cookie,before['csrf_token'])]:
            with self.subTest(cookie=passed),self.assertRaises(A.AuthError):self.auth.require(passed,csrf,mutation=True)
        self.assertEqual(self.auth.require(owner,status['csrf_token'],mutation=True),'nigel')
        value=A.cookie_header(token)
        self.assertIn('HttpOnly',value);self.assertIn('SameSite=Strict',value);self.assertIn('Path=/',value)

    def test_wrong_password_is_rejected_and_rate_limit_precedes_more_scrypt_work(self):
        A.configure_password(self.root,PASSWORD);status,cookie=self.session()
        for _ in range(10):
            with self.assertRaises(A.AuthError) as failed:self.auth.login('wrong password here',cookie,status['csrf_token'])
            self.assertEqual(failed.exception.status,401)
        with patch.object(A.hashlib,'scrypt',side_effect=AssertionError('Rate-limited request must not hash again')):
            with self.assertRaises(A.AuthError) as limited:self.auth.login(PASSWORD,cookie,status['csrf_token'])
        self.assertEqual(limited.exception.status,429)
        self.now+=901
        # The anonymous session lasts20min, so this token is still valid.
        status,token=self.auth.login(PASSWORD,cookie,status['csrf_token']);self.assertTrue(status['authenticated'])

    def test_setup_is_one_use_requires_local_bootstrap_and_never_returns_code(self):
        code='A'*43
        with patch.object(A.secrets,'token_urlsafe',return_value=code):path=A.create_bootstrap(self.root,now=self.now)
        self.assertEqual(path.name,'admin-setup-code');self.assertEqual(path.stat().st_mode&0o777,0o600)
        status,cookie=self.session();self.assertTrue(status['setup_available'])
        with self.assertRaises(A.AuthError):self.auth.setup('B'*43,PASSWORD,cookie,status['csrf_token'])
        result,token=self.auth.setup(code,PASSWORD,cookie,status['csrf_token'])
        self.assertTrue(result['authenticated']);self.assertFalse(result['setup_available'])
        self.assertFalse(path.exists());self.assertFalse((self.root/'.secrets/admin-bootstrap.json').exists())
        with self.assertRaises(A.AuthError):A.create_bootstrap(self.root)

    def test_setup_expiry_logout_session_expiry_and_local_password_reset(self):
        code='C'*43
        with patch.object(A.secrets,'token_urlsafe',return_value=code):A.create_bootstrap(self.root,now=self.now-A.BOOTSTRAP_SECONDS-1)
        status,cookie=self.session();self.assertFalse(status['setup_available'])
        with self.assertRaises(A.AuthError):self.auth.setup(code,PASSWORD,cookie,status['csrf_token'])
        result,owner=self.login()
        A.configure_password(self.root,'different owner passphrase',replace=True)
        self.assertFalse(self.auth.authenticated(owner),'Local reset invalidates existing owner sessions')
        status,cookie=self.session();status,token=self.auth.login('different owner passphrase',cookie,status['csrf_token']);owner=f'{A.COOKIE}={token}'
        self.auth.logout(owner,status['csrf_token']);self.assertFalse(self.auth.authenticated(owner))
        status,cookie=self.session();status,token=self.auth.login('different owner passphrase',cookie,status['csrf_token']);owner=f'{A.COOKIE}={token}'
        self.now+=A.SESSION_SECONDS+1;self.assertFalse(self.auth.authenticated(owner))

    def test_secret_symlinks_and_loose_credential_permissions_fail_closed(self):
        other=self.root/'outside';other.mkdir();(self.root/'.secrets').symlink_to(other,target_is_directory=True)
        with self.assertRaises(A.AuthError):self.auth.status()
        self.assertEqual(list(other.iterdir()),[])
        (self.root/'.secrets').unlink();A.configure_password(self.root,PASSWORD)
        credential=self.root/'.secrets/admin-auth.json';credential.chmod(0o644)
        with self.assertRaises(A.AuthError):self.auth.status()


if __name__=='__main__':unittest.main()
