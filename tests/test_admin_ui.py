"""Owner-desk client guards without a browser, real password, models or network."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AdminUI(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Existing Node runtime required')
    def test_actual_client_guards_send_once_and_bind_approval_to_current_revision(self):
        result = subprocess.run([shutil.which('node'), str(ROOT/'tests/test_admin_ui_client.cjs')],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_all_static_controller_fields_exist_once_and_public_navigation_has_no_admin_link(self):
        class IDs(HTMLParser):
            def __init__(self): super().__init__(); self.ids=[]
            def handle_starttag(self, tag, attrs):
                value=dict(attrs).get('id')
                if value: self.ids.append(value)
        parser=IDs(); parser.feed((ROOT/'admin/index.html').read_text())
        self.assertFalse([key for key,count in Counter(parser.ids).items() if count>1])
        lookups=set(re.findall(r"\$\('([^']+)'\)",(ROOT/'admin/admin.js').read_text()))
        self.assertFalse(lookups-set(parser.ids))
        for name in ['index.html','library.html','agents.html']:
            self.assertNotRegex((ROOT/'showcase'/name).read_text(),r'href=["\'][^"\']*/admin')


if __name__=='__main__': unittest.main()
