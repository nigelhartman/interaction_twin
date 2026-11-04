"""A workspace rebuild cannot expose truncated JSON to live browser readers."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_workspace as workspace


class AtomicWorkspaceCatalogTests(unittest.TestCase):
    def test_previous_complete_catalog_remains_visible_until_complete_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cases.json'
            path.write_text('{"generation":"previous"}\n')
            new = '{"generation":"new","cases":[1,2,3]}\n'
            original = os.replace
            def inspect_then_replace(source, destination):
                self.assertEqual(path.read_text(), '{"generation":"previous"}\n')
                self.assertEqual(Path(source).read_text(), new)
                original(source, destination)
            with patch.object(workspace.os, 'replace', side_effect=inspect_then_replace):
                workspace.atomic_text(path, new)
            self.assertEqual(path.read_text(), new)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_failed_publication_retains_previous_catalog_and_cleans_temporary_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cases.json'
            path.write_text('{"generation":"previous"}\n')
            with patch.object(workspace.os, 'replace', side_effect=OSError('fixture publication failure')):
                with self.assertRaises(OSError):
                    workspace.atomic_text(path, '{"generation":"new"}\n')
            self.assertEqual(path.read_text(), '{"generation":"previous"}\n')
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
