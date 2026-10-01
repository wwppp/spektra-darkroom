import unittest
import os
import sys

from PySide6.QtWidgets import QApplication
from version import VERSION_STRING, BUILD_NUMBER
from ui.preferences_dialog import PreferencesDialog
import config_manager


app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)


class TestV017Refinements(unittest.TestCase):

    def test_version_bumped_to_017(self):
        self.assertTrue(VERSION_STRING in ("0.1.7", "0.1.8", "0.1.9"))
        self.assertGreaterEqual(BUILD_NUMBER, 8)

    def test_silent_file_association_no_popup(self):
        pref = PreferencesDialog()
        # Trigger silent association
        pref._on_register_association()

        # Button must update directly without opening QMessageBox
        self.assertIn("已关联", pref.btn_assoc.text())


if __name__ == "__main__":
    unittest.main()
