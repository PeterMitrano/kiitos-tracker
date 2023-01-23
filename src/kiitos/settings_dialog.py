# noinspection PyUnresolvedReferences
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog


class SettingsDialog(QDialog):
    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle("Settings")
        uic.loadUi('src/kiitos/settings.ui', self)

        self.sound_effects_checkbox.setChecked(settings.value('sound_effects', True, type=bool))
