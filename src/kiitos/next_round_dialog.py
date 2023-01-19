from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel


class NextRoundDialog(QDialog):
    def __init__(self):
        super().__init__()
        next_round_msg = "Next Round"
        self.setWindowTitle(next_round_msg)

        message = QLabel(f"Round Over! Reset, then press {next_round_msg} to start the next round.")

        self.buttonBox = QDialogButtonBox()
        self.buttonBox.addButton(next_round_msg, QDialogButtonBox.AcceptRole)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
