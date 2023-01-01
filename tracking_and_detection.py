import numpy as np

from ocr import GoogleOCR
from video_capture import CaptureManager
import cv2
import time

confidence_inc = 0.25
confidence_dec = confidence_inc / 5
motion_alpha = 0.5

class CardTracker:

    def __init__(self, letter, position):
        self.letter = letter
        self.position = position
        self.confidence = 0.1
        self.reported = False

    def __repr__(self):
        return f'{self.letter} {self.position} {self.confidence} {self.reported}'


class NewCardDetector:

    def __init__(self):
        self.card_trackers = []
        self.cap = cv2.VideoCapture(2)
        self.cap_manager = CaptureManager(self.cap)
        while self.cap_manager.last_frame is None:
            time.sleep(1)

        self.ocr = GoogleOCR()
        # self.ocr = None

    def detect(self):
        frame = self.cap_manager.last_frame

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        if self.ocr is not None:
            annotated_frame, detected_letters, detected_positions = self.ocr.detect(frame)
            # cv2.imshow('OCR', annotated_frame)

            for letter, position in zip(detected_letters, detected_positions):
                tracker_found = False
                for card_tracker in self.card_trackers:
                    if card_tracker.letter == letter and np.linalg.norm(card_tracker.position - position) < 50:
                        tracker_found = True
                        break
                if not tracker_found:
                    self.card_trackers.append(CardTracker(letter, position))
                else:
                    card_tracker.position = motion_alpha * card_tracker.position + (1 - motion_alpha) * position
                    card_tracker.confidence = min(1, card_tracker.confidence + confidence_inc)

            to_remove = []
            for card_tracker in self.card_trackers:
                card_tracker.confidence -= confidence_dec
                if card_tracker.confidence <= 0:
                    to_remove.append(card_tracker)

            for to_remove_i in to_remove:
                self.card_trackers.remove(to_remove_i)

            for card_tracker in self.card_trackers:
                if card_tracker.confidence > 0.9 and not card_tracker.reported:
                    card_tracker.reported = True
                    return card_tracker.letter, None

        return None, annotated_frame
