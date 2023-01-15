import time

import cv2
import numpy as np

import annotation
from cnn_ocr import CNNOCR
from video_capture import CaptureManager

DISTANCE_THRESHOLD = 90

STATE_TEXT_COLOR = (220, 20, 220)
CONFIDENCE_THRESHOLD = 0.9

confidence_inc = 0.25
confidence_dec = confidence_inc / 6
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
        self.cap = cv2.VideoCapture(0)
        self.cap_manager = CaptureManager(self.cap)
        while self.cap_manager.last_frame is None:
            time.sleep(1)

        # self.ocr = GoogleOCR()
        self.ocr = CNNOCR()

    def detect(self, workspace_bbox):
        frame = self.cap_manager.last_frame

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        annotated_frame, detected_letters, detected_positions = self.ocr.detect(frame, workspace_bbox)

        for letter, position in zip(detected_letters, detected_positions):
            tracker_found = False
            for card_tracker in self.card_trackers:
                distance = np.linalg.norm(card_tracker.position - position)
                if card_tracker.letter == letter and distance < DISTANCE_THRESHOLD:
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

        new_letter = None
        for card_tracker in self.card_trackers:
            if card_tracker.confidence > CONFIDENCE_THRESHOLD and not card_tracker.reported:
                card_tracker.reported = True
                new_letter = card_tracker.letter
                break

        self.annotate_tracked_card_state(annotated_frame)

        return new_letter, annotated_frame

    def annotate_tracked_card_state(self, annotated_frame):
        for card_tracker in self.card_trackers:
            if card_tracker.confidence > CONFIDENCE_THRESHOLD:
                text_pos = tuple([int(cord) for cord in card_tracker.position])
                annotation.text(annotated_frame, card_tracker.letter, text_pos, 2, STATE_TEXT_COLOR, 4)
