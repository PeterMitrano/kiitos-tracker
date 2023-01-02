import cv2


def text(frame, text, position, scale, color, thickness):
    frame = cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
