import cv2

bbox_annotation_color = (0, 0, 255)
baseline_annotation_color = (0, 255, 0)


def annotate(input_img, text_and_vertices):
    annotated = input_img.copy()
    for letter, vertices in text_and_vertices:
        left_top = vertices[0]
        right_bottom = vertices[2]
        left_bottom = vertices[3]
        text_pos = (left_top[0] + 5, left_top[1] - 15)

        annotated = cv2.rectangle(annotated, left_top, right_bottom, bbox_annotation_color, 1)
        add_text(annotated, letter, text_pos, 1, bbox_annotation_color, 3)
        annotated = cv2.line(annotated, left_bottom, right_bottom, baseline_annotation_color, 5)
    return annotated


def add_text(frame, text, position, scale, color, thickness):
    frame = cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
