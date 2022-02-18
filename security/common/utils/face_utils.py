"""Crop face from image via landmarks
"""


def add_face_margin(x, y, w, h, margin=0.5):
    """Add marigin to face bounding box
    """
    x_marign = int(w * margin / 2)
    y_marign = int(h * margin / 2)

    x1 = x - x_marign
    x2 = x + w + x_marign
    y1 = y - y_marign
    y2 = y + h + y_marign

    return x1, x2, y1, y2


def get_face_box(img, landmarks, margin):
    """Get faca bounding box from landmarks

    Args:
        img (np.array): input image
        landmarks (np.array): face landmarks
        margin (float): margin for face box

    Returns:
        list: face bouding box
    """
    # load the positions of five landmarks
    x_list = [
        int(float(landmarks[6])),
        int(float(landmarks[8])),
        int(float(landmarks[10])),
        int(float(landmarks[12])),
        int(float(landmarks[14]))
    ]
    y_list = [
        int(float(landmarks[7])),
        int(float(landmarks[9])),
        int(float(landmarks[11])),
        int(float(landmarks[13])),
        int(float(landmarks[15]))
    ]

    x, y = min(x_list), min(y_list)
    w, h = max(x_list) - x, max(y_list) - y

    side = w if w > h else h

    # add margin
    x1, x2, y1, y2 = add_face_margin(x, y, side, side, margin)
    max_h, max_w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(max_w, x2)
    y2 = min(max_h, y2)

    return x1, x2, y1, y2
