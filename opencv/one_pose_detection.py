import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import cv2


# RetinaFace face detector
detector_model = tf.saved_model.load("./tf_retinaface_mbv2")

# Turtleneck Detection Model Loading
MODEL_PATH = "../opencv/sk_random_forest/model_938.joblib"
model = load(MODEL_PATH)


def one_face(frame, bbs, pointss):
    # process only one face (center ?)
    offsets = [
        (bbs[:, 0] + bbs[:, 2]) / 2 - frame.shape[1] / 2,
        (bbs[:, 1] + bbs[:, 3]) / 2 - frame.shape[0] / 2,
    ]
    offset_dist = np.sum(np.abs(offsets), 0)
    index = np.argmin(offset_dist)
    bb = bbs[index]
    points = pointss[:, index]
    return bb, points


def get_width_and_height(frame, bb, points):
    # draw rectangle and landmarks on face
    w = int(bb[2]) - int(bb[0])  # width
    h = int(bb[3]) - int(bb[1])  # height
    eye2box_ratio = (points[0] - bb[0]) / (bb[2] - points[1])
    list_size = [w, h, eye2box_ratio]
    return list_size


def find_roll(pts):
    return pts[6] - pts[5]


def find_yaw(pts):
    le2n = pts[2] - pts[0]
    re2n = pts[1] - pts[2]
    return le2n - re2n


def find_pitch(pts):
    eye_y = (pts[5] + pts[6]) / 2
    mou_y = (pts[8] + pts[9]) / 2
    e2n = eye_y - pts[7]
    n2m = pts[7] - mou_y
    return e2n / n2m


def face_detector(
    image,
    image_shape_max=640,
    score_min=None,
    pixel_min=None,
    pixel_max=None,
    Ain_min=None,
):
    """
    Performs face detection using retinaface method with speed boost and initial quality checks based on whole image size
    
    Parameters
    ----------
    image : uint8
        image for face detection.
    image_shape_max : int, optional
        maximum size (in pixels) of image. The default is None.
    score_min : float, optional
        minimum detection score (0 to 1). The default is None.
    pixel_min : int, optional
        mininmum face size based on heigth of bounding box. The default is None.
    pixel_max : int, optional
        maximum face size based on heigth of bounding box. The default is None.
    Ain_min : float, optional
        minimum area of face in bounding box. The default is None.

    Returns
    -------
    float array
        landmarks.
    float array
        bounding boxes.
    flaot array
        detection scores.
    float array
        face area in bounding box.

    """

    image_shape = image.shape[:2]

    # perform image resize for faster detection
    if image_shape_max:
        scale_factor = max([1, max(image_shape) / image_shape_max])
    else:
        scale_factor = 1

    if scale_factor > 1:
        scaled_image = cv2.resize(
            image, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor
        )
        bbs_all, points_all = retinaface(scaled_image)
        bbs_all[:, :4] *= scale_factor
        points_all *= scale_factor
    else:
        bbs_all, points_all = retinaface(image)

    bbs = bbs_all.copy()
    points = points_all.copy()

    # check detection score
    if score_min:
        mask = np.array(bbs[:, 4] > score_min)
        bbs = bbs[mask]
        points = points[mask]
        if len(bbs) == 0:
            return [], [], [], []

    # check pixel height
    if pixel_min:
        pixel = bbs[:, 3] - bbs[:, 1]
        mask = np.array(pixel > pixel_min)
        bbs = bbs[mask]
        points = points[mask]
        if len(bbs) == 0:
            return [], [], [], []

    if pixel_max:
        pixel = bbs[:, 3] - bbs[:, 1]
        mask = np.array(pixel < pixel_max)
        bbs = bbs[mask]
        points = points[mask]
        if len(bbs) == 0:
            return [], [], [], []

    # check face area in bounding box
    Ains = []
    for bb in bbs:
        Win = min(image_shape[1], bb[2]) - max(0, bb[0])
        Hin = min(image_shape[0], bb[3]) - max(0, bb[1])
        Abb = (bb[2] - bb[0]) * (bb[3] - bb[1])
        Ains.append(Win * Hin / Abb * 100 if Abb != 0 else 0)
    Ains = np.array(Ains)

    if Ain_min:
        mask = np.array(Ains >= Ain_min)
        bbs = bbs[mask]
        points = points[mask]
        Ains = Ains[mask]
        if len(bbs) == 0:
            return [], [], [], []

    scores = bbs[:, -1]
    bbs = bbs[:, :4]

    return points, bbs, scores, Ains


def retinaface(image):

    height = image.shape[0]
    width = image.shape[1]

    image_pad, pad_params = pad_input_image(image)
    image_pad = tf.convert_to_tensor(image_pad[np.newaxis, ...])
    image_pad = tf.cast(image_pad, tf.float32)

    outputs = detector_model(image_pad).numpy()

    outputs = recover_pad_output(outputs, pad_params)
    Nfaces = len(outputs)

    bbs = np.zeros((Nfaces, 5))
    lms = np.zeros((Nfaces, 10))

    bbs[:, [0, 2]] = outputs[:, [0, 2]] * width
    bbs[:, [1, 3]] = outputs[:, [1, 3]] * height
    bbs[:, 4] = outputs[:, -1]

    lms[:, 0:5] = outputs[:, [4, 6, 8, 10, 12]] * width
    lms[:, 5:10] = outputs[:, [5, 7, 9, 11, 13]] * height

    return bbs, lms


def pad_input_image(img, max_steps=32):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(
        img, 0, img_pad_h, 0, img_pad_w, cv2.BORDER_CONSTANT, value=padd_val.tolist()
    )
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * [
        (img_pad_w + img_w) / img_w,
        (img_pad_h + img_h) / img_h,
    ]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs


def discern_random_forest(model, Roll, Yaw, Pitch, width, height, eye2box):
    """
    df_temp = pd.DataFrame(
        {
            "Width": width,
            "Height": height,
            "Eye2Box": eye2box,
            "Roll": Roll,
            "Yaw": Yaw,
            "Pitch": Pitch,
        },
        index=[0],
    )
    predict_val = model.predict(df_temp.values)
    """
    input_values = [[width, height, eye2box, Roll, Yaw, Pitch]]
    tuple_predict_val = model.predict_proba(input_values)
    return tuple_predict_val


# ===========================================================================
def get_rf_prob(model, image_path):
    image = cv2.imread(image_path)

    pointss_all, bbs_all, scores_all, _ = face_detector(image)

    bbs_all = np.insert(bbs_all, bbs_all.shape[1], scores_all, axis=1)
    pointss_all = np.transpose(pointss_all)

    bbs = bbs_all.copy()
    pointss = pointss_all.copy()

    if len(bbs_all) > 0:  # if at least one face is detected
        # process only one face (center ?)
        bb, points = one_face(image, bbs, pointss)
        list_size = get_width_and_height(image, bb, points)  # width and height
        Roll = find_roll(points)
        Yaw = find_yaw(points)
        Pitch = find_pitch(points)

        turtle_value = discern_random_forest(
            model, Roll, Yaw, Pitch, list_size[0], list_size[1], list_size[2]
        )

        # print(turtle_value)
        if turtle_value[0, 0] > 0.45:
            print(turtle_value)
            return True
        else:
            return False

    else:
        return np.NaN


IMG2 = "/Users/noopy/turtleneck/Data/Good_total/1.jpg"
print(get_rf_prob(model, IMG2))  # True

IMG = "/Users/noopy/turtleneck/Data/Bad_total/1.jpg"
print(get_rf_prob(model, IMG))  # False

