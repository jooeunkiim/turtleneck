import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import cv2


# RetinaFace face detector
detector_model = tf.saved_model.load("./tf_retinaface_mbv2")

# Model Creation
df_good = pd.read_csv("./df_RYP/df_good.csv", index_col=0)  # set first column as index
df_bad = pd.read_csv("./df_RYP/df_bad.csv", index_col=0)  # set first column as index

df_good["label"] = 0
df_bad["label"] = 1

df = pd.concat([df_bad, df_good])

y = df["label"]
X = df.drop(["label"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("model score:", model.score(X_test, y_test))


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
    return predict_val


# ===========================================================================
# FONT SETTING (font style, size and color)
font = cv2.FONT_HERSHEY_COMPLEX  # Text in video
font_size = 0.9
blue = (225, 0, 0)
green = (0, 128, 0)
red = (0, 0, 255)
orange = (0, 140, 255)

# DEMO GUI SETTING
total_size = np.array([750, 1400], dtype=int)  # demo-gui size (resolution)
# complete/final frame to be shown on demo-gui
frame_show = np.ones((total_size[0], total_size[1], 3), dtype="uint8") * 255
logo_size = 150
show_size = 150  # Size showed detected faces
res_max = np.zeros((2), dtype=int)
res_resize = np.zeros((2), dtype=int)


# RECORDING SETTING (Recordings on/off)
image_save = False  # save face image
video_save = True  # save video
fps = 10.0
video_format = cv2.VideoWriter_fourcc("M", "J", "P", "G")
if video_save:
    video_file = "video_out.avi"
    video_out = cv2.VideoWriter(video_file, video_format, fps, (640, 480))

# CAMERA SETTING (video capture initialization)
camera = 0  # 0: internal, 1: external
cap = cv2.VideoCapture(camera)

res_actual = np.zeros((1, 2), dtype=int)  # actual resolution of the camera
res_actual[0, 0] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
res_actual[0, 1] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("\ncamera resolution: {}".format(res_actual))


# PROCESS FRAMES
while True:

    ret, frame = cap.read()  # read frame from camera
    if not (ret):
        break

    frame = np.array(frame)
    frame = cv2.flip(frame, 1)

    res_crop = np.asarray(frame.shape)[0:2]  # ?

    # bbs_all, pointss_all = detector.detect_faces(frame)# face detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pointss_all, bbs_all, scores_all, _ = face_detector(
        frame_rgb,
        image_shape_max=640,
        score_min=0.95,
        pixel_min=20,
        pixel_max=1000,
        Ain_min=90,
    )

    bbs_all = np.insert(bbs_all, bbs_all.shape[1], scores_all, axis=1)
    pointss_all = np.transpose(pointss_all)

    bbs = bbs_all.copy()
    pointss = pointss_all.copy()

    if len(bbs_all) > 0:  # if at least one face is detected
        # process only one face (center ?)
        bb, points = one_face(frame, bbs, pointss)
        list_size = get_width_and_height(frame, bb, points)  # width and height
        Roll = find_roll(points)
        Yaw = find_yaw(points)
        Pitch = find_pitch(points)

        cv2.putText(
            frame, f"Roll: {Roll} (-50 to +50)", (10, 60), font, font_size, red, 1
        )
        cv2.putText(
            frame, f"Yaw: {Yaw} (-100 to +100)", (10, 80), font, font_size, red, 1
        )
        cv2.putText(
            frame, f"Pitch: {Pitch} (0 to 4)", (10, 100), font, font_size, red, 1
        )

        turtle_value = discern_random_forest(
            model, Roll, Yaw, Pitch, list_size[0], list_size[1], list_size[2]
        )

        # print(turtle_value)
        if turtle_value == 0:
            cv2.putText(frame, "Position: Good", (10, 140), font, font_size, green, 1)
        else:
            cv2.putText(frame, "Position: Bad", (10, 140), font, font_size, green, 1)

    else:
        cv2.putText(
            frame_show, "no face", (10, logo_size + 200), font, font_size, blue, 2
        )

    res_max[0] = total_size[0]  # -show_size
    res_max[1] = total_size[1] - 2 * logo_size

    res_resize[1] = res_max[1]
    res_resize[0] = res_max[1] / res_crop[1] * res_crop[0]

    if res_resize[0] > res_max[0]:
        res_resize[0] = res_max[0]
        res_resize[1] = int(res_max[0] / res_crop[0] * res_crop[1] / 2) * 2

    frame_resize = cv2.resize(
        frame, (res_resize[1], res_resize[0]), interpolation=cv2.INTER_LINEAR
    )
    space_vert = (total_size[1] - res_resize[1]) // 2

    frame_show[: frame_resize.shape[0], space_vert:-space_vert, :] = frame_resize

    cv2.putText(frame_show, "q: quit", (10, 50), font, font_size, blue, 2)
    cv2.imshow("Pose Detection - Retina Face", frame_show)

    if video_save:
        video_out.write(frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    option = []
    options = ["Quit"]
    if key_pressed == ord("q"):
        break

cap.release()

if video_save:
    video_out.release()

cv2.destroyAllWindows()
