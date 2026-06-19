import cv2
import numpy as np
from tensorflow import keras
import os
import sys


IMG_SIZE = 380
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1792

MODEL_PATH = 'deepfake_video_model.h5'
TEST_VIDEO_PATH = r'E:\Fake Video\TESTING VIDS\agrmhtjdlk.mp4\agrmhtjdlk.mp4'


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if max_frames > 0 and len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def build_feature_extractor():

    feature_extractor = keras.applications.EfficientNetB4(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))

    outputs = feature_extractor(inputs)

    return keras.Model(inputs, outputs, name="feature_extractor")


def prepare_single_video(frames, feature_extractor):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype='bool')
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):

            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :], verbose=0)
        frame_mask[i, :length] = 1

    return frame_features, frame_mask



if __name__ == "__main__":

    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"[-] Error: Could not find video at '{TEST_VIDEO_PATH}'")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"[-] Error: Could not find model at '{MODEL_PATH}'")
        sys.exit(1)

    print("[+] Loading Deepfake Detection Model...")
    model = keras.models.load_model(MODEL_PATH)

    print("[+] Loading EfficientNetB4 Feature Extractor...")
    feature_extractor = build_feature_extractor()

    print(f"[+] Extracting frames from '{TEST_VIDEO_PATH}'...")

    frames = load_video(TEST_VIDEO_PATH, max_frames=MAX_SEQ_LENGTH)

    if len(frames) == 0:
        print("[-] Error: No frames extracted. The video might be corrupted or unreadable.")
        sys.exit(1)

    print(f"[+] Extracted {len(frames)} frames. Processing features...")
    frame_features, frame_mask = prepare_single_video(frames, feature_extractor)

    print("[+] Running inference...")

    prediction = model.predict([frame_features, frame_mask], verbose=0)[0][0]


    print("\n" + "=" * 40)
    print("           ANALYSIS COMPLETE")
    print("=" * 40)
    print(f"Target Video : {TEST_VIDEO_PATH}")
    print(f"Fake Score   : {prediction:.4f} (0=Real, 1=Fake)")

    if prediction >= 0.5:
        print("Verdict      : FAKE DETECTED 🚨")
    else:
        print("Verdict      : REAL VIDEO ✅")
    print("=" * 40 + "\n")