import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

class SignLanguageModel:
    def __init__(
        self,
        data_path="C:\\Users\\0107y\\OneDrive\\Masaüstü\\HVG\\MP_Data",
        actions=None,
        no_sequences=30,
        sequence_length=30,
        start_folder=0,
        min_detection_conf=0.5,
        min_tracking_conf=0.5
    ):
        """
        Initialize the sign language model workflow parameters.
        
        :param data_path: Base folder to store keypoint data
        :param actions: Tuple/list of actions to recognize
        :param no_sequences: Number of sequence videos to record per action
        :param sequence_length: Number of frames per sequence
        :param start_folder: Starting folder index for new recordings
        :param min_detection_conf: Minimum detection confidence for Mediapipe
        :param min_tracking_conf: Minimum tracking confidence for Mediapipe
        """
        self.data_path = Path(data_path)

        if actions is None:
            self.actions = None
        else:
            self.actions = np.array(actions)

        self.no_sequences = no_sequences
        self.sequence_length = sequence_length
        self.start_folder = start_folder
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf

        # Mediapipe utilities
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Placeholders for model and data
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        if actions is None:
            self.label_map = {}
        else:
            self.label_map = {label: num for num, label in enumerate(self.actions)}

    # ---------- 1. Mediapipe Utility Methods  ---------- #
    def mediapipe_detection(self, image, model):
        """Convert image to RGB, make predictions, then convert back to BGR."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        """Draw landmarks with custom styling for face, pose, and hands."""
        # Face
        self.mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            self.mp_holistic.FACEMESH_CONTOURS, # -----FACEMESH_CONTOURS OLARAK DEĞİŞTİRİLDİ SANIRIM ARTIK BÖYLE GEÇİYOR------
            self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        # Pose
        self.mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
        # Left hand
        self.mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
        # Right hand
        self.mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    def extract_keypoints(self, results):
        """Extract pose, face, left-hand, right-hand keypoints into a single array."""
        # Pose
        pose = (
            np.array([[res.x, res.y, res.z, res.visibility] 
                      for res in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks 
            else np.zeros(33*4)
        )
        # Face
        face = (
            np.array([[res.x, res.y, res.z] 
                      for res in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks 
            else np.zeros(468*3)
        )
        # Left Hand
        lh = (
            np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks 
            else np.zeros(21*3)
        )
        # Right Hand
        rh = (
            np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks 
            else np.zeros(21*3)
        )
        return np.concatenate([pose, face, lh, rh])

    # ---------- 2. Data Collection Method  ---------- #
    def collect_data_with_cam(self):
        """
        Collect data from the webcam for each action and store keypoints.
        This version mimics the original tutorial structure:
        - For each action
        - For each sequence (e.g., 30 sequences)
        - For each frame in that sequence (e.g., 30 frames)
        """
        # Ensure data_path exists
        self.data_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)

        with self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        ) as holistic:

            for action in self.actions:
                for sequence in range(self.start_folder, self.start_folder + self.no_sequences):
                    
                    # Create a folder named with the sequence number
                    sequence_folder = self.data_path / action / str(sequence)
                    sequence_folder.mkdir(parents=True, exist_ok=True)

                    for frame_num in range(self.sequence_length):
                        ret, frame = cap.read()
                        if not ret:
                            continue  # Skip if the frame is not read properly

                        # 1) Make Mediapipe detections
                        image, results = self.mediapipe_detection(frame, holistic)
                        
                        # 2) Draw landmarks
                        self.draw_styled_landmarks(image, results)

                        # 3) Display helpful text:
                        if frame_num == 0:
                            # At the start of each sequence, let the user see "STARTING COLLECTION"
                            cv2.putText(
                                image, 
                                'STARTING COLLECTION', 
                                (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 255, 0), 
                                4, 
                                cv2.LINE_AA
                            )
                            cv2.putText(
                                image, 
                                f'Collecting frames for {action} Video Number {sequence}', 
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 255), 
                                1, 
                                cv2.LINE_AA
                            )
                            # Show the frame
                            cv2.imshow('OpenCV Feed', image)
                            # Wait a bit so you have time to get into position
                            cv2.waitKey(1000)  # 1000 ms; you can set to 1000, etc.
                        else:
                            # For frames 1..(sequence_length-1), just show collecting
                            cv2.putText(
                                image, 
                                f'Collecting frames for {action} Video Number {sequence}', 
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 255), 
                                1, 
                                cv2.LINE_AA
                            )
                            cv2.imshow('OpenCV Feed', image)

                        # 4) Extract keypoints and save to an .npy file
                        keypoints = self.extract_keypoints(results)
                        npy_path = sequence_folder / f"{frame_num}.npy"
                        np.save(npy_path, keypoints)

                        # 5) Check if 'q' is pressed to exit early
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

                    # If 'q' was pressed during the frames loop, break the sequence loop
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                # If 'q' was pressed during the sequence loop, break the action loop
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    # ---------- 2.1 Collect Data from a Pre-defined Dictionary (manual) ---------- #
    def collect_data_from_videos(self, video_paths_by_action, target_frames=30):
        """
        Disk üzerindeki videolardan veri toplayıp (keypoint çıkartıp),
        data_path/action/sequence/frame.npy şeklinde kaydeder.

        :param video_paths_by_action: Sözlük (dict) yapısında,
                                    her bir action için bir veya daha fazla video yolu.
                                    Örneğin:
                                    {
                                        "hello": ["videos/hello1.mp4", "videos/hello2.mp4"],
                                        "thanks": ["videos/thanks1.mp4"],
                                        "iloveyou": ["videos/ily1.mp4", "videos/ily2.mp4"]
                                    }
        """

        # Klasörün varlığını garanti altına al
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Mediapipe modelini başlat
        with self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        ) as holistic:

            # 1) Her bir action için döngü
            for action in self.actions:
                # Bu action'a ait video dosyalarının listesini alç
                videos = video_paths_by_action.get(action, [])

                # 2) Her bir video dosyasını bir "sequence" gibi işle
                for sequence_idx, video_path in enumerate(videos):
                    cap = cv2.VideoCapture(video_path)

                    # Kaydedilecek klasör (ör. data_path/hello/0, data_path/hello/1 vb.)
                    sequence_folder = self.data_path / action / str(sequence_idx)
                    sequence_folder.mkdir(parents=True, exist_ok=True)

                    # 3) Uniform sampling ile frames_list al
                    frames_list = self.uniform_sample_video(cap, target_frames)
                    cap.release()

                    # frames_list tam 'target_frames' uzunluğunda
                    # Elemanlar -> frame (np.array) veya None
                    for i, frame in enumerate(frames_list):
                        if frame is None:
                            # Kısa video padding için -> sıfır landmark
                            keypoints = np.zeros(1662)  # pose+face+lh+rh
                        else:
                            # Mediapipe detections
                            image, results = self.mediapipe_detection(frame, holistic)

                            # İsteğe bağlı Landmark çizimi
                            self.draw_styled_landmarks(image, results)

                            # Ekranda gösterme (isteğe bağlı)
                            cv2.imshow("Video Feed", image)
                            # 'q' ile erken çıkış
                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break

                            # Keypoint çıkar
                            keypoints = self.extract_keypoints(results)

                        # frame i'e ait .npy dosyasını kaydet
                        npy_path = sequence_folder / f"{i}.npy"
                        np.save(npy_path, keypoints)

                    # Bir videoda 'q'ya basıldıysa
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            cv2.destroyAllWindows()

    def uniform_sample_video(self, cap, target_frames=30):
        """
        1) 'cap' bir VideoCapture nesnesi
        2) 'target_frames' kadar kare döndürülecek.
        
        Uzun video: Eşit aralıklarla 'target_frames' kare toplanır.
        Kısa video: Okuduğu kadar kare alır, geri kalanını 'None' olarak doldurur.
        
        Geri dönüş: 
        frames_list (length = target_frames)
        Her eleman ya gerçek bir frame (np.array) ya da None
        """
        frames_list = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            return frames_list  # Boş döneriz, ya da direk 0 boyutlu

        # A) Uzun video (>= target_frames)
        if total_frames >= target_frames:
            step = total_frames / float(target_frames)
            for i in range(target_frames):
                pos = int(round(i * step))
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    # Güvenlik amaçlı, okuyamadıysa 'None' koy
                    frames_list.append(None)
                else:
                    frames_list.append(frame)
        else:
            # B) Kısa video (< target_frames)
            frames_read = 0
            # Tüm kareleri oku
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_list.append(frame)
                frames_read += 1
                if frames_read >= total_frames:
                    break

            # Geri kalan kareleri 'None' ile doldur
            while len(frames_list) < target_frames:
                frames_list.append(None)

        return frames_list

    # ---------- 2.2 Collect Data by Automatically Parsing a Folder of Videos ---------- #
    def collect_data_from_folder_auto(self, video_folder,target_frames=30):
        """
        Bir klasördeki TÜM .mp4 dosyalarını toplayıp,
        dosya ismine bakarak 'action' ismini otomatik çıkarır.
        Sonra, collect_data_from_videos mantığıyla .npy dosyaları oluşturur.

        Ör:  'Anlamak.mp4', 'Anlamak1.mp4' => Action: 'Anlamak'
             'thanks1.mp4', 'thanks2.mp4' => Action: 'thanks'
        """
        video_folder = Path(video_folder)
        # data_path (ör: MP_Data) oluştur
        self.data_path.mkdir(parents=True, exist_ok=True)

        # 1) Tüm mp4 dosyalarını bul
        video_files = list(video_folder.rglob("*.mp4"))
        video_paths_by_action = {}

        for vf in video_files:
            stem = vf.stem  # Dosya adı uzantısız (ör: "Anlamak1")
            # Sona gelen sayı dizisini sil (ör: "Anlamak1" -> "Anlamak")
            action_name = re.sub(r"\d+$", "", stem)
            action_name = action_name.strip()

            # action_name'i dictionary'de tut
            if action_name not in video_paths_by_action:
                video_paths_by_action[action_name] = []
            video_paths_by_action[action_name].append(vf)

        # 2) Sözlükten actions listesini türet
        all_actions = list(video_paths_by_action.keys())
        self.actions = np.array(all_actions)
        # Label map'i güncelle
        self.label_map = {label: num for num, label in enumerate(self.actions)}

        # 3) collect_data_from_videos ile benzer mantığı burada yapalım
        with self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        ) as holistic:

            for action in self.actions:
                videos = video_paths_by_action[action]
                for sequence_idx, video_path in enumerate(videos):
                    cap = cv2.VideoCapture(str(video_path))

                    sequence_folder = self.data_path / action / str(sequence_idx)
                    sequence_folder.mkdir(parents=True, exist_ok=True)
                    # --- Burada uniform sampling metodumuzu çağırıyoruz ---
                    frames_list = self.uniform_sample_video(cap, target_frames)
                    cap.release()

                    # 'frames_list' tam 'target_frames' uzunluğunda
                    # Her eleman ya bir frame (ndarray) ya da None
                    for i, frm in enumerate(frames_list):
                        if frm is None:
                            # Padding
                            keypoints = np.zeros(1662)  # (pose+face+lh+rh) boyutu
                        else:
                            # Mediapipe
                            image, results = self.mediapipe_detection(frm, holistic)
                            # (isterseniz) self.draw_styled_landmarks(image, results)
                            keypoints = self.extract_keypoints(results)

                        npy_path = sequence_folder / f"{i}.npy"
                        np.save(npy_path, keypoints)

            cv2.destroyAllWindows()
    
    def get_next_sequence_index(self, action_name):
        """
        MP_Data/action_name klasöründeki alt klasörlerden en büyük sequence idx'ini bulur.
        Yoksa 0 döndürür.
        Örn: MP_Data/el/0, MP_Data/el/1, MP_Data/el/2 varsa en büyük 2'dir;
        geri dönüş 3 olur.
        """
        action_dir = self.data_path / action_name
        if not action_dir.exists():
            return 0  # Henüz klasör yoksa ilk sequence 0'dan başlasın

        seq_indices = []
        for f in action_dir.iterdir():
            if f.is_dir():
                try:
                    seq_indices.append(int(f.name))  # Alt klasör adı tam sayıya çevrilebiliyorsa ekle
                except ValueError:
                    pass

        if not seq_indices:
            return 0
        else:
            return max(seq_indices) + 1


    # ---------- 2.3 Collect Data with cam any time you want ---------- #
    def collect_data_with_cam_for_single_action(
    self, 
    no_sequences=5,
    sequence_length=30,
    action_name="background", 
    start_sequence=None):
        """
        Webcam'den veri toplayarak tek bir 'action' kaydeder.
        Daha önce var olan sequence klasörlerini bozmamak için, 
        mevcut en yüksek sequence'in üstüne yazmayacak şekilde kayıt yapar.
        """
        # Actions listesini güncelle
        if self.actions is None:
            self.actions = np.array([action_name])
        else:
            if action_name not in self.actions:
                self.actions = np.append(self.actions, action_name)

        # Label map'i güncelle
        self.label_map = {label: num for num, label in enumerate(self.actions)}

        # Varsayılan olarak, "mevcut en büyük sequence + 1" değerden başla
        if start_sequence is None:
            start_sequence = self.get_next_sequence_index(action_name)

        self.data_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        ) as holistic:

            for seq_num in range(start_sequence, start_sequence + no_sequences):
                sequence_folder = self.data_path / action_name / str(seq_num)
                sequence_folder.mkdir(parents=True, exist_ok=True)

                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Mediapipe
                    image, results = self.mediapipe_detection(frame, holistic)
                    self.draw_styled_landmarks(image, results)

                    # Görsel ipucu
                    if frame_num == 0:
                        cv2.putText(
                            image,
                            f"STARTING COLLECTION for {action_name} / seq {seq_num}",
                            (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA
                        )
                        cv2.imshow("OpenCV Feed", image)
                        cv2.waitKey(1000)
                    else:
                        cv2.putText(
                            image,
                            f"Collecting {action_name} - Seq {seq_num}, Frame {frame_num}",
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA
                        )
                        cv2.imshow("OpenCV Feed", image)

                    # Keypoint çıkarma
                    keypoints = self.extract_keypoints(results)
                    npy_path = sequence_folder / f"{frame_num}.npy"
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


    # ---------- 3. Preprocess Data and Create Labels  ---------- #
    def load_data(self):
        """
        Loads all data into NumPy arrays X and y for training/testing.
        Sets self.X_train, self.X_test, self.y_train, self.y_test.
        """
        sequences, labels = [], []
        
        for action in self.actions:
            action_folder = self.data_path / action
            # Each subfolder is a "sequence"
            for sequence_path in action_folder.iterdir():
                if sequence_path.is_dir():
                    window = []
                    for frame_num in range(self.sequence_length):
                        keypoints_file = sequence_path / f"{frame_num}.npy"
                        # In case some file is missing
                        if not keypoints_file.exists():
                            continue
                        res = np.load(keypoints_file)
                        window.append(res)
                    # Only if we have a full sequence of frames
                    if len(window) == self.sequence_length:
                        sequences.append(window)
                        labels.append(self.label_map[action])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.05, random_state=42
        )
        print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")


    # ---------- 4. Build and Train LSTM Model  ---------- #
    def build_model(self):
        """Build the LSTM model architecture."""
        self.model = Sequential()
        # 1662 input shape = 33*4 (pose) + 468*3 (face) + 21*3 (left hand) + 21*3 (right hand) = 1662
        self.model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 1662)))
        self.model.add(LSTM(128, return_sequences=True, activation="relu"))
        self.model.add(LSTM(64, return_sequences=False, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(len(self.actions), activation="softmax"))

        self.model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )

    def train_model(self, epochs=50):
        """
        Train the model on self.X_train and self.y_train without using TensorBoard.
        
        :param epochs: Number of training epochs
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        # Modeli eğit
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test)
        )
        print("Training complete.")


    # ---------- 5. Evaluate the Model  ---------- #
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print metrics.
        """
        if self.model is None:
            raise ValueError("Model is not available. Train or load a model first.")

        yhat = self.model.predict(self.X_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()

        # Print confusion matrices
        print("Confusion Matrices:")
        print(multilabel_confusion_matrix(ytrue, yhat))
        # Print overall accuracy
        print("Accuracy:", accuracy_score(ytrue, yhat))

    # ---------- 6. Real-Time Test Method  ---------- #
    def real_time_test(self, threshold=0.5):
        """
        Run real-time inference on webcam feed, using the trained model to predict actions.
        Press 'q' to quit.
        """
        if self.model is None:
            raise ValueError("Model is not available. Train or load a model first.")

        sequence = []
        sentence = []
        predictions = []

        # Colors for probability bars
        colors = [(245,117,16), (117,245,16), (16,117,245)]

        def prob_viz(res, input_frame):
            """Overlay probability bars of each action on the frame."""
            output_frame = input_frame.copy()
            for num, prob in enumerate(res):
        
                # Mod alma: büyük num'larda colors listesini tekrar döngüye sokar
                color = colors[num % len(colors)]
                cv2.rectangle(
                    output_frame,
                    (0, 60 + num*40),
                    (int(prob*100), 90 + num*40),
                    color,
                    -1
                )
                cv2.putText(
                    output_frame,
                    self.actions[num],
                    (0, 85 + num*40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            return output_frame

        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_styled_landmarks(image, results)

                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    # Visualization logic
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if self.actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(self.actions[np.argmax(res)])
                            else:
                                sentence.append(self.actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Draw probability bars
                    image = prob_viz(res, image)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(
                    image, 
                    " ".join(sentence), 
                    (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (255, 255, 255), 
                    2, 
                    cv2.LINE_AA
                )
                
                cv2.imshow("OpenCV Feed", image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    # ---------- 7. Save and Load Model Weights  ---------- #
    def save(self, filename="action.h5"):
        """Save the trained model weights to a file."""
        if self.model is None:
            raise ValueError("Model is not available to save. Train or load a model first.")
        self.model.save(filename)
        print(f"Model saved to {filename}.")

    def load(self, filename="action.h5"):
        """Load model weights from a file."""
        self.build_model()  # Rebuild architecture
        self.model.load_weights(filename)
        print(f"Model weights loaded from {filename}.")

    def set_actions_from_data_path(self):
        # MP_Data altındaki klasör isimlerini 'action' olarak al
        action_folders = [f.name for f in self.data_path.iterdir() if f.is_dir()]
        self.actions = np.array(action_folders)
        # Label map'i güncelle
        self.label_map = {label: num for num, label in enumerate(self.actions)}

if __name__ == "__main__":
    # Example usage:
    sign_model = SignLanguageModel(
        data_path="C:\\Users\\0107y\\OneDrive\\Masaüstü\\HVG\\try_data",
        actions=["hello" , "thanks" , "iloveyou"],
        no_sequences=10,       # for a quick demo
        sequence_length=30,    # for a quick demo
        start_folder=0
    )
    
    # 1. Collect data (Uncomment if you want to record new data)
    #sign_model.collect_data_with_cam()
    '''
    sign_model.collect_data_from_videos({"hello": ["C:\\Users\\0107y\\OneDrive\\Masaüstü\\merhaba1.mp4","C:\\Users\\0107y\\OneDrive\\Masaüstü\\merhaba2.mp4","C:\\Users\\0107y\\OneDrive\\Masaüstü\\HVG\\indirilen_videolar\\E\\Ebe1.mp4"],
                                         "thanks": ["C:\\Users\\0107y\\OneDrive\\Masaüstü\\thanks1.mp4","C:\\Users\\0107y\\OneDrive\\Masaüstü\\thanks2.mp4"],
                                         "iloveyou": ["C:\\Users\\0107y\\OneDrive\\Masaüstü\\iloveyou1.mp4","C:\\Users\\0107y\\OneDrive\\Masaüstü\\iloveyou2.mp4"]})
    '''
    #sign_model.collect_data_from_folder_auto("C:\\Users\\0107y\\OneDrive\\Masaüstü\\HVG\\indirilen_videolar_test")
    '''
    sign_model.collect_data_with_cam_for_single_action(
        action_name="background", 
        no_sequences=5, 
        sequence_length=30
    )
    '''
    #sign_model.set_actions_from_data_path()
    '''
    # 2. Load data
    sign_model.load_data()

    # 3. Build model
    sign_model.build_model()

    # 4. Train model
    sign_model.train_model(epochs=20)

    # 5. Evaluate model
    sign_model.evaluate_model()
    '''
    sign_model.load("C:\\Users\\0107y\\Downloads\\action.h5")       # Nicholas Renotte's Model

    # 6. (Optional) Real-time test
    sign_model.real_time_test(threshold=0.5)

    # 7. Save model
    # sign_model.save("action.h5")
    