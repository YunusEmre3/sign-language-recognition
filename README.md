# sign-language-recognition

## Credits

This project was inspired by Nicholas Renotte's tutorial on sign language recognition using Mediapipe and LSTM. 
Some parts of the original code were adapted and further developed for this project.

Prototype for a sign language recognition system using Mediapipe and LSTM. Not fully functional yet; under development and open for improvements.

# Sign Language Recognition with Mediapipe and LSTM

> ⚠ **Note:** This project is a prototype and may not fully work as expected. It is under development and open for contributions.

This project is designed to recognize sign language gestures in real-time using Mediapipe for keypoint extraction and an LSTM-based deep learning model.

## Features
- Keypoint extraction using Mediapipe (pose, face, and hands landmarks)
- LSTM deep learning model for sequence classification
- Real-time prediction and visualization
- Data collection tools from webcam or videos
- 
## Dataset Note

⚠ Due to file size limitations, the full `MP_Data` directory is not included in this repository. 
Please collect your own data using the provided collection methods or contact the maintainer for sample data.

## Usage

```python
from SignLanguageModel import SignLanguageModel

sign_model = SignLanguageModel(
    data_path="MP_Data",
    actions=["hello", "thanks", "iloveyou"]
)

sign_model.load_data()
sign_model.build_model()
sign_model.train_model(epochs=20)
sign_model.evaluate_model()
sign_model.real_time_test(threshold=0.5)
