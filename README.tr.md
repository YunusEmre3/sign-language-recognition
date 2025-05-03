# Sign Language Recognition with Mediapipe and LSTM

> ⚠ **Not:** Bu proje bir prototiptir ve tam olarak çalışmayabilir. Geliştirmeye açıktır.

Bu proje, işaret dili hareketlerini gerçek zamanlı olarak tanımak için geliştirilmiş bir sistemdir.

## Özellikler
- Mediapipe ile anahtar nokta çıkarımı
- LSTM tabanlı derin öğrenme modeli
- Gerçek zamanlı tahmin
- Kendi videolarınız veya webcam üzerinden veri toplama

## Kullanım

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
