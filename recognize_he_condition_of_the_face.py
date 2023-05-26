# ----------------Info Developer-------------
# -Last Name : Lecheheb
# -First Name : Djaafar
# -Country : Algeria
# -Age : 26
# -Skills : Python - HTML - CSS - C
# -instagram : @ddos_attack_co
# ------------Fallowed Me for instagram-------

# import OpenCV library
import cv2

# import Keras library
from keras.models import load_model

# import NumPy library
import numpy as np

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# تحميل نموذج FER المدرب سابقًا
model = load_model('FER_model.h5')

# تعريف تصنيفات المشاعر الممكنة
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']


# تحميل معالم وجه Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# تهيئة كاميرا الكمبيوتر
video = cv2.VideoCapture(0)
i=0
while True:
    # قراءة إطار الكاميرا
    ret, frame = video.read()
    
    # تحويل الإطار إلى الأبيض والأسود
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # تحديد الوجوه في الإطار
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # التعرف على المشاعر في الوجوه المكتشفة
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=3)
        face = face / 255.0
        
        # التنبؤ بالمشاعر
        predicted_class = np.argmax(model.predict(face))
        print(predicted_class)
        emotion = emotions[predicted_class]
        
        # رسم مربع حول الوجه وكتابة المشاعر
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if emotion == "Happy":
            i+=1
            cv2.imwrite(f'happy_face_{i}.jpg', frame)
            print("تم حفظ الصورة.")
        if emotion == "Surprise":
            i+=1
            cv2.imwrite(f'Surprise_face_{i}.jpg', frame)
            print("تم حفظ الصورة.")
        if emotion == "Sad":
            i+=1
            cv2.imwrite(f'Sad_face_{i}.jpg', frame)
            print("تم حفظ الصورة.")
    
    # عرض الإطار المعالج
    cv2.imshow('Project Djaafar Lecheheb', frame)
    # انتظار الضغط على مفتاح ESC للخروج
    if cv2.waitKey(1) == 27:
        break


# إغلاق كاميرا الكمبيوتر وتدمير النوافذ
video.release()
cv2.destroyAllWindows