from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

app = Flask(__name__)

with open ("lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

# sign language "J" 와 "Z"는 동적인 문자이기 때문에 demo1 데이터셋에서 제외하였음.
actions = [chr(x) for x in range(65, 91) if chr(x) not in ["J", "Z", "P", "Q", "T"]]
actions.append("I Love You!")

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                
                v1 = joint[[0,1,2,3,0,5,6,7,5,0,9,10,11,9,0,13,14,15,13,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,9,10,11,12,13,13,14,15,16,17,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [23, 3]

                # 20개의 관절 정보를 정규화한다
                # (손가락 관절의 길이와 상관 없이 인식할 수 있도록)
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,4,8,9,10,11,9,13,14,15,16,14,18,19,20,21],:], 
                v[[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],:])) # [21,]

                angle = np.degrees(angle) # Convert radian to degree
                d = np.concatenate([joint[:,3], v.flatten(), angle])
                input_data = np.expand_dims(d, axis=0)
                y_pred = actions[int(model.predict(input_data)[0])]

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.putText(img, f"this is : {y_pred}", 
                            org=(10, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                            fontScale=1, color=(0, 255, 0), 
                            thickness=2)
                
                frame = cv2.imencode('.jpg', img)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                key = cv2.waitKey(20)
                if key == 27:
                    break

@app.route('/video_feed')
def video_feed():
    return Response(gen(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)