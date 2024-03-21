import cv2
import mediapipe as mp
import numpy as np
import time, os

# 알파벳 A-Z로 액션을 정의한다.
# actions = [chr(x) for x in range(65, 91)]
actions = ["A", "B", "C"]
# seq_length = 30

# 각 동작을 30초의 loop를 돌며 기록한다.
secs_for_action = 10

# Mediapipe 인식 model을 정의한다.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, # 한 손만을 인식할 수 있도록 한다.
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 손 동작 학습을 위한 웹캠을 정의한다.
cap = cv2.VideoCapture(0)

# 새롭게 생성할 데이터셋의 생성시점을 정의하고
# 데이터셋을 정의할 폴더를 생성한다.
created_time = int(time.time())
os.makedirs('demo1/dataset', exist_ok=True)

while cap.isOpened():

    # A-Z 액션의 데이터를 수집하도록 한다.
    for idx, action in enumerate(actions):

        # 예) data에는 'A' 액션의 정보가 저장된다.
        data = []

        # 현재 웹캠의 이미지를 찍는다
        ret, img = cap.read()

        # 이미지 반전
        img = cv2.flip(img, 1)
        
        # 이미지를 화면에 표시
        cv2.putText(img, f"take '{action}' please", 
                    org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)

        # 3초간 대기했다가 본격적으로 사진을 촬영할 수 있도록 한다.
        cv2.waitKey(2000)

        start_time = time.time()
        
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 촬영한 이미지로부터 mediapipe - 관절 할당하기
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 캡쳐된 이미지에 손이 인식되었다면
            if result.multi_hand_landmarks is not None:

                # 여기서 multi_hand_landmarks는 한 이미지에 감지된 손 각각을 의미
                for res in result.multi_hand_landmarks:

                    # 해당 이미지의 joint 정보를 저장할 벡터 생성
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):

                        # 21개 joint point의 x/y/z 좌표와 visibility를 저장
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # 인접한 joint 간의 관계를 계산하여 벡터화
                    # 20개의 마디를 저장한다고 이해할 수 있음
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]

                    # 20개의 관절 정보를 정규화한다
                    # (손가락 관절의 길이와 상관 없이 인식할 수 있도록)
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    
                    
                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx) # [16, ]

                    # 21개의 관절 정보 * 3개의 차원 = 63개
                    # 21개의 각 관절 포인트 visibility = 21개
                    # angle_label = 16개
                    # ==> d = [100, ]
                    # 근데 v를 넣었으면 넣었지 joint는 왜 데이터로 쓰는거지 ...?
                    d = np.concatenate([joint.flatten(), angle_label])
                    data.append(d)
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print("=" * 30)
        print(action, data.shape)
        print("=" * 30)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

    # 웹캠 off
    break
