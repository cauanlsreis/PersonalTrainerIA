import cv2
import mediapipe as mp
from mediapipe.tasks import python #processa dados da análise das imagens
from mediapipe.tasks.python import vision
import numpy as np #para manipulação de arrays
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import threading
import queue
import math

class PersonalAI:
    def __init__(self, file_name='IMG_2149.mp4'):
        self.file_name = file_name
        self.image_q = queue.Queue()
        model_path = 'pose_landmarker_full.task'

        #Estima onde estão os membros da pessoa do video
        self.options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO)  # define que o exemplo utilizado no projeto é um video pronto

    #Encontra o ângulo entre 3 pontos
    def find_angle(self, frame, landmarks, p1, p2, p3, draw):
        land = landmarks.pose_landmarks[0]
        h, w, c = frame.shape
        x1, y1 = (land[p1].x, land[p1].y)
        x2, y2 = (land[p2].x, land[p2].y)
        x3, y3 = (land[p3].x, land[p3].y)

        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                                math.atan2(y1-y2, x1-x2))
        position = (int(x2 * w + 10), int(y2 * h +10))
        if draw:
            frame = cv2.putText(frame, str(int(angle)), position,
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
        return frame, angle

        

    #método de desenho
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
              annotated_image,
              pose_landmarks_proto,
              solutions.pose.POSE_CONNECTIONS,
              solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def process_video(self, draw, display):
        #Processamento de imagem
        with vision.PoseLandmarker.create_from_options(self.options) as landmarker: #detector
            cap = cv2.VideoCapture(self.file_name)
            calc_timestamps = [0.0]
            #executando o video
            while (cap.isOpened()): #executa enquanto o arquivo do video estiver iterado
                #ret (retorna true ou false para o final do video) frame recolhe os pixels da imagem
                ret, frame = cap.read()
                fps = cap.get(cv2.CAP_PROP_FPS) #pegando o FPS

                if ret == True:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    calc_timestamps.append(int(calc_timestamps[-1]+1000/fps)) #calculando o tempo do video com o FPS
                    detection_result = landmarker.detect_for_video(mp_image, calc_timestamps[-1])

                    if draw:
                        frame = self.draw_landmarks_on_image(frame, detection_result) #executa o desenho das conexões entre os pontos


                    if display:
                        cv2.imshow('Frame', frame)
                        #Aperta Q para fechar o video
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.image_q.put((frame, detection_result, calc_timestamps[-1]))
                else:
                    break
        
        self.image_q((1, 1, "done"))
        cap.release() #encerra a captura de video
        cv2.destroyAllWindows() #fecha as janelas criadas pelo OpenCV

    def run(self, draw, display=False):
        t1 = threading.Thread(target=self.process_video, args=(draw, display))
        t1.start()

if __name__ == "__main__":
    personalAI = PersonalAI()
    personalAI.process_video(False, True)