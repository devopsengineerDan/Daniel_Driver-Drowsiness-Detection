'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
from kivymd.app import MDApp
import kivy
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture

import cv2
import dlib
import playsound
from scipy.spatial import distance as dist
from imutils import face_utils


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./datasets/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# defining EYE_AR_Thresh
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 5
COUNTER = 0

# eye_aspect_ratio function
def eye_aspect_ratio(eye):
    # compute the euclidean verticle distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear



def alarm(self):
    # play alarm sound
    playsound.playsound('audio/alert.mp3')



def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


class MyCamera(Camera):
    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)

    def _camera_loaded(self, *largs):
        if kivy.platform=='android':
            self.texture = Texture.create(size=self.resolution,colorfmt='rgb')
            self.texture_size = list(self.texture.size)
        else:
            super(MyCamera, self)._camera_loaded()


    def on_tex(self, *l):
        if kivy.platform == 'android':
            buf = self._camera.grab_frame()
            if not buf:
                return
            frame = self._camera.decode_frame(buf)
            buf = self.process_frame(frame)
        else:
            ret, frame = self._camera._device.read()

        if frame is None:
            print("No")

        buf = self.process_frame(frame)
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        super(MyCamera, self).on_tex(*l)

    def process_frame(self, frame):

        faces = detector(frame)

        for face in faces:


            landmarks = predictor(frame, face)
            landmarks = face_utils.shape_to_np(landmarks)  # converting to NumPy Array

            leftEye = landmarks[lStart:lEnd]
            rightEye = landmarks[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)



            print(ear)

            # check if the person in the frame is starting to show symptoms of drowsiness
            if ear < EYE_AR_THRESH:
                global COUNTER
                COUNTER += 1
                print(COUNTER)

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    alarm(self)

            else:
                COUNTER = 0

        return frame.tostring()


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images
        '''
        camera = self.ids['camera']

        print("Captured")


Builder.load_string('''
<CameraClick>:
    size: root.size
    orientation: 'vertical'
    MyCamera:
        id: camera
        index: 0
        #resolution: (640, 480)
        size_hint: 1, 1
        play: False
    ToggleButton:
        text: 'Play/pause'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
''')



class AlertDriver(MDApp):
    def build(self):
        return CameraClick()


AlertDriver().run()

