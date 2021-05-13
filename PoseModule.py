import cv2
import mediapipe as mp
import time
import math


class PoseDetector:
    def __init__(self, mode=False, upBody=True, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def find_angle(self, img, p1, p2, p3, draw=True):
        # Get the landmark
        x1, y1, = self.lmList[p1][1:]
        x2, y2, = self.lmList[p2][1:]
        x3, y3, = self.lmList[p3][1:]
        # calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 12, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 12, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 12, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle)), (x2 - 15, y2 + 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
        return angle


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = PoseDetector()
    while True:
        succes, img = cap.read()
        img = detector.find_pose(img)
        lmList = detector.find_position(img)
        # Analizar solo un punto
        '''
        print(lmList[14])
        if lmList != 0:
            cv2.circle(img,(lmList[14][1],lmList[14][2]), 15, (0,0,255), cv2.FILLED)
        '''
        print(lmList)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


'''
if __name__ =="__main__":
    main()
'''
