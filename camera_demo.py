"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Live Camera Demo using opencv dnn face detection & Emotion Recognition
"""
import enum
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from numpy.lib.type_check import imag
import torch
from torch.functional import norm
import torchvision.transforms.transforms as transforms
from face_detector.face_detector import DnnDetector, HaarCascadeDetector

from model.model import Mini_Xception
from utils import get_label_emotion, normalization, histogram_equalization, standerlization, get_label_age
from face_alignment.face_alignment import FaceAlignment

sys.path.insert(1, 'face_detector')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # Model
    mini_xception = Mini_Xception().to(device)
    mini_xception.eval()

    mini_xception_age = Mini_Xception(5).to(device)
    mini_xception_age.eval()

    # Load model
    checkpoint = torch.load(args.pretrained, map_location=device)
    checkpoint_age = torch.load(args.pretrained_age, map_location=device)

    mini_xception.load_state_dict(checkpoint['mini_xception'])
    mini_xception_age.load_state_dict(checkpoint_age['mini_xception'])

    face_alignment = FaceAlignment()

    # Face detection
    root = 'face_detector'
    face_detector = None
    if args.haar:
        face_detector = HaarCascadeDetector(root)
    else:
        face_detector = DnnDetector(root)

    video = None
    isOpened = False
    if not args.image:
        if args.path:
            video = cv2.VideoCapture(args.path) 
        else:
            video = cv2.VideoCapture(0) # 480, 640
        isOpened = video.isOpened()
        print('video.isOpened:', isOpened)
    
    t1 = 0
    t2 = 0
    
    while args.image or isOpened:
        if args.image:
            frame = cv2.imread(args.path)
        else:
            _, frame = video.read()
            isOpened = video.isOpened()    
        # if loaded video or image (not live camera) .. resize it 
        if args.path:
            frame = cv2.resize(frame, (640, 480))

        # time
        t2 = time.time()
        fps = round(1/(t2-t1))
        t1 = t2

        # faces
        faces = face_detector.detect_faces(frame)

        for face in faces:
            (x,y,w,h) = face

            # preprocessing
            input_face = face_alignment.frontalize_face(face, frame)
            input_face = cv2.resize(input_face, (48,48))

            input_face = histogram_equalization(input_face)
            cv2.imshow('input face', cv2.resize(input_face, (120, 120)))

            input_face = transforms.ToTensor()(input_face).to(device)
            input_face = torch.unsqueeze(input_face, 0)

            with torch.no_grad():
                input_face = input_face.to(device)
                t = time.time()
                emotion = mini_xception(input_face)
                age = mini_xception_age(input_face)

                # print(f'\ntime={(time.time()-t) * 1000 } ms')

                torch.set_printoptions(precision=6)
                softmax = torch.nn.Softmax()

                emotions_soft = softmax(emotion.squeeze()).reshape(-1,1).cpu().detach().numpy()
                emotions_soft = np.round(emotions_soft, 3)

                ages_soft = softmax(emotion.squeeze()).reshape(-1,1).cpu().detach().numpy()
                ages_soft = np.round(emotions_soft, 3)

                for i, em in enumerate(emotions_soft):
                    em = round(em.item(),3)
                    # print(f'{get_label_emotion(i)} : {em}')

                for i, ag in enumerate(ages_soft):
                    ag = round(ag.item(), 3)
                    # print(f'{get_label_emotion(i)} : {em}')

                emotion = torch.argmax(emotion)
                age = torch.argmax(age)

                percentage = round(emotions_soft[emotion].item(), 2)
                percentage_age = round(ages_soft[age].item(), 2)

                emotion = emotion.squeeze().cpu().detach().item()
                age = age.squeeze().cpu().detach().item()

                emotion = get_label_emotion(emotion)
                age = get_label_age(age)

                # draw emotion info
                frame[y-60:y, x:x+w] = (50,50,50)
                cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200))
                cv2.putText(frame, str(percentage), (x + w - 40,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200,200,0))

                # draw age info
                cv2.putText(frame, age, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200))
                cv2.putText(frame, str(percentage_age), (x + w - 40, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 0))

                # enclose face
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
    
        cv2.putText(frame, str(fps), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        cv2.imshow("Video", frame)   
        if cv2.waitKey(1) & 0xff == 27:
            video.release()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--haar', action='store_true', help='run the haar cascade face detector')
    parser.add_argument('--pretrained',type=str,default='custom_models/73_dataset_hybrid_64_0.001_40_1e-06.pth.tar'
                        ,help='load weights')
    parser.add_argument('--pretrained_age',type=str,default='custom_models/69_dataset_age_15_0.001_40_1e-06.pth.tar'
                        ,help='load weights')
    parser.add_argument('--head_pose', action='store_true', help='visualization of head pose euler angles')
    parser.add_argument('--path', type=str, default='', help='path to video to test')
    parser.add_argument('--image', action='store_true', help='specify if you test image or not')
    args = parser.parse_args()

    main(args)

