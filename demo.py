import cv2
import time

def video_capture():
    cap = cv2.VideoCapture(0)
    classes = ['Normal', 'Laughing', 'scream-shout']
    start_sec = time.time()
    text = None
    flag5 = True
    count = 1
    while (True):
        # Capture frames in the video
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX

        if (time.time() - start_sec) < 5 and flag5 == True:
            text = 'Silence'
        elif (time.time() - start_sec) >= 5 and flag5 == True:
            flag5 = False
            start_sec = time.time()
            text = classes[0]
        elif (time.time() - start_sec) >= 10 and flag5 == False:
            start_sec = time.time()
            text = classes[count]
            count += 1

        cv2.putText(frame, 'Audio-Emotion', (150, 50), font, 2, (0, 0, 255), 3, )
        cv2.putText(frame, text, (150, 250), font, 2, (0, 255, 0), 3, )
        cv2.putText(frame, time.strftime("%H:%M:%S-%Y:%m:%d"), (950, 650), font, .6, (0, 0, 0), 2, )
        # cv2.putText(frame, text, (250,50), font, 1, (0, 0, 255), 2, )
        # Display the resulting frame
        cv2.imshow('video', frame)

        # creating 'q' as the quit
        # button for the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # np.save('buffer_normal_2.npy',buffer)
            break

video_capture()