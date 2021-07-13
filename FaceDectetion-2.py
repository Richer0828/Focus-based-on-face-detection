import cv2
import math
import numpy as np
import face_recognition
import os
import time

def writefile(s):
    #print(s)
    f = open('test.txt',mode='a',encoding='utf-8')
    f.writelines(s)
    f.close()

# 检测眼睛大小，返回R,R是一个数组，包含图中所有人脸的眼睛大小的参数，并将每个人连眼睛大小写在该人脸的下方
def eyedetection(img):
    landmarks = face_recognition.face_landmarks(img)
    # print('landmarks: ',landmarks)
    # print('landmarks length:',len(landmarks))
    print('face_locations:', face_recognition.face_locations(img))
    R = []
    for faces in landmarks:
        s = faces['right_eye']
        # print('faces[right_eye]: ',s)
        x1 = s[0][0]
        x2 = s[1][0]
        x3 = s[2][0]
        x4 = s[3][0]
        x5 = s[4][0]
        x6 = s[5][0]
        y1 = s[0][1]
        y2 = s[1][1]
        y3 = s[2][1]
        y4 = s[3][1]
        y5 = s[4][1]
        y6 = s[5][1]
        t1 = ((x2 - x6) ** 2 + (y2 - y6) ** 2) ** 0.5
        t2 = ((x3 - x5) ** 2 + (y3 - y5) ** 2) ** 0.5
        t3 = ((x1 - x4) ** 2 + (y1 - y4) ** 2) ** 0.5
        R1 = ((t1 + t2) / t3) / 2
        s = faces['left_eye']
        # print('faces[letf_eye]: ',s)
        x1 = s[0][0]
        x2 = s[1][0]
        x3 = s[2][0]
        x4 = s[3][0]
        x5 = s[4][0]
        x6 = s[5][0]
        y1 = s[0][1]
        y2 = s[1][1]
        y3 = s[2][1]
        y4 = s[3][1]
        y5 = s[4][1]
        y6 = s[5][1]
        t1 = ((x2 - x6) ** 2 + (y2 - y6) ** 2) ** 0.5
        t2 = ((x3 - x5) ** 2 + (y3 - y5) ** 2) ** 0.5
        t3 = ((x1 - x4) ** 2 + (y1 - y4) ** 2) ** 0.5
        R2 = ((t1 + t2) / t3) / 2
        R.append((R1 + R2) / 2)
        print('R = ', R)

    for i in range(0, len(face_recognition.face_locations(img))):
        if 0.28 > R[i] >= 0.22:
            print('眼睛微闭')
            '''cv2.putText(img, "eyes closed a little, R:{}".format(round(R[i], 3)), (
            face_recognition.face_locations(img)[i][3] + 1, face_recognition.face_locations(img)[i][2] - 1),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)'''
        elif R[i] >= 0.28:
            print('眼睛正常')
            '''cv2.putText(img, "eyes are normal, R:{}".format(round(R[i], 3)), (
            face_recognition.face_locations(img)[i][3] + 1, face_recognition.face_locations(img)[i][2] - 1),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)'''
        else:
            print('闭眼')
            '''cv2.putText(img, "eyes closed R:{}".format(round(R[i], 3)), (
            face_recognition.face_locations(img)[i][3] + 1, face_recognition.face_locations(img)[i][2] - 1),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)'''
    return R


# 检测嘴巴大小，返回R数组
def mouthdetection(img):
    landmarks = face_recognition.face_landmarks(img)
    # print('landmarks: ',landmarks)
    R = []
    for faces in landmarks:
        s = faces['top_lip']
        # print('faces[top_lip]: ',s)
        x1 = s[10][0]
        y1 = s[10][1]
        x3 = s[8][0]
        y3 = s[8][1]
        s = faces['bottom_lip']
        # print('faces[bottom_lip]: ', s)
        x2 = s[8][0]
        y2 = s[8][1]
        x4 = s[10][0]
        y4 = s[10][1]
        x5 = s[0][0]
        y5 = s[0][1]
        x6 = s[6][0]
        y6 = s[6][1]
        t1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        t2 = ((x3 - x4) ** 2 + (y3 - y4) ** 2) ** 0.5
        t3 = ((x5 - x6) ** 2 + (y5 - y6) ** 2) ** 0.5
        R.append(((t1 + t2) / 2) / t3)
        print('R = ', R)
    for i in range(0, len(face_recognition.face_locations(img))):
        if R[i] > 0.17:
            print('嘴巴张开')
            '''cv2.putText(img, "mouth opened, R:{}".format(round(R[i], 3)), (
            face_recognition.face_locations(img)[i][3] + 1, face_recognition.face_locations(img)[i][2] - 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)'''
        elif R[i] <= 0.06:
            print('嘴巴闭合')
            '''cv2.putText(img, "mouth closed, R:{}".format(round(R[i], 3)), (
            face_recognition.face_locations(img)[i][3] + 1, face_recognition.face_locations(img)[i][2] - 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)'''
        else:
            print('嘴巴微张')
            '''cv2.putText(img, "mouth opened a little, R:{}".format(round(R[i], 3)), (
            face_recognition.face_locations(img)[i][3] + 1, face_recognition.face_locations(img)[i][2] - 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)'''
    return R


# 检测头部旋转角度，返回pitch数组
def rotatedetection(img):
    size = img.shape
    landmarks = face_recognition.face_landmarks(img)
    # 2D image points. If you change the image, you need to change vector
    if landmarks == []:
        return
    face_locations = face_recognition.face_locations(img)
    roll = []
    pitch = []
    yaw = []
    X = []
    Y = []
    Z = []
    for i in range(0, len(face_locations)):
        image_points = np.array([
            landmarks[i]['nose_bridge'][-1],  # Nose tip
            landmarks[i]['chin'][int(len(landmarks[0]['chin']) / 2)],  # Chin
            landmarks[i]['left_eye'][0],  # Left eye left corner
            landmarks[i]['right_eye'][int(len(landmarks[0]['right_eye']) / 2)],  # Right eye right corne
            landmarks[i]['top_lip'][0],  # Left Mouth corner
            landmarks[i]['top_lip'][7]  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        # Camera internals

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # print("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs)

        # print("Rotation Vector:\n {0}".format(rotation_vector))  # 旋转向量
        # print("Translation Vector:\n {0}".format(translation_vector))  # 平移向量
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        #cv2.line(img, p1, p2, (255, 0, 0), 2)
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)

        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta

        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        # print('t0:{}, t1:{}'.format(t0, t1))
        pitch.append(math.atan2(t0, t1))

        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw.append(math.asin(t2))

        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll.append(math.atan2(t3, t4))

        # 单位转换：将弧度转换为度
        Y.append((pitch[i] / math.pi) * 180)  # 点头，低头抬头
        X.append((yaw[i] / math.pi) * 180)  # 左右扭头
        Z.append((roll[i] / math.pi) * 180)  # 摆头
        print('pitch:{}, yaw:{}, roll:{}'.format(Y[i], X[i], Z[i]))
        if pitch[i] > 0:
            Y[i] = round(180 - abs(Y[i]), 3) * 2
        else:
            Y[i] = -round(180 - abs(Y[i]), 3) * 2
        print('NO.', i, 'face_locations = ', face_locations, '\n', )
        '''cv2.putText(img, 'head rotated:{}'.format(Y[i]), (face_locations[i][3] + 1, face_locations[i][2] - 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)'''
    print('pitch = ', pitch)
    return pitch


# 通过所给路径对图片进行检测，无返回值
def picture(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
    R1 = eyedetection(img)
    R2 = mouthdetection(img)
    pitch = rotatedetection(img)
    focus_degree = concentraion_caculate(R1, R2, pitch, img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#全局变量
pre_time = 0

pre_R_mouth = []
pre_R_eyes = []
pre_pitch = 0
flag = 0

# 通过所给的眼睛大小、嘴巴大小、头部偏转角作为参数，返回专注度评判标准分数focus_degree数组
def concentraion_caculate(R_eyes, R_mouth, pitch, img):
    # print("R_EYES = ",R_eyes,"\nR_MOUTH = ",R_mouth,"\nPITCH = ",pitch,'\n\n\n')
    face_locations = face_recognition.face_locations(img)
    focus_degree = []
    global pre_R_mouth
    global pre_pitch
    global pre_R_eyes
    global pre_time
    global flag
    local_time = time.perf_counter()

    #flag = 0 #表示没有检测到发呆

    if len(pre_R_mouth) == 0 or len(pre_pitch) == 0 or len(pre_R_eyes) == 0:
        pre_R_eyes = R_eyes
        pre_pitch = pitch
        pre_R_mouth = R_mouth
        #pre_time = local_time
    else:
        for (top, right, bottom, left),i in zip(face_locations,range(0,len(pre_R_mouth))):

            if abs(pre_R_mouth[i] - R_mouth[i])<0.1 and abs(pre_R_eyes[i] - R_eyes[i])<0.1 and abs(pre_pitch[i] - pitch[i])<0.1 and local_time-pre_time>3:
                print('检测到发呆')
                print('face_names = ',face_names)


                cv2.putText(img, 'distracted!!!', (left, top + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                flag = 1
                print('flag = {}'.format(flag))
            else:
                pre_pitch[i] = pitch[i]
                pre_R_mouth[i] = R_mouth[i]
                pre_R_eyes[i] = R_eyes[i]
                print('flag = {}'.format(flag))
                if flag == 1:
                    if i < len(face_names) and face_names[i] != 'Unknown':
                        writefile(time.strftime("%H:%M:%S", time.localtime()) + '检测到{}发呆'.format(face_names[i])+str(round(local_time-pre_time,0))+'秒钟\n')
                    pre_time = local_time
                    flag = 0
        if len(R_eyes)>len(pre_R_mouth):
            for i in range(len(pre_R_eyes),len(R_eyes)):
                pre_R_eyes.append(R_eyes[i])
                pre_pitch.append(pitch[i])
                pre_R_mouth.append(R_mouth[i])

    print('pre_R_mouth:{}'.format(pre_R_mouth))
    print('pre_R_eyes:{}'.format(pre_R_eyes))
    print('pre_pitch:{}'.format(pre_pitch))
    print('pre_time:{}'.format(pre_time))
    print('local_time:{}'.format(local_time))



    for (top, right, bottom, left), i in zip(face_locations, range(0, len(face_locations))):

        if R_eyes[i] == None or R_mouth[i] == None or pitch[i] == None:
            return None
        else:
            focus_degree.append(((R_eyes[i]*0.8 / 0.28 - R_mouth[i]*0.5 / 0.06) * 0.6 + (1 - abs(pitch[i]) / 60.0) * 0.4)*100)
            if focus_degree[i] < 0:
                focus_degree[i] = 0
            cv2.putText(img, 'foucus degree:{}'.format(focus_degree[i]), (left + 1, bottom - 45),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0),1)
        if R_eyes[i] < 0.28 and R_mouth[i] > 0.17:
            print('检测到打哈欠行为')
            if i < len(face_names) and face_names[i] != 'Unknown':
                writefile(time.strftime("%H:%M:%S", time.localtime())+'检测到{}打哈欠\n'.format(face_names[i]))

            cv2.putText(img, 'yawn!!!', (left, top + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
    return focus_degree


# 调用摄像头进行人脸检测
def Vedio():
    capture = cv2.VideoCapture(0)

    while True:
        ret, img = capture.read()
        img = cv2.flip(img, 1)
        R_eyes = eyedetection(img)
        R_mouth = mouthdetection(img)
        pitch = rotatedetection(img)
        rotatedetection(img)
        concentraion_caculate(R_eyes, R_mouth, pitch, img)
        cv2.imshow('vedio', img)
        print('\n\n')
        t = cv2.waitKey(5)
        if t == 27:
            break
    cv2.destroyAllWindows()

face_names = []

def Vedio_Rectangle():
    dir_path = r'D:\GraduateProject\Img\FaceDetecion_Rectangle/'
    listdir = os.listdir(dir_path)  # listdir是当前文件夹下图片的名称

    known_img = []  # known_img是要存放通过face_recognition加载到的信息
    known_img_encode = []  # 对my_img中的所有人脸进行编码处理
    known_name = []  # 文件夹中的图片需要按照学生的名字来命名

    for dirname in listdir:
        known_name.append(dirname.split('.')[0])
        known_img.append(face_recognition.load_image_file(dir_path + dirname))

    if len(known_img) <= 0:
        print('未检测到人脸')

    for i in range(len(known_img)):
        known_img_encode.append(face_recognition.face_encodings(known_img[i])[0])
    # print(listdir)
    # print(known_img)
    # print(known_img_encode)
    # print(known_name)

    capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    global face_names
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        R_eyes = eyedetection(frame)
        R_mouth = mouthdetection(frame)
        pitch = rotatedetection(frame)
        print("R_EYES = ", R_eyes, "\nR_MOUTH = ", R_mouth, "\nPITCH = ", pitch)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame,model='cnn')
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_img_encode, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_img_encode, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_name[best_match_index]

                face_names.append(name)
        concentraion_caculate(R_eyes, R_mouth, pitch, frame)  # 检测哈欠在此函数中


        process_this_frame = not process_this_frame


 # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) == 27:
            break
        print('\n\n')

    # Release handle to the webcam
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # picture('Img/10.jpeg')
    time.asctime(time.localtime(time.time()))
    writefile(time.asctime(time.localtime(time.time())) + '开始上课\n')

    Vedio_Rectangle()

    time.asctime(time.localtime(time.time()))
    writefile(time.asctime(time.localtime(time.time())) + '上课结束\n\n')