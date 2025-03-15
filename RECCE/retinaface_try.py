# import retinaface
import cv2

from retinaface import RetinaFace

def extract_frames(video_path,):
    # 打开视频文件
    # video_path = '/data3/FaceForensics++/original_sequences/youtube/c23/videos/000.mp4'
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    images=[]
    while cap.isOpened():
        ret, frame = cap.read()
        # if not ret:
            # break

        # 检测人脸
        faces = RetinaFace.detect_faces(frame)
        # exit()
        # 提取人脸并保存
        # print(faces.values())
        faces=sorted(list(faces.values()), key=lambda x : x['score'], reverse=True)
        # print(faces)
        # break
        # for face in faces.values():
        face=faces[0]
        x1, y1, x2, y2 = face['facial_area']
        face_image = frame[y1:y2, x1:x2]

        # idx=os.path.join(save_dir, f'{frame_count}.jpg')
        cv2.imwrite('output.png', face_image)
        # images.append((idx, target))
        frame_count += 1
        # exit()

    cap.release()
    cv2.destroyAllWindows()
    return images

if __name__ == '__main__':
    extract_frames('/data4/FaceForensics++/original_sequences/youtube/c40/videos/001.mp4')