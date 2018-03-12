import numpy as np
import pandas as pd
import time
import cv2
import face_recognition
import pyaudio
import wave

# <editor-fold desc="class Face">
class Face:
    """
    Class Face
    Object of the class contains spatial information about the Face as well as the .csv file name connected with it.
    Class also contains static list of detected Faces as well as the methods that serve this list.
    """
    # These are columns used when creating each .csv file connected with a new Face object
    col = np.concatenate((['Frame'], np.array(['x{1}'.format(1, i) for i in np.arange(1, 25)])
                          , np.array(['y{1}'.format(1, i) for i in np.arange(1, 25)])))

    # This is a static variable storing the id of the new Face object
    face_id = 0

    # This is a static variable storing the id of the recording experiment,
    #  user has to change it manualy for every new recording
    record_id = 2

    # Stores a path where .csv files are stored, keep in mind this is not relative path
    record_path = "C:\\Users\\Boris\\PycharmProjects\\facerecognition\\rc\\{1}.csv"

    # List of detected Face objects
    faces = []

    @staticmethod
    def get_Faces()->list:
        """
        returns list of currently detected Face objects
        """
        return Face.faces
    @staticmethod
    def remove_Face(face)->None:
        """
        removes specified Face object from the list
        """
        Face.faces.remove(face)
    @staticmethod
    def find_Face(x:int,y:int,w:int) -> 'Face':
        """
        Returns the Face object connected with givven spatial parameters.
        It finds the Face whos center is close enough to the provided center compared to the givven widths,
        if existing Face object is found, it's spatial information is updated by function parameters,
        if not it creates new Face objects and returns the reference to it.
        :param x: (x,y) is a center of new-found Face
        :param y: (x,y) is a center of new-found Face
        :param w: Width of a new-found Face (calculated as 1/3 of a diagonal of the Face retangle)
        :return:
        """
        found = None
        dmin  = 100000
        for face in Face.faces:
            d = (abs(face.x-x)+abs(face.y-y))
            if( face.w >= d and d < dmin ):
                dmin = d
                found = face
        if (found == None):
            found = Face(x, y, w)
        else:
            found.x = x
            found.y = y
            found.w = w
        return found


    def __init__(self,x:int,y:int,half_width:int):
        """
        Creates a new Face object and adds it to the list. Also creates empty .csv file connected to it.
        :param x: (x,y) is a center of new-found Face
        :param y: (x,y) is a center of new-found Face
        :param half_width: Width of a new-found Face (calculated as 1/3 of a diagonal of the Face retangle)
        """
        Face.face_id += 1
        self.df = pd.DataFrame(columns=Face.col)
        self.csv_name = 'lips{1}{2}{3}'.format(1,Face.record_id,'_',Face.face_id)
        self.df.to_csv(path_or_buf=Face.record_path.format(1,self.csv_name)
                       ,index=False)
        self.x = x
        self.y = y
        self.w = half_width
        self.id = Face.face_id
        Face.faces.append(self)
    def lip_record(self,frame:int,face_landmarks:dict)->None:
        """
        Performs a writing of a single frame lip signal of a givven Face.
        :param frame: number of the current frame
        :param face_landmarks: structure which contains all face landmarks
        """
        a = np.array([list(elem) for elem in face_landmarks['bottom_lip']])
        b = np.array([list(elem) for elem in face_landmarks['top_lip']])
        self.df.loc[0] = np.concatenate(([frame], a[:, 0], b[:, 0],
                                         a[:, 1], b[:, 1]))
        self.df.to_csv(path_or_buf=Face.record_path.format(1, self.csv_name),
                       index=False, header=False, mode='a')

    def __str__(self):
        return self.csv_name
    def __repr__(self):
        return self.csv_name
    def __eq__(self, other):
        if (other == None):
            return False
        return self.id == other.id
# </editor-fold>

# <editor-fold desc="MAIN">
# Number of audio samples captured each frame
CHUNKSIZE = 5755#int(44100/7.66) # fixed chunk size

# contains audio samples, later to be written in .wav file
audio = []

# Frame counter, gets incremented each new frame
frame_cnt = 0

time.sleep(2)

# init audio capture (microphone)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True,
                frames_per_buffer=CHUNKSIZE)
# init video capture (default camera)
video_capture = cv2.VideoCapture(0)
print("RECORDING")

while True:
    ret, frame = video_capture.read() #capture video frame
    audio.append(stream.read(CHUNKSIZE))#capture audio samples
    if (frame_cnt % 3 == 0): #every third frame perform face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # reduce the picture size for cheaper detection
        rgb_small_frame = small_frame[:, :, ::-1] #transform to rgb color format
        face_locations = face_recognition.face_locations(rgb_small_frame, 2) #face detection
    func_start = time.process_time()
    for (top, right, bottom, left) in face_locations:
        top    *= 4 #returns the coordinates to original frame size
        right  *= 4
        bottom *= 4
        left   *= 4
        x = int((left+right)/2)
        y = int((top+bottom)/2)
        w = int((abs(top-bottom)+abs(right-left))/3)
        found = Face.find_Face(x, y, w) #find or create Face object
        face_landmarks = face_recognition.face_landmarks(frame,
                        face_locations=[[top, right, bottom, left]])[0] #extract face landmarks
        found.lip_record(frame_cnt+1,face_landmarks)#write lip signal
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) #draw face retangle
    print(time.process_time() - func_start)
    cv2.imshow('Video', frame) #show the frame
    #press 'q' for stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_cnt += 1

#Stop the audio stream
stream.stop_stream()
stream.close()
p.terminate()
#stop video stream
video_capture.release()
cv2.destroyAllWindows()

#write .wav file
waveFile = wave.open("rc\\lips{1}_audio.wav".format(1,Face.record_id), 'wb')
waveFile.setnchannels(2)
waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
waveFile.setframerate(44100)
waveFile.writeframes(b''.join(audio))
waveFile.close()
# </editor-fold>