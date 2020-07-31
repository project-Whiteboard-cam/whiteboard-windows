import pyaudio
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from threading import Thread
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

captions = ['*', '*', '*', '*']
ifchange = False

from tkinter import *

master = Tk()

master.geometry('800x100')
master.title('IBM Speech To Text')

F = ("arial", 20)

w = Label(master, text="Hello, world!", font=F)

w.pack()


def update():
    w.config(text=captions[-2] + "\n" + captions[-1])


from queue import Queue, Full

# Initalize queue to store the recordings
CHUNK = 1024 * 4

BUF_MAX_SIZE = CHUNK * 10
# Buffer for storing store audio
q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK)))

# instance of AudioSource
audio_source = AudioSource(q, True, True)

# Prepare Speech to Text Service

# initialize speech to text service
authenticator = IAMAuthenticator("<Watson speech to text API key>")
speech_to_text = SpeechToTextV1(authenticator=authenticator)
speech_to_text.set_service_url(
    '<service url>')


# define callback for the speech to text service
class MyRecognizeCallback(RecognizeCallback):

    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
        global captions
        global w
        captions.pop(0)
        captions.append(transcript[0]['transcript'])
        w.after(1, update)
        print(transcript)

    def on_connected(self):
        print('Connection was successful')

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

    def on_listening(self):
        print('Service is listening')

    def on_hypothesis(self, hypothesis):
        # print(hypothesis)
        pass

    def on_data(self, data):
        # print(data)
        pass

    def on_close(self):
        print("Connection closed")


# this function will initiate the recognize service and pass in the AudioSource
def recognize_using_weboscket(*args):
    mycallback = MyRecognizeCallback()
    speech_to_text.recognize_using_websocket(audio=audio_source,
                                             content_type='audio/l16; rate=44100',
                                             recognize_callback=mycallback,
                                             interim_results=True)
    return mycallback


# Prepare the for recording using Pyaudio

# Variables for recording the speech
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


# define callback for pyaudio to store the recording in queue
def pyaudio_callback(in_data, frame_count, time_info, status):
    try:
        q.put(in_data)
    except Full:
        pass  # discard
    return (None, pyaudio.paContinue)


# instantiate pyaudio
audio = pyaudio.PyAudio()

# open stream using callback
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=pyaudio_callback,
    start=False
)

# Start the recording and start service to recognize the stream

print("Enter CTRL+C to end recording...")
stream.start_stream()

try:
    recognize_thread = Thread(target=recognize_using_weboscket, args=())

    recognize_thread.start()

    master.mainloop()

except KeyboardInterrupt:
    # stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    audio_source.completed_recording()