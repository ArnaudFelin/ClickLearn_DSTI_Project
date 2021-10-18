from pydub import AudioSegment 
from pydub.utils import make_chunks
import os

def process_sudio(file_name):
    '''takes .wav files in folder and creates 1min chunks. Returns the chunks inside a folder called chunked.'''
    myaudio = AudioSegment.from_file(file_name, "wav") 
    chunk_length_ms = 60000 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of 1min, 
    for i, chunk in enumerate(chunks): 
        chunk_name = './chunked/' + file_name + "_{0}.wav".format(i) 
        print ("exporting", chunk_name) 
        chunk.export(chunk_name, format="wav") 

all_file_names = os.listdir()
try:
    os.makedirs('chunked') # creating a folder named chunked
except:
    pass
for each_file in all_file_names:
    if ('.wav' in each_file):
        process_sudio(each_file)