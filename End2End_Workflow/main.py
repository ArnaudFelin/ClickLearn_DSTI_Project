import datetime
from multiprocessing import Process, Value

import ctypes
import logging2
import os, shutil
from time import sleep

# ===================================================================================================================================
# =========== Utils ================================================================================================================
# ===================================================================================================================================
    
class ProgressBar:

    def __init__ (self, valmax, maxbar, title):
        if valmax == 0:  valmax = 1
        if maxbar > 200: maxbar = 200
        self.valmax = valmax
        self.maxbar = maxbar
        self.title  = title
    
    def update(self, val):
        import sys
        # format
        if val > self.valmax: val = self.valmax
        
        # process
        perc  = round((float(val) / float(self.valmax)) * 100)
        scale = 100.0 / float(self.maxbar)
        bar   = int(perc / scale)
  
        # render 
        out = '\r %20s [%s%s] %s' % (self.title, 'x' * bar, ' ' * (self.maxbar - bar), " Type 'q' then 'ENTER' to quit !")
        #original render
        #out = '\r %20s [%s%s] %3d %%' % (self.title, 'x' * bar, ' ' * (self.maxbar - bar), perc)
        sys.stdout.write(out)

def my_logger(msg, log_type = 'default', log_title = ''):
    
    if log_type == 'default':
        print(f'=====> {msg}')                   
    elif log_type == 'title':
        print(f'\n==================== {log_title} ====================')
        print(f'{msg}\n')

# ===================================================================================================================================
# =========== Record sound process ==================================================================================================
# ===================================================================================================================================

def record_sound_process(terminate_process):
    
    logger = logging2.Logger('record_sound_process')
    handler = logging2.FileHandler('./log/record_sound_process.log')
    logger.add_handler(handler)
    	
    logger.info('==========> record_sound_process started')

    while not terminate_process.value:
        
        d = datetime.datetime.utcnow()
        status = os.system(f"arecord -D hw:2,0 -d 60 -c 1 -r 32000 -f S16_LE -q ./audio/audioTemp/record_{d.isoformat(sep='-', timespec='milliseconds')}.wav")
        
        if status == 0:
            logger.info(f"new file record_{d.isoformat(sep='-', timespec='milliseconds')}.wav created")
            # Move one wav file from audioTemp folder to audioToParse before process
            shutil.move(f"./audio/audioTemp/record_{d.isoformat(sep='-', timespec='milliseconds')}.wav", './audio/audioToParse')
        else:
            logger.error(f'something wrong with the record, check the device. Erreur code : {status}')
 
    if terminate_process.value: logger.info('==========> record_sound_process ended')

# ===================================================================================================================================
# =========== Inference process =====================================================================================================
# ===================================================================================================================================

def list_wav_file():
    filesList = []
    filesList = os.listdir('./audio/audioToParse')
    filesList.sort()
    return filesList

def unload_wav_file(filesList, index):
    shutil.move('./audio/audioToParse/'+filesList[index], './audio/audioParsed')

def remove_wav_file(filesList, index):
    shutil.os.remove('./audio/audioToParse/'+filesList[index])

def inference_process(terminate_process):
 
    from ketos.audio.audio_loader import AudioFrameLoader
    from ketos.neural_networks.resnet import ResNetInterface
    from ketos.neural_networks.dev_utils.detection import process, save_detections, merge_overlapping_detections
    
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    
    # Set up db connection
    
    # My pc : 
    #token = "YJUzVrp6nYljzVwnRlTgzeoys1zYrcqJEwMMsX08S_RLpkrXh-3-pZK5Vk57_AickAnn88dRv9j-9IEVlKeSZA=="
    #org = "ClickLearn"
    #bucket = "ClickLearn_data"
    #client = InfluxDBClient(url="http://192.168.0.14:8086", token=token)
    
    # Manuel pc : 
    token = "NKMleslxPoSeutLZFI5pjlWO8JU2BqW1Dr5jrt9XJZGRiWUe2tnUNaUusUDfhyLMiHrl7ckM2FXsoDQQ6sSTww=="
    org = "clicklearn"
    bucket = "clicklearn"
    client = InfluxDBClient(url="http://clicklearn.gawert.de:8086", token=token)

    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    # Set up logger
    logger = logging2.Logger('inference_process')
    handler = logging2.FileHandler('./log/inference_process.log')
    logger.add_handler(handler)
    
    logger.info('==========> inference_process started')
    
    # load the classifier and the spectrogram parameters
    model, audio_repr = ResNetInterface.load_model_file('./model/narw.kt', './model/narw_tmp_folder', load_audio_repr=True)
    spec_config = audio_repr[0]['spectrogram'] 

    while not terminate_process.value:
        # List all wav file name in audioToParse folder
        wavFiles = list_wav_file()
        logger.info(f'List all wav file name in audioToParse folder :{wavFiles}')
        
        if len(wavFiles) != 0:
            for i,wavFileName in enumerate(wavFiles):
                # Initialize the audio loader
                audio_loader = AudioFrameLoader(frame=spec_config['duration'], step=1.5, path='./audio/audioToParse', filename=wavFileName, repres=spec_config)

                # Process the audio file
                detections = process(provider=audio_loader, model=model, batch_size=64, buffer=0.0, threshold=0.55, group=True, win_len=5, progress_bar=False)
                detections = merge_overlapping_detections(detections)
                
                if len(detections)!=0:
                    logger.info('**** Detections found, write it in DB ****')
                    #Save the detections
                    save_detections(detections=detections, save_to='./detections/detections_no_overlap.csv')
                    
                    for detection in detections:
                                                
                        #Retrieve the date time of the file and add delta (when the sound is detected by the model) 
                        #in order to stock it in DB
                        dt_wavFile_str = wavFileName.split('_')[1][:-4]
                        dt_obj = datetime.datetime.strptime(dt_wavFile_str, '%Y-%m-%d-%H:%M:%S.%f')
                        dt_obj = dt_obj + datetime.timedelta(seconds=detection[1])
                            
                        point = Point('upcall')\
                        .tag('species', 'narw')\
                        .tag('sea', 'atlantic')\
                        .tag('coastal_sea', 'manche')\
                        .tag('country', 'France')\
                        .tag('buoys_group', 'saint-malo_001')\
                        .tag('buoy', '001')\
                        .tag('geographic_coordinates_lat', '48.682771')\
                        .tag('geographic_coordinates_long', '-2.093856')\
                        .tag('file_name', detection[0])\
                        .tag('date_time_start', dt_obj.isoformat(sep='T'))\
                        .tag('start', detection[1])\
                        .tag('duration', detection[2])\
                        .field('score', round(detection[3],2))
                        
                        logger.info(f'write this detection in DB : {detection}')
                       
                        write_api.write(bucket, org, point)
                        sleep(2)
                        
                    # Move one wav file from audioToParse folder to audioParsed after process
                    logger.info (f'Move one wav file : {wavFileName} from audioToParse folder to audioParsed after process')
                    unload_wav_file(wavFiles, i)
                
                else:
                    
                    # No detection so remove the wav file from audioToParse folder
                    logger.info (f'No detection, remove one wav file : {wavFileName} from audioToParse folder')
                    remove_wav_file(wavFiles, i)               

        # Sleep before load next wav file
        sleep(60)
       
    if terminate_process.value: 
        logger.info('==========> inference_process ended')
        # Close DB connection
        client.close()
    
# ===================================================================================================================================
# =========== WatchDog process =====================================================================================================
# ===================================================================================================================================

def watchDog_all_process(terminate_watchDog):
    
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS    

    # shared variable with other process
    terminate_process = Value(ctypes.c_bool, False) 

    # Set up db connection
    
    # My pc : 
    #token = "YJUzVrp6nYljzVwnRlTgzeoys1zYrcqJEwMMsX08S_RLpkrXh-3-pZK5Vk57_AickAnn88dRv9j-9IEVlKeSZA=="
    #org = "ClickLearn"
    #bucket = "ClickLearn_data"
    #client = InfluxDBClient(url="http://192.168.0.14:8086", token=token)
    
    # Manuel pc : 
    token = "NKMleslxPoSeutLZFI5pjlWO8JU2BqW1Dr5jrt9XJZGRiWUe2tnUNaUusUDfhyLMiHrl7ckM2FXsoDQQ6sSTww=="
    org = "clicklearn"
    bucket = "clicklearn"
    client = InfluxDBClient(url="http://clicklearn.gawert.de:8086", token=token)
    
    write_api = client.write_api(write_options=SYNCHRONOUS)

    
    logger = logging2.Logger('watchDog_all_process')
    handler = logging2.FileHandler('./log/watchDog_all_process.log')
    logger.add_handler(handler)
    
    logger.info('==========> watchDog_all_process started')
    logger.info('start other process')
    
    # Start other process   
    p_record_sound_process = Process(target=record_sound_process, args=(terminate_process,))
    p_record_sound_process.start()
    
    p_inference_process = Process(target=inference_process, args=(terminate_process,))
    p_inference_process.start()
    
    bar = ProgressBar(100, 50, 'All process are alive :')
    j = 0
    all_process_alive = True
    
    # Wait just for start the progress bar after some not wanted log !
    sleep(80)
    # Some empty print just a matter of visual aspect
    print('')
    print('')
    
    #Infinite loop in order to :
    # - show main process and watchDog process are alive with a bar update
    # - check if the other process are well alive
    while not terminate_watchDog.value:

        #this takes ~1min
        if all_process_alive:
            for i in range(1,101):
                bar.update(i)
                sleep(0.6)
                if terminate_watchDog.value:
                    break
        else:
            print("\rError : all the process are not alive !                                    Type 'q' then 'ENTER' to quit !", end='')
            sleep(60)
        
        #check all ~5 min if the other process are alive 
        #and send a flag to say all is ok to the DB
        if j>4:
            j=0
            if p_record_sound_process.is_alive() & p_inference_process.is_alive():
                logger.info('watchDog all process are alive')
                
                point = Point('upcall')\
                .tag('species', 'narw')\
                .tag('sea', 'atlantic')\
                .tag('coastal_sea', 'manche')\
                .tag('country', 'France')\
                .tag('buoys_group', 'saint-malo_001')\
                .tag('buoy', '001')\
                .tag('geographic_coordinates_lat', '48.682771')\
                .tag('geographic_coordinates_long', '-2.093856')\
                .field('process_alive', True)

                write_api.write(bucket, org, point)    
                
            if not p_record_sound_process.is_alive():
                logger.error('record_sound_process is not alive !')
                all_process_alive = False
                
            if not p_inference_process.is_alive():
                logger.error('inference_process is not alive !')
                all_process_alive = False
        else:
            j+=1
        
         #if not all_process_alive: TODO see if we can restart all the process or the jetson ?
        
        if terminate_watchDog.value:
            logger.info('received the command to terminate all the process')
            terminate_process.value = True
            
            # Close DB connection
            client.close()
            
            p_record_sound_process.join()
            p_inference_process.join()

# ===================================================================================================================================
# =========== Main ==================================================================================================================
# ===================================================================================================================================
def main():
    # shared variable with watchDog process
    terminate_watchDog = Value(ctypes.c_bool, False) 

    my_logger('', 'title','ClickLearn main program started')
   
    # Start process    
    p_watchDog_all_process = Process(target=watchDog_all_process, args=(terminate_watchDog,))
    p_watchDog_all_process.start()
    
    while True:     
        ch = input()
    
        if ch == 'q':
            print('')
            print('Wait for the end of the process')
            terminate_watchDog.value = True
            p_watchDog_all_process.join()
            print('')
            print('All process ended')
            break
        else:
            print("\rYou hit the wrong keybord key ! If you want to quit : Type 'q' then 'ENTER' !                             ", end='')
                
    my_logger('', 'title','ClickLearn main program ended')
    

if __name__ == '__main__':
    main()
    
    
    

