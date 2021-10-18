import os, shutil
from time import sleep

from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.resnet import ResNetInterface
from ketos.neural_networks.dev_utils.detection import process, save_detections, merge_overlapping_detections

def list_wav_file():
    print("List all wav file name in audioTemp folder")
    filesList = []
    filesList = os.listdir("./audioTemp")
    filesList.sort()
    return filesList

def load_wav_file(filesList, index):
    print("Move 1 wav file from audioTemp folder to audioToParse before process")
    shutil.move("./audioTemp/"+filesList[index], "./audioToParse")


def unload_wav_file(filesList, index):
    print("Move 1 wav file from audioToParse folder to audioParsed after process")
    shutil.move("./audioToParse/"+filesList[index], "./audioParsed")

def main():
    
    # load the classifier and the spectrogram parameters
    model, audio_repr = ResNetInterface.load_model_file('narw.kt', './narw_tmp_folder', load_audio_repr=True)
    spec_config = audio_repr[0]['spectrogram']    
    
    # load all the name of the audio wav files
    wavFiles = list_wav_file()

    for i,n in enumerate(wavFiles):

	load_wav_file(wavFiles, n)

        # initialize the audio loader
        audio_loader = AudioFrameLoader(frame=spec_config['duration'], step=1.5, path='audioToParse', filename=None, repres=spec_config)

	# process the audio data
        detections = process(provider=audio_loader, model=model, batch_size=64, buffer=0.0, threshold=0.55, group=True, win_len=5, progress_bar=True)

        detections = merge_overlapping_detections(detections)

        # save the detections
        if os.path.isfile(args.output): os.remove(args.output) #remove, if already exists
        print(f'{len(detections)} detections saved to {args.output}')
        save_detections(detections=detections, save_to=args.output)

	unload_wav_file(wavFiles, n)

	# Sleep before load next wav file
        sleep(30)


if __name__ == "__main__":
    main()
