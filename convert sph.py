from sphfile import SPHFile
import glob
import os
""""Convert SPH file to wav"""

dialects_path = "/home/marcin/Pobrane/TIMIT"
root_dir = os.path.join(dialects_path, '**/*.WAV')
wav_files = glob.glob(root_dir, recursive=True)

for wav_file in wav_files:
    sph = SPHFile(wav_file)
    txt_file = ""
    txt_file = wav_file[:-3] + "TXT"

    f = open(txt_file,'r')
    for line in f:
        words = line.split(" ")
        start_time = (int(words[0])/16000)
        end_time = (int(words[1])/16000)
    print("writing file ", wav_file)
    sph.write_wav(wav_file.replace(".WAV",".wav"),start_time,end_time)