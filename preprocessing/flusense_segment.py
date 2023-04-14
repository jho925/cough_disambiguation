from pydub import AudioSegment
import os 
import pandas as pd

df = pd.read_csv('flusense_metadata.csv')
file_list = []
labels_list = []
id_list = []
wav_list = []
for wav in os.listdir('flusense_data_og/'):
    f = os.path.join('flusense_data_og/', wav)
    if f.endswith(".wav"):
        audio = AudioSegment.from_file(file = f,
                                  format = "wav")
        df2 = df.loc[df['filename'] == f[17:]]
        i=0
        if df2.shape[0] == 1:
            if df2['label'].iloc[0] == 'cough' or df2['label'].iloc[0] == 'speech':
                new_filename = 'flusense_data/' + f[17:]
                
                if df2['label'].iloc[0] == 'cough':
                    labels_list.append('cough')
                elif df2['label'].iloc[0] == 'speech':
                    labels_list.append('speech')

                file_list.append(new_filename)
                id_list.append(wav[:-4])
                wav_list.append(f)
                curr_chunk = audio[df2['start'].iloc[0]*1000:df2['end'].iloc[0]*1000]
                curr_chunk.export('flusense_data/' + f[17:], format="wav")
        else:
            for index, row in df2.iterrows():
                if row['label'] == 'cough' or row['label'] == 'speech':
                    i+=1
                    new_filename = 'flusense_data/' + f[17:-4] + 'segment' + str(i) + '.wav'
                    file_list.append(new_filename)
                    if row['label'] == 'cough':
                        labels_list.append('cough')
                    elif row['label'] == 'speech':
                        labels_list.append('speech')
                    wav_list.append(f)
                    id_list.append(wav[:-4])
                    curr_chunk = audio[row['start']*1000:row['end']*1000]
                    curr_chunk.export(new_filename, format="wav")

df = pd.DataFrame(
    {'id': id_list,
     'wav_list': wav_list,
     'path': file_list,
     'is_cough': labels_list,
    })




df.to_csv('flusense_labels.csv',index=False)








