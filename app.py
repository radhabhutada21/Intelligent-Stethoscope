from flask import Flask, request, render_template
import tensorflow
import librosa
import numpy as np
import warnings
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Disable AutoGraph warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def func(mp):
    def extract_features(file_name):
        max_pad_len = 862
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20) 
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None 
        return mfccs
    @tensorflow.autograph.experimental.do_not_convert
    def predict(mp):
        D_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
        features = [] 
        data = extract_features(mp)
        features.append(data)
        print('Finished feature extraction from ', mp)
        features = np.array(features)
        model = tensorflow.keras.models.load_model('model.h5')
    
        result=model.predict(np.array([features[0]]),verbose=0)
        dp=list(zip(D_names,list(*result)))
        s=''' '''
        for i,j in dp:
            print(i,':',j)
            s+=str(i)+':'+str(j)+'\n'
        res=max(dp,key=lambda x:x[1])
        #print('The predicted Disease for the patient was:',res[0],':{:.2f}'.format(res[1]),'%')
        ss=str(res[0])+':{:.2%}'.format(res[1])
        return  ss
    return predict(mp)
    
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the uploaded file
            file = request.files['file']

            # Save the file to disk
            file_path = 'audio.wav'
            file.save(file_path)

            # Extract audio features using MFCCs
            prediction = func(file_path)

            # Return the prediction as a string
            return render_template('index.html',prediction=f'The predicted Disease is {prediction}')

        except Exception as e:
            # Log the error message instead of printing it
            logging.exception('Error occurred while processing file')
            error_msg = f'Error: could not process file. Line {e.__traceback__.tb_lineno}'
            return error_msg

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
