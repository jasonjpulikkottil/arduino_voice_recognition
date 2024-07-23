# Voice Recognition with Machine Learning on Arduino Nano 33 BLE Sense

This is a project on voice recognition with machine learning on the Arduino Nano 33 BLE Sense. The project covers various aspects, including hardware and software requirements, capturing audio samples, training a machine learning model, and deploying it to Arduino.


## Hardware and Software Requirements: 

You’ll need an Arduino Nano 33 BLE Sense, 0.96 Inch I2C/IIC 4-Pin OLED Display Module, Arduino IDE and Python installed on your computer, along with Python modules scikit-learn and micromlgen.

To install the software, open your terminal and install the libraries.

pip install -U scikit-learn
pip install -U micromlgen

## Capturing Audio Samples: 

It involves using the microphone on the Arduino Nano 33 BLE Sense, which utilizes pulse-density modulation, and recording short words like ‘yes’, ‘no’, ‘play’, ‘stop’. Variations in voice intensity and distance from the microphone are recommended for robust data collection. The collected data is then saved in CSV format.

To flash the code to your board, copy the provided code (arduino_audio_capture.ino), open the Arduino IDE, and follow the on-screen instructions.

## Training the Machine Learning Model: 

Now train a classifier model using the collected data. The script uses the scikit-learn library and an SVM (Support Vector Machine) classifier, which showed the best accuracy in this case. The trained model is then exported as a plain C code using the micromlgen library.

## Deploying the Model to Arduino: 

The final step involves uploading the C code (arduino_audio_classification.ino) of the trained model to the Arduino Nano 33 BLE Sense.
Once uploaded, the Arduino should be able to recognize and classify the words it was trained on in real-time.
