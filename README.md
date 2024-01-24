# Voice Recognition with Machine Learning on Arduino Nano 33 BLE Sense

This is a comprehensive and detailed post on voice recognition with machine learning on the Arduino Nano 33 BLE Sense. The post covers various aspects, including hardware and software requirements, capturing audio samples, training a machine learning model, and deploying it to Arduino.


Hardware and Software Requirements: You’ll need an Arduino Nano 33 BLE Sense, 0.96 Inch I2C/IIC 4-Pin OLED Display Module, Arduino IDE and Python installed on your computer, along with Python modules scikit-learn and micromlgen.

To install the software, open your terminal and install the libraries.

pip install -U scikit-learn
pip install -U micromlgen
Capturing Audio Samples: The tutorial instructs on how to capture audio samples of specific words using the Arduino board. It involves using the microphone on the Arduino Nano 33 BLE Sense, which utilizes pulse-density modulation, and recording short words like ‘yes’, ‘no’, ‘play’, ‘stop’. Variations in voice intensity and distance from the microphone are recommended for robust data collection. The collected data is then saved in CSV format.

To flash the code to your board, copy the provided code (arduino audio capture.ino), open the Arduino IDE, and follow the on-screen instructions.

Training the Machine Learning Model: Now train a classifier model using the collected data. The script uses the scikit-learn library and an SVM (Support Vector Machine) classifier, which showed the best accuracy in this case. The trained model is then exported as a plain C code using the micromlgen library.

ArduinoWordClassification
  |-- train_classifier.py
  |-- data/
  |---- yes.csv
  |---- no.csv
  |---- play.csv
  |---- any other .csv file you recorded
  
# file: train_classifier.py

import numpy as np
from os.path import basename
from glob import glob
from sklearn.svm import SVC
from micromlgen import port
from sklearn.model_selection import train_test_split

def load_features(folder):
    dataset = None
    classmap = {}
    for class_idx, filename in enumerate(glob('%s/*.csv' % folder)):
        class_name = basename(filename)[:-4]
        classmap[class_idx] = class_name
        samples = np.loadtxt(filename, dtype=float, delimiter=',')
        labels = np.ones((len(samples), 1)) * class_idx
        samples = np.hstack((samples, labels))
        dataset = samples if dataset is None else np.vstack((dataset, samples))
    return dataset, classmap

np.random.seed(0)
dataset, classmap = load_features('data')
X, y = dataset[:, :-1], dataset[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SVC(kernel='poly', degree=2, gamma=0.1, C=100)
clf.fit(X_train, y_train)

print('Accuracy', clf.score(X_test, y_test))
print('Exported classifier to plain C')
print(port(clf, classmap=classmap))
// File: Classifier.h

#pragma once
namespace Eloquent {
    namespace ML {
        namespace Port {
            class SVM {
            public:
                /**
                * Predict class for features vector
                */
                int predict(float *x) {
                    float kernels[35] = { 0 };
                    float decisions[6] = { 0 };
                    int votes[4] = { 0 };
                    kernels[0] = compute_kernel(x,   33.0  , 41.0  , 47.0  , 54.0  , 59.0  , 61.0  , 56.0  , 51.0  , 50.0  , 51.0  , 44.0  , 32.0  , 23.0  , 15.0  , 12.0  , 8.0  , 5.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 5.0  , 3.0  , 5.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 );
                    kernels[1] = compute_kernel(x,   40.0  , 50.0  , 51.0  , 60.0  , 56.0  , 57.0  , 58.0  , 53.0  , 50.0  , 45.0  , 42.0  , 34.0  , 23.0  , 16.0  , 10.0  , 7.0  , 3.0  , 3.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 14.0  , 3.0  , 8.0  , 0.0  , 0.0  , 3.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 3.0  , 0.0  , 0.0  , 5.0  , 3.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 3.0  , 0.0  , 5.0  , 3.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 3.0  , 0.0  , 0.0  , 0.0  , 3.0 );
                    kernels[2] = compute_kernel(x,   56.0  , 68.0  , 78.0  , 91.0  , 84.0  , 84.0  , 84.0  , 74.0  , 69.0  , 64.0  , 57.0  , 44.0  , 33.0  , 18.0  , 12.0  , 8.0  , 5.0  , 9.0  , 15.0  , 12.0  , 12.0  , 9.0  , 12.0  , 7.0  , 3.0  , 10.0  , 12.0  , 6.0  , 3.0  , 0.0  , 0.0  , 0.0  , 0.0  , 6.0  , 3.0  , 6.0  , 10.0  , 10.0  , 8.0  , 3.0  , 9.0  , 9.0  , 9.0  , 8.0  , 9.0  , 9.0  , 11.0  , 3.0  , 8.0  , 9.0  , 8.0  , 8.0  , 8.0  , 6.0  , 7.0  , 3.0  , 3.0  , 8.0  , 5.0  , 3.0  , 0.0  , 3.0  , 0.0  , 0.0 );

                    // ...many other kernels computations...

                    decisions[0] = 0.722587775297
                                   + kernels[1] * 3.35855e-07
                                   + kernels[2] * 1.64612e-07
                                   + kernels[4] * 6.00056e-07
                                   + kernels[5] * 3.5195e-08
                                   + kernels[7] * -4.2079e-08
                                   + kernels[8] * -4.2843e-08
                                   + kernels[9] * -9.994e-09
                                   + kernels[10] * -5.11065e-07
                                   + kernels[11] * -5.979e-09
                                   + kernels[12] * -4.4672e-08
                                   + kernels[13] * -1.5606e-08
                                   + kernels[14] * -1.2941e-08
                                   + kernels[15] * -2.18903e-07
                                   + kernels[17] * -2.31635e-07
                            ;
                    decisions[1] = -1.658344586719
                                   + kernels[0] * 2.45018e-07
                                   + kernels[1] * 4.30223e-07
                                   + kernels[3] * 1.00277e-07
                                   + kernels[4] * 2.16524e-07
                                   + kernels[18] * -4.81187e-07
                                   + kernels[20] * -5.10856e-07
                            ;
                    decisions[2] = -1.968607562265
                                   + kernels[0] * 3.001833e-06
                                   + kernels[3] * 4.5201e-08
                                   + kernels[4] * 1.54493e-06
                                   + kernels[5] * 2.81834e-07
                                   + kernels[25] * -5.93581e-07
                                   + kernels[26] * -2.89779e-07
                                   + kernels[27] * -1.73958e-06
                                   + kernels[28] * -1.09552e-07
                                   + kernels[30] * -3.09126e-07
                                   + kernels[31] * -1.294219e-06
                                   + kernels[32] * -5.37961e-07
                            ;
                    decisions[3] = -0.720663029823
                                   + kernels[6] * 1.4362e-08
                                   + kernels[7] * 6.177e-09
                                   + kernels[9] * 1.25e-08
                                   + kernels[10] * 2.05478e-07
                                   + kernels[12] * 2.501e-08
                                   + kernels[15] * 4.363e-07
                                   + kernels[16] * 9.147e-09
                                   + kernels[18] * -1.82182e-07
                                   + kernels[20] * -4.93707e-07
                                   + kernels[21] * -3.3084e-08
                            ;
                    decisions[4] = -1.605747746589
                                   + kernels[6] * 6.182e-09
                                   + kernels[7] * 1.3853e-08
                                   + kernels[8] * 2.12e-10
                                   + kernels[9] * 1.1243e-08
                                   + kernels[10] * 7.80681e-07
                                   + kernels[15] * 8.347e-07
                                   + kernels[17] * 1.64985e-07
                                   + kernels[23] * -4.25014e-07
                                   + kernels[25] * -1.134803e-06
                                   + kernels[34] * -2.52038e-07
                            ;
                    decisions[5] = -0.934328303475
                                   + kernels[19] * 3.3529e-07
                                   + kernels[20] * 1.121946e-06
                                   + kernels[21] * 3.44683e-07
                                   + kernels[22] * -6.23056e-07
                                   + kernels[24] * -1.4612e-07
                                   + kernels[28] * -1.24025e-07
                                   + kernels[29] * -4.31701e-07
                                   + kernels[31] * -9.2146e-08
                                   + kernels[33] * -3.8487e-07
                            ;
                    votes[decisions[0] > 0 ? 0 : 1] += 1;
                    votes[decisions[1] > 0 ? 0 : 2] += 1;
                    votes[decisions[2] > 0 ? 0 : 3] += 1;
                    votes[decisions[3] > 0 ? 1 : 2] += 1;
                    votes[decisions[4] > 0 ? 1 : 3] += 1;
                    votes[decisions[5] > 0 ? 2 : 3] += 1;
                    int val = votes[0];
                    int idx = 0;

                    for (int i = 1; i < 4; i++) {
                        if (votes[i] > val) {
                            val = votes[i];
                            idx = i;
                        }
                    }

                    return idx;
                }

                /**
                * Convert class idx to readable name
                */
                const char* predictLabel(float *x) {
                    switch (predict(x)) {
                        case 0:
                            return "no";
                        case 1:
                            return "stop";
                        case 2:
                            return "play";
                        case 3:
                            return "yes";
                        default:
                            return "we have a problem";
                    }
                }

            protected:
                
                float compute_kernel(float *x, ...) {
                    va_list w;
                    va_start(w, 64);
                    float kernel = 0.0;

                    for (uint16_t i = 0; i < 64; i++) {
                        kernel += x[i] * va_arg(w, double);
                    }

                    return pow((0.1 * kernel) + 0.0, 2);
                }
            };
        }
    }
}

## Deploying the Model to Arduino: 

The final step involves uploading the C code (arduino audio classification.ino) of the trained model to the Arduino Nano 33 BLE Sense.
Once uploaded, the Arduino should be able to recognize and classify the words it was trained on in real-time.
