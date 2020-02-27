# Drowsiness_Detection
This project is about detecting drowsiness using dlib library and opencv. For this conda is used project, created a new env (info in requirements.txt) and installed required libraries. It uses the coordinates around eyes and mouth (for yawning) to detect drowsiness.It uses a pre-trained model to get the coordinates for face landmarks (total 68) and then opencv to process the data and frames from video camera feed to detect face and face landmarks.

Usage:
```
python drowsiness_detection.py -p shape_predictor_68_face_landmarks.dat -a alarm.wav
```

link to download the pre-trained [model](https://drive.google.com/open?id=1YTRX15tplq9URBEOfcLhOu3ljI3UjhMX)
