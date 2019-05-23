Made by Bastien Gonzales and Gauthier Perrod

Requirements : 
keras
tensorflow gpu
numpy
pandas
pickle
tqdm
sklearn
gc
scipy
re
itertools
time



need a folder data with inside the cooking stack echange dataset provided (raw)
an empty folder params
an empty folder resuts

-First create the dataset using : python parse.py
- Then you can train the LSTM (warning it needs around 100 epochs to converge, 18s by epochs on my GTX1070)
	by running python lstm.py
- You can obtain the metrics corresponding : python lstm_metrics.py
- And you can run the logreg with : python logreg.py





