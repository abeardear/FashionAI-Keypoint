cd fusemodel
python ./argsoftmax_predict.py
cd ../crop
python ./predict_crop.py
cd ../
python merge.py