rsync -a --progress --exclude-from 'exclude.txt' server@192.168.1.173:/home/server/Documents/projects/NumberRecognition ./

rsync -a --progress server@192.168.1.173:/home/server/Documents/projects/NumberRecognition/AttetionWithTorch/model4/model/training/simple_model_with_fake_image_2016_06_07_15_23_34.t7 .