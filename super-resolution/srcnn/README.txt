module load python/anaconda
pip install --upgrade scikit-image

cnn build dataset:
python3 build_dataset.py --data_dir ../data/High --output_dir ../data/CNN_Output --input_size 144 --output_size 144

cnn train model:
python3 train_cnn.py --data_dir ../data/CNN_Output --model_dir ../srcnn/training --model srcnn --cuda cuda0 --optim adam

cnn evaluate model:
python3 evaluate_cnn.py --data_dir ../data/CNN_Output --model_dir ..srcnn/training --model srcnn --cuda cuda0

cnn super resolution:
python3 super_resolution.py --input_dir ../data/Low --output_dir ../data/SRCNN_Resolved --model_path ../srcnn/training/best.pth.tar