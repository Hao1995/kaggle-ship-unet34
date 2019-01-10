IMG_PATH = 'data/train/ship/'
SEG_PATH = 'data/label/'
SAVE_WEIGHTS_PATH = 'results/'
SEG_FILE = 'data/train_ship_segmentations_v2.csv'

TEST_IMG_PATH = 'data/train/ship/'
SEG_RESULT_PATH = 'data/seg_results/'
SCORE_RESULT_PATH = 'data/score_results/'

img_size_ori = 768
img_size_target = 768
batch_size = 1 # 256:64, 384:32, 768:6(8)
epochs = 10

threshold = 0.5

