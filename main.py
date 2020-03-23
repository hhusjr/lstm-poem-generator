"""
主程序
@author Junru Shen
"""
from config import Config
from model import PoemModel
from preprocessing import Preprocessing


def main():
    preprocessing = Preprocessing(Config.train_poems_location)
    preprocessing.preprocess()
    model = PoemModel(
        preprocessed=preprocessing,
        weight_file=Config.weight_file,
        window_size=Config.window_size,
        learning_rate=0.001,
        batch_size=32
    )

if __name__ == '__main__':
    main()
