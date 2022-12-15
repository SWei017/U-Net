import argparse
import sys


class BaseOptions:

    def __init__(self):
        self.initialized = False
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    def initialize(self):
        self.parser.add_argument('--model', default='SegNet', help='Define model name')
        self.parser.add_argument('--img_directory', default='/content/drive/MyDrive/Colab Notebooks', help='Crack images directory')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
        self.parser.add_argument('--result_saving_dir', type=str, default='./results',
                                 help='Saving directory for results, ie predicted image, graph')
        self.parser.add_argument('--graph_directory', type=str, default='graphs',
                                 help='Directory for loss, f1 score, etc graphs')
        self.parser.add_argument('--in_channels', type=int, default=3, help='Input channels for model')
        self.parser.add_argument('--out_channels', type=int, default=1, help='Output channels for model')
        self.parser.add_argument('--spec_images_directory', default='Training Images',
                                 help='Directory for training images')
        self.parser.add_argument('--input_image_directory', default='overlapped', help='Directory for input images')
        self.parser.add_argument('--mask_directory', default='labelled', help='Directory for masked images')
        self.parser.add_argument('--dilate_directory', default='dilated', help='Directory for dilated images')
        self.parser.add_argument('--isTrain', type=bool, default=True, help='Training or Testing')
        self.initialized = True

    def get_options(self):
        if not self.initialized:
            self.initialize()

        return self.parser.parse_args()

    def parse(self):
        if not self.initialized:
            self.initialize()
        return self.parser


if __name__ == '__main__':
    parser = BaseOptions()
    parser.get_options()
    parser = parser.get_options()
    parser.batch_size = 16
    print(parser.batch_size)




