from options.base_options import BaseOptions


class TrainOptions(BaseOptions):

    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser = BaseOptions().parse()
        self.initialized = False

    def add_argument(self):
        self.parser.add_argument('--save_model_frequency', type=int, default=5,
                                 help='Frequency of saving model at every epochs')
        self.parser.add_argument('--model_weight_save_path', default='./checkpoints',
                                 help='Saving folder path for model weights')
        self.parser.add_argument('--epochs', type=int, default=100, help='Total epochs to run.')
        self.parser.add_argument('--spec_images_directory', default='Training Images', help='Directory for training images')
        self.parser.add_argument('--input_image_directory', default='overlapped', help='Directory for input images')
        self.parser.add_argument('--mask_directory', default='labelled', help='Directory for masked images')
        self.parser.add_argument('--dilate_directory', default='dilated', help='Directory for dilated images')
        self.parser.add_argument('--optimizer', default='Adamax', help='Optimizer used for neural network model')
        self.parser.add_argument('--loss_function', default='dice_loss',
                                 help='Loss function used for neural network model')
        self.parser.add_argument('--scheduler', default='ReduceLROnPlateau', help='Learning rate scheduler')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
        self.parser.add_argument('--phase', default=['train', 'valid'], help='Phase for training or validating')
        self.parser.add_argument('--last_epoch', default=0, help='Epoch from previous training')
        self.parser.add_argument('--save_latest_frequency', default=1,
                                 help='Saving frequency results such as loss, f1 score, accuracy, etc')
        self.parser.add_argument('--save_result_images_frequency', default=5, help='Saving frequency for result image')
        self.parser.add_argument('--model_state_dict', type=str, default='model_state_dict.pth',
                                 help='File name for saving model')
        self.parser.add_argument('--best_model_state_dict', type=str, default='best_model_state_dict.pth',
                                 help='File name for saving best model')
        self.parser.add_argument('--result_data_filename', type=str, default='results_data.pkl',
                                 help='File name for stored datas, ie loss, accuracy...')
        self.parser.add_argument('--cont_train', default=False, help='Continue last training')
        self.parser.add_argument('--result_image_folder', default='images', help='Saving folder name for result image')
        self.parser.add_argument('--ricap_probability', type=int, default=0.3, help='Probability of implementing RICAP')
        self.parser.add_argument('--filename_suffix', type=str, default='', help='Input file name suffix (Optional)')
        self.parser.add_argument('--isTrain', type=bool, default=True, help='Training or testing phase')
        self.initialized = True

    def get_options(self):
        if not self.initialized:
            self.add_argument()
        return self.parser.parse_args()

    def parse(self):
        if not self.initialized:
            self.add_argument()
        return self.parser


if __name__ == '__main__':
    parser = TrainOptions()
    opt = parser.get_options()
    print(opt)
