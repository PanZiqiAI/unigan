
import os
from custom_pkg.basic.operations import chk_d
from custom_pkg.io.config import CanonicalConfigPyTorch


########################################################################################################################
# Config for Train
########################################################################################################################

class ConfigTrainModel(CanonicalConfigPyTorch):
    """
    The config for training models.
    """
    def __init__(self, exp_dir=os.path.join(os.path.split(os.path.realpath(__file__))[0], '../STORAGE/experiments'), **kwargs):
        super(ConfigTrainModel, self).__init__(exp_dir, **kwargs)

    def _set_directory_args(self):
        super(ConfigTrainModel, self)._set_directory_args()
        self.args.eval_vis_dir = os.path.join(self.args.ana_train_dir, 'visuals')
        self.args.eval_sv_dir = os.path.join(self.args.ana_train_dir, 'jacob_svs')

    def _add_root_args(self):
        super(ConfigTrainModel, self)._add_root_args()
        self.parser.add_argument("--version",                       type=str,   default='v2',   choices=['v1', 'v2'])
        self.parser.add_argument("--dataset",                       type=str,   default='celeba-hq')

    def _add_tree_args(self, args_dict):
        ################################################################################################################
        # Datasets
        ################################################################################################################
        if args_dict['dataset'] == 'single-mnist':
            self.parser.add_argument("--dataset_category",          type=int,   default=8)
        if args_dict['dataset'] == 'mnist':
            self.parser.set("n_classes", 10)
        if args_dict['dataset'] in ['single-mnist', 'mnist']:
            self.parser.set(['img_nc', 'img_size'], [1, 32])
        if args_dict['dataset'] in ['celeba-hq', 'afhq', 'ffhq'] or args_dict['dataset'].startswith('lsun'):
            self.parser.set('img_nc', 3)
            self.parser.add_argument("--img_size",                  type=int,   default=64)
            self.parser.add_argument("--img_n_bits",                type=int,   default=5)
            self.parser.add_argument("--dataset_maxsize",           type=int,   default=float("inf"))
        if args_dict['dataset'] == 'afhq':
            self.parser.add_argument("--dataset_category",          type=str,   default='cat')
            self.parser.add_argument("--dataset_maxsize",           type=int,   default=float("inf"))
        self.parser.add_argument("--dataset_num_threads",           type=int,   default=0)
        self.parser.add_argument("--dataset_drop_last",             type=bool,  default=True)
        self.parser.add_argument("--dataset_shuffle",               type=bool,  default=True)
        ################################################################################################################
        # Modules
        ################################################################################################################
        # 1. Generator
        self.parser.add_argument("--init_size",                     type=int,   default=4)
        self.parser.add_argument("--ncs",                           type=int,   nargs='+',  default=[512, 256, 128, 64, 16])
        self.parser.add_argument("--hidden_ncs",                    type=int,   nargs='+',  default=[512, 256, 128, 64, 32])
        self.parser.add_argument("--middle_ns_flows",               type=int,   nargs='+',  default=[3]*4)
        # 2. Discriminator.
        self.parser.add_argument("--disc",                          type=str,   default='vanilla')
        if chk_d(args_dict, 'disc', '==', 'vanilla'):
            self.parser.add_argument("--ndf",                       type=int,   default=64)
        if chk_d(args_dict, 'disc', '==', 'stylegan2'):
            self.parser.add_argument("--disc_capacity",             type=int,   default=64)
            self.parser.add_argument("--disc_max_nc",               type=int,   default=512)
            self.parser.add_argument("--disc_blur",                 type=bool,  default=True)
        ################################################################################################################
        # Optimization
        ################################################################################################################
        self.parser.add_argument("--n_grad_accum",                  type=int,   default=1)
        # Latent codes
        self.parser.add_argument("--nz",                            type=int,   default=64)
        self.parser.add_argument("--random_type",                   type=str,   default='uni')
        if chk_d(args_dict, 'random_type', '==', 'uni'):
            self.parser.add_argument("--random_uni_radius",         type=float, default=3.0)
        # Constraints
        # (1) Adversarial.
        self.parser.add_argument("--trigger_disc",                  type=int,   default=5)
        self.parser.add_argument("--trigger_gen",                   type=int,   default=10)
        self.parser.add_argument("--lambda_gen_adv",                type=float, default=1.0)
        self.parser.add_argument("--lambda_disc_adv",               type=float, default=1.0)
        if chk_d(args_dict, 'disc', '==', 'stylegan2'):
            self.parser.add_argument("--freq_step_disc_gp",         type=int,   default=4)
            self.parser.add_argument("--weight_disc_gp",            type=float, default=10.0)
            self.parser.add_argument("--lambda_disc_gp",            type=float, default=1.0)
        # (2) Generator uniformity.
        # 1> Regularization.
        self.parser.add_argument("--trigger_gunif",                 type=int,   default=0)
        self.parser.add_argument("--freq_step_gunif",               type=int,   default=4)
        self.parser.add_argument("--sn_power",                      type=int,   default=3)
        self.parser.add_argument("--lambda_gen_gunif",              type=float, default=1.0)
        self.parser.add_argument("--loss_gunif_reduction",          type=str,   default='mean',  choices=['sum', 'mean'])
        # 2> Update EMA target.
        if args_dict['version'] == 'v2':
            self.parser.add_argument("--trigger_update_sv_ema",     type=int,   default=0)
            self.parser.add_argument("--freq_step_update_sv_ema",   type=int,   default=10)
            self.parser.add_argument("--update_sv_ema_n_samples",   type=int,   default=32)
            self.parser.add_argument("--update_sv_ema_batch_size",  type=int,   default=8)
            self.parser.add_argument("--update_sv_ema_jacob_bsize", type=int,   default=96)
        # Update Generator EMA.
        self.parser.add_argument("--trigger_update_gen_ema",        type=int,   default=20000)
        self.parser.add_argument("--freq_step_update_gen_ema",      type=int,   default=10)
        ################################################################################################################
        # Evaluation
        ################################################################################################################
        # Visualization
        self.parser.add_argument("--freq_counts_step_eval_vis",             type=int,   default=1000)
        self.parser.add_argument("--eval_vis_sqrt_num",                     type=int,   default=8)
        # Jacob.
        self.parser.add_argument("--freq_counts_step_eval_jacob",           type=int,   default=10)
        self.parser.add_argument("--eval_jacob_num_samples",                type=int,   default=40)
        self.parser.add_argument("--eval_jacob_batch_size",                 type=int,   default=4)
        self.parser.add_argument("--eval_jacob_ag_bsize",                   type=int,   default=96)

    def _add_additional_args(self):
        # Epochs & batch size
        self.parser.add_argument("--steps",                         type=int,   default=200000)
        self.parser.add_argument("--batch_size",                    type=int,   default=96)
        # Learning rate
        self.parser.add_argument("--learning_rate",                 type=float, default=0.0001)
        # Frequency
        self.parser.add_argument("--freq_counts_step_log",          type=int,   default=1000)
        self.parser.add_argument("--freq_counts_step_chkpt",        type=int,   default=50)
