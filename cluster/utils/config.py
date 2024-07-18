import configargparse


def parse_args():
    def add_common_cmdline_args(parser):
        # for cluster runs
        #parser.add_argument('--cfg', required=False, is_config_file=True, help='cfg file path')
        parser.add_argument('--opts', default=[], nargs='*', help='additional options to update config')
        parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
        parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
        parser.add_argument('--bid', type=int, default=10, help='amount of bid for cluster')
        parser.add_argument('--memory', type=int, default=64000, help='memory amount for cluster')
        parser.add_argument('--gpu_min_mem', type=int, default=12000, help='minimum amount of GPU memory')
        parser.add_argument('--gpu_arch', default=['tesla', 'quadro', 'rtx'],
                            nargs='*', help='additional options to update config')
        parser.add_argument('--num_cpus', type=int, default=8, help='num cpus for cluster')
        parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing SMPL mesh files (.npz).')
        parser.add_argument('--output-dir', type=str, required=True, help='Output directory for STAR parameter files (.npz).')
        parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
        return parser

    # For Blender main parser
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'BEDLAM SMPL to STAR conversion using BoMoTo'

    parser = configargparse.ArgumentParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='star-conversion-bomoto',)
    parser.add_argument('--object_key', type=int, default=None,
                        help='the object key to run the experiment on')
    parser.add_argument('--cluster_batch_size', type=int, default=None,
                        help='batch size to split the total meshes for parallel runs on cluster')
    parser.add_argument('--cluster_start_idx', type=int, default=0,
                        help='start index for the cluster batch')


    parser = add_common_cmdline_args(parser)

    args = parser.parse_args()
    print(args, end='\n\n')

    return args
