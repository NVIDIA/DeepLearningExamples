import argparse

def get_args(params):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("NPZ", type=str, help="input distograms and anglegrams (NN predictions)")
    parser.add_argument("FASTA", type=str, help="input sequence")
    parser.add_argument("OUT", type=str, help="output model (in PDB format)")

    parser.add_argument('-r', type=int, dest='nrestarts', default=params['NRUNS'], help='number of noisy restrarts')
    parser.add_argument('-pd', type=float, dest='pcut', default=params['PCUT'], help='min probability of distance restraints')
    parser.add_argument('-m', type=int, dest='mode', default=2, choices=[0,1,2], help='0: sh+m+l, 1: (sh+m)+l, 2: (sh+m+l)')
    parser.add_argument('-bb', type=str, dest='bb', default='', help='predicted backbone torsions')
    parser.add_argument('-sg', type=str, dest='sg', default='', help='window size and order for a Savitzky-Golay filter (comma-separated)')
    parser.add_argument('-n', type=int, dest='steps', default=1000, help='number of minimization steps')
    parser.add_argument('--save_chk', dest='save_chk', default=False, action='store_true', help='save checkpoint files to restart')
    parser.add_argument('--orient', dest='use_orient', action='store_true', help='use orientations')
    parser.add_argument('--no-orient', dest='use_orient', action='store_false')
    parser.add_argument('--fastrelax', dest='fastrelax', action='store_true', help='perform FastRelax')
    parser.add_argument('--no-fastrelax', dest='fastrelax', action='store_false')
    parser.add_argument('--roll', dest='roll', action='store_true', help='circularly shift 6d coordinate arrays by 1')
    parser.add_argument('--no-roll', dest='roll', action='store_false')
    parser.set_defaults(use_orient=True)
    parser.set_defaults(fastrelax=True)
    parser.set_defaults(roll=False)

    args = parser.parse_args()

    params['PCUT'] = args.pcut
    params['USE_ORIENT'] = args.use_orient
    params['NRUNS'] = args.nrestarts
    params['ROLL'] = args.roll

    params['NPZ'] = args.NPZ
    params['BB'] = args.bb
    params['SG'] = args.sg

    return args
