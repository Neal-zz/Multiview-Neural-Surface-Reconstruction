import sys
sys.path.append('../code')
import argparse

from training.idr_train import IDRTrainRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/ys_fixed_cameras.conf')
    parser.add_argument('--is_continue', default=True, action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    opt = parser.parse_args()

    trainrunner = IDRTrainRunner(conf=opt.conf,
                                batch_size=opt.batch_size,
                                nepochs=opt.nepoch,
                                exps_folder_name='exps',
                                is_continue=opt.is_continue,
                                timestamp=opt.timestamp,
                                checkpoint=opt.checkpoint,
                                scan_id=opt.scan_id
                                )

    trainrunner.run()
