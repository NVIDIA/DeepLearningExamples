import argparse
import torch

from tacotron2.data_function import TextMelLoader
from common.utils import load_filepaths_and_text

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-files', required=True,
                        type=str, help='Path to filelist with audio paths and text')
    parser.add_argument('--mel-files', required=True,
                        type=str, help='Path to filelist with mel paths and text')
    parser.add_argument('--text-cleaners', nargs='*',
                        default=['english_cleaners'], type=str,
                        help='Type of text cleaners for input text')
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')

    return parser


def audio2mel(dataset_path, audiopaths_and_text, melpaths_and_text, args):

    melpaths_and_text_list = load_filepaths_and_text(dataset_path, melpaths_and_text)
    audiopaths_and_text_list = load_filepaths_and_text(dataset_path, audiopaths_and_text)

    data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args)

    for i in range(len(melpaths_and_text_list)):
        if i%100 == 0:
            print("done", i, "/", len(melpaths_and_text_list))

        mel = data_loader.get_mel(audiopaths_and_text_list[i][0])
        torch.save(mel, melpaths_and_text_list[i][0])

def main():

    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args = parser.parse_args()
    args.load_mel_from_disk = False

    audio2mel(args.dataset_path, args.wav_files, args.mel_files, args)

if __name__ == '__main__':
    main()
