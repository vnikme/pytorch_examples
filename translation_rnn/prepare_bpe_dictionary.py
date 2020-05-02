import sentencepiece as spm
import sys


def main(input_path, model_prefix, vocab_size):
    spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={}'.format(input_path, model_prefix, vocab_size))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))

