import argparse
from transformer.data.process_data import _VOCAB_FILE

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/tmp/translate_ende",
      help="[default: %(default)s] Directory containing training and "
           "evaluation data, and vocab file used for encoding.",
      metavar="<DD>")
  parser.add_argument(
      "--vocab_file", "-vf", type=str, default=_VOCAB_FILE,
      help="[default: %(default)s] Name of vocabulary file.",
      metavar="<vf>")
  parser.add_argument(
      "--model_dir", "-md", type=str, default="/tmp/transformer_model",
      help="[default: %(default)s] Directory to save Transformer model "
           "training checkpoints",
      metavar="<MD>")
  parser.add_argument(
      "--init_checkpoint", "-ic", type=str, default=None,
      help="Initial checkpoint (usually from a pre-trained BERT model).",
      metavar="<ic>")
  parser.add_argument(
      "--save_checkpoints_steps", "-sc", type=int, default=0,
      help="Save checkpoint every <SC> train iterations.",
      metavar="<SC>")
  parser.add_argument(
      "--params", "-p", type=str, default="big", choices=["base", "big"],
      help="[default: %(default)s] Parameter set to use when creating and "
           "training the model.",
      metavar="<P>")
  parser.add_argument(
      "--num_cpu_cores", "-nc", type=int, default=4,
      help="[default: %(default)s] Number of CPU cores to use in the input "
           "pipeline.",
      metavar="<NC>")
  parser.add_argument(
      "--batch_size", "-b", type=int, default=0,
      help="Override default batch size parameter in prams",
      metavar="<B>")
  parser.add_argument(
      "--learning_rate", "-lr", default='0.0',
      help="Override default learning rate parameter in params",
      metavar="<LR>")
  parser.add_argument(
      "--warmup_steps", "-ws", type=int, default=4000,
      help="Override default warmup_steps parameter in params",
      metavar="<WS>")

  # Flags for training with steps
  parser.add_argument(
      "--train_steps", "-ts", type=int, default=5000,
      help="Total number of training steps. If both --train_epochs and "
           "--train_steps are not set, the model will train for 10 epochs.",
      metavar="<TS>")
  parser.add_argument(
      "--steps_between_eval", "-sbe", type=int, default=1000,
      help="[default: %(default)s] Number of training steps to run between "
           "evaluations.",
      metavar="<SBE>")

  # BLEU score computation
  parser.add_argument(
      "--bleu_source", "-bs", type=str, default=None,
      help="Path to source file containing text translate when calculating the "
           "official BLEU score. Both --bleu_source and --bleu_ref must be "
           "set. The BLEU score will be calculated during model evaluation.",
      metavar="<BS>")
  parser.add_argument(
      "--bleu_ref", "-br", type=str, default=None,
      help="Path to file containing the reference translation for calculating "
           "the official BLEU score. Both --bleu_source and --bleu_ref must be "
           "set. The BLEU score will be calculated during model evaluation.",
      metavar="<BR>")
  parser.add_argument(
      "--bleu_threshold", "-bt", type=float, default=None,
      help="Stop training when the uncased BLEU score reaches this value. "
           "Setting this overrides the total number of steps or epochs set by "
           "--train_steps or --train_epochs.",
      metavar="<BT>")

  # Dataset options
  parser.add_argument(
          "--sentencepiece", "-sp", action='store_true',
          help="Use SentencePiece tokenizer. Warning: In order to use SP "
               "you have to preprocess dataset with SP as well")

  parser.add_argument(
      "--random_seed", "-rs", type=int, default=1,
      help="The random seed to use.", metavar="<SEED>")
  parser.add_argument(
      "--enable_xla", "-enable_xla", action="store_true",
      help="Enable JIT compile with XLA.")
  parser.add_argument(
      "--enable_amp", "-enable_amp", action="store_true",
      help="Enable mixed-precision, fp16 where possible.")
  parser.add_argument(
      "--enable_horovod", "-enable_hvd", action="store_true",
      help="Enable multi-gpu training with horovod")
  parser.add_argument(
      "--report_loss", "-rl", action="store_true",
      help="Report throughput and loss in alternative format")

  return parser

