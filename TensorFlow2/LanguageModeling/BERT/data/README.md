Steps to reproduce datasets from web

1) Build the container
  * docker build -t bert_tf2 .
2) Run the container interactively
  * nvidia-docker run -it --ipc=host bert_tf2
  * Optional: Mount data volumes
    * -v yourpath:/workspace/bert_tf2/data/wikipedia_corpus/download
    * -v yourpath:/workspace/bert_tf2/data/wikipedia_corpus/extracted_articles
    * -v yourpath:/workspace/bert_tf2/data/wikipedia_corpus/raw_data
    * -v yourpath:/workspace/bert_tf2/data/wikipedia_corpus/intermediate_files
    * -v yourpath:/workspace/bert_tf2/data/wikipedia_corpus/final_text_file_single
    * -v yourpath:/workspace/bert_tf2/data/wikipedia_corpus/final_text_files_sharded
    * -v yourpath:/workspace/bert_tf2/data/wikipedia_corpus/final_tfrecords_sharded
    * -v yourpath:/workspace/bert_tf2/data/bookcorpus/download
    * -v yourpath:/workspace/bert_tf2/data/bookcorpus/final_text_file_single
    * -v yourpath:/workspace/bert_tf2/data/bookcorpus/final_text_files_sharded
    * -v yourpath:/workspace/bert_tf2/data/bookcorpus/final_tfrecords_sharded
  * Optional: Select visible GPUs
    * -e CUDA_VISIBLE_DEVICES=0

** Inside of the container starting here**
3) Download pretrained weights (they contain vocab files for preprocessing) and SQuAD
  * bash data/create_datasets_from_start.sh squad
5) "One-click" Wikipedia data download and prep (provides tfrecords)
  * bash data/create_datasets_from_start.sh pretrained wiki_only
6) "One-click" Wikipedia and BookCorpus data download and prep (provided tfrecords)
  * bash data/create_datasets_from_start.sh pretrained wiki_books
