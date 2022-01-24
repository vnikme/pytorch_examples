Input data: text file

1. Prepare BPE dictionary.<br>
 a. Sample a subset from the huge input file: shuf -n 7848080 russia_all_addrs.txt > russia_all_addrs_small.txt<br>
 b. Make BPE dictionary: python3 prepare_bpe_dictionary.py russia_all_addrs_small.txt bpe1k 1000<br>

2. Train a model.<br>
 a. Run training: python3 seq2seq.py russia_all_addrs_tiny.txt models/model1 -1 0.00020012017 100<br>
 b. Wait sufficient time.<br>

3. Run the model in generation mode.<br>

