Input data: file with json lines of format {'en': en_text, 'ru': ru_text}

1. Prepare BPE dictionary.<br>
 a. Convert json to plain text: python3 json_to_txt.py data.json > data.txt<br>
 b. Make BPE dictionary: python3 prepare_bpe_dictionary.py data.txt bpe1k 1000<br>

2. Train a model.<br>
 a. Run training: python3 translation.py data.json model0 -1 0.00020012017 100.0<br>
 b. Wait sufficient time.<br>

3. Run the model in generation mode.<br>

