## T5-like span-masked language modeling

In the following, we demonstrate how to train a T5 model using the span-masked language model 
objective as proposed in the [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).
More specifically, we demonstrate how JAX/Flax can be leveraged 
to pre-train [**`google/t5-v1_1-base`**](https://huggingface.co/google/t5-v1_1-base)
in Norwegian on a single TPUv3-8 pod.

The example script uses the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

Let's start by creating a model repository to save the trained model and logs.
Here we call the model `"norwegian-t5-base"`, but you can change the model name as you like.

To setup all relevant files for training, let's create a directory.

```bash
cd ./norwegian-t5-base
```

### Train tokenizer

In the first step, we train a tokenizer to efficiently process the text input for the model. 
We make use of the [tokenizers](https://github.com/huggingface/tokenizers) library to train 
a sentencepiece unigram tokenizer as shown in [t5_tokenizer_model.py](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling/t5_tokenizer_model.py) 
which is heavily inspired from [yandex-research/DeDLOC's tokenizer model](https://github.com/yandex-research/DeDLOC/blob/5c994bc64e573702a9a79add3ecd68b38f14b548/sahajbert/tokenizer/tokenizer_model.py) .

The tokenizer is trained on the complete Norwegian dataset of OSCAR
and consequently saved in the cloned model directory.
This can take up to 120 minutes depending on your hardware ☕☕☕ .

```python
import datasets

from t5_tokenizer_model import SentencePieceUnigramTokenizer


vocab_size = 32_000
input_sentence_size = None

# Initialize a dataset
dataset = datasets.load_dataset("oscar", name="unshuffled_deduplicated_no", split="train")

tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i: i + batch_length]["text"]


# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)

# Save files to disk
tokenizer.save("./norwegian-t5-base/tokenizer.json")
```

### Create configuration

Next, we create the model's configuration file. This is as simple 
as loading and storing [`**google/t5-v1_1-base**`](https://huggingface.co/google/t5-v1_1-base)
in the local model folder:

```python
from transformers import T5Config

config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
config.save_pretrained("./norwegian-t5-base")
```

Great, we have set up our model repository. During training, we will automatically
push the training logs and model weights to the repo.

### Train model

Next we can run the example script to pretrain the model:

```bash
python run_t5_mlm_flax.py \
	--output_dir="./norwegian-t5-base" \
	--model_type="t5" \
	--config_name="./norwegian-t5-base" \
	--tokenizer_name="./norwegian-t5-base" \
	--dataset_name="oscar" \
	--dataset_config_name="unshuffled_deduplicated_no" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" \
	--push_to_hub
```

Training should converge at a loss and accuracy 
of 2.36 and 57.0 respectively after 3 epochs on a single TPUv3-8.
This should take around 4.5 hours.
Training statistics can be accessed on directly on the 🤗 [hub](https://huggingface.co/patrickvonplaten/t5-base-norwegian/tensorboard)

