#https://keras.io/examples/generative/text_generation_with_miniature_gpt/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import string
import random



def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)



#基礎注意力層
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()



#因果自註意力層
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x



#前饋網絡
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x



#編碼器
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, ff_dim, d_model, dff, rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers= num_layers

        #self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=rate)

##        self.ffn = keras.Sequential(
##            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
##        )
        self.ffn = FeedForward(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        #self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, causal_mask):
        #attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        x = self.causal_self_attention(x=inputs)
        attention_output = self.dropout1(x)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        #ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2



class GPT(layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 num_layers,
                 d_model,
                 dff,
                 rate=0.1):
        super().__init__()

        self.num_layers= num_layers

##        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
##        self.ffn = keras.Sequential(
##            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
##        )
##        
##        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
##        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
##        self.dropout1 = layers.Dropout(rate)
##        self.dropout2 = layers.Dropout(rate)

        self.decoder = [
            Decoder(num_heads=num_heads,
                    ff_dim=ff_dim,
                    d_model=d_model,
                    dff=dff,
                    rate=rate)
            for _ in range(num_layers)]

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        for i in range(self.num_layers):
            out = self.decoder[i](inputs, causal_mask)
          
##        for i in range(self.num_layers):
##            attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
##            attention_output = self.dropout1(attention_output)
##            out1 = self.layernorm1(inputs + attention_output)
##            ffn_output = self.ffn(out1)
##            ffn_output = self.dropout2(ffn_output)
##            out2 = self.layernorm2(out1 + ffn_output)

        return out

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

vocab_size = 20000  # Only consider the top 20k words
maxlen = 120  # Max sequence size
embed_dim = 256  # Embedding size for each token
d_model = 256
dff = 512
num_layers = 24
num_heads = 24  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    gpt = GPT(embed_dim,
              num_heads,
              feed_forward_dim,
              num_layers,
              d_model,
              dff)
    x = gpt(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
##    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
##    model.compile(
##        "adam", loss=[loss_fn, None],
##    )  # No loss and optimization based on word embeddings from transformer block
    return model

batch_size = 128

# The dataset contains each review in a separate text file
# The text files are present in four different folders
# Create a list all files
filenames = []
directories = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    "aclImdb/test/pos",
    "aclImdb/test/neg",
]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

# Create a dataset from text files
random.shuffle(filenames)
text_ds = tf.data.TextLineDataset(filenames)     #用檔案位置載入檔案
text_ds = text_ds.shuffle(buffer_size=256)     #randon
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """ Remove html line-break tags and handle punctuation """
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# Create a vectorization layer and adapt it to the text     #kind of bert
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices


def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)     #增加維度
    tokenized_sentences = vectorize_layer(text)     #change to bert  maxlen 80
    #print(tokenized_sentences)
    x = tokenized_sentences[:, :-1]
    #print(x)
    y = tokenized_sentences[:, 1:]
    #print(y)
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)
##print(text_ds)
##print(type(text_ds))
##for example in text_ds:
##    print(example[0].numpy())
##    print(example[1].numpy())

class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 100
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)

model = create_model()

model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    "adam", loss=[loss_fn, None],
)

model.load_weights('GPT_4.0_en_50.h5')

model.fit(text_ds, verbose=1, epochs=50, callbacks=[text_gen_callback])

model.save_weights('GPT_4.0.h5')
