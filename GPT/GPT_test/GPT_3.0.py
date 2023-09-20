import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import string
import random



#編碼位置加成
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)



#查找添加位置
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x



#基礎注意力層
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()



#全局自註意力層
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            return_attention_scores=True)

        self.last_attn_scores = attn_scores
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x



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



#解碼器層
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                *,
                d_model,
                num_heads,
                dff,
                dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.global_self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.causal_self_attention(x=x)
        x = self.global_self_attention(x=x)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.global_self_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x



#解碼器
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, maxlen, embed_dim,
               dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)

        self.embedding_layer = TokenAndPositionEmbedding(maxlen=maxlen,
                                                   vocab_size=vocab_size,
                                                   embed_dim=embed_dim)
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x):
        # `x` is token-IDs shape (batch, target_seq_len)
        #x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        #x = self.embedding_layer(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
          x  = self.dec_layers[i](x)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x



#Transformer
class GPT(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, maxlen, embed_dim, dropout_rate=0.1):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen=maxlen,
                                                   vocab_size=input_vocab_size,
                                                   embed_dim=embed_dim)

        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               maxlen=maxlen,
                               embed_dim=embed_dim,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(input_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        #x  = inputs
        x = self.embedding_layer(inputs)

        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size=batch_size,
                                            n_dest=seq_len,
                                            n_src=seq_len,
                                            dtype=tf.bool)

        attention_output = self.att(x, x, attention_mask=causal_mask)

        x = self.decoder(attention_output)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        # Return the final output and the attention weights.
        return logits


##def causal_attention_mask(batch_size, n_dest, n_src, dtype):
##    """
##    Mask the upper half of the dot product matrix in self attention.
##    This prevents flow of information from future tokens to current token.
##    1's in the lower triangle, counting from the lower right corner.
##    """
##    i = tf.range(n_dest)[:, None]
##    j = tf.range(n_src)
##    m = i >= j - n_src + n_dest
##    mask = tf.cast(m, dtype)
##    mask = tf.reshape(mask, [1, n_dest, n_src])
##    mult = tf.concat(
##        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
##    )
##    return tf.tile(mask, mult)
##
##
##class TransformerBlock(layers.Layer):
##    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
##        super().__init__()
##        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
##        self.ffn = keras.Sequential(
##            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
##        )
##        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
##        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
##        self.dropout1 = layers.Dropout(rate)
##        self.dropout2 = layers.Dropout(rate)
##
##    def call(self, inputs):
##        input_shape = tf.shape(inputs)
##        batch_size = input_shape[0]
##        seq_len = input_shape[1]
##        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
##        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
##        attention_output = self.dropout1(attention_output)
##        out1 = self.layernorm1(inputs + attention_output)
##        ffn_output = self.ffn(out1)
##        ffn_output = self.dropout2(ffn_output)
##        return self.layernorm2(out1 + ffn_output)
##
##class TokenAndPositionEmbedding(layers.Layer):
##    def __init__(self, maxlen, vocab_size, embed_dim):
##        super().__init__()
##        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
##        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
##
##    def call(self, x):
##        maxlen = tf.shape(x)[-1]
##        positions = tf.range(start=0, limit=maxlen, delta=1)
##        positions = self.pos_emb(positions)
##        x = self.token_emb(x)
##        return x + positions

vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_layers = 4
d_model = 256
dff = 512
num_heads = 8
dropout_rate = 0.1
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

##def create_model():
##    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
##    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
##    x = embedding_layer(inputs)
##    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
##    x = transformer_block(x)
##    outputs = layers.Dense(vocab_size)(x)
##    model = keras.Model(inputs=inputs, outputs=[outputs, x])
##    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
##    model.compile(
##        "adam", loss=[loss_fn, None],
##    )  # No loss and optimization based on word embeddings from transformer block
##    return model

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
            y = self.model.predict(x)
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
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)

#model = create_model()

model = GPT(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=len(vocab),
    maxlen=maxlen,
    embed_dim=embed_dim,
    dropout_rate=dropout_rate)

model(tf.convert_to_tensor([[1]]))

model.summary()



#自訂優化器
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



#優化器
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)



#loss
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss



#accuracy
def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)



#train
model.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

##loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
##model.compile("adam", loss=[loss_fn, None])

model.fit(text_ds, verbose=1, epochs=25, callbacks=[text_gen_callback])

model.save_weights('GPT_3.0.h5')
