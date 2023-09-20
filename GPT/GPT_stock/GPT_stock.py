#https://keras.io/examples/generative/text_generation_with_miniature_gpt/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import string
import random
from transformers import BertTokenizer



def causal_attention_mask(batch_size, n_dest, n_src, dtype):
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

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=rate)

        self.ffn = FeedForward(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)

    def call(self, inputs, causal_mask):
        x = self.causal_self_attention(x=inputs)
        attention_output = self.dropout1(x)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
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


vocab_size = 10000  # Only consider the top 20k words
maxlen = 180  # Max sequence size
embed_dim = 256  # Embedding size for each token
d_model = 256
dff = 512
num_layers = 32
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
    return model

batch_size = 128

filenames = []
directories = [
    "D_data",
]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

random.shuffle(filenames)
text_ds = tf.data.TextLineDataset(filenames)


datasets_input = []
datasets_output = []
for text in text_ds:
    datas = str(text.numpy())[2:-2]
    datas = datas.split(' ')
    datas = [int(float(i) * 100) for i in datas]
    if maxlen>=len(datas):
        datas = datas + [0] * (maxlen-len(datas)+1)
        
    for i in range(maxlen,len(datas)):
        data = datas[i-maxlen:i+1]
        datasets_input.append(tf.convert_to_tensor(data[:-1]))
        datasets_output.append(tf.convert_to_tensor(data[1:]))
    
##    ran = random.randint(maxlen,len(datas)-1)
##    data = datas[ran-maxlen:ran+1]
##    datasets_input.append(tf.convert_to_tensor(data[:-1]))
##    datasets_output.append(tf.convert_to_tensor(data[1:]))
    #break

text_ds = tf.data.Dataset.from_tensor_slices((datasets_input,datasets_output))


text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

class TextGenerator(keras.callbacks.Callback):
    def __init__(
        self, max_tokens, start_tokens, top_k=10, print_every=1):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        
        if (epoch + 1) % 10 == 0:
            self.model.save_weights('GPT_5.0_'+ str(epoch + 1 +100) +'.h5')
            
        if (epoch + 1) % self.print_every != 0:
            return
        
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[len(start_tokens) - maxlen:]
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

        for i in start_tokens:
            print(round(i / 100,5))


start_prompt = "0.001 0.02 0.003"
start_tokens = start_prompt.split(' ')
start_tokens = [int(float(i) * 100) for i in start_tokens]
num_tokens_generated = 100
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens)

model = create_model()

model.summary()


###自訂優化器
##class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
##    def __init__(self, d_model, warmup_steps=4000):
##        super().__init__()
##
##        self.d_model = d_model
##        self.d_model = tf.cast(self.d_model, tf.float32)
##
##        self.warmup_steps = warmup_steps
##
##    def __call__(self, step):
##        step = tf.cast(step, dtype=tf.float32)
##        arg1 = tf.math.rsqrt(step)
##        arg2 = step * (self.warmup_steps ** -1.5)
##        
##        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
##
##
##
###優化器
##learning_rate = CustomSchedule(d_model)
##
##optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
##                                     epsilon=1e-9)
##
##
##
#####優化器圖表
####plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
####plt.ylabel('Learning Rate')
####plt.xlabel('Train Step')
####plt.show()
##
##
##
###loss
##def masked_loss(label, pred):
##    mask = label != 0
##    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
##    from_logits=True, reduction='none')
##    loss = loss_object(label, pred)
##
##    mask = tf.cast(mask, dtype=loss.dtype)
##    loss *= mask
##
##    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
##    return loss
##
##
##
###accuracy
##def masked_accuracy(label, pred):
##    pred = tf.argmax(pred, axis=2)
##    label = tf.cast(label, pred.dtype)
##    match = label == pred
##
##    mask = label != 0
##
##    match = match & mask
##
##    match = tf.cast(match, dtype=tf.float32)
##    mask = tf.cast(mask, dtype=tf.float32)
##    return tf.reduce_sum(match)/tf.reduce_sum(mask)
##
###transformer.load_weights('GPT.h5')
##
###train
##model.compile(
##    loss=masked_loss,
##    optimizer=optimizer,
##    metrics=[masked_accuracy])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
"adam", loss=[loss_fn, None],
)

model.load_weights('GPT_5.0_100.h5')

model.fit(text_ds, verbose=1, epochs=50, callbacks=[text_gen_callback])
