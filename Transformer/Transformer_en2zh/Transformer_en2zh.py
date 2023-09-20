#https://www.tensorflow.org/text/tutorials/transformer#training
#datasets
#https://huggingface.co/datasets/zetavg/coct-en-zh-tw-translations-twp-300k
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
#import tensorflow.compat.v1 as tf

import tensorflow_text

import pandas as pd

from transformers import BertTokenizer

tf.compat.v1.enable_eager_execution()



###BertTokenizer 中文編碼
tokenizer_zh = BertTokenizer.from_pretrained('bert-base-chinese')
#print(tokenizer_zh.tokenize('你好阿'))
#print(tokenizer_zh.vocab_size)

###BertTokenizer 英文編碼
tokenizer_en = BertTokenizer.from_pretrained('bert-base-cased')
#print(tokenizer_en.tokenize('How are you?'))



#載入parquet檔
ds = pd.read_parquet(r'C:\Users\yinyo\OneDrive\Desktop\notebook\deep-learning\Transformer\datasets_zh_to_en\train-00000-of-00001-0516df60d57d3da8.parquet')

MAX_TOKENS=128

train_examples = ds.head(int(ds.shape[0]*0.80))

train_example_en_list=[]
for train_example in train_examples['en']:
    #en = tokenizer_en.tokenize(train_example)     #to token
    en = tokenizer_en.encode(train_example)     #to id
    for i in range(len(en),MAX_TOKENS):
        en.append(0)
    en = en[:MAX_TOKENS]
    train_example_en_list.append(tf.convert_to_tensor(en))
    #break

train_example_zh_inputs_list=[]
train_example_zh_labels_list=[]
for train_example in train_examples['ch']:
    zh = tokenizer_zh.encode(train_example)
    for i in range(len(zh),MAX_TOKENS+1):
        zh.append(0)
    zh = zh[:(MAX_TOKENS+1)]
    train_example_zh_inputs_list.append(tf.convert_to_tensor(zh[:-1]))
    train_example_zh_labels_list.append(tf.convert_to_tensor(zh[1:]))
    #break
    
#train_examples = tf.data.Dataset.from_tensor_slices((train_examples['en'].tolist(),train_examples['ch'].tolist()))
train_examples = tf.data.Dataset.from_tensor_slices(((train_example_en_list,train_example_zh_inputs_list),train_example_zh_labels_list))
#train_examples = tf.stack([((train_example_en_list,train_example_zh_inputs_list),train_example_zh_labels_list)], 0)
train_examples = train_examples.prefetch(1)



val_examples = ds.tail(int(ds.shape[0]*0.20))

val_examples_en_list=[]
for val_example in val_examples['en']:
    en = tokenizer_en.encode(val_example)
    for i in range(len(en),MAX_TOKENS):
        en.append(0)
    en = en[:MAX_TOKENS]
    val_examples_en_list.append(tf.convert_to_tensor(en))
    #break

val_examples_zh_inputs_list=[]
val_examples_zh_labels_list=[]
for val_example in val_examples['ch']:
    zh = tokenizer_zh.encode(val_example)
    for i in range(len(zh),MAX_TOKENS+1):
        zh.append(0)
    zh = zh[:(MAX_TOKENS+1)]
    val_examples_zh_inputs_list.append(tf.convert_to_tensor(zh[:-1]))
    val_examples_zh_labels_list.append(tf.convert_to_tensor(zh[1:]))
    #break
    
#train_examples = tf.data.Dataset.from_tensor_slices((train_examples['en'].tolist(),train_examples['ch'].tolist()))
val_examples = tf.data.Dataset.from_tensor_slices(((val_examples_en_list,val_examples_zh_inputs_list),val_examples_zh_labels_list))
val_examples = val_examples.prefetch(1)



###datasets
##examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
##                               with_info=True,
##                               as_supervised=True)
##
##train_examples, val_examples = examples['train'], examples['validation']
##
##print(type(train_examples))
##
###BertTokenizer 文字轉編碼
##model_name = 'ted_hrlr_translate_pt_en_converter'
##tf.keras.utils.get_file(
##    f'{model_name}.zip',
##    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
##    cache_dir='.', cache_subdir='', extract=True
##)
##
##tokenizers = tf.saved_model.load(model_name)
##
##[item for item in dir(tokenizers.en) if not item.startswith('_')]
##
##
##
#####test print
####for pt_examples, en_examples in train_examples.batch(3).take(1):
####
####    #print pt en
####    print('> Examples in Portuguese:')
####    for pt in pt_examples.numpy():
####        print(pt.decode('utf-8'))
####    print()
####
####    print('> Examples in English:')
####    for en in en_examples.numpy():
####        print(en.decode('utf-8'))
####
####    #原先string
####    print('> This is a batch of strings:')
####    for en in en_examples.numpy():
####        print(en.decode('utf-8'))
####
####    #轉成數據 編碼
####    encoded = tokenizers.en.tokenize(en_examples)
####
####    print('> This is a padded-batch of token IDs:')
####    for row in encoded.to_list():
####      print(row)
####
####    #轉回文本 解碼
####    round_trip = tokenizers.en.detokenize(encoded)
####
####    print('> This is human-readable text:')
####    for line in round_trip.numpy():
####        print(line.decode('utf-8'))
####
####    #解碼成另種格式
####    print('> This is the text split into tokens:')
####    tokens = tokenizers.en.lookup(encoded)
####    print(tokens)
##
##
##
#####文本tokens建表格
####lengths = []
####
####for pt_examples, en_examples in train_examples.batch(1024):
####    pt_tokens = tokenizers.pt.tokenize(pt_examples)
####    lengths.append(pt_tokens.row_lengths())
####
####    en_tokens = tokenizers.en.tokenize(en_examples)
####    lengths.append(en_tokens.row_lengths())
####    print('.', end='', flush=True)
####
####all_lengths = np.concatenate(lengths)
####
####plt.hist(all_lengths, np.linspace(0, 500, 101))
####plt.ylim(plt.ylim())
####max_length = max(all_lengths)
####plt.plot([max_length, max_length], plt.ylim())
####plt.title(f'Maximum tokens per example: {max_length}')
####plt.show()



###適合訓練的datasets
##MAX_TOKENS=128
##
##def prepare_batch(en, zh):
##    en = tokenizer_en.tokenize(en)      # Output is ragged.
##    en = en[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
##    en = en.to_tensor()  # Convert to 0-padded dense Tensor
##
##    zh = tokenizer_zh.tokenize(zh)
##    zh = zh[:, :(MAX_TOKENS+1)]
##    zh_inputs = zh[:, :-1].to_tensor()  # Drop the [END] tokens
##    zh_labels = zh[:, 1:].to_tensor()   # Drop the [START] tokens
##
##    return (en, zh_inputs), zh_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 16

def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        #.map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))



# Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)



#pt、en
for (pt, en), en_labels in train_batches.take(1):
    break

##print(pt.shape)
##print(en.shape)
##print(en_labels.shape)
##
##print(en[0][:10])
##print(en_labels[0][:10])



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



###函數圖表
##pos_encoding = positional_encoding(length=2048, depth=512)
##
### Check the shape.
##print(pos_encoding.shape)
##
### Plot the dimensions.
##plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
##plt.ylabel('Depth')
##plt.xlabel('Position')
##plt.colorbar()
##plt.show()



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



#套用並測試
embed_pt = PositionalEmbedding(vocab_size=tf.convert_to_tensor(tokenizer_en.vocab_size), d_model=512)
embed_en = PositionalEmbedding(vocab_size=tf.convert_to_tensor(tokenizer_zh.vocab_size), d_model=512)

pt_emb = embed_pt(pt)
en_emb = embed_en(en)

##print(en_emb._keras_mask)



#基礎注意力層
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()



#交叉注意力層
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

###test
##sample_ca = CrossAttention(num_heads=2, key_dim=512)
##
##print(pt_emb.shape)
##print(en_emb.shape)
##print(sample_ca(en_emb, pt_emb).shape)



#全局自註意力層
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

###test
##sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)
##
##print(pt_emb.shape)
##print(sample_gsa(pt_emb).shape)



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

###test
##sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)
##
##print(en_emb.shape)
##print(sample_csa(en_emb).shape)



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

###test
##sample_ffn = FeedForward(512, 2048)
##
##print(en_emb.shape)
##print(sample_ffn(en_emb).shape)



#編碼器層
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

###test
##sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
##
##print(pt_emb.shape)
##print(sample_encoder_layer(pt_emb).shape)



#解碼器
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
          x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.

###test
##sample_encoder = Encoder(num_layers=4,
##                         d_model=512,
##                         num_heads=8,
##                         dff=2048,
##                         vocab_size=8500)
##
##sample_encoder_output = sample_encoder(pt, training=False)
##
##print(pt.shape)
##print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.



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

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x

###test
##sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)
##
##sample_decoder_layer_output = sample_decoder_layer(
##    x=en_emb, context=pt_emb)
##
##print(en_emb.shape)
##print(pt_emb.shape)
##print(sample_decoder_layer_output.shape)  # `(batch_size, seq_len, d_model)`



#解碼器
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
          x  = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

###test
##sample_decoder = Decoder(num_layers=4,
##                         d_model=512,
##                         num_heads=8,
##                         dff=2048,
##                         vocab_size=8000)
##
##output = sample_decoder(
##    x=en,
##    context=pt_emb)
##
##print(en.shape)
##print(pt_emb.shape)
##print(output.shape)



#Transformer
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

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



#參數
num_layers = 24
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1



#model
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizer_en.vocab_size,
    target_vocab_size=tokenizer_zh.vocab_size,
    dropout_rate=dropout_rate)

transformer((pt, en))

###test
##print(en.shape)
##print(pt.shape)
##print(output.shape)
##
##attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
##print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)

transformer.summary()



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



###優化器圖表
##plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
##plt.ylabel('Learning Rate')
##plt.xlabel('Train Step')
##plt.show()



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


#transformer.load_weights('transformer_zh_to_en_20.h5')
#train
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

transformer.fit(train_batches,
                epochs=10,
                validation_data=val_batches)

transformer.save_weights('translator.h5')



#運行
class Translator(tf.Module):
    def __init__(self, tokenizer_zh, tokenizer_en, transformer):
        self.tokenizer_zh = tokenizer_zh
        self.tokenizer_en = tokenizer_en
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
            
        sentence = np.array2string(sentence.numpy())
        sentence = sentence[:-2]
        sentence = sentence[3:]
        sentence = self.tokenizer_en.encode(sentence)
        sentence = tf.convert_to_tensor(sentence)
        sentence = tf.reshape(sentence,[1,sentence.shape[0]])
        
        encoder_input = sentence

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        start_end = self.tokenizer_zh.encode('')
        start = tf.convert_to_tensor([start_end[0]], dtype=tf.int64)
        end = tf.convert_to_tensor([start_end[1]], dtype=tf.int64)

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = output.numpy().tolist()[0]  # Shape: `()`.        

        tokens = tokenizer_zh.decode(output.numpy().tolist()[0])
        
        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:,:-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights



#輸出格式
def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens}')
    print(f'{"Ground truth":15s}: {ground_truth}')



#測試
translator = Translator(tokenizer_zh, tokenizer_en, transformer)

sentence = 'this is a problem we have to solve .'
ground_truth = 'this is a problem we have to solve .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_tokens, ground_truth)



###save model
##class ExportTranslator(tf.Module):
##    def __init__(self, translator):
##        self.translator = translator
##
##    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
##    def __call__(self, sentence):
##        (result,
##         tokens,
##         attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)
##
##        return result
##
##translator = ExportTranslator(translator)
##
##tf.saved_model.save(translator, export_dir='translator')
##
##print(translator(tf.constant('this is a problem we have to solve .')).numpy())
##
##reloaded = tf.saved_model.load('translator')
##
##print(reloaded(tf.constant('this is a problem we have to solve .')).numpy())
