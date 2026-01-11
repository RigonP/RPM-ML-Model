# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

print("Duke filluar trajnimin e modelit...")

# 1. Lexoni dataset-in
print("Po lexoj dataset-in...")
df = pd.read_csv('../data/train.csv')

# Për shembull, përdorim vetëm kolonën 'toxic'
# Nëse doni të detektoni të gjitha kategoritë, mund t'i kombinoni
texts = df['comment_text'].values
labels = df['toxic'].values

print(f"Dataset u ngarkua: {len(texts)} komente")

# 2. Ndani të dhënat
print("Po ndaj të dhënat në training dhe test...")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 3. Tokenizimi
print("Po bëj tokenizimin...")
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

print(f"Training samples: {len(X_train_pad)}")
print(f"Test samples: {len(X_test_pad)}")

# 4. Ndërtimi i modelit
print("Po ndërtoj modelin...")
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Arkitektura e modelit:")
model.summary()

# 5. Trajnimi
print("\nPo filloj trajnimin (mund të zgjasë 20-30 minuta)...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 6. Vlerësimi
print("\nPo vlerësoj modelin...")
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# 7. Ruajtja e modelit
print("\nPo ruaj modelin dhe tokenizer-in...")

# Krijoni folder për modelin nëse nuk ekziston
os.makedirs('../api', exist_ok=True)

model.save('../api/toxicity_model.h5')
print("Modeli u ruajt në: ../api/toxicity_model.h5")

with open('../api/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer-i u ruajt në: ../api/tokenizer.pickle")

# Ruaj edhe max_len për reference
config = {'max_len': max_len}
with open('../api/config.pickle', 'wb') as handle:
    pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅ Trajnimi përfundoi me sukses!")
print(f"✅ Saktësia finale: {accuracy * 100:.2f}%")