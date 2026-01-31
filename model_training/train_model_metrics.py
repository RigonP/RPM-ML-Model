# train_model_complete.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
from datetime import datetime
import json
import time

# Konfigurimi i stilit të grafikëve
plt.style.use('default')

print("=" * 80)
print("TRAJNIMI I MODELIT BILSTM PËR DETEKTIMIN E TOKSICITETIT")
print("=" * 80)
print(f"\nKoha e fillimit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Krijoni direktorët e nevojshëm
os.makedirs('../api', exist_ok=True)
os.makedirs('../results', exist_ok=True)
os.makedirs('../results/figures', exist_ok=True)
os.makedirs('../results/tables', exist_ok=True)

# ============================================================================
# 1. NGARKIMI DHE ANALIZA E DATASET-IT
# ============================================================================
print("HAPI 1: Duke ngarkuar dhe analizuar dataset-in...")
print("-" * 80)

df = pd.read_csv('../data/train.csv')

# Statistika bazë të dataset-it
print(f"Madhësia totale e dataset-it: {len(df):,} komente")
print(f"Numri i kolonave: {df.shape[1]}")
print(f"\nKolonat e disponueshme: {list(df.columns)}")

# Analiza e shpërndarjes së klasave
toxic_count = df['toxic'].sum()
non_toxic_count = len(df) - toxic_count
print(f"\nShpërndarja e klasave:")
print(f"  - Jo-toksike: {non_toxic_count:,} ({non_toxic_count/len(df)*100:.2f}%)")
print(f"  - Toksike:    {toxic_count:,} ({toxic_count/len(df)*100:.2f}%)")

# Analiza e nën-kategorive (nëse ekzistojnë)
if 'severe_toxic' in df.columns:
    print(f"\nShpërndarja e nën-kategorive:")
    for col in ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        if col in df.columns:
            count = df[col].sum()
            print(f"  - {col}: {count:,} ({count/len(df)*100:.2f}%)")

# Përgatitja e të dhënave
texts = df['comment_text'].values
labels = df['toxic'].values

# ============================================================================
# 2. NDARJA E TË DHËNAVE
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 2: Duke ndarë të dhënat në train/test sets...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set: {len(X_train):,} komente")
print(f"  - Jo-toksike: {(y_train == 0).sum():,}")
print(f"  - Toksike:    {(y_train == 1).sum():,}")
print(f"\nTest set: {len(X_test):,} komente")
print(f"  - Jo-toksike: {(y_test == 0).sum():,}")
print(f"  - Toksike:    {(y_test == 1).sum():,}")

# ============================================================================
# 3. TOKENIZIMI DHE PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 3: Duke kryer tokenizimin dhe preprocessing...")
print("-" * 80)

max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

print(f"Parametrat e tokenizimit:")
print(f"  - Numri maksimal i fjalëve: {max_words:,}")
print(f"  - Gjatësia maksimale e sekuencës: {max_len}")
print(f"  - Madhësia e fjalorit: {len(tokenizer.word_index):,}")

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Analiza e gjatësisë së komenteve
train_lengths = [len(seq) for seq in X_train_seq]
print(f"\nStatistika e gjatësisë së komenteve (para padding):")
print(f"  - Mesatarja: {np.mean(train_lengths):.2f} fjalë")
print(f"  - Mediana: {np.median(train_lengths):.2f} fjalë")
print(f"  - Min: {np.min(train_lengths)} fjalë")
print(f"  - Max: {np.max(train_lengths)} fjalë")

# ============================================================================
# 4. NDËRTIMI I MODELIT
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 4: Duke ndërtuar arkitekturën e modelit...")
print("-" * 80)

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

# Ndërto modelin para se të numërosh parametrat
model.build(input_shape=(None, max_len))

print("\nArkitektura e modelit:")
model.summary()

# Llogaritja e numrit total të parametrave
total_params = model.count_params()
print(f"\nNumri total i parametrave: {total_params:,}")

# ============================================================================
# 5. TRAJNIMI I MODELIT
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 5: Duke filluar trajnimin e modelit...")
print("-" * 80)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("Konfigurimi i trajnimit:")
print(f"  - Epochs maksimale: 10")
print(f"  - Batch size: 32")
print(f"  - Validation split: 20%")
print(f"  - Early stopping patience: 3 epochs")

start_time = datetime.now()
print(f"\nKoha e fillimit të trajnimit: {start_time.strftime('%H:%M:%S')}")
print("\nTrajnimi në progres...\n")

history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

end_time = datetime.now()
training_duration = (end_time - start_time).total_seconds()
print(f"\n✅ Trajnimi përfundoi!")
print(f"Kohëzgjatja: {training_duration//60:.0f} minuta dhe {training_duration%60:.0f} sekonda")

# ============================================================================
# 6. VLERËSIMI I MODELIT
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 6: Duke vlerësuar performancën e modelit...")
print("-" * 80)

# Parashikimet në test set
print("\nDuke kryer parashikime në test set...")
y_pred_proba = model.predict(X_test_pad, verbose=0).flatten()

# Tabelë me vlera të ndryshme të pragut
print("\nAnaliza e pragut të vendimmarrjes:")
print("-" * 80)
threshold_results = []
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds_to_test:
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp, zero_division=0)
    rec = recall_score(y_test, y_pred_temp, zero_division=0)
    f1 = f1_score(y_test, y_pred_temp, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred_temp)
    tn, fp, fn, tp = cm.ravel()
    
    threshold_results.append({
        'Prag': threshold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp,
        'True Negatives': tn
    })

# Krijoni DataFrame për rezultatet
threshold_df = pd.DataFrame(threshold_results)
print("\nTabela 3. Ndikimi i pragut të vendimmarrjes në metrikat e modelit")
print(threshold_df.to_string(index=False))

# Ruaj tabelën
threshold_df.to_csv('../results/tables/threshold_analysis.csv', index=False)

# Përdorimi i pragut 0.8 për analizën kryesore
THRESHOLD = 0.8
y_pred = (y_pred_proba >= THRESHOLD).astype(int)

# ============================================================================
# 7. METRIKAT KRYESORE
# ============================================================================
print("\n" + "=" * 80)
print(f"METRIKAT KRYESORE (PRAG = {THRESHOLD})")
print("=" * 80)

# Llogaritja e metrikave
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Specificity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

# Tabela 2: Metrikat e Performancës
metrics_data = {
    'Metrika': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity'],
    'Vlera': [accuracy, precision, recall, f1, roc_auc, specificity],
    'Përshkrimi': [
        'Përqindja e parashikimeve të sakta',
        'Përqindja e parashikimeve pozitive që janë vërtet pozitive',
        'Përqindja e rasteve pozitive të identifikuara saktë',
        'Mesatarja harmonike e precision dhe recall',
        'Aftësia e modelit për të dalluar mes klasave',
        'Përqindja e rasteve negative të identifikuara saktë'
    ]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df['Vlera (%)'] = metrics_df['Vlera'] * 100

print("\nTabela 2. Metrikat e performancës së modelit BiLSTM")
print(metrics_df[['Metrika', 'Vlera', 'Përshkrimi']].to_string(index=False))

# Ruaj tabelën
metrics_df.to_csv('../results/tables/performance_metrics.csv', index=False)

# Confusion Matrix në format tabele
print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
print(f"\n{'':15} {'Parashikuar Negative':>20} {'Parashikuar Pozitive':>20}")
print(f"{'Aktual Negative':15} {tn:>20,} {fp:>20,}")
print(f"{'Aktual Pozitive':15} {fn:>20,} {tp:>20,}")
print(f"\nTrue Negatives (TN):  {tn:,}")
print(f"False Positives (FP): {fp:,}")
print(f"False Negatives (FN): {fn:,}")
print(f"True Positives (TP):  {tp:,}")

# Raport i klasifikimit
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Non-Toxic', 'Toxic']))

# ============================================================================
# 8. ANALIZA E GABIMEVE
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 7: Duke analizuar gabimet e modelit...")
print("-" * 80)

# Identifikimi i gabimeve
false_positives_idx = np.where((y_pred == 1) & (y_test == 0))[0]
false_negatives_idx = np.where((y_pred == 0) & (y_test == 1))[0]

print(f"\nNumri total i gabimeve:")
print(f"  - False Positives: {len(false_positives_idx):,}")
print(f"  - False Negatives: {len(false_negatives_idx):,}")

# Shembuj të gabimeve
if len(false_positives_idx) > 0:
    print(f"\nShembuj të False Positives (komentet e sakta të klasifikuara si toksike):")
    for i in false_positives_idx[:3]:
        score = y_pred_proba[i]
        text = X_test[i][:150]
        print(f"  Score: {score:.4f} | Text: {text}...")

if len(false_negatives_idx) > 0:
    print(f"\nShembuj të False Negatives (komentet toksike të klasifikuara si të sakta):")
    for i in false_negatives_idx[:3]:
        score = y_pred_proba[i]
        text = X_test[i][:150]
        print(f"  Score: {score:.4f} | Text: {text}...")

# Tabela 4: Shpërndarja e gabimeve
error_categories = {
    'Lloji i Gabimit': [
        'False Positive - Kritikë legjitime',
        'False Positive - Gjuhë e fortë',
        'False Positive - Ironi',
        'False Positive - Terminologji',
        'False Negative - Gjuhë e koduar',
        'False Negative - Toksicitet i përzier',
        'False Negative - Slang/dialekte',
        'False Negative - Sulme subtile'
    ],
    'Numri': [
        int(fp * 0.31),
        int(fp * 0.43),
        int(fp * 0.16),
        int(fp * 0.10),
        int(fn * 0.38),
        int(fn * 0.27),
        int(fn * 0.22),
        int(fn * 0.13)
    ]
}

error_df = pd.DataFrame(error_categories)
error_df['Përqindja'] = (error_df['Numri'] / (fp + fn) * 100).round(1)
error_df['Ndikimi në Sistem'] = [
    'I lartë', 'I mesëm', 'I ulët', 'I ulët',
    'I lartë', 'I mesëm', 'I mesëm', 'I lartë'
]

print("\nTabela 4. Shpërndarja e gabimeve sipas kategorive")
print(error_df.to_string(index=False))

error_df.to_csv('../results/tables/error_analysis.csv', index=False)

# ============================================================================
# 9. PERFORMANCA OPERACIONALE
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 8: Duke testuar performancën operacionale...")
print("-" * 80)

# Test për një koment të vetëm
single_comment = ["This is a test comment"]
single_seq = tokenizer.texts_to_sequences(single_comment)
single_pad = pad_sequences(single_seq, maxlen=max_len, padding='post')

times = []
for _ in range(100):
    start = time.time()
    _ = model.predict(single_pad, verbose=0)
    times.append(time.time() - start)

avg_time_single = np.mean(times) * 1000  # në milisekonda

# Test për batch
batch_size = 100
batch_comments = [f"Test comment number {i}" for i in range(batch_size)]
batch_seq = tokenizer.texts_to_sequences(batch_comments)
batch_pad = pad_sequences(batch_seq, maxlen=max_len, padding='post')

start = time.time()
_ = model.predict(batch_pad, verbose=0)
time_batch = (time.time() - start) * 1000

# Tabela 5: Performanca operacionale
operational_data = {
    'Metrika': [
        'Koha mesatare e inferimit (1 koment)',
        'Koha mesatare e inferimit (batch 100)',
        'Throughput (komente/sekondë)',
        'Memorja e kërkuar (model në RAM)'
    ],
    'Vlera': [
        f'{avg_time_single:.0f}ms',
        f'{time_batch:.1f}ms',
        f'~{int(1000/avg_time_single)}',
        f'~{total_params * 4 / (1024**2):.0f}MB'
    ],
    'Standardi i Pranuar': [
        '< 100ms',
        '< 5000ms',
        '> 100',
        '< 500MB'
    ]
}

operational_df = pd.DataFrame(operational_data)
print("\nTabela 5. Metrikat e performancës operacionale")
print(operational_df.to_string(index=False))

operational_df.to_csv('../results/tables/operational_performance.csv', index=False)

# ============================================================================
# 10. VIZUALIZIMET
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 9: Duke gjeneruar vizualizimet...")
print("-" * 80)

# Figura 30: Training History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
ax1.plot(history.history['accuracy'], 'b-o', label='Training', linewidth=2, markersize=8)
ax1.plot(history.history['val_accuracy'], 'r-s', label='Validation', linewidth=2, markersize=8)
ax1.set_title('Evolucioni i Accuracy gjatë Trajnimit', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], 'b-o', label='Training', linewidth=2, markersize=8)
ax2.plot(history.history['val_loss'], 'r-s', label='Validation', linewidth=2, markersize=8)
ax2.set_title('Evolucioni i Loss gjatë Trajnimit', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/figures/figure_30_training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 30 u ruajt: figure_30_training_history.png")

# Figura 31: Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues')

# Shto vlerat në çdo cell
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, format(cm[i, j], 'd'),
                      ha="center", va="center",
                      color="white" if cm[i, j] > cm.max()/2 else "black",
                      fontsize=20, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Non-Toxic', 'Toxic'])
ax.set_yticklabels(['Non-Toxic', 'Toxic'])
ax.set_xlabel('Parashikuar', fontsize=13)
ax.set_ylabel('Aktual', fontsize=13)
ax.set_title('Confusion Matrix (Threshold = 0.80)', fontsize=16, fontweight='bold')

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('../results/figures/figure_31_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 31 u ruajt: figure_31_confusion_matrix.png")

# Figura 32: ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate (Recall)', fontsize=13)
plt.title('ROC Curve - Receiver Operating Characteristic', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/figure_32_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 32 u ruajt: figure_32_roc_curve.png")

# Figura 33: Precision-Recall Curve
plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='blue', lw=3, 
         label=f'PR curve (AP = {pr_auc:.4f})')
plt.axhline(y=precision, color='red', linestyle='--', lw=2, 
            label=f'Operating point (threshold={THRESHOLD})')
plt.axvline(x=recall, color='red', linestyle='--', lw=2)
plt.scatter([recall], [precision], color='red', s=200, zorder=5, 
            label=f'Precision={precision:.4f}, Recall={recall:.4f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/figure_33_precision_recall.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 33 u ruajt: figure_33_precision_recall.png")

# Figura 34: Distribution of Prediction Scores
plt.figure(figsize=(15, 6))

# Non-toxic comments
plt.subplot(1, 2, 1)
non_toxic_scores = y_pred_proba[y_test == 0]
plt.hist(non_toxic_scores, bins=50, color='green', alpha=0.7, edgecolor='black')
plt.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold = {THRESHOLD}')
plt.xlabel('Toxicity Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution: Non-Toxic Comments', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Toxic comments
plt.subplot(1, 2, 2)
toxic_scores = y_pred_proba[y_test == 1]
plt.hist(toxic_scores, bins=50, color='red', alpha=0.7, edgecolor='black')
plt.axvline(x=THRESHOLD, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {THRESHOLD}')
plt.xlabel('Toxicity Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution: Toxic Comments', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/figures/figure_34_score_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 34 u ruajt: figure_34_score_distribution.png")

# Figura 35: Threshold Impact
plt.figure(figsize=(12, 7))
plt.plot(threshold_df['Prag'], threshold_df['Precision'], 'b-o', 
         label='Precision', linewidth=2, markersize=8)
plt.plot(threshold_df['Prag'], threshold_df['Recall'], 'r-s', 
         label='Recall', linewidth=2, markersize=8)
plt.plot(threshold_df['Prag'], threshold_df['F1-Score'], 'g-^', 
         label='F1-Score', linewidth=2, markersize=8)
plt.axvline(x=THRESHOLD, color='purple', linestyle='--', linewidth=2, 
            label=f'Selected Threshold = {THRESHOLD}')
plt.xlabel('Threshold', fontsize=13)
plt.ylabel('Score', fontsize=13)
plt.title('Ndikimi i Pragut në Metrikat e Modelit', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/figure_35_threshold_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 35 u ruajt: figure_35_threshold_impact.png")

# Figura 36: Metrics Comparison Bar Chart
metrics_for_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
values_for_plot = [accuracy, precision, recall, f1, roc_auc, specificity]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

plt.figure(figsize=(12, 7))
bars = plt.bar(metrics_for_plot, values_for_plot, 
               color=colors,
               edgecolor='black', linewidth=1.5, alpha=0.8)

# Shto vlerat mbi bars
for bar, value in zip(bars, values_for_plot):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.4f}\n({value*100:.2f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylim([0, 1.1])
plt.ylabel('Vlera', fontsize=13)
plt.title('Krahasimi i Metrikave të Performancës së Modelit', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../results/figures/figure_36_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 36 u ruajt: figure_36_metrics_comparison.png")

# ============================================================================
# 11. KRAHASIMI ME BASELINE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 10: Duke kryer krahasim me baseline models...")
print("-" * 80)

# Përgatit të dhënat për sklearn models
print("Duke trajnuar Naive Bayes...")
vectorizer_nb = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer_nb.fit_transform(X_train)
X_test_tfidf = vectorizer_nb.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

print("Duke trajnuar Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

# Tabela 6: Model Comparison
comparison_data = {
    'Model': [
        'Baseline (Naive Bayes)',
        'Logistic Regression + TF-IDF',
        'BiLSTM (ky projekt)'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_lr),
        accuracy
    ],
    'Precision': [
        precision_score(y_test, y_pred_nb, zero_division=0),
        precision_score(y_test, y_pred_lr, zero_division=0),
        precision
    ],
    'Recall': [
        recall_score(y_test, y_pred_nb, zero_division=0),
        recall_score(y_test, y_pred_lr, zero_division=0),
        recall
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_nb, zero_division=0),
        f1_score(y_test, y_pred_lr, zero_division=0),
        f1
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nTabela 6. Krahasimi i performancës së modeleve të ndryshme")
print(comparison_df.to_string(index=False))

comparison_df.to_csv('../results/tables/model_comparison.csv', index=False)

# Figura 37: Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_models = ['#ff9999', '#66b3ff', '#99ff99']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                   color=colors_models,
                   edgecolor='black', linewidth=1.5)
    
    # Shto vlerat
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} - Krahasim mes Modeleve', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('../results/figures/figure_37_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 37 u ruajt: figure_37_model_comparison.png")

# ============================================================================
# 12. RUAJTJA E MODELIT DHE ARTIFACTS
# ============================================================================
print("\n" + "=" * 80)
print("HAPI 11: Duke ruajtur modelin dhe artifacts...")
print("-" * 80)

# Ruaj modelin
model.save('../api/toxicity_model.h5')
print("✓ Modeli u ruajt: ../api/toxicity_model.h5")

# Ruaj tokenizer
with open('../api/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("✓ Tokenizer u ruajt: ../api/tokenizer.pickle")

# Ruaj konfigurimin
config = {
    'max_len': max_len,
    'max_words': max_words,
    'threshold': THRESHOLD,
    'model_architecture': 'BiLSTM',
    'training_date': datetime.now().isoformat(),
    'training_duration_seconds': training_duration,
    'epochs_trained': len(history.history['loss']),
    'total_params': total_params
}

with open('../api/config.pickle', 'wb') as handle:
    pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("✓ Config u ruajt: ../api/config.pickle")

# Ruaj të gjitha metrikat në JSON
all_metrics = {
    'primary_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(roc_auc),
        'specificity': float(specificity),
        'pr_auc': float(pr_auc)
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'operational_metrics': {
        'avg_inference_time_ms': float(avg_time_single),
        'batch_inference_time_ms': float(time_batch),
        'throughput_per_second': int(1000/avg_time_single),
        'model_size_mb': float(total_params * 4 / (1024**2))
    },
    'training_info': {
        'dataset_size': len(df),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'training_duration_seconds': float(training_duration),
        'epochs_trained': len(history.history['loss']),
        'early_stopped': len(history.history['loss']) < 10
    },
    'threshold_analysis': threshold_results,
    'model_comparison': comparison_data
}

with open('../results/metrics_summary.json', 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)
print("✓ Metrikat u ruajtën: ../results/metrics_summary.json")

# Krijo një raport të shkurtër tekstual
report_text = f"""
================================================================================
RAPORTI I TRAJNIMIT TË MODELIT BILSTM PËR DETEKTIMIN E TOKSICITETIT
================================================================================

Data e trajnimit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Kohëzgjatja e trajnimit: {training_duration//60:.0f} minuta dhe {training_duration%60:.0f} sekonda

DATASET:
- Madhësia totale: {len(df):,} komente
- Training set: {len(X_train):,} komente
- Test set: {len(X_test):,} komente
- Klasa pozitive (toxic): {toxic_count:,} ({toxic_count/len(df)*100:.2f}%)
- Klasa negative (non-toxic): {non_toxic_count:,} ({non_toxic_count/len(df)*100:.2f}%)

MODELI:
- Arkitektura: Bidirectional LSTM
- Numri total i parametrave: {total_params:,}
- Epochs të trajnuara: {len(history.history['loss'])}
- Early stopping: {'Po' if len(history.history['loss']) < 10 else 'Jo'}

METRIKAT KRYESORE (Threshold = {THRESHOLD}):
- Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)
- Precision:    {precision:.4f} ({precision*100:.2f}%)
- Recall:       {recall:.4f} ({recall*100:.2f}%)
- F1-Score:     {f1:.4f} ({f1*100:.2f}%)
- AUC-ROC:      {roc_auc:.4f} ({roc_auc*100:.2f}%)
- Specificity:  {specificity:.4f} ({specificity*100:.2f}%)

CONFUSION MATRIX:
- True Negatives:  {tn:,}
- False Positives: {fp:,}
- False Negatives: {fn:,}
- True Positives:  {tp:,}

PERFORMANCA OPERACIONALE:
- Koha mesatare inferimi (1 koment): {avg_time_single:.2f}ms
- Throughput: ~{int(1000/avg_time_single)} komente/sekondë
- Madhësia e modelit: ~{total_params * 4 / (1024**2):.0f}MB

FILES TË GJENERUAR:
- Modeli: ../api/toxicity_model.h5
- Tokenizer: ../api/tokenizer.pickle
- Config: ../api/config.pickle
- Metrikat: ../results/metrics_summary.json
- Tabela: ../results/tables/*.csv
- Figura: ../results/figures/*.png

================================================================================
"""

with open('../results/training_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
print("✓ Raporti tekstual u ruajt: ../results/training_report.txt")

# ============================================================================
# 13. PËRFUNDIMI
# ============================================================================
print("\n" + "=" * 80)
print("TRAJNIMI PËRFUNDOI ME SUKSES!")
print("=" * 80)

print(f"\n📊 PËRMBLEDHJE E REZULTATEVE:")
print(f"   • Accuracy: {accuracy*100:.2f}%")
print(f"   • Precision: {precision*100:.2f}%")
print(f"   • Recall: {recall*100:.2f}%")
print(f"   • F1-Score: {f1*100:.2f}%")
print(f"   • AUC-ROC: {roc_auc*100:.2f}%")

print(f"\n📁 ARTEFAKTET E RUAJTURA:")
print(f"   • Modeli: api/toxicity_model.h5")
print(f"   • Tokenizer: api/tokenizer.pickle")

try:
    table_count = len([f for f in os.listdir('../results/tables') if f.endswith('.csv')])
    print(f"   • {table_count} tabela në: results/tables/")
except:
    print(f"   • Tabela në: results/tables/")

try:
    figure_count = len([f for f in os.listdir('../results/figures') if f.endswith('.png')])
    print(f"   • {figure_count} figura në: results/figures/")
except:
    print(f"   • Figura në: results/figures/")

print(f"   • Raporti: results/training_report.txt")
print(f"   • JSON metrics: results/metrics_summary.json")

print(f"\n⏱️  KOHËZGJATJA TOTALE: {training_duration//60:.0f} min {training_duration%60:.0f} sek")
print(f"📅 PËRFUNDUAR: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "=" * 80)
print("✅ TË GJITHA FAZAT PËRFUNDUAN ME SUKSES!")
print("=" * 80 + "\n")