# create_professional_albanian_dataset.py
import pandas as pd
import random
import numpy as np

print("=" * 70)
print("KRIJIMI I DATASET-IT PROFESIONAL SHQIP")
print("Struktura: e ngjashme me Jigsaw Toxic Comment Dataset")
print("=" * 70)

# ============================================================================
# KATEGORITË E TOKSICITETIT (si Jigsaw)
# ============================================================================
# toxic - përmbajtje ofenduese ose e papërshtatshme
# severe_toxic - shumë ofenduese, agresive
# obscene - e pahijshme, vulgare
# threat - kërcënim
# insult - fyer, sharje
# identity_hate - urrejtje ndaj identitetit (kombësi, religjion, etj)

# ============================================================================
# 1. KOMENTE TOKSIKE - KATEGORI TË NDRYSHME
# ============================================================================

# INSULT - Fyerje të thjeshta
insults = [
    ("Je budalla", [1, 0, 0, 0, 1, 0]),
    ("Je idiot", [1, 0, 0, 0, 1, 0]),
    ("Je trap", [1, 0, 0, 0, 1, 0]),
    ("Je faqezi", [1, 0, 0, 0, 1, 0]),
    ("Je kreteni", [1, 0, 0, 0, 1, 0]),
    ("Mësues i paaftë", [1, 0, 0, 0, 1, 0]),
    ("Profesor i dobët", [1, 0, 0, 0, 1, 0]),
    ("Je kot", [1, 0, 0, 0, 1, 0]),
    ("Nuk vlen asgjë", [1, 0, 0, 0, 1, 0]),
    ("Je i paarsimuar", [1, 0, 0, 0, 1, 0]),
    ("Je i prapambetur", [1, 0, 0, 0, 1, 0]),
    ("Je i pashkolluar", [1, 0, 0, 0, 1, 0]),
    ("Nuk di asgjë", [1, 0, 0, 0, 1, 0]),
    ("Je zero në punën tënde", [1, 0, 0, 0, 1, 0]),
    ("S'ke nivel", [1, 0, 0, 0, 1, 0]),
]

# SEVERE_TOXIC + OBSCENE - Shumë fyese dhe vulgare
severe_toxic = [
    ("Je budalla e madhe dhe idiot", [1, 1, 0, 0, 1, 0]),
    ("Ky profesor është trash", [1, 1, 0, 0, 1, 0]),
    ("Je pleh", [1, 1, 1, 0, 1, 0]),
    ("Je derr", [1, 1, 1, 0, 1, 0]),
    ("Je bythqir", [1, 1, 1, 0, 1, 0]),
    ("Mësues horr", [1, 1, 1, 0, 1, 0]),
    ("Ky mësues është një pleh i madh", [1, 1, 1, 0, 1, 0]),
    ("Je një budalla e paarsimuar", [1, 1, 0, 0, 1, 0]),
    ("Profesor trash dhe i paaftë", [1, 1, 0, 0, 1, 0]),
    ("Hajde more trap", [1, 1, 0, 0, 1, 0]),
    ("Mos u bë kreteni", [1, 1, 0, 0, 1, 0]),
    ("Ik more", [1, 1, 0, 0, 1, 0]),
    ("Hajt tutje", [1, 1, 0, 0, 1, 0]),
]

# THREAT - Kërcënime
threats = [
    ("Do të pendohesh për këtë", [1, 1, 0, 1, 0, 0]),
    ("Do të marr masa kundër teje", [1, 0, 0, 1, 0, 0]),
    ("Më mirë ki kujdes", [1, 0, 0, 1, 0, 0]),
    ("Do të të shoh jashtë", [1, 1, 0, 1, 0, 0]),
    ("Do të paguash për këtë", [1, 1, 0, 1, 0, 0]),
]

# TOXIC - Të papërshtatshme por jo shumë të rënda
mild_toxic = [
    ("Mësues i keq", [1, 0, 0, 0, 0, 0]),
    ("S'di të mësosh", [1, 0, 0, 0, 0, 0]),
    ("Nuk je për këtë punë", [1, 0, 0, 0, 1, 0]),
    ("Kurse je tallur me ne", [1, 0, 0, 0, 0, 0]),
    ("Mësues i tmerrshëm", [1, 0, 0, 0, 1, 0]),
    ("Ka lidhje me punën", [1, 0, 0, 0, 0, 0]),
    ("Je i pa zot", [1, 0, 0, 0, 1, 0]),
]

# KOMBINIME KOMPLEKSE
complex_toxic = [
    ("Ky profesor është trap dhe faqezi", [1, 1, 0, 0, 1, 0]),
    ("Je budalla dhe nuk di të mësosh", [1, 1, 0, 0, 1, 0]),
    ("Mësues i paaftë dhe i prapambetur", [1, 1, 0, 0, 1, 0]),
    ("Je idiot i madh që nuk ka nivel", [1, 1, 0, 0, 1, 0]),
    ("Profesor trash që nuk vlen asgjë", [1, 1, 0, 0, 1, 0]),
    ("Je kreteni dhe s'ke arsim", [1, 1, 0, 0, 1, 0]),
]

# HATE - Urrejtje (të rralla, por të përfshira)
hate = [
    ("Të urrej shumë", [1, 1, 0, 0, 0, 1]),
    ("Nuk të duroj dot", [1, 0, 0, 0, 0, 1]),
    ("Të mallkoft zoti", [1, 1, 1, 0, 0, 1]),
]

# ============================================================================
# 2. KOMENTE NORMALE (JO-TOKSIKE)
# ============================================================================

normal_comments_data = [
    ("Mësuesi shpjegon mirë konceptet", [0, 0, 0, 0, 0, 0]),
    ("Mësimet janë interesante", [0, 0, 0, 0, 0, 0]),
    ("Është shumë i durueshëm me studentët", [0, 0, 0, 0, 0, 0]),
    ("Na ndihmon të kuptojmë më mirë", [0, 0, 0, 0, 0, 0]),
    ("Përgatit mirë leksionet", [0, 0, 0, 0, 0, 0]),
    ("Është i respektuar nga të gjithë", [0, 0, 0, 0, 0, 0]),
    ("Merr kohë për çdo student", [0, 0, 0, 0, 0, 0]),
    ("Shpjegon në mënyrë të qartë", [0, 0, 0, 0, 0, 0]),
    ("Është profesionist", [0, 0, 0, 0, 0, 0]),
    ("Ka njohuri të mira", [0, 0, 0, 0, 0, 0]),
    ("Është mësues i mirë", [0, 0, 0, 0, 0, 0]),
    ("Na motivon për të mësuar", [0, 0, 0, 0, 0, 0]),
    ("Krijon atmosferë pozitive", [0, 0, 0, 0, 0, 0]),
    ("Është i drejtë me të gjithë", [0, 0, 0, 0, 0, 0]),
    ("Ka metodologji të mirë", [0, 0, 0, 0, 0, 0]),
    ("Përdor shembuj praktikë", [0, 0, 0, 0, 0, 0]),
    ("Na inkurajon të pyesim", [0, 0, 0, 0, 0, 0]),
    ("Është i hapur për diskutime", [0, 0, 0, 0, 0, 0]),
    ("Respekton opinionet tona", [0, 0, 0, 0, 0, 0]),
    ("Është mentor i shkëlqyer", [0, 0, 0, 0, 0, 0]),
    ("Mëson me pasion", [0, 0, 0, 0, 0, 0]),
    ("Është model për ne", [0, 0, 0, 0, 0, 0]),
    ("Ka qasje moderne", [0, 0, 0, 0, 0, 0]),
    ("Përdor teknologji në mësim", [0, 0, 0, 0, 0, 0]),
    ("Bën mësimet argëtuese", [0, 0, 0, 0, 0, 0]),
    ("Është i qartë në shpjegime", [0, 0, 0, 0, 0, 0]),
    ("Na jep feedback konstruktiv", [0, 0, 0, 0, 0, 0]),
    ("Organizon mirë orët", [0, 0, 0, 0, 0, 0]),
    ("Përdor metoda interaktive", [0, 0, 0, 0, 0, 0]),
    ("Është burim frymëzimi", [0, 0, 0, 0, 0, 0]),
    ("Ka përvojë të madhe", [0, 0, 0, 0, 0, 0]),
    ("Është i përkushtuar", [0, 0, 0, 0, 0, 0]),
    ("Shpjegon me durim", [0, 0, 0, 0, 0, 0]),
    ("Ka komunikim të mirë", [0, 0, 0, 0, 0, 0]),
    ("Është i aftë", [0, 0, 0, 0, 0, 0]),
    ("Na inkurajon", [0, 0, 0, 0, 0, 0]),
    ("Është inspirues", [0, 0, 0, 0, 0, 0]),
    ("Ka njohuri të thella", [0, 0, 0, 0, 0, 0]),
    ("Është i talentuar", [0, 0, 0, 0, 0, 0]),
    ("Na ndihmon shumë", [0, 0, 0, 0, 0, 0]),
    ("Profesor i shkëlqyer", [0, 0, 0, 0, 0, 0]),
    ("Mësues i mirë", [0, 0, 0, 0, 0, 0]),
    ("Ka energji pozitive", [0, 0, 0, 0, 0, 0]),
    ("Është i zgjuar", [0, 0, 0, 0, 0, 0]),
    ("Ka vizion", [0, 0, 0, 0, 0, 0]),
    ("Është i kujdesshëm", [0, 0, 0, 0, 0, 0]),
    ("Na kupton", [0, 0, 0, 0, 0, 0]),
    ("Është i ngrohtë", [0, 0, 0, 0, 0, 0]),
    ("Ka qasje të mirë", [0, 0, 0, 0, 0, 0]),
    ("Është mbështetës", [0, 0, 0, 0, 0, 0]),
]

# ============================================================================
# 3. GJENERIMI I DATASET-IT
# ============================================================================

print("\nPo gjeneroj dataset...")

# Kombino të gjitha kategoritë toksike
all_toxic = (
    insults * 30 + 
    severe_toxic * 25 + 
    threats * 15 + 
    mild_toxic * 20 + 
    complex_toxic * 20 +
    hate * 10
)

# Repliko komentet normale për balancim
all_normal = normal_comments_data * 40

print(f"  ✓ Komente toksike: {len(all_toxic)}")
print(f"  ✓ Komente normale: {len(all_normal)}")

# Kombino të gjitha
all_comments = all_toxic + all_normal

# Krijoni DataFrame
data = []
for idx, (comment, labels) in enumerate(all_comments):
    data.append({
        'id': f'sq_{idx:010d}',
        'comment_text': comment,
        'toxic': labels[0],
        'severe_toxic': labels[1],
        'obscene': labels[2],
        'threat': labels[3],
        'insult': labels[4],
        'identity_hate': labels[5]
    })

df = pd.DataFrame(data)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================================
# 4. STATISTIKA
# ============================================================================

print("\n" + "=" * 70)
print("STATISTIKA E DATASET-IT")
print("=" * 70)
print(f"  Totali i komenteve: {len(df):,}")
print(f"  Komente toksike: {df['toxic'].sum():,} ({df['toxic'].mean()*100:.1f}%)")
print(f"  Komente normale: {(1-df['toxic']).sum():,} ({(1-df['toxic']).mean()*100:.1f}%)")
print()
print("  Nën-kategoritë:")
print(f"    - Severe Toxic: {df['severe_toxic'].sum():,} ({df['severe_toxic'].mean()*100:.1f}%)")
print(f"    - Obscene: {df['obscene'].sum():,} ({df['obscene'].mean()*100:.1f}%)")
print(f"    - Threat: {df['threat'].sum():,} ({df['threat'].mean()*100:.1f}%)")
print(f"    - Insult: {df['insult'].sum():,} ({df['insult'].mean()*100:.1f}%)")
print(f"    - Identity Hate: {df['identity_hate'].sum():,} ({df['identity_hate'].mean()*100:.1f}%)")
print("=" * 70)

# ============================================================================
# 5. RUAJTJA
# ============================================================================

# Ruaj në format CSV (si Jigsaw)
df.to_csv('../data/albanian_toxic_comments.csv', index=False, encoding='utf-8')
print(f"\n✅ Dataset u ruajt në: ../data/albanian_toxic_comments.csv")

# Krijo edhe një version të thjeshtuar (vetëm toxic/non-toxic)
df_simple = df[['comment_text', 'toxic']].copy()
df_simple.to_csv('../data/albanian_toxic.csv', index=False, encoding='utf-8')
print(f"✅ Versioni i thjeshtuar u ruajt në: ../data/albanian_toxic.csv")

# ============================================================================
# 6. SHEMBUJ
# ============================================================================

print("\n" + "=" * 70)
print("DISA SHEMBUJ NGA DATASET-I")
print("=" * 70)

print("\n🔴 KOMENTE TOKSIKE:")
toxic_samples = df[df['toxic'] == 1].sample(min(10, df['toxic'].sum()))
for _, row in toxic_samples.iterrows():
    labels = []
    if row['severe_toxic']: labels.append('severe')
    if row['obscene']: labels.append('obscene')
    if row['threat']: labels.append('threat')
    if row['insult']: labels.append('insult')
    if row['identity_hate']: labels.append('hate')
    label_str = ', '.join(labels) if labels else 'toxic'
    print(f"  [{label_str}] {row['comment_text']}")

print("\n🟢 KOMENTE NORMALE:")
normal_samples = df[df['toxic'] == 0].sample(min(10, (df['toxic'] == 0).sum()))
for _, row in normal_samples.iterrows():
    print(f"  {row['comment_text']}")

print("\n" + "=" * 70)
print("✅ DATASET-I U KRIJUA ME SUKSES!")
print("=" * 70)
print("\n📌 Hapi tjetër:")
print("   python train_model.py")
print("=" * 70)