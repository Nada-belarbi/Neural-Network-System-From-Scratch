"""
Script pour générer différents datasets CSV pour tester le système de réseaux de neurones.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_circles, make_moons, make_blobs
import os

# Créer le dossier datasets s'il n'existe pas
os.makedirs('datasets', exist_ok=True)

# Fixer la seed pour la reproductibilité
np.random.seed(42)

# 1. Dataset de classification binaire linéaire
print("Génération du dataset de classification binaire linéaire...")
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                         n_redundant=0, n_clusters_per_class=1, 
                         flip_y=0.1, class_sep=2.0, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y
df.to_csv('datasets/linear_classification.csv', index=False)
print("✓ linear_classification.csv créé")

# 2. Dataset de classification multi-classes
print("\nGénération du dataset de classification multi-classes...")
X, y = make_classification(n_samples=200, n_features=3, n_informative=3,
                         n_redundant=0, n_classes=3, n_clusters_per_class=1,
                         flip_y=0.1, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y
df.to_csv('datasets/multiclass_classification.csv', index=False)
print("✓ multiclass_classification.csv créé")

# 3. Dataset de régression
print("\nGénération du dataset de régression...")
X, y = make_regression(n_samples=200, n_features=2, noise=10.0, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y
df.to_csv('datasets/regression.csv', index=False)
print("✓ regression.csv créé")

# 4. Dataset de cercles concentriques (non-linéaire)
print("\nGénération du dataset de cercles concentriques...")
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y
df.to_csv('datasets/circles_classification.csv', index=False)
print("✓ circles_classification.csv créé")

# 5. Dataset en forme de lunes (non-linéaire)
print("\nGénération du dataset en forme de lunes...")
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y
df.to_csv('datasets/moons_classification.csv', index=False)
print("✓ moons_classification.csv créé")

# 6. Dataset de clusters (blobs)
print("\nGénération du dataset de clusters...")
X, y = make_blobs(n_samples=200, n_features=2, centers=4, 
                  cluster_std=1.0, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y
df.to_csv('datasets/blobs_classification.csv', index=False)
print("✓ blobs_classification.csv créé")

# 7. Dataset avec caractéristiques plus complexes (5 features)
print("\nGénération du dataset complexe...")
X, y = make_classification(n_samples=200, n_features=5, n_informative=4,
                         n_redundant=1, n_clusters_per_class=2,
                         flip_y=0.05, random_state=42)
df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(5)])
df['target'] = y
df.to_csv('datasets/complex_classification.csv', index=False)
print("✓ complex_classification.csv créé")

# 8. Dataset de fonction sinus (régression non-linéaire)
print("\nGénération du dataset de fonction sinus...")
X = np.random.uniform(-3, 3, (200, 1))
y = np.sin(2 * X).ravel() + np.random.normal(0, 0.1, X.shape[0])
df = pd.DataFrame(X, columns=['feature1'])
df['target'] = y
df.to_csv('datasets/sine_regression.csv', index=False)
print("✓ sine_regression.csv créé")

# 9. Dataset spirale (classification très non-linéaire)
print("\nGénération du dataset spirale...")
n = 100  # points par classe
theta = np.sqrt(np.random.rand(n)) * 2 * np.pi  # angle
r_a = 2 * theta + np.pi
r_b = -2 * theta - np.pi

# Classe 0
x_a = r_a * np.cos(theta) + np.random.randn(n) * 0.5
y_a = r_a * np.sin(theta) + np.random.randn(n) * 0.5

# Classe 1
x_b = r_b * np.cos(theta) + np.random.randn(n) * 0.5
y_b = r_b * np.sin(theta) + np.random.randn(n) * 0.5

X = np.vstack([np.column_stack([x_a, y_a]), np.column_stack([x_b, y_b])])
y = np.hstack([np.zeros(n), np.ones(n)])

df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y.astype(int)
df.to_csv('datasets/spiral_classification.csv', index=False)
print("✓ spiral_classification.csv créé")

# 10. Dataset de portes logiques (AND, OR)
print("\nGénération du dataset de portes logiques...")
# Générer des points autour des coins du carré unitaire
corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
X = []
y_and = []
y_or = []

for _ in range(50):  # 50 points par coin
    for i, corner in enumerate(corners):
        point = corner + np.random.normal(0, 0.1, 2)
        X.append(point)
        # AND gate
        y_and.append(1 if corner[0] == 1 and corner[1] == 1 else 0)
        # OR gate
        y_or.append(1 if corner[0] == 1 or corner[1] == 1 else 0)

X = np.array(X)
# Dataset AND
df = pd.DataFrame(X, columns=['input1', 'input2'])
df['target'] = y_and
df.to_csv('datasets/and_gate.csv', index=False)
print("✓ and_gate.csv créé")

# Dataset OR
df = pd.DataFrame(X, columns=['input1', 'input2'])
df['target'] = y_or
df.to_csv('datasets/or_gate.csv', index=False)
print("✓ or_gate.csv créé")

print("\n✅ Tous les datasets ont été générés avec succès!")
print(f"📁 Les fichiers sont dans le dossier: {os.path.abspath('datasets')}")

# Afficher un résumé des datasets créés
print("\n📊 Résumé des datasets créés:")
datasets_info = [
    ("linear_classification.csv", "Classification binaire linéaire", "2 features, 2 classes"),
    ("multiclass_classification.csv", "Classification multi-classes", "3 features, 3 classes"),
    ("regression.csv", "Régression linéaire", "2 features, valeurs continues"),
    ("circles_classification.csv", "Cercles concentriques", "2 features, 2 classes non-linéaires"),
    ("moons_classification.csv", "Forme de lunes", "2 features, 2 classes non-linéaires"),
    ("blobs_classification.csv", "Clusters (blobs)", "2 features, 4 classes"),
    ("complex_classification.csv", "Classification complexe", "5 features, 2 classes"),
    ("sine_regression.csv", "Fonction sinus", "1 feature, régression non-linéaire"),
    ("spiral_classification.csv", "Spirale", "2 features, 2 classes très non-linéaires"),
    ("and_gate.csv", "Porte logique AND", "2 features binaires, 2 classes"),
    ("or_gate.csv", "Porte logique OR", "2 features binaires, 2 classes")
]

for filename, name, description in datasets_info:
    print(f"  - {filename}: {name} ({description})")