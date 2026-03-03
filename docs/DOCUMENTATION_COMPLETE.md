# Documentation Complète du Projet de Réseaux de Neurones

## Table des Matières
1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Architecture du système](#architecture-du-système)
3. [Implémentation du Machine Learning](#implémentation-du-machine-learning)
4. [Design Patterns utilisés](#design-patterns-utilisés)
5. [Principes SOLID appliqués](#principes-solid-appliqués)
6. [Structure des fichiers](#structure-des-fichiers)
7. [Fonctionnalités implémentées](#fonctionnalités-implémentées)
---

## Vue d'ensemble du projet

### Objectif
Développer un système d'apprentissage supervisé basé sur des réseaux de neurones artificiels, en appliquant les principes du génie logiciel pour créer des composants réutilisables et maintenables.

### Technologies utilisées
- **Langage** : Python 3.12
- **Bibliothèques principales** :
  - NumPy : Calculs matriciels et opérations mathématiques
  - Matplotlib : Visualisation des données et des performances
  - Tkinter : Interface graphique native
  - CSV/JSON : Gestion des données

### Points clés du projet
- Architecture MVC (Model-View-Controller)
- Implémentation from scratch des réseaux de neurones
- Support de multiples fonctions d'activation
- Interface graphique et ligne de commande
- Système de sauvegarde/chargement des modèles
- Génération et chargement de datasets variés

---

## Architecture du système

### 1. Architecture MVC

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      Views      │     │   Controllers   │     │     Models      │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ - GUI Interface │────▶│ - NetworkBuilder│────▶│ - Neuron        │
│ - CLI Interface │     │ - TrainingCtrl  │     │ - Layer         │
│ - Base View     │     │                 │     │ - Network       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ▲                       │                        │
         │                       ▼                        ▼
         │              ┌─────────────────┐     ┌─────────────────┐
         └──────────────│      Data       │     │      Utils      │
                        ├─────────────────┤     ├─────────────────┤
                        │ - DataLoader    │     │ - Visualization │
                        │ - DataGenerator │     │ - Metrics       │
                        └─────────────────┘     └─────────────────┘
```

### 2. Architecture en couches

**Couche Présentation (Views)**
- `gui_interface.py` : Interface graphique Tkinter avec onglets
- `cli_interface.py` : Interface en ligne de commande interactive
- Communication avec les contrôleurs via des méthodes standardisées

**Couche Logique (Controllers)**
- `network_builder.py` : Construction des réseaux (Pattern Builder)
- `training_controller.py` : Gestion de l'entraînement et de l'évaluation
- Orchestration entre les modèles et les vues

**Couche Modèle (Models)**
- `neuron.py` : Unité de base avec poids, biais et activation
- `layer.py` : Couche de neurones avec propagation avant/arrière
- `network.py` : Réseau complet avec méthodes d'apprentissage
- `activation_functions.py` : Fonctions d'activation (Strategy Pattern)

**Couche Données (Data)**
- `data_loader.py` : Chargement CSV/JSON, normalisation, split
- `data_generator.py` : Génération de datasets synthétiques

---

## Implémentation du Machine Learning

### 1. Structure d'un Neurone

```python
class Neuron:
    def __init__(self, input_size, activation_function):
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = 0.0
        self.activation = activation_function
        self.learning_rate = 0.01
        
    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation.activate(z)
        self.inputs = inputs
        return self.output
```

### 2. Propagation Avant (Forward Propagation)

Le processus suit ces étapes :
1. **Entrée** → Multiplication par les poids
2. **Somme pondérée** → Addition du biais
3. **Fonction d'activation** → Sortie du neurone

```
Entrée: x = [x1, x2, ..., xn]
Calcul: z = Σ(wi * xi) + b
Sortie: y = f(z) où f est la fonction d'activation
```

### 3. Rétropropagation (Backpropagation)

Algorithme d'apprentissage basé sur la descente de gradient :

1. **Calcul de l'erreur** : δ = (sortie_prédite - sortie_réelle)
2. **Propagation de l'erreur** : Chaque couche calcule son gradient
3. **Mise à jour des poids** : w = w - α * ∂E/∂w

```python
def backward(self, error):
    # Gradient de la fonction d'activation
    gradient = error * self.activation.derivative(self.output)
    
    # Mise à jour des poids
    self.weights -= self.learning_rate * gradient * self.inputs
    self.bias -= self.learning_rate * gradient
    
    # Propagation de l'erreur vers la couche précédente
    return np.dot(gradient, self.weights)
```

### 4. Fonctions d'Activation Implémentées

**Sigmoid**
- Formule : f(x) = 1 / (1 + e^(-x))
- Dérivée : f'(x) = f(x) * (1 - f(x))
- Usage : Classification binaire, sorties entre 0 et 1

**ReLU (Rectified Linear Unit)**
- Formule : f(x) = max(0, x)
- Dérivée : f'(x) = 1 si x > 0, sinon 0
- Usage : Couches cachées, évite le vanishing gradient

**Tanh (Tangente Hyperbolique)**
- Formule : f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- Dérivée : f'(x) = 1 - f(x)²
- Usage : Sorties centrées entre -1 et 1

**Linear**
- Formule : f(x) = x
- Dérivée : f'(x) = 1
- Usage : Régression, couche de sortie

### 5. Fonction de Perte

**Mean Squared Error (MSE)**
```python
def calculate_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)
```

---

## Design Patterns utilisés

### 1. **Builder Pattern** (NetworkBuilder)
```python
builder = NetworkBuilder()
network = (builder
    .add_layer(2, activation='sigmoid')
    .add_layer(4, activation='relu')
    .add_layer(1, activation='sigmoid')
    .set_learning_rate(0.1)
    .build())
```

**Avantages** :
- Construction étape par étape
- API fluide et intuitive
- Validation à la construction

### 2. **Strategy Pattern** (ActivationFunction)
```python
class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x):
        pass
    
    @abstractmethod
    def derivative(self, x):
        pass

class Sigmoid(ActivationFunction):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
```

**Avantages** :
- Ajout facile de nouvelles fonctions
- Interchangeabilité à l'exécution
- Séparation des responsabilités

### 3. **Factory Pattern** (ActivationFactory)
```python
class ActivationFactory:
    @staticmethod
    def create(name):
        activations = {
            'sigmoid': Sigmoid(),
            'relu': ReLU(),
            'tanh': Tanh()
        }
        return activations.get(name.lower())
```

### 4. **Observer Pattern** (Implicite dans GUI)
- Les vues observent les changements du modèle
- Mise à jour automatique de l'interface
- Découplage vue/modèle

### 5. **MVC Pattern** (Architecture globale)
- **Model** : Logique métier (neurones, réseaux)
- **View** : Interfaces utilisateur (GUI, CLI)
- **Controller** : Orchestration et logique de contrôle

---

## Principes SOLID appliqués

### 1. **Single Responsibility Principle (SRP)**
- `Neuron` : Gère uniquement la logique d'un neurone
- `Layer` : Gère uniquement une couche de neurones
- `DataLoader` : Responsable uniquement du chargement des données
- `TrainingController` : Gère uniquement l'entraînement

### 2. **Open/Closed Principle (OCP)**
```python
# Ouvert à l'extension via l'héritage
class ActivationFunction(ABC):
    pass

# Fermé à la modification
class NewActivation(ActivationFunction):
    def activate(self, x):
        # Nouvelle implémentation
```

### 3. **Liskov Substitution Principle (LSP)**
- Toutes les fonctions d'activation sont interchangeables
- Les interfaces View peuvent être substituées sans impact

### 4. **Interface Segregation Principle (ISP)**
```python
# Interfaces spécifiques et focalisées
class IDataSource:
    def load_data(self, path): pass

class ITrainable:
    def train(self, data): pass
    def evaluate(self, data): pass
```

### 5. **Dependency Inversion Principle (DIP)**
- Les classes de haut niveau dépendent d'abstractions
- Injection de dépendances dans les constructeurs
```python
class TrainingController:
    def __init__(self, network: Network, data_loader: IDataSource):
        self.network = network
        self.data_loader = data_loader
```

---

## Structure des fichiers

```
projet_geni_logiciel/
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neuron.py              # Classe Neuron
│   │   ├── layer.py               # Classe Layer
│   │   ├── network.py             # Classe Network
│   │   └── activation_functions.py # Fonctions d'activation
│   │
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── network_builder.py     # Builder pour les réseaux
│   │   └── training_controller.py # Contrôleur d'entraînement
│   │
│   ├── views/
│   │   ├── __init__.py
│   │   ├── gui_interface.py       # Interface graphique
│   │   └── cli_interface.py       # Interface CLI
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Chargement des données
│   │   └── data_generator.py      # Génération de données
│   │
│   └── utils/
│       ├── __init__.py
│       └── visualization.py       # Outils de visualisation
│
├── datasets/                      # Datasets CSV générés
├── docs/                          # Documentation
├── tests/                         # Tests unitaires
├── examples/                      # Exemples d'utilisation
│
├── main_gui.py                    # Point d'entrée GUI
├── main_cli.py                    # Point d'entrée CLI
├── requirements.txt               # Dépendances
└── README.md                      # Documentation principale
```

---

## Fonctionnalités implémentées

### 1. **Construction de réseaux**
- Architecture flexible (nombre de couches et neurones)
- Support de différentes fonctions d'activation par couche
- Initialisation aléatoire des poids (Xavier/He)

### 2. **Entraînement**
- Algorithme de rétropropagation complet
- Support du batch learning et online learning
- Métriques en temps réel (loss, accuracy)
- Early stopping pour éviter le surapprentissage

### 3. **Gestion des données**
- Chargement CSV avec détection automatique des colonnes
- Normalisation Min-Max et Z-score
- Division train/test configurable
- Support des problèmes de classification et régression

### 4. **Visualisation**
- Graphiques de perte pendant l'entraînement
- Visualisation des frontières de décision (2D)
- Matrice de confusion pour la classification
- Export des graphiques

### 5. **Persistance**
- Sauvegarde/chargement des modèles en JSON
- Export des poids et architecture
- Historique d'entraînement

### 6. **Interfaces utilisateur**
- GUI intuitive avec onglets
- CLI interactive avec menus
- Feedback en temps réel
- Gestion des erreurs utilisateur

---

## Commandes d'utilisation

### Interface Graphique
```bash
python main_gui.py
```

### Interface CLI
```bash
python main_cli.py
```

### Génération de datasets
```bash
python generate_datasets.py
```

### Tests du système
```bash
python test_system.py
```

---

## Conclusion

Ce projet démontre l'application réussie des principes du génie logiciel à un problème de machine learning. L'architecture modulaire, l'utilisation de design patterns et le respect des principes SOLID permettent de créer un système :

- **Extensible** : Ajout facile de nouvelles fonctionnalités
- **Maintenable** : Code clair et bien organisé
- **Réutilisable** : Composants indépendants et génériques
- **Robuste** : Gestion d'erreurs et validation des entrées
- **Performant** : Optimisations NumPy et algorithmes efficaces

Le système peut servir de base pour des applications plus complexes tout en restant un excellent outil pédagogique pour comprendre les réseaux de neurones.