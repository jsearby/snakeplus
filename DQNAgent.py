import numpy as np
import random
import os
import pickle
from collections import deque
from contants import *

# Paramètres DQN
TAILLE_ETAT = 8
TAILLE_ACTION = 4
TAILLE_CACHEE = 64
MEMOIRE_MAX = 5000
BATCH_SIZE = 64
GAMMA = 0.95
EPSILON_INIT = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TAUX_APPRENTISSAGE = 0.01

class DQNAgent:
    def __init__(self):

        self.memoire = deque(maxlen=MEMOIRE_MAX)
        self.w1 = np.random.randn(TAILLE_ETAT, TAILLE_CACHEE) * 0.1
        self.b1 = np.zeros((1, TAILLE_CACHEE))
        self.w2 = np.random.randn(TAILLE_CACHEE, TAILLE_ACTION) * 0.1
        self.b2 = np.zeros((1, TAILLE_ACTION))
        self.load_model()

    def predict(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        return z2

    def act(self, etat):
        if np.random.rand() < self.epsilon:
            return random.randint(0, TAILLE_ACTION - 1)
        q_values = self.predict(etat)
        return np.argmax(q_values)

    def remember(self, etat, action, recompense, nouvel_etat, termine):
        self.memoire.append((etat, action, recompense, nouvel_etat, termine))

    def replay(self):
        if len(self.memoire) < BATCH_SIZE:
            return
        lot = random.sample(self.memoire, BATCH_SIZE)
        for etat, action, recompense, nouvel_etat, termine in lot:
            cible = self.predict(etat)
            if termine:
                cible[0][action] = recompense
            else:
                futur_q = np.max(self.predict(nouvel_etat))
                cible[0][action] = recompense + GAMMA * futur_q
            self._train(etat, cible)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def _train(self, x, y):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        erreur = z2 - y
        dz2 = erreur
        dw2 = np.dot(a1.T, dz2)
        db2 = dz2
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (1 - np.tanh(z1) ** 2)
        dw1 = np.dot(x.T, dz1)
        db1 = dz1
        self.w1 -= TAUX_APPRENTISSAGE * dw1
        self.b1 -= TAUX_APPRENTISSAGE * db1
        self.w2 -= TAUX_APPRENTISSAGE * dw2
        self.b2 -= TAUX_APPRENTISSAGE * db2

    def convertir_action_en_direction(self, action, dx, dy):
        if action == 0 and dy == 0:
            return 0, -TAILLE_CASE
        elif action == 1 and dy == 0:
            return 0, TAILLE_CASE
        elif action == 2 and dx == 0:
            return -TAILLE_CASE, 0
        elif action == 3 and dx == 0:
            return TAILLE_CASE, 0
        return dx, dy

    def extraire_etat(self, x, y, pomme_x, pomme_y, dx, dy, serpent):
        distance_x = (pomme_x - x) / LARGEUR
        distance_y = (pomme_y - y) / HAUTEUR
        danger_gauche = 1 if [x - TAILLE_CASE, y] in serpent or x - TAILLE_CASE < 0 else 0
        danger_droite = 1 if [x + TAILLE_CASE, y] in serpent or x + TAILLE_CASE >= LARGEUR else 0
        danger_haut = 1 if [x, y - TAILLE_CASE] in serpent or y - TAILLE_CASE < 0 else 0
        danger_bas = 1 if [x, y + TAILLE_CASE] in serpent or y + TAILLE_CASE >= HAUTEUR else 0
        return np.array([[distance_x, distance_y, dx / TAILLE_CASE, dy / TAILLE_CASE,
                          danger_gauche, danger_droite, danger_haut, danger_bas]])

    def save_model(self, nom_fichier="dqn_model.pkl"):
        with open(nom_fichier, "wb") as f:
            pickle.dump((self.w1, self.b1, self.w2, self.b2), f)

    def load_model(self, nom_fichier="dqn_model.pkl"):
        if os.path.exists(nom_fichier):
            print(f"Loading model")
            self.epsilon = EPSILON_MIN
            with open(nom_fichier, "rb") as f:
                self.w1, self.b1, self.w2, self.b2 = pickle.load(f)
        else:
            self.epsilon = EPSILON_INIT
            print(f"Le fichier model n'existe pas")

