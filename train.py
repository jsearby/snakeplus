import pygame
import sys
import random
import numpy as np
from DQNAgent import DQNAgent
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

# Couleurs
NOIR = (0, 0, 0)
VERT = (0, 255, 0)
ROUGE = (255, 0, 0)
BLANC = (255, 255, 255)


def afficher_infos(surface, score, epsilon, generation):
    font = pygame.font.SysFont(None, 24)
    texte = font.render(f"Score: {score}  Epsilon: {epsilon:.3f}  Génération: {generation}", True, BLANC)
    surface.blit(texte, (10, 10))

def main():
    pygame.init()
    fenetre = pygame.display.set_mode((LARGEUR, HAUTEUR))
    pygame.display.set_caption("Snake DQN")
    horloge = pygame.time.Clock()
    agent = DQNAgent()
    generation = 0

    while generation < 20:
        generation += 1
        x, y = LARGEUR // 2, HAUTEUR // 2
        dx, dy = 0, -TAILLE_CASE
        serpent = [[x, y]]
        pomme = [random.randrange(0, LARGEUR, TAILLE_CASE), random.randrange(0, HAUTEUR, TAILLE_CASE)]
        score = 0
        pas_sans_manger = 0
        termine = False

        while not termine:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent.save_model()
                    pygame.quit()
                    sys.exit()

            etat = agent.extraire_etat(x, y, pomme[0], pomme[1], dx, dy, serpent)
            action = agent.act(etat)
            dx, dy = agent.convertir_action_en_direction(action, dx, dy)

            x += dx
            y += dy
            pas_sans_manger += 1

            if [x, y] in serpent or x < 0 or x >= LARGEUR or y < 0 or y >= HAUTEUR or pas_sans_manger > 200:
                recompense = -1
                nouvel_etat = agent.extraire_etat(x, y, pomme[0], pomme[1], dx, dy, serpent)
                agent.remember(etat, action, recompense, nouvel_etat, True)
                agent.replay()
                termine = True
                continue

            serpent.insert(0, [x, y])
            if x == pomme[0] and y == pomme[1]:
                score += 1
                recompense = 3
                pomme = [random.randrange(0, LARGEUR, TAILLE_CASE), random.randrange(0, HAUTEUR, TAILLE_CASE)]
                pas_sans_manger = -0.5
            else:
                serpent.pop()
                dist_avant = np.linalg.norm([etat[0][0], etat[0][1]])
                nouvel_etat = agent.extraire_etat(x, y, pomme[0], pomme[1], dx, dy, serpent)
                dist_apres = np.linalg.norm([nouvel_etat[0][0], nouvel_etat[0][1]])
                recompense = 0 if dist_apres < dist_avant else -0.5

            nouvel_etat = agent.extraire_etat(x, y, pomme[0], pomme[1], dx, dy, serpent)
            agent.remember(etat, action, recompense, nouvel_etat, False)
            agent.replay()

            fenetre.fill(NOIR)
            for segment in serpent:
                pygame.draw.rect(fenetre, VERT, pygame.Rect(segment[0], segment[1], TAILLE_CASE, TAILLE_CASE))
            pygame.draw.rect(fenetre, ROUGE, pygame.Rect(pomme[0], pomme[1], TAILLE_CASE, TAILLE_CASE))
            afficher_infos(fenetre, score, agent.epsilon, generation)
            pygame.display.flip()
            horloge.tick(100)

    agent.save_model()

if __name__ == "__main__":
    main()

