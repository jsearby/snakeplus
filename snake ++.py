import pygame
import sys
import time
import math
import numpy as np
import random
from collections import deque
import torch.nn as nn
import torch.optim as optim
from DQNAgent import DQNAgent
from contants import *
from train import load_model

couleur_serpent = (50, 200, 50)  # Vert par défaut


# ========== FONCTION UTILITAIRE POUR L'ÉTAT ==========
def extraire_etat(x, y, pomme_x, pomme_y, dx, dy, serpent, largeur, hauteur):
    distance_x = (pomme_x - x) / largeur
    distance_y = (pomme_y - y) / hauteur
    danger_gauche = 1 if [x - 20, y] in serpent or x - 20 < 0 else 0
    danger_droite = 1 if [x + 20, y] in serpent or x + 20 >= largeur else 0
    danger_haut = 1 if [x, y - 20] in serpent or y - 20 < 0 else 0
    return np.array([[distance_x, distance_y, dx / 20, dy / 20, danger_gauche, danger_droite, danger_haut]])

def convertir_action_en_direction(action, dx, dy):
    if action == 0 and dy == 0:
        return 0, -20  # haut
    elif action == 1 and dy == 0:
        return 0, 20   # bas
    elif action == 2 and dx == 0:
        return -20, 0  # gauche
    elif action == 3 and dx == 0:
        return 20, 0   # droite
    return dx, dy

# ========== INITIALISATION ==========
pygame.init()
fenetre = pygame.display.set_mode((LARGEUR, HAUTEUR))
pygame.display.set_caption("Snake Deluxe 🐍")
fond_image = pygame.Surface((LARGEUR, HAUTEUR))
fond_image.fill((25, 25, 25))

# ========== COULEURS ==========
NOIR = (0, 0, 0)
BLANC = (255, 255, 255)
VERT = (50, 200, 50)
BLEU = (0, 160, 255)
JAUNE = (250, 200, 20)
ROUGE = (200, 50, 50)
VIOLET = (138, 43, 226)
ORANGE = (255, 165, 0)
ROSE = (25, 105, 180)
GRIS = (128, 128, 128)
FOND = (25, 25, 25)

# ========== POLICES ==========
police_titre = pygame.font.SysFont("comicsansms", 60, bold=True)
police_bouton = pygame.font.SysFont("arial", 30)
police_score = pygame.font.SysFont("arial", 26)

# ========== VARIABLES ==========
vitesse = 12

# ========== UTILITAIRES ==========
def quitter():
    pygame.quit()
    sys.exit()

def changer_vitesse(val):
    global vitesse
    vitesse = val

def dessiner_bouton(texte, x, y, l, h, couleur, couleur_hover, action=None):
    souris = pygame.mouse.get_pos()
    clique = pygame.mouse.get_pressed()
    rect = pygame.Rect(x, y, l, h)
    couleur_finale = couleur_hover if rect.collidepoint(souris) else couleur
    pygame.draw.rect(fenetre, couleur_finale, rect, border_radius=8)
    texte_rendu = police_bouton.render(texte, True, NOIR)
    texte_rect = texte_rendu.get_rect(center=rect.center)
    fenetre.blit(texte_rendu, texte_rect)
    if rect.collidepoint(souris) and clique[0]:
        pygame.time.delay(200)
        if action:
            action()

def dessiner_segment_glow(surface, x, y, taille, couleur=None, intensite=3):
    if couleur is None:
        couleur = couleur_serpent
    for i in range(intensite, 0, -1):
        alpha = int(50 / i)
        glow_surface = pygame.Surface((taille + i * 5, taille + i * 5), pygame.SRCALPHA)
        pygame.draw.rect(
            glow_surface,
            (*couleur, alpha),
            (0, 0, taille + i * 5, taille + i * 5),
            border_radius=taille // 2
        )
        surface.blit(glow_surface, (x - i * 2, y - i * 2))
    pygame.draw.rect(surface, couleur, (x, y, taille, taille), border_radius=taille // 2)


# ========== ÉCRANS ==========
def menu_classement():
    setattr(sys.modules[__name__], '_menu_exit', False)

    while True:
        fenetre.fill((20, 20, 30))
        titre = police_titre.render("Classement", True, JAUNE)
        fenetre.blit(titre, (LARGEUR // 2 - titre.get_width() // 2, 70))

        try:
            with open("scores.txt", "r") as f:
                scores = [int(l.strip()) for l in f.readlines()]
        except:
            scores = []

        scores = sorted(scores, reverse=True)[:5]
        for i, s in enumerate(scores):
            ligne = police_score.render(f"{i + 1}. {s} points", True, BLANC)
            fenetre.blit(ligne, (LARGEUR // 2 - ligne.get_width() // 2, 160 + i * 40))

        dessiner_bouton("Retour", 300, 480, 200, 50, ROUGE, JAUNE,
                        lambda: setattr(sys.modules[__name__], '_menu_exit', True))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()

        pygame.display.update()

        if getattr(sys.modules[__name__], '_menu_exit', False):
            break



def menu_reglages():
    setattr(sys.modules[__name__], '_menu_exit', False)

    while True:
        fenetre.fill(FOND)
        texte = police_titre.render("Réglages", True, BLANC)
        fenetre.blit(texte, (LARGEUR // 2 - texte.get_width() // 2, 80))

        dessiner_bouton("Vitesse : Lente", 300, 200, 200, 50, BLEU, VERT, lambda: changer_vitesse(8))
        dessiner_bouton("Vitesse : Moyenne", 300, 270, 200, 50, BLEU, VERT, lambda: changer_vitesse(12))
        dessiner_bouton("Vitesse : Rapide", 300, 340, 200, 50, BLEU, VERT, lambda: changer_vitesse(20))
        dessiner_bouton("Vitesse : Extrême", 300, 410, 200, 50, BLEU, VERT, lambda: changer_vitesse(28))

        dessiner_bouton("Retour", 300, 480, 200, 50, ROUGE, JAUNE,
                        lambda: setattr(sys.modules[__name__], '_menu_exit', True))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()

        pygame.display.update()

        if getattr(sys.modules[__name__], '_menu_exit', False):
            break


def afficher_ecran_game_over(score):
    setattr(sys.modules[__name__], '_menu_exit', False)

    while True:
        fenetre.fill((10, 10, 10))
        titre = police_titre.render("GAME OVER", True, ROUGE)
        fenetre.blit(titre, (LARGEUR // 2 - titre.get_width() // 2, 180))
        score_txt = police_bouton.render(f"Ton score : {score}", True, BLANC)
        fenetre.blit(score_txt, (LARGEUR // 2 - score_txt.get_width() // 2, 260))

        dessiner_bouton("Rejouer", 300, 300, 200, 50, VERT, BLEU, jouer)
        dessiner_bouton("Menu", 300, 370, 200, 50, ROUGE, JAUNE,
                        lambda: setattr(sys.modules[__name__], '_menu_exit', True))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()

        pygame.display.update()

        if getattr(sys.modules[__name__], '_menu_exit', False):
            break

def afficher_ecran_game_over_ia(score):
    setattr(sys.modules[__name__], '_menu_exit', False)

    while True:
        fenetre.fill((10, 10, 10))
        titre = police_titre.render("GAME OVER", True, ROUGE)
        fenetre.blit(titre, (LARGEUR // 2 - titre.get_width() // 2, 180))
        score_txt = police_bouton.render(f"Ton score : {score}", True, BLANC)
        fenetre.blit(score_txt, (LARGEUR // 2 - score_txt.get_width() // 2, 260))

        dessiner_bouton("Rejouer", 300, 300, 200, 50, VERT, BLEU, jouer_ia)
        dessiner_bouton("Menu", 300, 370, 200, 50, ROUGE, JAUNE,
                        lambda: setattr(sys.modules[__name__], '_menu_exit', True))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()

        pygame.display.update()

        if getattr(sys.modules[__name__], '_menu_exit', False):
            break

def afficher_ecran_victoire_ia(score):
    setattr(sys.modules[__name__], '_menu_exit', False)

    while True:
        fenetre.fill((10, 10, 10))
        titre = police_titre.render("VICTOIRE", True, JAUNE)
        fenetre.blit(titre, (LARGEUR // 2 - titre.get_width() // 2, 180))
        score_txt = police_bouton.render(f"Ton score : {score}", True, BLANC)
        fenetre.blit(score_txt, (LARGEUR // 2 - score_txt.get_width() // 2, 260))

        dessiner_bouton("Rejouer", 300, 300, 200, 50, VERT, BLEU, jouer_ia)
        dessiner_bouton("Menu", 300, 370, 200, 50, ROUGE, JAUNE,
                        lambda: setattr(sys.modules[__name__], '_menu_exit', True))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()

        pygame.display.update()

        if getattr(sys.modules[__name__], '_menu_exit', False):
            break


def animation_game_over(score):
    alpha = 0
    surface_noire = pygame.Surface((LARGEUR, HAUTEUR))
    surface_noire.fill((0, 0, 0))
    while alpha < 255:
        surface_noire.set_alpha(alpha)
        fenetre.blit(surface_noire, (0, 0))
        pygame.display.update()
        alpha += 5
        pygame.time.delay(20)
    afficher_ecran_game_over(score)

def animation_game_over_ia(score):
    alpha = 0
    surface_noire = pygame.Surface((LARGEUR, HAUTEUR))
    surface_noire.fill((0, 0, 0))
    while alpha < 255:
        surface_noire.set_alpha(alpha)
        fenetre.blit(surface_noire, (0, 0))
        pygame.display.update()
        alpha += 5
        pygame.time.delay(20)
    afficher_ecran_game_over_ia(score)

def animation_victoire_ia(score):
    alpha = 0
    surface_noire = pygame.Surface((LARGEUR, HAUTEUR))
    surface_noire.fill((0, 0, 0))
    while alpha < 255:
        surface_noire.set_alpha(alpha)
        fenetre.blit(surface_noire, (0, 0))
        pygame.display.update()
        alpha += 5
        pygame.time.delay(20)
    afficher_ecran_victoire_ia(score)



# ========== FONCTION DE JEU CLASSIQUE ==========
def jouer():
    global vitesse
    x = LARGEUR // 2
    y = HAUTEUR // 2
    dx, dy = 0, 0
    serpent = []
    longueur = 1
    score = 0
    debut_temps = time.time()
    pomme_x = random.randrange(0, LARGEUR - TAILLE, TAILLE)
    pomme_y = random.randrange(0, HAUTEUR - TAILLE, TAILLE)
    horloge = pygame.time.Clock()
    en_jeu = True
    temps_depart_pulsation = time.time()
    while en_jeu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and dx == 0:
                    dx, dy = -TAILLE, 0
                elif event.key == pygame.K_RIGHT and dx == 0:
                    dx, dy = TAILLE, 0
                elif event.key == pygame.K_UP and dy == 0:
                    dx, dy = 0, -TAILLE
                elif event.key == pygame.K_DOWN and dy == 0:
                    dx, dy = 0, TAILLE
        x += dx
        y += dy
        if x < 0 or x >= LARGEUR or y < 0 or y >= HAUTEUR:
            en_jeu = False
        tete = [x, y]
        serpent.append(tete)
        if len(serpent) > longueur:
            del serpent[0]
        for segment in serpent[:-1]:
            if segment == tete:
                en_jeu = False
        fenetre.blit(fond_image, (0, 0))
        ROUGE_FLASH = (255, 0, 0)
        temps_actuel = time.time() - temps_depart_pulsation
        facteur_pulsation = 1 + 0.1 * math.sin(temps_actuel * 4)
        taille_pomme = int(TAILLE * facteur_pulsation)
        offset = (taille_pomme - TAILLE) // 2
        pygame.draw.rect(fenetre, ROUGE_FLASH, (pomme_x - offset, pomme_y - offset, taille_pomme, taille_pomme),
                         border_radius=6)

        for s in serpent:
            dessiner_segment_glow(fenetre, s[0], s[1], TAILLE)
        texte_score = police_score.render(f"Score : {score}", True, BLANC)
        fenetre.blit(texte_score, (10, 10))
        temps_ecoule = int(time.time() - debut_temps)
        texte_temps = police_score.render(f"Temps : {temps_ecoule}s", True, BLANC)
        fenetre.blit(texte_temps, (LARGEUR - 160, 10))
        pygame.display.update()
        if x == pomme_x and y == pomme_y:
            pomme_x = random.randrange(0, LARGEUR - TAILLE, TAILLE)
            pomme_y = random.randrange(0, HAUTEUR - TAILLE, TAILLE)
            longueur += 1
            score += 10
        horloge.tick(vitesse)
    with open("scores.txt", "a") as f:
        f.write(str(score) + "\n")
    animation_game_over(score)

#=========== FONCTION JEU IA ==========
def jouer_ia():
    global vitesse
    TAILLE = 20

    # Joueur humain
    x1, y1 = LARGEUR // 4, HAUTEUR // 2
    dx1, dy1 = TAILLE, 0
    serpent1 = []
    longueur1 = 1
    score1 = 0
    vies1 = 3
    invincible1 = False
    temps_invincible1 = 0

    # Agent IA
    x2, y2 = 3 * LARGEUR // 4, HAUTEUR // 2 + 60
    dx2, dy2 = -TAILLE, 0
    serpent2 = []
    longueur2 = 1
    score2 = 0
    vies2 = 3
    invincible2 = False
    temps_invincible2 = 0

    agent = DQNAgent()

    # Pomme
    pomme_x = random.randrange(0, LARGEUR - TAILLE, TAILLE)
    pomme_y = random.randrange(0, HAUTEUR - TAILLE, TAILLE)

    horloge = pygame.time.Clock()
    debut_temps = time.time()
    en_jeu = True

    while en_jeu:
        temps_actuel = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and dx1 == 0:
                    dx1, dy1 = -TAILLE, 0
                elif event.key == pygame.K_RIGHT and dx1 == 0:
                    dx1, dy1 = TAILLE, 0
                elif event.key == pygame.K_UP and dy1 == 0:
                    dx1, dy1 = 0, -TAILLE
                elif event.key == pygame.K_DOWN and dy1 == 0:
                    dx1, dy1 = 0, TAILLE

        # IA : décision
        etat_ia = agent.extraire_etat(x2, y2, pomme_x, pomme_y, dx2, dy2, serpent2)
        action = agent.act(etat_ia)
        dx2, dy2 = agent.convertir_action_en_direction(action, dx2, dy2)

        # Mise à jour positions
        if not invincible1:
            x1 += dx1
            y1 += dy1
        if not invincible2:
            x2 += dx2
            y2 += dy2

        # Vérification collisions
        def collision(x, y, serpent):
            return x < 0 or x >= LARGEUR or y < 0 or y >= HAUTEUR or [x, y] in serpent

        if not invincible1 and (collision(x1, y1, serpent1) or [x1, y1] in serpent2):
            vies1 -= 1
            if vies1 == 0:
                animation_game_over_ia(score1)
                return
            invincible1 = True
            temps_invincible1 = temps_actuel
            x1, y1 = LARGEUR // 4, HAUTEUR // 2
            dx1, dy1 = TAILLE, 0
            serpent1 = []
            longueur1 = 1

        if not invincible2 and (collision(x2, y2, serpent2) or [x2, y2] in serpent1):
            vies2 -= 1
            if vies2 == 0:
                afficher_ecran_victoire_ia(score1)  # joueur gagne
                return
            invincible2 = True
            temps_invincible2 = temps_actuel
            x2, y2 = 3 * LARGEUR // 4, HAUTEUR // 2 + 60
            dx2, dy2 = -TAILLE, 0
            serpent2 = []
            longueur2 = 1

        # Fin de l'invincibilité
        if invincible1 and temps_actuel - temps_invincible1 > 5:
            invincible1 = False
        if invincible2 and temps_actuel - temps_invincible2 > 5:
            invincible2 = False

        # Mise à jour serpents
        if not invincible1:
            tete1 = [x1, y1]
            serpent1.append(tete1)
            if len(serpent1) > longueur1:
                del serpent1[0]

        if not invincible2:
            tete2 = [x2, y2]
            serpent2.append(tete2)
            if len(serpent2) > longueur2:
                del serpent2[0]

        # Pomme mangée
        if x1 == pomme_x and y1 == pomme_y:
            longueur1 += 1
            score1 += 10
            pomme_x = random.randrange(0, LARGEUR - TAILLE, TAILLE)
            pomme_y = random.randrange(0, HAUTEUR - TAILLE, TAILLE)
        elif x2 == pomme_x and y2 == pomme_y:
            longueur2 += 1
            score2 += 10
            pomme_x = random.randrange(0, LARGEUR - TAILLE, TAILLE)
            pomme_y = random.randrange(0, HAUTEUR - TAILLE, TAILLE)

        # Affichage
        fenetre.blit(fond_image, (0, 0))
        pygame.draw.rect(fenetre, ROUGE, (pomme_x, pomme_y, TAILLE, TAILLE), border_radius=6)

        for s in serpent1:
            dessiner_segment_glow(fenetre, s[0], s[1], TAILLE, VERT)
        for s in serpent2:
            dessiner_segment_glow(fenetre, s[0], s[1], TAILLE, BLEU)

        texte_score1 = police_score.render(f"Joueur : {score1} | Vies : {vies1}", True, BLANC)
        texte_score2 = police_score.render(f"IA : {score2} | Vies : {vies2}", True, BLANC)
        fenetre.blit(texte_score1, (10, 10))
        fenetre.blit(texte_score2, (LARGEUR - 250, 10))

        pygame.display.update()
        horloge.tick(vitesse)


# ========== MENU PRINCIPAL ==========
def menu_principal():
    while True:
        fenetre.fill(FOND)
        titre = police_titre.render("SNAKE DELUXE 🐍", True, JAUNE)
        fenetre.blit(titre, (LARGEUR // 2 - titre.get_width() // 2, 100))
        dessiner_bouton("Jouer", 300, 220, 200, 50, VERT, BLEU, jouer)
        dessiner_bouton("Classement", 300, 290, 200, 50, BLEU, VERT, menu_classement)
        dessiner_bouton("Réglages", 300, 360, 200, 50, BLEU, VERT, menu_reglages)
        dessiner_bouton("Quitter", 300, 430, 200, 50, ROUGE, JAUNE, quitter)
        dessiner_bouton("VS IA", 150, 325, 100, 50, JAUNE, VERT, jouer_ia)
        dessiner_bouton("Casier", 570, 325, 100, 50, VIOLET, VERT, menu_casier)
        #dessiner_bouton("?", 570, 385, 100, 50, JAUNE, VERT, )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()
        pygame.display.update()

# ========== MENU FOND D'ECRAN ==========
#def menu_fond_ecran()

    #fond_disponible = pygame.image.load(C:\Users\james\PyCharmMiscProject\fond_pixel_nature.png)

    #while True:
       # fenetre.fill(FOND)
      #  titre = police_titre.render("fond d'écrans", True, ROSE)
     #   fenetre.blit(titre, (LARGEUR // 2 - titre.get_width() // 2, 60))

    #dessiner_bouton("Retour", 300, 500, 200, 50, ROUGE, JAUNE,
     #               lambda: setattr(sys.modules[__name__], '_menu_exit', True))
# ========== MENU CASIER ==========
def menu_casier():
    global couleur_serpent
    setattr(sys.modules[__name__], '_menu_exit', False)

    couleurs_disponibles = [
        (50, 200, 50), (0, 160, 255), (250, 200, 20), (200, 50, 50),
        (255, 105, 180), (138, 43, 226), (255, 165, 0), (0, 255, 127),
        (255, 255, 255), (128, 128, 128)
    ]

    while True:
        fenetre.fill(FOND)
        titre = police_titre.render("Casier 🎨", True, JAUNE)
        fenetre.blit(titre, (LARGEUR // 2 - titre.get_width() // 2, 60))

        for i, couleur in enumerate(couleurs_disponibles):
            x = 100 + (i % 5) * 130
            y = 180 + (i // 5) * 130
            rect = pygame.Rect(x, y, 100, 100)
            pygame.draw.rect(fenetre, couleur, rect, border_radius=12)
            if rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(fenetre, BLANC, rect, 4, border_radius=12)
                if pygame.mouse.get_pressed()[0]:
                    couleur_serpent = couleur
                    pygame.time.delay(200)

        dessiner_bouton("Retour", 300, 500, 200, 50, ROUGE, JAUNE,
                        lambda: setattr(sys.modules[__name__], '_menu_exit', True))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitter()

        pygame.display.update()

        if getattr(sys.modules[__name__], '_menu_exit', False):
            break



# ========== DÉMARRAGE ==========
menu_principal()

