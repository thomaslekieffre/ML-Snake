import os
import pygame
import random
import sys

# --------------------
# CONFIGURATION INITIALE
# --------------------
config = {
    'grid_cell_size': 20,
    'window_width': 600,
    'window_height': 600,
    'fps': 10,
}

# --------------------
# COULEURS
# --------------------
BLACK  = (0, 0, 0)
GREEN  = (0, 200, 0)
RED    = (200, 0, 0)
WHITE  = (255, 255, 255)
GREY   = (50, 50, 50)

# --------------------
# CLASSE SNAKEGAME
# --------------------
class SnakeGame:
    def __init__(self, config):
        pygame.init()
        self.config = config
        self.cell_size = config['grid_cell_size']
        self.window_width = config['window_width']
        self.window_height = config['window_height']
        self.grid_width = self.window_width // self.cell_size
        self.grid_height = self.window_height // self.cell_size
        self.fps = config['fps']

        self.display = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20, bold=True)

        self.reset()

    def reset(self):
        """
        Réinitialise le jeu :
        - Place le serpent au centre
        - Réinitialise la direction et le score
        - Génère une nouvelle pomme
        """
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = (0, -1)
        self.spawn_food()
        self.score = 0
        self.running = True

    def spawn_food(self):
        """
        Génère une nouvelle position pour la nourriture, en s'assurant qu'elle
        ne se trouve pas sur le serpent.
        """
        while True:
            self.food = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if self.food not in self.snake:
                break

    def move(self):
        """
        Met à jour la position du serpent en fonction de la direction actuelle.
        Vérifie les collisions avec les murs ou avec lui-même.
        """
        x, y = self.snake[0]
        dx, dy = self.direction
        new_head = (x + dx, y + dy)
        
        if new_head in self.snake or not (0 <= new_head[0] < self.grid_width and 0 <= new_head[1] < self.grid_height):
            self.running = False
            return
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()

    def update_ui(self):
        """
        Met à jour l'affichage du jeu, dessine le serpent, la nourriture,
        ainsi que le score.
        """
        self.display.fill(BLACK)
        
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(self.display, GREY, (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.cell_size):
            pygame.draw.line(self.display, GREY, (0, y), (self.window_width, y))
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0]*self.cell_size, pt[1]*self.cell_size, self.cell_size, self.cell_size))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0]*self.cell_size, self.food[1]*self.cell_size, self.cell_size, self.cell_size))
        
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, (10, 10))
        
        pygame.display.flip()

    def handle_events(self):
        """
        Gère les événements du clavier pour permettre au joueur de contrôler
        la direction du serpent.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != (0, 1):
                    self.direction = (0, -1)
                elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                    self.direction = (0, 1)
                elif event.key == pygame.K_LEFT and self.direction != (1, 0):
                    self.direction = (-1, 0)
                elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                    self.direction = (1, 0)

    def run(self):
        """
        Boucle principale du jeu :
        - Gère les entrées du joueur
        - Met à jour le jeu
        - Rafraîchit l'affichage
        """
        while self.running:
            self.handle_events()
            self.move()
            self.update_ui()
            self.clock.tick(self.fps)

        self.show_game_over()

    def show_game_over(self):
        """
        Affiche un écran de fin de partie et redémarre le jeu après quelques secondes.
        """
        self.display.fill(BLACK)
        over_text = self.font.render("GAME OVER", True, RED)
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(over_text, (self.window_width//2 - over_text.get_width()//2, 100))
        self.display.blit(score_text, (self.window_width//2 - score_text.get_width()//2, 150))
        pygame.display.flip()
        pygame.time.delay(2000)
        self.reset()
        self.run()

if __name__ == "__main__":
    game = SnakeGame(config)
    game.run()
