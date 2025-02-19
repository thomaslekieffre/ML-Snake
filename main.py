import os
import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys
import copy  # Pour copier en profondeur le state_dict

# --------------------
# CONFIGURATION INITIALE
# --------------------
config = {
    'max_epsilon': 80,        # Valeur de départ pour l'exploration
    'learning_rate': 0.001,
    'gamma': 0.9,
    'grid_cell_size': 20,
    'window_width': 600,
    'window_height': 600,
    'fps': 100,
    'generation_length': 10,  # Nombre d'épisodes par génération
    'mutation_rate': 0.02     # Taux de mutation (optionnel)
}

# --------------------
# COULEURS
# --------------------
BLACK  = (0, 0, 0)
GREEN  = (0, 200, 0)
RED    = (200, 0, 0)
WHITE  = (255, 255, 255)
GREY   = (50, 50, 50)
BLUE   = (0, 0, 255)
YELLOW = (255, 255, 0)

# --------------------
# INITIALISATION DU SON
# --------------------
pygame.mixer.init()
try:
    pygame.mixer.music.load("background.mp3")
    pygame.mixer.music.play(-1)
except Exception as e:
    print("Background music non trouvé :", e)

try:
    sound_food = pygame.mixer.Sound("food.wav")
except Exception as e:
    sound_food = None
    print("Sound food non trouvé :", e)

try:
    sound_game_over = pygame.mixer.Sound("gameover.wav")
except Exception as e:
    sound_game_over = None
    print("Sound game over non trouvé :", e)

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
        pygame.display.set_caption("Snake IA - YouTube Edition 🚀🐍")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20, bold=True)

        # Modes possibles: 'menu', 'play', 'replay', 'game_over'
        self.mode = 'menu'
        self.replay_frames = []    # Pour le replay slow‑motion
        self.best_score = 0
        self.episode = 0
        self.epsilon = config['max_epsilon']
        self.generation = 1        # Système de générations
        self.commentary = ""
        self.scores_history = []   # Historique des scores pour le graphique

        # BONUS POWER-UP
        self.bonus_food = None     # Coordonnées bonus (si présent)
        self.bonus_timer = 0       # Durée de vie du bonus

        # PARTICULES pour animation bonus
        self.particles = []

    def show_menu(self):
        menu = True
        selected = 0
        options = [
            "Max Epsilon: " + str(self.config['max_epsilon']),
            "Learning Rate: " + str(self.config['learning_rate']),
            "Gamma: " + str(self.config['gamma']),
            "Start Game"
        ]
        while menu:
            self.display.fill(BLACK)
            title_surface = self.font.render("Configuration du Jeu", True, WHITE)
            self.display.blit(title_surface, (self.window_width//2 - title_surface.get_width()//2, 50))
            for i, option in enumerate(options):
                color = GREEN if i == selected else WHITE
                option_surface = self.font.render(option, True, color)
                self.display.blit(option_surface, (self.window_width//2 - option_surface.get_width()//2, 150 + i * 40))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected = (selected - 1) % len(options)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(options)
                    elif event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                        if selected == 0:
                            self.config['max_epsilon'] += 5
                            options[0] = "Max Epsilon: " + str(self.config['max_epsilon'])
                        elif selected == 1:
                            self.config['learning_rate'] = round(self.config['learning_rate'] + 0.0005, 4)
                            options[1] = "Learning Rate: " + str(self.config['learning_rate'])
                        elif selected == 2:
                            self.config['gamma'] = round(min(self.config['gamma'] + 0.05, 0.99), 2)
                            options[2] = "Gamma: " + str(self.config['gamma'])
                        elif selected == 3:
                            menu = False
            self.clock.tick(10)

    def reset(self):
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = (0, -1)  # Initialement vers le haut
        self.spawn_food()
        self.score = 0
        self.frame_iteration = 0
        self.commentary = ""
        self.replay_frames = []
        self.bonus_food = None
        self.bonus_timer = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, self.grid_width - 1),
                         random.randint(0, self.grid_height - 1))
            if self.food not in self.snake:
                break

    def spawn_bonus(self):
        # Spawner un bonus avec une faible probabilité s'il n'y en a pas déjà un
        if self.bonus_food is None and random.random() < 0.005:
            while True:
                bonus = (random.randint(0, self.grid_width - 1),
                         random.randint(0, self.grid_height - 1))
                if bonus not in self.snake and bonus != self.food:
                    self.bonus_food = bonus
                    self.bonus_timer = 200  # Le bonus dure 200 frames
                    break

    def update_particles(self):
        # Mise à jour et affichage des particules
        for particle in self.particles[:]:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)
            else:
                pygame.draw.circle(self.display, YELLOW,
                                   (int(particle['pos'][0]), int(particle['pos'][1])), 3)

    def play_step(self, action):
        self.frame_iteration += 1
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # Mouvement du snake
        self.move(action)
        self.snake.insert(0, self.head)
        reward = 0
        if self.head == self.food:
            self.score += 1
            reward = 10
            if sound_food:
                sound_food.play()
            self.spawn_food()
        else:
            self.snake.pop()
        # BONUS power-up
        self.spawn_bonus()
        if self.bonus_food:
            self.bonus_timer -= 1
            if self.bonus_timer <= 0:
                self.bonus_food = None
            if self.head == self.bonus_food:
                self.score += 100
                reward = 100
                self.bonus_food = None
                for _ in range(20):
                    self.particles.append({
                        'pos': [self.head[0]*self.cell_size + self.cell_size/2,
                                self.head[1]*self.cell_size + self.cell_size/2],
                        'vel': [random.uniform(-2, 2), random.uniform(-2, 2)],
                        'life': 30,
                        'max_life': 30
                    })
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            if sound_game_over:
                sound_game_over.play()
        if self.check_danger():
            self.commentary = "Attention, risque de collision !"
        else:
            self.commentary = ""
        if self.score > self.best_score:
            self.best_score = self.score
            self.commentary = "Score record en cours !"
        # Pénalité temporelle pour éviter que l'agent tourne en rond
        reward -= 0.01
        # Condition d'arrêt supplémentaire : si trop de frames se sont écoulées
        if self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
        # Enregistrer la frame pour le replay
        self.replay_frames.append({
            'snake': self.snake.copy(),
            'food': self.food,
            'bonus_food': self.bonus_food,
            'score': self.score,
            'frame': self.frame_iteration,
            'commentary': self.commentary
        })
        self.update_ui()
        self.clock.tick(self.fps)
        return self.get_state(), reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt[0] < 0 or pt[0] >= self.grid_width or pt[1] < 0 or pt[1] >= self.grid_height:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def check_danger(self):
        head = self.snake[0]
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for d in directions:
            pt = (head[0] + d[0], head[1] + d[1])
            if self.is_collision(pt):
                return True
        return False

    def update_ui(self):
        self.display.fill(BLACK)
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(self.display, GREY, (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.cell_size):
            pygame.draw.line(self.display, GREY, (0, y), (self.window_width, y))
        for pt in self.snake:
            rect = pygame.Rect(pt[0]*self.cell_size, pt[1]*self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.display, GREEN, rect)
            pygame.draw.rect(self.display, BLACK, rect, 1)
        food_rect = pygame.Rect(self.food[0]*self.cell_size, self.food[1]*self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.display, RED, food_rect)
        if self.bonus_food:
            bonus_rect = pygame.Rect(self.bonus_food[0]*self.cell_size, self.bonus_food[1]*self.cell_size,
                                     self.cell_size, self.cell_size)
            pygame.draw.rect(self.display, YELLOW, bonus_rect)
        info_lines = [
            f"Episode  : {self.episode}",
            f"Score    : {self.score}",
            f"Best     : {self.best_score}",
            f"Epsilon  : {self.epsilon:.2f}",
            f"Frame    : {self.frame_iteration}",
            f"Génération: {self.generation}",
            f"{self.commentary}"
        ]
        y_offset = 10
        for line in info_lines:
            text_surface = self.font.render(line, True, WHITE)
            self.display.blit(text_surface, (10, y_offset))
            y_offset += 25
        if self.scores_history:
            graph_x = self.window_width - 150
            graph_y = 10
            graph_width = 140
            graph_height = 100
            pygame.draw.rect(self.display, GREY, (graph_x, graph_y, graph_width, graph_height), 1)
            title_text = self.font.render("Historique", True, WHITE)
            self.display.blit(title_text, (graph_x + (graph_width - title_text.get_width())//2, graph_y - 25))
            max_score = max(self.scores_history) if max(self.scores_history) > 0 else 1
            label_top = self.font.render(str(max_score), True, WHITE)
            label_bottom = self.font.render("0", True, WHITE)
            self.display.blit(label_top, (graph_x - label_top.get_width() - 5, graph_y - label_top.get_height()//2))
            self.display.blit(label_bottom, (graph_x - label_bottom.get_width() - 5, graph_y + graph_height - label_bottom.get_height()))
            if len(self.scores_history) > 1:
                for i in range(len(self.scores_history)-1):
                    x1 = graph_x + int(i / (len(self.scores_history)-1) * graph_width)
                    y1 = graph_y + graph_height - int(self.scores_history[i] / max_score * graph_height)
                    x2 = graph_x + int((i+1) / (len(self.scores_history)-1) * graph_width)
                    y2 = graph_y + graph_height - int(self.scores_history[i+1] / max_score * graph_height)
                    pygame.draw.line(self.display, BLUE, (x1, y1), (x2, y2), 2)
        if hasattr(self, "latest_q_values") and self.latest_q_values is not None:
            q_text = self.font.render(f"Q: {np.around(self.latest_q_values, 2)}", True, WHITE)
            self.display.blit(q_text, (self.window_width - q_text.get_width() - 10,
                                       self.window_height - q_text.get_height() - 10))
        self.update_particles()
        pygame.display.flip()

    def move(self, action):
        clock_wise = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        idx = clock_wise.index(self.direction)
        if action == [1, 0, 0]:
            new_dir = clock_wise[idx]
        elif action == [0, 1, 0]:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.snake[0][0] + self.direction[0]
        y = self.snake[0][1] + self.direction[1]
        self.head = (x, y)

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)
      
        # Danger autour
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)
        danger_straight = (dir_r and self.is_collision(point_r)) or \
                          (dir_l and self.is_collision(point_l)) or \
                          (dir_u and self.is_collision(point_u)) or \
                          (dir_d and self.is_collision(point_d))
      
        # Danger à droite
        if self.direction == (0, -1):
            point_r = (head[0] + 1, head[1])
        elif self.direction == (1, 0):
            point_r = (head[0], head[1] + 1)
        elif self.direction == (0, 1):
            point_r = (head[0] - 1, head[1])
        else:
            point_r = (head[0], head[1] - 1)
        danger_right = self.is_collision(point_r)
      
        # Danger à gauche
        if self.direction == (0, -1):
            point_l = (head[0] - 1, head[1])
        elif self.direction == (1, 0):
            point_l = (head[0], head[1] - 1)
        elif self.direction == (0, 1):
            point_l = (head[0] + 1, head[1])
        else:
            point_l = (head[0], head[1] + 1)
        danger_left = self.is_collision(point_l)
      
        # Informations pour la nourriture classique
        food_left = int(self.food[0] < head[0])
        food_right = int(self.food[0] > head[0])
        food_up = int(self.food[1] < head[1])
        food_down = int(self.food[1] > head[1])
      
        # Informations sur le bonus
        bonus_present = int(self.bonus_food is not None)
        if bonus_present:
            bonus_left = int(self.bonus_food[0] < head[0])
            bonus_right = int(self.bonus_food[0] > head[0])
            bonus_up = int(self.bonus_food[1] < head[1])
            bonus_down = int(self.bonus_food[1] > head[1])
        else:
            bonus_left, bonus_right, bonus_up, bonus_down = 0, 0, 0, 0
      
        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            food_left,
            food_right,
            food_up,
            food_down,
            bonus_present,
            bonus_left,
            bonus_right,
            bonus_up,
            bonus_down
        ]
        return np.array(state, dtype=int)

    def show_game_over(self):
        over = True
        start_time = pygame.time.get_ticks()
        while over:
            self.display.fill(BLACK)
            over_text = self.font.render("GAME OVER", True, RED)
            score_text = self.font.render(f"Score: {self.score}", True, WHITE)
            best_text = self.font.render(f"Meilleur Score: {self.best_score}", True, WHITE)
            replay_text = self.font.render("Appuyez sur R pour replay slow-motion", True, WHITE)
            new_game_text = self.font.render("Appuyez sur N pour nouvelle partie", True, WHITE)
            auto_text = self.font.render("Lancement auto dans 2s...", True, WHITE)
            self.display.blit(over_text, (self.window_width//2 - over_text.get_width()//2, 100))
            self.display.blit(score_text, (self.window_width//2 - score_text.get_width()//2, 150))
            self.display.blit(best_text, (self.window_width//2 - best_text.get_width()//2, 200))
            self.display.blit(replay_text, (self.window_width//2 - replay_text.get_width()//2, 250))
            self.display.blit(new_game_text, (self.window_width//2 - new_game_text.get_width()//2, 300))
            self.display.blit(auto_text, (self.window_width//2 - auto_text.get_width()//2, 350))
            pygame.display.flip()
            current_time = pygame.time.get_ticks()
            if current_time - start_time >= 2000:
                over = False
                self.mode = 'play'
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        over = False
                        self.mode = 'replay'
                    elif event.key == pygame.K_n:
                        over = False
                        self.mode = 'play'
            self.clock.tick(10)

    def run_replay(self):
        for frame in self.replay_frames:
            self.display.fill(BLACK)
            for x in range(0, self.window_width, self.cell_size):
                pygame.draw.line(self.display, GREY, (x, 0), (x, self.window_height))
            for y in range(0, self.window_height, self.cell_size):
                pygame.draw.line(self.display, GREY, (0, y), (self.window_width, y))
            for pt in frame['snake']:
                rect = pygame.Rect(pt[0]*self.cell_size, pt[1]*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.display, GREEN, rect)
                pygame.draw.rect(self.display, BLACK, rect, 1)
            food_rect = pygame.Rect(frame['food'][0]*self.cell_size, frame['food'][1]*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.display, RED, food_rect)
            if frame.get('bonus_food'):
                bonus_rect = pygame.Rect(frame['bonus_food'][0]*self.cell_size, frame['bonus_food'][1]*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.display, YELLOW, bonus_rect)
            info_lines = [
                f"Replay - Frame: {frame['frame']}",
                f"Score: {frame['score']}",
                f"{frame['commentary']}"
            ]
            y_offset = 10
            for line in info_lines:
                text_surface = self.font.render(line, True, WHITE)
                self.display.blit(text_surface, (10, y_offset))
                y_offset += 25
            pygame.display.flip()
            self.clock.tick(5)
        self.show_game_over()

# --------------------
# RÉSEAU DE NEURONES & AGENT DQN
# --------------------
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class Agent:
    def __init__(self, config):
        self.n_games = 0
        self.epsilon = config['max_epsilon']
        self.gamma = config['gamma']
        self.memory = deque(maxlen=100_000)
        # On utilise 16 entrées pour inclure l'info du bonus
        self.model = Linear_QNet(16, 256, 3)
        self.trainer = QTrainer(self.model, lr=config['learning_rate'], gamma=self.gamma)
        self.scores_history = []
        self.loss_history = []
        self.latest_q_values = None
        self.generation = 1
        self.generation_length = config['generation_length']
        self.mutation_rate = config['mutation_rate']
        self.best_score_generation = -float('inf')
        self.best_model_state = None

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss_val = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.loss_history.append(loss_val)

    def train_short_memory(self, state, action, reward, next_state, done):
        loss_val = self.trainer.train_step(state, action, reward, next_state, done)
        self.loss_history.append(loss_val)

    def get_action(self, state):
        self.epsilon = config['max_epsilon'] - self.n_games
        final_move = [0, 0, 0]
        is_random = False
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            is_random = True
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.latest_q_values = prediction.detach().numpy()
        return final_move, is_random

    def mutate(self):
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * self.mutation_rate
                param.add_(noise)
        print(f"Mutation appliquée pour la génération {self.generation}.")

    def save_checkpoint(self, filename="checkpoint.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'n_games': self.n_games,
            'generation': self.generation,
            'best_score_generation': self.best_score_generation,
            'scores_history': self.scores_history,
        }, filename)
        print("Checkpoint sauvegardé.")

    def load_checkpoint(self, filename="checkpoint.pth"):
        checkpoint = torch.load(filename)
        state_dict = checkpoint['model_state_dict']
        # Si le checkpoint est issu d'une version avec 11 entrées, on adapte le poids du premier layer
        if state_dict['linear1.weight'].size(1) != 16:
            print("Mise à jour des poids du premier layer depuis le checkpoint existant.")
            old_weight = state_dict['linear1.weight']  # shape [256, 11]
            new_weight = torch.zeros(256, 16)
            new_weight[:, :old_weight.size(1)] = old_weight
            # Initialiser aléatoirement les nouvelles colonnes
            new_weight[:, old_weight.size(1):] = torch.randn(256, 16 - old_weight.size(1)) * 0.01
            state_dict['linear1.weight'] = new_weight
        self.model.load_state_dict(state_dict)
        # Essayer de charger l'état de l'optimizer, sinon le réinitialiser
        try:
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print("Impossible de charger l'état de l'optimizer, réinitialisation...", e)
        self.n_games = checkpoint.get('n_games', 0)
        self.generation = checkpoint.get('generation', 1)
        self.best_score_generation = checkpoint.get('best_score_generation', -float('inf'))
        self.scores_history = checkpoint.get('scores_history', [])
        print("Checkpoint chargé.")

def main():
    resume = False
    if os.path.exists("checkpoint.pth"):
        print("Checkpoint trouvé. Reprendre l'entraînement ? (y/n)")
        answer = input()
        if answer.lower().startswith("y"):
            resume = True
    game_instance = SnakeGame(config)
    game_instance.show_menu()
    agent = Agent(config)
    if resume:
        agent.load_checkpoint("checkpoint.pth")
    while True:
        state_old = game_instance.reset()
        game_instance.episode = agent.n_games
        game_instance.epsilon = agent.epsilon
        game_instance.generation = agent.generation
        game_over = False
        while not game_over:
            state_old = game_instance.get_state()
            action, is_random = agent.get_action(state_old)
            if is_random:
                game_instance.commentary = "L'IA explore de nouvelles stratégies !"
            state_new, reward, game_over, score = game_instance.play_step(action)
            agent.train_short_memory(state_old, action, reward, state_new, game_over)
            agent.remember(state_old, action, reward, state_new, game_over)
        agent.n_games += 1
        agent.train_long_memory()
        agent.scores_history.append(score)
        game_instance.scores_history = agent.scores_history
        if score > agent.best_score_generation:
            agent.best_score_generation = score
            agent.best_model_state = copy.deepcopy(agent.model.state_dict())
        if agent.n_games % agent.generation_length == 0:
            print(f"Fin de génération {agent.generation}. Meilleur score: {agent.best_score_generation}")
            if agent.best_model_state is not None:
                agent.model.load_state_dict(copy.deepcopy(agent.best_model_state))
            # Optionnel : activer agent.mutate() ici si désiré
            agent.generation += 1
            agent.best_score_generation = -float('inf')
            agent.best_model_state = None
            agent.save_checkpoint("checkpoint.pth")
        game_instance.show_game_over()
        if game_instance.mode == 'replay':
            game_instance.run_replay()

if __name__ == '__main__':
    main()
