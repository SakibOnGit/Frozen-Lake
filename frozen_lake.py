import gymnasium as gym
import numpy as np
import random
import pygame
import time

# GA Parameters
POP_SIZE = 150
INITIAL_LENGTH = 1
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100

# Environment Setup
custom_map = [
    "FSFFF",
    "FFHFH",
    "HFFFF",
    "FFHFF",
    "FHGFF"
]

ROWS = len(custom_map)
COLS = len(custom_map[0])
START_POS = None
GOAL_POS = None

for r in range(ROWS):
    for c in range(COLS):
        if custom_map[r][c] == 'S':
            START_POS = (r, c)
        elif custom_map[r][c] == 'G':
            GOAL_POS = (r, c)

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
ACTION_SPACE = [0, 1, 2, 3]
ACTION_NAMES = {0: "L", 1: "D", 2: "R", 3: "U"}

# GA Functions
def calculate_fitness(chromosome):
    state, _ = env.reset()
    fitness_score = 0
    reached_goal = False
    
    for action in chromosome:
        prev_row = state // COLS
        prev_col = state % COLS
        
        state, reward, done, truncated, _ = env.step(action)
        
        curr_row = state // COLS
        curr_col = state % COLS
        
        if curr_row == prev_row and curr_col == prev_col:
            fitness_score -= 5
        
        if done:
            if reward == 1.0:
                fitness_score += 100
                reached_goal = True
            else:
                fitness_score -= 10
            break
    
    if not reached_goal and fitness_score > -10:
        row = state // COLS
        col = state % COLS
        dist = abs(GOAL_POS[0] - row) + abs(GOAL_POS[1] - col)
        fitness_score += (1.0 / (dist + 1)) * 10

    return fitness_score, reached_goal

def select(population, fitness):
    fitness = np.array(fitness)
    min_fit = np.min(fitness)
    if min_fit < 0:
        fitness = fitness - min_fit + 0.01
    total = np.sum(fitness)
    if total == 0: return random.choice(population)
    probs = fitness / total
    if np.any(np.isnan(probs)): probs = np.ones(len(population))/len(population)
    idx = np.random.choice(len(population), p=probs)
    return population[idx]

def crossover(parent1, parent2):
    min_len = min(len(parent1), len(parent2))
    if min_len < 2: return parent1, parent2
    point = random.randint(1, min_len - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(chromosome)-1)
        chromosome[idx] = random.choice(ACTION_SPACE)
    return chromosome

# Main GA Loop
population = [[random.choice(ACTION_SPACE) for _ in range(INITIAL_LENGTH)] for _ in range(POP_SIZE)]
best_chromosome = None
best_fitness = -9999
first_goal_generation = None

print(f"\nMap Size: {ROWS}x{COLS}")
print(f"Training started...")
print("-" * 50)

for generation in range(MAX_GENERATIONS):
    fitness = []
    reached_flags = []
    
    for ind in population:
        fit, reached = calculate_fitness(ind)
        fitness.append(fit)
        reached_flags.append(reached)
    
    gen_best_idx = np.argmax(fitness)
    gen_best_fit = fitness[gen_best_idx]
    gen_best_chrom = population[gen_best_idx]
    
    if gen_best_fit > best_fitness:
        best_fitness = gen_best_fit
        best_chromosome = gen_best_chrom
    
    seq_str = " ".join([ACTION_NAMES[act] for act in gen_best_chrom])
    if True in reached_flags:
        if first_goal_generation is None:
            first_goal_generation = generation + 1
        print(f"Gen {generation+1:03}: Fit={gen_best_fit:.2f} | Seq: [{seq_str}] [GOAL HIT!]")
    else:
        print(f"Gen {generation+1:03}: Fit={gen_best_fit:.2f} | Seq: [{seq_str}]")

    new_population = [best_chromosome]
    while len(new_population) < POP_SIZE:
        parent1 = select(population, fitness)
        parent2 = select(population, fitness)
        c1, c2 = crossover(parent1, parent2)
        new_population.append(mutate(c1))
        if len(new_population) < POP_SIZE:
            new_population.append(mutate(c2))
    
    grown_population = []
    for crom in new_population:
        new_gene = random.choice(ACTION_SPACE)
        extended_crom = crom + [new_gene]
        grown_population.append(extended_crom)
    population = grown_population

env.close()
print("-" * 50)
if first_goal_generation:
    print(f"First goal reached at generation: {first_goal_generation}")
print(f"Training completed after {MAX_GENERATIONS} generations")

# Pygame Visualization
if best_chromosome is not None:
    final_path = " ".join([ACTION_NAMES[a] for a in best_chromosome])
    print(f"Simulating Best Path: {final_path}")
else:
    print("No solution found.")
    exit()

# UI Constants & Colors
TILE = 80
HEADER_HEIGHT = 120
WINDOW_WIDTH = max(COLS * TILE, 600)
WINDOW_HEIGHT = (ROWS * TILE) + HEADER_HEIGHT + 50
FPS = 5

COLOR_BG_TOP = (7, 21, 47)
COLOR_BG_BOTTOM = (10, 30, 63)
COLOR_HEADER_BG = (16, 27, 51)
COLOR_ACCENT = (58, 190, 255)
COLOR_CARD_BG = (23, 42, 58)

COLOR_TEXT_MAIN = (241, 245, 249)
COLOR_TEXT_SUB = (148, 163, 184)
COLOR_TEXT_GOLD = (250, 204, 21)

COLOR_ICE = (224, 244, 255)
COLOR_HOLE_OUTER = (17, 49, 77)
COLOR_HOLE_INNER = (5, 10, 20)

COLOR_PENGUIN_BODY = (30, 30, 40)
COLOR_PENGUIN_BELLY = (240, 240, 255)
COLOR_BEAK = (255, 165, 0)
COLOR_HOUSE_WALL = (160, 82, 45)
COLOR_HOUSE_ROOF = (178, 34, 34)
COLOR_DOOR = (101, 67, 33)
COLOR_IGLOO = (200, 230, 255)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Genetic Algorithm Simulation - Frozen Lake")
clock = pygame.time.Clock()

font_title = pygame.font.SysFont("verdana", 24, bold=True)
font_subtitle = pygame.font.SysFont("verdana", 12)
font_label = pygame.font.SysFont("verdana", 11)
font_value = pygame.font.SysFont("verdana", 16, bold=True)

agent_r, agent_c = START_POS
step_idx = 0
running = True
finished_anim = False
last_move_time = pygame.time.get_ticks()

def draw_gradient_background():
    for y in range(WINDOW_HEIGHT):
        ratio = y / WINDOW_HEIGHT
        r = COLOR_BG_TOP[0] + ratio * (COLOR_BG_BOTTOM[0] - COLOR_BG_TOP[0])
        g = COLOR_BG_TOP[1] + ratio * (COLOR_BG_BOTTOM[1] - COLOR_BG_TOP[1])
        b = COLOR_BG_TOP[2] + ratio * (COLOR_BG_BOTTOM[2] - COLOR_BG_TOP[2])
        pygame.draw.line(screen, (int(r), int(g), int(b)), (0, y), (WINDOW_WIDTH, y))

def draw_penguin(x, y, size):
    cx, cy = x + size // 2, y + size // 2
    pygame.draw.ellipse(screen, COLOR_BEAK, (cx - 15, cy + 15, 12, 8))
    pygame.draw.ellipse(screen, COLOR_BEAK, (cx + 3, cy + 15, 12, 8))
    pygame.draw.ellipse(screen, COLOR_PENGUIN_BODY, (cx - 18, cy - 20, 36, 45))
    pygame.draw.ellipse(screen, COLOR_PENGUIN_BELLY, (cx - 12, cy - 10, 24, 30))
    pygame.draw.circle(screen, (255, 255, 255), (cx - 7, cy - 12), 4)
    pygame.draw.circle(screen, (255, 255, 255), (cx + 7, cy - 12), 4)
    pygame.draw.circle(screen, (0, 0, 0), (cx - 7, cy - 12), 2)
    pygame.draw.circle(screen, (0, 0, 0), (cx + 7, cy - 12), 2)
    pygame.draw.polygon(screen, COLOR_BEAK, [(cx - 4, cy - 5), (cx + 4, cy - 5), (cx, cy)])

def draw_house(x, y, size):
    margin = 15
    wall_rect = (x + margin, y + size//2 - 5, size - 2*margin, size//2)
    pygame.draw.rect(screen, COLOR_HOUSE_WALL, wall_rect)
    door_rect = (x + size//2 - 8, y + size - 25, 16, 20)
    pygame.draw.rect(screen, COLOR_DOOR, door_rect)
    roof_points = [(x + 5, y + size//2 - 5), (x + size - 5, y + size//2 - 5), (x + size//2, y + 10)]
    pygame.draw.polygon(screen, COLOR_HOUSE_ROOF, roof_points)

def draw_igloo(x, y, size):
    margin = 10
    rect = (x + margin, y + 20, size - 2*margin, size - 30)
    pygame.draw.arc(screen, COLOR_IGLOO, rect, 0, 3.14, 50)
    pygame.draw.ellipse(screen, COLOR_IGLOO, rect)
    door_rect = (x + size//2 - 10, y + size - 30, 20, 20)
    pygame.draw.ellipse(screen, (50, 80, 120), door_rect)
    pygame.draw.line(screen, (180, 210, 230), (x+20, y+40), (x+size-20, y+40), 1)
    pygame.draw.line(screen, (180, 210, 230), (x+15, y+55), (x+size-15, y+55), 1)

def draw_hole(x, y, size):
    cx, cy = x + size // 2, y + size // 2
    pygame.draw.ellipse(screen, COLOR_HOLE_OUTER, (cx - size//3 - 5, cy - size//3 - 5, size*2//3 + 10, size*2//3 + 10))
    pygame.draw.circle(screen, COLOR_HOLE_INNER, (cx, cy), size//3)
    pygame.draw.circle(screen, (200, 230, 255), (cx - size//3 - 3, cy - size//3 + 5), 4)
    pygame.draw.circle(screen, (200, 230, 255), (cx + size//3 + 3, cy + size//3 - 5), 4)

def draw_ui():
    pygame.draw.rect(screen, COLOR_HEADER_BG, (0, 0, WINDOW_WIDTH, HEADER_HEIGHT))
    pygame.draw.line(screen, COLOR_ACCENT, (0, HEADER_HEIGHT-2), (WINDOW_WIDTH, HEADER_HEIGHT-2), 2)

    margin = 20
    title_surf = font_title.render("FROZEN LAKE", True, COLOR_TEXT_MAIN)
    sub_surf = font_subtitle.render("Genetic Algorithm Simulation", True, COLOR_ACCENT)
    screen.blit(title_surf, (margin, 20))
    screen.blit(sub_surf, (margin, 50))

    prof_w, prof_h = 160, 70
    prof_x, prof_y = WINDOW_WIDTH - prof_w - 20, 25
    pygame.draw.rect(screen, COLOR_CARD_BG, (prof_x, prof_y, prof_w, prof_h), border_radius=10)
    pygame.draw.rect(screen, COLOR_ACCENT, (prof_x, prof_y, prof_w, prof_h), 1, border_radius=10)
    name_surf = font_value.render("Naimur Sakib", True, COLOR_TEXT_GOLD)
    id_lbl = font_label.render("ID: 22235103630", True, COLOR_TEXT_SUB)
    screen.blit(name_surf, (prof_x + 15, prof_y + 15))
    screen.blit(id_lbl, (prof_x + 15, prof_y + 40))

def draw_board():
    grid_width = COLS * TILE
    offset_x = (WINDOW_WIDTH - grid_width) // 2
    grid_padding_top = 20

    for r in range(ROWS):
        for c in range(COLS):
            x = offset_x + c * TILE
            y = HEADER_HEIGHT + r * TILE + grid_padding_top
            rect = (x + 4, y + 4, TILE - 8, TILE - 8)
            char = custom_map[r][c]
            
            if char in ['F', 'S', 'G']:
                pygame.draw.rect(screen, COLOR_ICE, rect, border_radius=8)
            if char == 'S':
                draw_igloo(x, y, TILE)
            elif char == 'G':
                draw_house(x, y, TILE)
            elif char == 'H':
                draw_hole(x, y, TILE)

    agent_px = offset_x + agent_c * TILE
    agent_py = HEADER_HEIGHT + agent_r * TILE + grid_padding_top
    draw_penguin(agent_px, agent_py, TILE)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_gradient_background()
    draw_ui()
    draw_board()
    pygame.display.flip()

    current_time = pygame.time.get_ticks()
    if not finished_anim and current_time - last_move_time > 500:
        if step_idx < len(best_chromosome):
            action = best_chromosome[step_idx]
            new_r, new_c = agent_r, agent_c
            if action == 0: new_c -= 1
            elif action == 1: new_r += 1
            elif action == 2: new_c += 1
            elif action == 3: new_r -= 1
            if 0 <= new_r < ROWS and 0 <= new_c < COLS:
                agent_r, agent_c = new_r, new_c
            step_idx += 1
            last_move_time = current_time
            tile = custom_map[agent_r][agent_c]
            if tile == 'H':
                print("Agent fell in a hole during replay!")
                finished_anim = True
            elif tile == 'G':
                print("Agent reached the goal!")
                finished_anim = True
        else:
            finished_anim = True
    clock.tick(FPS)

pygame.quit()
