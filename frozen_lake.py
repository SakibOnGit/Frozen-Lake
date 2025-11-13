import gymnasium as gym
import numpy as np
import random
import pygame
import time

POP_SIZE = 100        
CHROMOSOME_LENGTH = 10  
GENERATIONS = 80        
MUTATION_RATE = 0.3     


custom_map = [
    "FSFFH",
    "FHFFH",
    "FFHHF",
    "HFFGF"
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



def evaluate(chromosome):
    state, _ = env.reset()
    total_reward = 0
    reached_goal = False
    
    for action in chromosome:

        prev_row = state // COLS
        prev_col = state % COLS
        
        state, reward, done, truncated, _ = env.step(action)
       
        row = state // COLS
        col = state % COLS
        
   
        if row == prev_row and col == prev_col:
            total_reward -= 2 
 
        dist = abs(GOAL_POS[0] - row) + abs(GOAL_POS[1] - col)
        total_reward += (1.0 / (dist + 1))

        if done:
            if reward == 1.0:
                total_reward += 100 
                reached_goal = True
            else:
                total_reward -= 5  
            break
            
    return total_reward, reached_goal

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
    if len(parent1) < 2: return parent1, parent2
    point = random.randint(1, len(parent1) - 2)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = random.choice(ACTION_SPACE)
    return chromosome

population = [np.random.choice(ACTION_SPACE, CHROMOSOME_LENGTH) for _ in range(POP_SIZE)]
best_chromosome = None
best_fitness = -9999

print(f"\nMap Size: {ROWS}x{COLS}")
print(f"Training with Wall Penalty Logic...")
print("-" * 50)

for generation in range(GENERATIONS):
    fitness = []
    reached_flags = []
    
    for ind in population:
        fit, reached = evaluate(ind)
        fitness.append(fit)
        reached_flags.append(reached)
    
    gen_best_idx = np.argmax(fitness)
    gen_best_fit = fitness[gen_best_idx]
    gen_best_chrom = population[gen_best_idx]
    
    if gen_best_fit > best_fitness:
        best_fitness = gen_best_fit
        best_chromosome = gen_best_chrom
        
    seq_str = " ".join([ACTION_NAMES[act] for act in gen_best_chrom])
    status = ""
    if True in reached_flags:
        status = " [GOAL HIT!]"
        
    print(f"Gen {generation+1:03}: Fit={gen_best_fit:.2f} | Seq: [{seq_str}]{status}")

    new_population = []
    
    sorted_indices = np.argsort(fitness)[::-1]
    new_population.append(population[sorted_indices[0]]) 
    new_population.append(population[sorted_indices[1]]) 
    
    while len(new_population) < POP_SIZE:
        parent1 = select(population, fitness)
        parent2 = select(population, fitness)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1))
        if len(new_population) < POP_SIZE:
            new_population.append(mutate(child2))
            
    population = np.array(new_population)

env.close()
print("-" * 50)

if best_chromosome is not None:
    final_path = " ".join([ACTION_NAMES[a] for a in best_chromosome])
    print(f"Simulating Best Path: {final_path}")
else:
    print("No solution found.")
    exit()

TILE = 80
WINDOW_WIDTH = COLS * TILE
WINDOW_HEIGHT = ROWS * TILE
FPS = 5 

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Frozen Lake Evolution Replay")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 24)
font_big = pygame.font.SysFont("arial", 40, bold=True)

COLOR_BG = (30, 30, 30)
COLOR_ICE = (220, 240, 255)
COLOR_HOLE = (20, 20, 40)
COLOR_GOAL_BG = (255, 235, 100)
COLOR_START_BG = (150, 255, 150)
COLOR_AGENT = (255, 60, 60)

agent_r, agent_c = START_POS
step_idx = 0
running = True
finished_anim = False
last_move_time = pygame.time.get_ticks()

def draw_board():
    screen.fill(COLOR_BG)
    for r in range(ROWS):
        for c in range(COLS):
            rect = (c*TILE+2, r*TILE+2, TILE-4, TILE-4)
            char = custom_map[r][c]
            
            if char == 'F':
                pygame.draw.rect(screen, COLOR_ICE, rect, border_radius=5)
            elif char == 'H':
                pygame.draw.rect(screen, COLOR_HOLE, rect, border_radius=5)
                pygame.draw.circle(screen, (50,50,60), (c*TILE+TILE//2, r*TILE+TILE//2), TILE//4)
            elif char == 'S':
                pygame.draw.rect(screen, COLOR_START_BG, rect, border_radius=5)
                text_s = font_big.render("S", True, (0, 100, 0))
                screen.blit(text_s, (c*TILE + TILE//2 - text_s.get_width()//2, r*TILE + TILE//2 - text_s.get_height()//2))
            elif char == 'G':
                pygame.draw.rect(screen, COLOR_GOAL_BG, rect, border_radius=5)
                text_g = font_big.render("G", True, (200, 100, 0))
                screen.blit(text_g, (c*TILE + TILE//2 - text_g.get_width()//2, r*TILE + TILE//2 - text_g.get_height()//2))
                
    cx = agent_c * TILE + TILE // 2
    cy = agent_r * TILE + TILE // 2
    pygame.draw.circle(screen, COLOR_AGENT, (cx, cy), TILE//3)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_board()
    info_surf = font.render(f"Step: {step_idx}/{len(best_chromosome)}", True, (0,0,0))
    screen.blit(info_surf, (10, 10))

    if finished_anim:
        end_surf = font.render("Simulation Complete", True, (200, 0, 0))
        screen.blit(end_surf, (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT - 40))
    
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