import pygame
import numpy as np
from numba import cuda
from sklearn.cluster import KMeans

# Parameters
NUM_VOTERS = 200
NUM_CANDIDATES = 5
FIELD_SIZE = 600
MOVE_SPEED = 8.0
NOISE_INTENSITY = 2.0
FPS = 24

# Colors
WHITE = (255, 255, 255)

# Hyperparameters
k = 10
n = 2
CLUSTER_SCALING = 1.5
CAMPAIGN_MOVE_SPEED = 2.0
REPULSION_STRENGTH = 50000  
REPULSION_RADIUS = 100     



# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((FIELD_SIZE, FIELD_SIZE))
pygame.display.set_caption("Election Potential Field with KMeans Clustering")
clock = pygame.time.Clock()

# Randomly generate voters and candidates
voters = np.random.rand(NUM_VOTERS, 2) * FIELD_SIZE
candidates = np.random.rand(NUM_CANDIDATES, 2) * FIELD_SIZE
candidate_strengths = np.random.randint(5, 15, NUM_CANDIDATES)
candidate_colors = [
    (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
    for _ in range(NUM_CANDIDATES)
]

candidate_policies = np.random.rand(NUM_CANDIDATES, 3)
voter_preferences = np.random.rand(NUM_VOTERS, 3)

# Voter Demographics
demographics = np.random.choice(['youth', 'adult', 'senior'], NUM_VOTERS)
income_levels = np.random.choice(['low', 'middle', 'high'], NUM_VOTERS)
education_levels = np.random.choice(['high_school', 'bachelor', 'advanced'], NUM_VOTERS)
ideologies = np.random.choice(['progressive', 'moderate', 'conservative'], NUM_VOTERS)

# Candidate Traits (affinities for demographics)
candidate_traits = {
    0: {'youth': 1.5, 'low_income': 1.2, 'progressive': 1.3},  # Youth-focused, progressive
    1: {'adult': 1.4, 'middle_income': 1.3, 'moderate': 1.2},   # Moderate with middle-income focus
    2: {'senior': 1.6, 'high_income': 1.5, 'conservative': 1.4}, # Senior-focused, conservative
    3: {'youth': 1.2, 'bachelor': 1.5, 'progressive': 1.4},     # Youth with education focus
    4: {'adult': 1.3, 'middle_income': 1.1, 'advanced': 1.3}    # Adult with advanced education focus
}

# Calculate Attraction Weights
def calculate_attraction_weights():
    weights = np.ones(NUM_VOTERS)
    for i, voter in enumerate(voters):
        voter_traits = {
            'youth': demographics[i] == 'youth',
            'adult': demographics[i] == 'adult',
            'senior': demographics[i] == 'senior',
            'low_income': income_levels[i] == 'low',
            'middle_income': income_levels[i] == 'middle',
            'high_income': income_levels[i] == 'high',
            'high_school': education_levels[i] == 'high_school',
            'bachelor': education_levels[i] == 'bachelor',
            'advanced': education_levels[i] == 'advanced',
            'progressive': ideologies[i] == 'progressive',
            'moderate': ideologies[i] == 'moderate',
            'conservative': ideologies[i] == 'conservative',
        }

        # Aggregate attraction weights
        for trait, attraction in candidate_traits[np.random.randint(NUM_CANDIDATES)].items():
            if voter_traits.get(trait, False):
                weights[i] *= attraction

    return weights

# KMeans Clustering
def calculate_cluster_effect():
    kmeans = KMeans(n_clusters=NUM_CANDIDATES, n_init=10).fit(voters)
    labels = kmeans.labels_

    # Calculate cluster density (voter count per cluster)
    cluster_counts = np.zeros(NUM_CANDIDATES)
    for i in range(NUM_CANDIDATES):
        cluster_counts[i] = np.sum(labels == i)

    return cluster_counts / np.max(cluster_counts)

# CUDA Kernel for Potential Field Calculation
@cuda.jit
def compute_candidate_field_cuda(field, candidates, strengths, cluster_effects):
    x, y = cuda.grid(2)
    if x < FIELD_SIZE and y < FIELD_SIZE:
        potential = 0.0
        for i in range(NUM_CANDIDATES):
            cx, cy = candidates[i]
            strength = strengths[i] * (1 + CLUSTER_SCALING * cluster_effects[i])
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            potential += 5000 * strength / (distance / n + k)
        field[x, y] = potential

# GPU-Accelerated Potential Field Calculation
def compute_candidate_fields(cluster_effects):
    field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
    d_field = cuda.to_device(field)
    d_candidates = cuda.to_device(candidates)
    d_strengths = cuda.to_device(candidate_strengths)
    d_cluster_effects = cuda.to_device(cluster_effects)

    threadsperblock = (16, 16)
    blockspergrid_x = (FIELD_SIZE + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (FIELD_SIZE + threadsperblock[1] - 1) // threadsperblock[1]

    compute_candidate_field_cuda[(blockspergrid_x, blockspergrid_y), threadsperblock](
        d_field, d_candidates, d_strengths, d_cluster_effects
    )

    return d_field.copy_to_host()

# Calculate gradient for voter movement
def compute_gradient(field, position):
    x, y = int(position[0]), int(position[1])
    grad_x = field[min(x + 1, FIELD_SIZE - 1), y] - field[max(x - 1, 0), y]
    grad_y = field[x, min(y + 1, FIELD_SIZE - 1)] - field[x, max(y - 1, 0)]
    gradient = np.array([grad_x, grad_y])
    norm = np.linalg.norm(gradient)
    if norm != 0:
        gradient = (gradient / norm) * MOVE_SPEED
    noise = np.random.normal(0, NOISE_INTENSITY, 2)
    return gradient + noise

# Candidate Repulsion Logic
def apply_repulsion():
    for i in range(NUM_CANDIDATES):
        repulsion_vector = np.zeros(2)
        for j in range(NUM_CANDIDATES):
            if i != j:
                direction = candidates[i] - candidates[j]
                distance = np.linalg.norm(direction)
                if distance < REPULSION_RADIUS and distance > 1e-3:
                    repulsion_strength = REPULSION_STRENGTH / (distance ** 2)
                    repulsion_vector += (direction / distance) * repulsion_strength

        # Apply repulsion force
        candidates[i] += repulsion_vector * 0.1  # Scale down for stability


# Candidate Targeted Campaigning Strategy
def targeted_campaigning():
    for i in range(NUM_CANDIDATES):
        alignment_scores = np.dot(voter_preferences, candidate_policies[i])
        high_alignment_indices = np.where(alignment_scores > 0.7)[0]

        if len(high_alignment_indices) > 0:
            aligned_voters = voters[high_alignment_indices]
            centroid = np.mean(aligned_voters, axis=0)
            direction = centroid - candidates[i]
            candidates[i] += (direction / np.linalg.norm(direction)) * CAMPAIGN_MOVE_SPEED


# Draw combined potential field
def draw_potential_field(field):
    max_val = np.max(field)
    pixel_array = np.zeros((FIELD_SIZE, FIELD_SIZE, 3), dtype=np.uint8)
    pixel_array[:, :, 0] = (field / max_val * 255).astype(np.uint8)
    pygame.surfarray.blit_array(screen, pixel_array)

def draw_elements():
    for voter in voters:
        pygame.draw.circle(screen, WHITE, voter.astype(int), 4)
    for candidate, color in zip(candidates, candidate_colors):
        pygame.draw.circle(screen, color, candidate.astype(int), 8)

# Compute cluster effects and candidate fields before the loop
running = True
while running:
    attraction_weights = calculate_attraction_weights()
    total_field = compute_candidate_fields(attraction_weights)

    screen.fill(WHITE)
    draw_potential_field(total_field)
    draw_elements()

    for i, voter in enumerate(voters):
        gradient = compute_gradient(total_field, voter)
        voters[i] += gradient
        voters[i][0] = np.clip(voters[i][0], 0, FIELD_SIZE - 1)
        voters[i][1] = np.clip(voters[i][1], 0, FIELD_SIZE - 1)

    targeted_campaigning()
    apply_repulsion()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
