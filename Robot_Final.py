import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VECTOR HELPERS
# ============================================================

def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([0.0, 0.0])
    return v / n


def cross_2d(a, b):
    """2D cross product magnitude/sign: a_x*b_y - a_y*b_x."""
    return a[0] * b[1] - a[1] * b[0]


def rotate_left(v):
    return np.array([-v[1], v[0]])


def rotate_right(v):
    return np.array([v[1], -v[0]])


# ============================================================
# OBSTACLE CLASSES
#
# Important architecture idea:
# The WORLD and SENSOR are allowed to know obstacle geometry.
# The ROBOT should not directly call obs.g() for navigation.
# ============================================================

class Obstacle:
    def g(self, pos):
        """
        Signed-ish distance function.
        g(pos) > 0 outside obstacle
        g(pos) = 0 on obstacle boundary
        g(pos) < 0 inside obstacle
        """
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError


class CircleObstacle(Obstacle):
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def g(self, pos):
        return np.linalg.norm(pos - self.center) - self.radius

    def draw(self):
        circle = plt.Circle(self.center, self.radius, fill=False, linewidth=3)
        plt.gca().add_patch(circle)


class SegmentObstacle(Obstacle):
    def __init__(self, start, end, thickness=0.05):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.thickness = thickness

    def closest_point(self, pos):
        segment_vec = self.end - self.start
        pos_vec = pos - self.start
        denom = np.dot(segment_vec, segment_vec)

        if denom < 1e-9:
            return self.start

        t = np.dot(pos_vec, segment_vec) / denom
        t = np.clip(t, 0.0, 1.0)
        return self.start + t * segment_vec

    def g(self, pos):
        closest = self.closest_point(pos)
        return np.linalg.norm(pos - closest) - self.thickness

    def draw(self):
        plt.plot(
            [self.start[0], self.end[0]],
            [self.start[1], self.end[1]],
            linewidth=4
        )


# ============================================================
# WORLD
# ============================================================

class World:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def inside_bounds(self, pos):
        x, y = pos
        return 0 <= x <= self.width and 0 <= y <= self.height

    def collides_with_obstacle(self, pos):
        """
        This is used by the simulated sensor/world, not by the robot brain.
        """
        if not self.inside_bounds(pos):
            return True

        for obs in self.obstacles:
            if obs.g(pos) <= 0:
                return True

        return False

    def draw(self):
        for obs in self.obstacles:
            obs.draw()


# ============================================================
# RANGE SENSOR
# ============================================================

class RayScan:
    def __init__(self, directions, distances, hit_points, max_range):
        self.directions = directions
        self.distances = distances
        self.hit_points = hit_points
        self.max_range = max_range

    def hit_mask(self):
        return self.distances < self.max_range

    def closest_hit(self):
        hits = self.hit_mask()
        if not np.any(hits):
            return None, None, None

        hit_indices = np.where(hits)[0]
        best_index = hit_indices[np.argmin(self.distances[hit_indices])]
        return self.directions[best_index], self.distances[best_index], self.hit_points[best_index]

    def obstacle_angularly_closest_to(self, direction):
        """
        Return the sensed obstacle ray most aligned with desired direction.
        This is used in NORMAL mode to see the obstacle portion most relevant
        to the signal-gradient direction.
        """
        direction = normalize(direction)
        hits = self.hit_mask()

        if np.linalg.norm(direction) < 1e-9 or not np.any(hits):
            return None, None, None

        best_score = -float("inf")
        best_index = None

        for i, hit in enumerate(hits):
            if not hit:
                continue

            alignment = np.dot(self.directions[i], direction)

            # Ignore obstacles behind the robot relative to desired direction.
            if alignment <= 0:
                continue

            # Strong alignment matters most; closer obstacles are slightly more urgent.
            score = alignment / max(self.distances[i], 1e-6)

            if score > best_score:
                best_score = score
                best_index = i

        if best_index is None:
            return None, None, None

        return self.directions[best_index], self.distances[best_index], self.hit_points[best_index]

    def corridor_risk_cluster(self, desired_dir, half_width, corridor_length):
        """
        Risk-first method for AVOID/NORMAL decisions.

        It checks every hit ray and keeps only rays that lie inside the robot's
        forward swept corridor along desired_dir.

        Returns:
            risk_exists, avg_risk_dir, closest_risk_dist
        """
        desired_dir = normalize(desired_dir)
        hits = self.hit_mask()

        if np.linalg.norm(desired_dir) < 1e-9 or not np.any(hits):
            return False, None, None

        risk_dirs = []
        risk_dists = []

        for ray_dir, ray_dist, hit in zip(self.directions, self.distances, hits):
            if not hit:
                continue

            relative = ray_dir * ray_dist
            forward = np.dot(relative, desired_dir)
            lateral = abs(cross_2d(relative, desired_dir))

            if 0.0 < forward < corridor_length and lateral < half_width:
                risk_dirs.append(ray_dir)
                risk_dists.append(ray_dist)

        if not risk_dirs:
            return False, None, None

        risk_dirs = np.array(risk_dirs)
        risk_dists = np.array(risk_dists)

        weights = 1.0 / np.maximum(risk_dists, 1e-6)
        avg_risk_dir = normalize(np.sum(risk_dirs * weights[:, None], axis=0))
        closest_risk_dist = np.min(risk_dists)

        return True, avg_risk_dir, closest_risk_dist

    def closest_cluster_direction(self, tolerance=0.15):
        """
        Single shortest ray can jitter. This averages rays whose distances are
        close to the closest distance.
        """
        hits = self.hit_mask()
        if not np.any(hits):
            return None, None

        closest_dist = np.min(self.distances[hits])
        cluster = hits & (self.distances <= closest_dist + tolerance)

        dirs = self.directions[cluster]
        dists = self.distances[cluster]

        # Weight closer rays more.
        weights = 1.0 / np.maximum(dists, 1e-6)
        avg_dir = normalize(np.sum(dirs * weights[:, None], axis=0))
        avg_dist = np.average(dists, weights=weights)

        return avg_dir, avg_dist


class RangeSensor:
    def __init__(self, max_range=1.2, num_rays=180, ray_step=0.02):
        self.max_range = max_range
        self.num_rays = num_rays
        self.ray_step = ray_step

        self.angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
        self.local_directions = np.column_stack((np.cos(self.angles), np.sin(self.angles)))

    def scan(self, robot_pos, world):
        """
        Simulated 360-degree ray casting.

        The sensor uses world geometry internally.
        The robot only receives distances and directions.
        """
        directions = self.local_directions.copy()
        distances = np.full(self.num_rays, self.max_range, dtype=float)
        hit_points = np.full((self.num_rays, 2), np.nan, dtype=float)

        ray_distances = np.arange(self.ray_step, self.max_range + self.ray_step, self.ray_step)

        for i, direction in enumerate(directions):
            for d in ray_distances:
                p = robot_pos + d * direction
                if world.collides_with_obstacle(p):
                    distances[i] = d
                    hit_points[i] = p
                    break

        return RayScan(directions, distances, hit_points, self.max_range)


# ============================================================
# SIGNAL FIELD
# ============================================================

class SignalField:
    def __init__(self, source_pos, noise_level=0.002, h=0.1, samples=5):
        self.source_pos = np.array(source_pos, dtype=float)
        self.noise_level = noise_level
        self.h = h
        self.samples = samples

    def true_signal(self, x, y):
        distance = np.sqrt((x - self.source_pos[0])**2 + (y - self.source_pos[1])**2)
        return 1.0 / (1.0 + distance)

    def measure_signal(self, x, y):
        return self.true_signal(x, y) + np.random.normal(0.0, self.noise_level)

    def averaged_signal(self, x, y):
        total = 0.0
        for _ in range(self.samples):
            total += self.measure_signal(x, y)
        return total / self.samples

    def gradient_direction(self, pos):
        x, y = pos
        h = self.h

        dS_dx = (self.averaged_signal(x + h, y) - self.averaged_signal(x - h, y)) / (2.0 * h)
        dS_dy = (self.averaged_signal(x, y + h) - self.averaged_signal(x, y - h)) / (2.0 * h)

        return normalize(np.array([dS_dx, dS_dy]))

    def draw_background(self, width, height, resolution=150):
        x = np.linspace(0, width, resolution)
        y = np.linspace(0, height, resolution)
        X, Y = np.meshgrid(x, y)
        signal = self.true_signal(X, Y)
        plt.imshow(signal, extent=[0, width, 0, height], origin="lower")
        plt.colorbar(label="signal strength")


# ============================================================
# ROBOT
# ============================================================

class Robot:
    def __init__(self, start_pos, body_width=0.35, body_length=0.45):
        self.pos = np.array(start_pos, dtype=float)
        self.body_width = body_width
        self.body_length = body_length

        self.mode = "NORMAL"
        self.bug_side = "RIGHT"  # fixed side for committed wall following

        self.path = [self.pos.copy()]
        self.mode_history = []
        self.signal_history = []

        self.best_pos = self.pos.copy()
        self.best_signal = -float("inf")

    def required_lateral_clearance(self, safety_margin):
        return self.body_width / 2.0 + safety_margin

    def signal_corridor_blocked(self, scan, signal_dir, safety_margin, corridor_length):
        """
        Checks whether any sensed obstacle point lies inside the robot's
        forward swept corridor along signal_dir.
        """
        signal_dir = normalize(signal_dir)
        if np.linalg.norm(signal_dir) < 1e-9:
            return False

        half_width = self.required_lateral_clearance(safety_margin)

        for direction, distance in zip(scan.directions, scan.distances):
            if distance >= scan.max_range:
                continue

            relative = direction * distance
            forward = np.dot(relative, signal_dir)
            lateral = abs(cross_2d(relative, signal_dir))

            if 0.0 < forward < corridor_length and lateral < half_width:
                return True

        return False

    def lateral_clearance_risk(self, scan, signal_dir, safety_margin, avoid_activation_distance):
        """
        Risk-first AVOID trigger.

        Instead of first choosing the ray most aligned with the signal direction,
        this checks all rays and finds the cluster of rays that actually lie
        inside the robot's forward swept corridor.
        """
        half_width = self.required_lateral_clearance(safety_margin)

        risk, risk_dir, closest_risk_dist = scan.corridor_risk_cluster(
            desired_dir=signal_dir,
            half_width=half_width,
            corridor_length=avoid_activation_distance
        )

        return risk, risk_dir, closest_risk_dist

    def avoid_direction(self, obstacle_dir, signal_dir):
        """
        Local dodge: choose the tangent direction that still agrees most with
        the signal gradient.
        """
        if obstacle_dir is None:
            return signal_dir

        # obstacle_dir points from robot to obstacle.
        # normal_away points from obstacle toward robot.
        normal_away = -normalize(obstacle_dir)

        tangent_1 = rotate_left(normal_away)
        tangent_2 = -tangent_1

        # Choose the tangent that preserves uphill signal progress.
        if np.dot(tangent_1, signal_dir) >= np.dot(tangent_2, signal_dir):
            tangent = tangent_1
        else:
            tangent = tangent_2

        return normalize(0.55 * signal_dir + 0.85 * tangent)

    def bug_direction(self, scan, following_distance, lambda_gain, cluster_tolerance):
        """
        Committed wall-following mode using only ray-scan data.
        """
        closest_dir, closest_dist = scan.closest_cluster_direction(tolerance=cluster_tolerance)

        if closest_dir is None:
            # Lost the wall. Caller should usually switch back to NORMAL.
            return None

        # closest_dir points from robot to obstacle.
        normal_away = -normalize(closest_dir)

        tangent_right = rotate_right(normal_away)
        tangent_left = -tangent_right

        if self.bug_side == "RIGHT":
            tangent = tangent_right
        else:
            tangent = tangent_left

        # If too close, error is positive and pushes away.
        # If too far, error is negative and pulls toward obstacle.
        error = following_distance - closest_dist
        direction = tangent + lambda_gain * error * normal_away

        return normalize(direction)

    def update(
        self,
        scan,
        signal_dir,
        signal_value,
        step_size,
        world,
        safety_margin=0.08,
        following_distance=0.35,
        lambda_gain=2.0,
        avoid_activation_distance=0.9,
        corridor_length=0.8,
        cluster_tolerance=0.15
    ):
        # Track best signal seen so far. Useful later for stuck recovery.
        self.signal_history.append(signal_value)
        if signal_value > self.best_signal:
            self.best_signal = signal_value
            self.best_pos = self.pos.copy()

        signal_dir = normalize(signal_dir)

        blocked = self.signal_corridor_blocked(
            scan=scan,
            signal_dir=signal_dir,
            safety_margin=safety_margin,
            corridor_length=corridor_length
        )

        clearance_risk, relevant_obstacle_dir, _ = self.lateral_clearance_risk(
            scan=scan,
            signal_dir=signal_dir,
            safety_margin=safety_margin,
            avoid_activation_distance=avoid_activation_distance
        )

        # -------------------------
        # MODE SWITCHING
        # -------------------------
        if self.mode == "NORMAL":
            if blocked:
                self.mode = "BUG"
                print("Switching to BUG")
            elif clearance_risk:
                self.mode = "AVOID"
                print("Switching to AVOID")

        elif self.mode == "AVOID":
            if blocked:
                self.mode = "BUG"
                print("AVOID failed. Switching to BUG")
            elif not clearance_risk:
                self.mode = "NORMAL"
                print("Returning to NORMAL")

        elif self.mode == "BUG":
            # Leave bug mode only when signal direction is genuinely clear.
            if not blocked:
                self.mode = "NORMAL"
                print("Signal path clear. Returning to NORMAL")

        # -------------------------
        # DIRECTION DECISION
        # -------------------------
        if self.mode == "NORMAL":
            direction = signal_dir

        elif self.mode == "AVOID":
            direction = self.avoid_direction(relevant_obstacle_dir, signal_dir)

        elif self.mode == "BUG":
            direction = self.bug_direction(
                scan=scan,
                following_distance=following_distance,
                lambda_gain=lambda_gain,
                cluster_tolerance=cluster_tolerance
            )
            if direction is None:
                self.mode = "NORMAL"
                direction = signal_dir

        else:
            direction = signal_dir

        # -------------------------
        # MOVE
        # -------------------------
        direction = normalize(direction)
        new_pos = self.pos + step_size * direction

        # Simple bounds handling.
        new_pos[0] = np.clip(new_pos[0], 0.0, world.width)
        new_pos[1] = np.clip(new_pos[1], 0.0, world.height)

        self.pos = new_pos
        self.path.append(self.pos.copy())
        self.mode_history.append(self.mode)


# ============================================================
# SIMULATION SETUP
# ============================================================

np.random.seed(2)

width = 10
height = 10

source_pos = np.array([5.4, 6.0])

obstacles = [
    SegmentObstacle([5.0, 2.0], [5.0, 8.0], thickness=0.05),
    SegmentObstacle([5.0, 8.0], [8.0, 8.0], thickness=0.05),
    CircleObstacle([3.8, 5.2], radius=1.0)
]

world = World(width=width, height=height, obstacles=obstacles)
field = SignalField(source_pos=source_pos, noise_level=0.002, h=0.1, samples=5)
# Faster scan settings for debugging.
# Original was num_rays=180, ray_step=0.02, which is accurate but slow.
sensor = RangeSensor(max_range=1.3, num_rays=72, ray_step=0.05)
robot = Robot(start_pos=[3.7, 1.0], body_width=0.35, body_length=0.45)

step_size = 0.12
num_steps = 250

safety_margin = 0.08
following_distance = 0.35
lambda_gain = 2.0
avoid_activation_distance = 0.9
corridor_length = 0.85
cluster_tolerance = 0.18


# ============================================================
# MAIN LOOP
# ============================================================

for step in range(num_steps):
    scan = sensor.scan(robot.pos, world)
    signal_dir = field.gradient_direction(robot.pos)
    signal_value = field.averaged_signal(robot.pos[0], robot.pos[1])

    robot.update(
        scan=scan,
        signal_dir=signal_dir,
        signal_value=signal_value,
        step_size=step_size,
        world=world,
        safety_margin=safety_margin,
        following_distance=following_distance,
        lambda_gain=lambda_gain,
        avoid_activation_distance=avoid_activation_distance,
        corridor_length=corridor_length,
        cluster_tolerance=cluster_tolerance
    )


# ============================================================
# VISUALIZATION
# ============================================================

path = np.array(robot.path)

plt.figure(figsize=(8, 7))
field.draw_background(width, height, resolution=150)

plt.scatter(source_pos[0], source_pos[1], marker="x", s=120, label="True Signal Source")
plt.scatter(robot.best_pos[0], robot.best_pos[1], marker="*", s=160, label="Best Signal Seen")

world.draw()

plt.plot(
    path[:, 0],
    path[:, 1],
    marker="o",
    markersize=2.5,
    linewidth=1.5,
    label="Robot Path"
)

plt.title("Sensor-Based Signal Gradient Navigation With AVOID and BUG Modes")
plt.xlabel("x position")
plt.ylabel("y position")
plt.legend()
plt.axis("equal")
plt.xlim(0, width)
plt.ylim(0, height)
plt.show()


# Optional mode plot
plt.figure(figsize=(8, 2.5))
mode_to_number = {"NORMAL": 0, "AVOID": 1, "BUG": 2}
mode_numbers = [mode_to_number[m] for m in robot.mode_history]
plt.plot(mode_numbers)
plt.yticks([0, 1, 2], ["NORMAL", "AVOID", "BUG"])
plt.xlabel("step")
plt.ylabel("mode")
plt.title("Robot Mode History")
plt.grid(True)
plt.show()

print("Best signal:", robot.best_signal)
print("Best position:", robot.best_pos)
print("Final position:", robot.pos)
