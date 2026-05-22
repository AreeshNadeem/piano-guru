"""
dynamic_difficulty.py  —  Piano Guru AI Extension
==================================================
Dynamic Difficulty Matching with Q-Learning

Q-Learning integration:
  - State  : (difficulty_bucket, performance_bucket)
              difficulty split into 5 buckets: [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
              performance split into 3 buckets: poor (<0.4), average (0.4-0.7), good (>0.7)
              → 15 unique states total
  - Action : index into the ranked song library (which song to recommend)
  - Reward : performance score after playing the recommended song
              bonus reward if difficulty was appropriately challenging
              penalty if song was too easy or too hard for user's level
  - Policy : epsilon-greedy (explores random songs early, exploits learned
              best choices as confidence grows)

Backwards compatible:
  - All original dataclasses, helpers, and CSV loader are unchanged
  - recommend_next_song() now routes through Q-Learning agent
  - Falls back to heuristic if agent has no Q-table entries yet
  - Hardware-free demo and argparse CLI still work as before
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Song:
    name: str
    notes: str
    genre: str = "neutral"
    difficulty: Optional[float] = None


@dataclass(frozen=True)
class PerformanceRecord:
    song_name: str
    accuracy: float
    total_notes: int
    correct_notes: int
    mistakes: int
    elapsed_seconds: Optional[float] = None
    expected_seconds: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def parse_note_tokens(notes: str) -> List[str]:
    if not notes:
        return []
    text = notes.strip()
    if not text:
        return []
    if re.search(r"[\s,|;-]", text):
        parts = re.split(r"[\s,|;-]+", text)
        return [p for p in parts if p]
    return list(text)


def token_positions(tokens: List[str]) -> List[int]:
    if not tokens:
        return []
    unique_sorted = sorted(set(tokens))
    index_map = {token: idx for idx, token in enumerate(unique_sorted)}
    return [index_map[t] for t in tokens]


def compute_song_difficulty(notes: str) -> float:
    if not notes:
        return 0.0
    tokens = parse_note_tokens(notes)
    if not tokens:
        return 0.0
    token_count  = len(tokens)
    unique_count = len(set(tokens))
    positions    = token_positions(tokens)

    length_score  = clamp((token_count - 8) / 24.0)
    variety_ratio = unique_count / max(token_count, 1)
    variety_score = clamp(variety_ratio * 2.0)
    jumps = [abs(positions[i] - positions[i - 1]) for i in range(1, len(positions))]
    if jumps:
        max_possible_jump = max(unique_count - 1, 1)
        jump_score = clamp(sum(jumps) / len(jumps) / max_possible_jump)
    else:
        jump_score = 0.0

    difficulty = (0.5 * length_score) + (0.25 * variety_score) + (0.25 * jump_score)
    return round(clamp(difficulty), 4)


def performance_score(record: PerformanceRecord) -> float:
    accuracy_component  = clamp(record.accuracy / 100.0)
    total               = max(record.total_notes, 1)
    mistake_rate        = clamp(record.mistakes / total)
    precision_component = 1.0 - mistake_rate
    timing_component    = 1.0
    if (record.elapsed_seconds is not None
            and record.expected_seconds is not None
            and record.expected_seconds > 0):
        ratio            = record.elapsed_seconds / record.expected_seconds
        timing_component = clamp(1.0 / max(1.0, ratio))
    score = (0.6 * accuracy_component) + (0.3 * precision_component) + (0.1 * timing_component)
    return round(clamp(score), 4)


def rolling_user_score(history: List[PerformanceRecord], window: int = 5) -> float:
    if not history:
        return 0.5
    recent = history[-window:]
    return round(sum(performance_score(r) for r in recent) / len(recent), 4)


def rank_songs_with_difficulty(song_library: List[Song]) -> List[Song]:
    ranked = []
    for song in song_library:
        diff = song.difficulty if song.difficulty is not None else compute_song_difficulty(song.notes)
        ranked.append(Song(name=song.name, notes=song.notes, genre=song.genre, difficulty=diff))
    return sorted(ranked, key=lambda s: s.difficulty if s.difficulty is not None else 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# AI HELPERS (HMM + K-MEANS + SVD) FOR DIFFICULTY FUSION
# ══════════════════════════════════════════════════════════════════════════════

DIFFICULTY_LEVEL_MIN = 0
DIFFICULTY_LEVEL_MAX = 5
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180


def _score_to_level(score: float) -> int:
    """Map continuous score [0,1] to discrete difficulty level [0,5]."""
    bounded = clamp(score)
    return int(round(bounded * DIFFICULTY_LEVEL_MAX))


def _level_to_servo_angle(level: int) -> int:
    """Map difficulty level [0,5] to servo angle [0,180]."""
    safe_level = max(DIFFICULTY_LEVEL_MIN, min(DIFFICULTY_LEVEL_MAX, level))
    ratio = safe_level / DIFFICULTY_LEVEL_MAX if DIFFICULTY_LEVEL_MAX > 0 else 0.0
    return int(round(SERVO_MIN_ANGLE + ratio * (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)))


def hmm_skill_probability(history: List[PerformanceRecord]) -> float:
    """
    Lightweight Hidden Markov Model estimate of being in a "high-skill" hidden state.
    Uses two hidden states: low_skill (0) and high_skill (1).
    """
    if not history:
        return 0.5

    transition = (
        (0.85, 0.15),  # from low -> (low, high)
        (0.10, 0.90),  # from high -> (low, high)
    )
    emission = (
        (0.80, 0.20),  # low state emits (low_obs, high_obs)
        (0.25, 0.75),  # high state emits (low_obs, high_obs)
    )

    # Start with neutral prior.
    p_low, p_high = 0.5, 0.5
    for record in history[-12:]:
        obs_high = 1 if performance_score(record) >= 0.65 else 0
        obs_low = 1 - obs_high

        pred_low = p_low * transition[0][0] + p_high * transition[1][0]
        pred_high = p_low * transition[0][1] + p_high * transition[1][1]

        like_low = emission[0][0] if obs_low else emission[0][1]
        like_high = emission[1][0] if obs_low else emission[1][1]

        post_low = pred_low * like_low
        post_high = pred_high * like_high
        norm = max(post_low + post_high, 1e-9)
        p_low, p_high = post_low / norm, post_high / norm

    return round(clamp(p_high), 4)


def _kmeans_1d(values: List[float], k: int = 3, iterations: int = 7) -> Tuple[List[float], List[int]]:
    """Tiny 1D K-means (no external deps) returning (centroids, assignments)."""
    if not values:
        return [0.5] * k, []
    ordered = sorted(clamp(v) for v in values)
    if len(ordered) == 1:
        return [ordered[0]] * k, [0]

    centroids = []
    for i in range(k):
        idx = int(round(i * (len(ordered) - 1) / max(k - 1, 1)))
        centroids.append(ordered[idx])

    assignments = [0] * len(values)
    for _ in range(iterations):
        for i, v in enumerate(values):
            assignments[i] = min(range(k), key=lambda c: abs(v - centroids[c]))
        for c in range(k):
            members = [values[i] for i, a in enumerate(assignments) if a == c]
            if members:
                centroids[c] = sum(members) / len(members)
    return centroids, assignments


def kmeans_cluster_difficulty_signal(history: List[PerformanceRecord]) -> float:
    """Cluster recent performance scores and return normalized cluster trend signal."""
    if not history:
        return 0.5
    scores = [performance_score(r) for r in history[-15:]]
    centroids, assignments = _kmeans_1d(scores, k=3)
    hardest_cluster = max(range(len(centroids)), key=lambda i: centroids[i])
    recent_window = assignments[-5:] if assignments else []
    if not recent_window:
        return 0.5
    ratio_in_hard_cluster = sum(1 for a in recent_window if a == hardest_cluster) / len(recent_window)
    return round(clamp(0.4 * (sum(centroids) / len(centroids)) + 0.6 * ratio_in_hard_cluster), 4)


def svd_song_affinity_signal(song_library: List[Song], history: List[PerformanceRecord]) -> float:
    """
    Matrix factorization proxy:
    Build user-song interaction matrix and use rank-1 SVD style approximation
    to estimate user's current preferred difficulty region.
    """
    if not history or not song_library:
        return 0.5

    ranked = rank_songs_with_difficulty(song_library)
    idx_by_name = {s.name: i for i, s in enumerate(ranked)}
    n = len(ranked)

    # Single-user interaction vector over songs.
    interactions = [0.0] * n
    counts = [0] * n
    for rec in history[-25:]:
        idx = idx_by_name.get(rec.song_name)
        if idx is None:
            continue
        interactions[idx] += performance_score(rec)
        counts[idx] += 1

    for i in range(n):
        if counts[i] > 0:
            interactions[i] /= counts[i]

    # Rank-1 approximation in 1D is equivalent to projecting onto normalized vector.
    norm = math.sqrt(sum(v * v for v in interactions))
    if norm <= 1e-9:
        return 0.5
    user_pref = [v / norm for v in interactions]

    difficulty_weighted = 0.0
    for i, song in enumerate(ranked):
        difficulty_weighted += user_pref[i] * (song.difficulty or 0.0)
    return round(clamp(difficulty_weighted), 4)


def estimate_ai_difficulty_level(
    current_diff: float,
    history: List[PerformanceRecord],
    song_library: List[Song],
) -> Tuple[int, Dict[str, float]]:
    """
    Fuse HMM, K-means, and SVD signals into final difficulty level [0,5].
    """
    hmm_signal = hmm_skill_probability(history)
    kmeans_signal = kmeans_cluster_difficulty_signal(history)
    svd_signal = svd_song_affinity_signal(song_library, history)
    base_signal = clamp(current_diff)

    fused = clamp(
        0.25 * base_signal +
        0.35 * hmm_signal +
        0.20 * kmeans_signal +
        0.20 * svd_signal
    )
    level = _score_to_level(fused)
    return level, {
        "base_signal": round(base_signal, 4),
        "hmm_signal": round(hmm_signal, 4),
        "kmeans_signal": round(kmeans_signal, 4),
        "svd_signal": round(svd_signal, 4),
        "fused_signal": round(fused, 4),
    }


def apply_servo_for_difficulty(
    difficulty_level: int,
    servo_writer: Optional[Callable[[int], None]] = None,
) -> int:
    """
    Computes servo angle for difficulty level and optionally writes it to hardware.
    `servo_writer(angle)` can be passed from your ESP/servo module without changing this file's imports.
    """
    angle = _level_to_servo_angle(difficulty_level)
    if servo_writer is not None:
        servo_writer(angle)
    return angle


def build_esp_difficulty_command(
    latest_accuracy: float,
    recommended_difficulty: float,
    song_index: int,
) -> str:
    """
    Build serial payload for ESP firmware command:
    DIFFUPDATE:<perf_percent>,<song_diff_0to1>,<song_index>
    """
    perf_percent = round(max(0.0, min(100.0, latest_accuracy)), 2)
    song_diff_01 = round(clamp(recommended_difficulty), 4)
    safe_song_idx = max(0, int(song_index))
    return f"DIFFUPDATE:{perf_percent:.2f},{song_diff_01:.4f},{safe_song_idx}"


# ══════════════════════════════════════════════════════════════════════════════
# Q-LEARNING AGENT
# ══════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Tabular Q-Learning agent for song recommendation.

    State space (15 states):
      difficulty_bucket × performance_bucket
      difficulty: 0=very_easy, 1=easy, 2=medium, 3=hard, 4=very_hard
      performance: 0=poor, 1=average, 2=good

    Action space:
      Integer index into the sorted song library (0 … N-1).
      The library is ranked by difficulty so action 0 = easiest song.

    Q-table:
      Dict mapping (diff_bucket, perf_bucket, action) → Q-value (float).
      Stored as JSON so progress persists between sessions.

    Reward function:
      base_reward   = performance score after playing (0.0–1.0)
      zone_bonus    = +0.2  if song difficulty was in the "sweet spot"
                             (user_score between 0.4 and 0.85 on that song)
      too_easy_pen  = -0.15 if user_score > 0.9 AND song difficulty < 0.4
      too_hard_pen  = -0.15 if user_score < 0.3 AND song difficulty > 0.6

    Hyperparameters:
      alpha   = 0.1   learning rate
      gamma   = 0.9   discount factor (rewards future performance)
      epsilon = 0.3   initial exploration rate
      epsilon_decay = 0.99  applied after each update
      epsilon_min   = 0.05  floor so agent never stops exploring entirely
    """

    DIFF_BUCKETS = 5    # [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    PERF_BUCKETS = 3    # poor < 0.4, 0.4 <= average <= 0.7, good > 0.7

    def __init__(
        self,
        alpha: float         = 0.1,
        gamma: float         = 0.9,
        epsilon: float       = 0.3,
        epsilon_decay: float = 0.99,
        epsilon_min: float   = 0.05,
        qtable_path: str     = "qtable.json",
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.qtable_path   = qtable_path
        self.q_table: Dict[str, float] = {}
        self._load_qtable()
        print(f"[QLearningAgent] Ready. ε={self.epsilon:.3f}, "
              f"Q-table entries={len(self.q_table)}.")

    # ── state / action encoding ───────────────────────────────────────────────

    @staticmethod
    def difficulty_bucket(diff: float) -> int:
        """Map 0.0–1.0 difficulty to bucket index 0–4."""
        return min(int(diff * QLearningAgent.DIFF_BUCKETS), QLearningAgent.DIFF_BUCKETS - 1)

    @staticmethod
    def performance_bucket(score: float) -> int:
        """Map 0.0–1.0 performance score to bucket index 0–2."""
        if score < 0.4:
            return 0   # poor
        elif score <= 0.7:
            return 1   # average
        else:
            return 2   # good

    @staticmethod
    def _key(diff_b: int, perf_b: int, action: int) -> str:
        return f"{diff_b},{perf_b},{action}"

    def get_q(self, diff_b: int, perf_b: int, action: int) -> float:
        return self.q_table.get(self._key(diff_b, perf_b, action), 0.0)

    def set_q(self, diff_b: int, perf_b: int, action: int, value: float) -> None:
        self.q_table[self._key(diff_b, perf_b, action)] = round(value, 6)

    # ── action selection (epsilon-greedy) ─────────────────────────────────────

    def select_action(self, diff_b: int, perf_b: int, n_actions: int) -> int:
        """
        Explore with probability epsilon, otherwise exploit best known action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, n_actions - 1)   # explore

        # Exploit: pick action with highest Q-value for this state
        q_values = [self.get_q(diff_b, perf_b, a) for a in range(n_actions)]
        max_q    = max(q_values)

        # Break ties randomly so agent doesn't always pick the first song
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    # ── reward calculation ────────────────────────────────────────────────────

    @staticmethod
    def compute_reward(play_score: float, song_difficulty: float) -> float:
        """
        Reward = base performance score
               + zone bonus  (song was appropriately challenging)
               - penalties   (song too easy or too hard)
        """
        reward = play_score

        in_sweet_spot = 0.4 <= play_score <= 0.85
        if in_sweet_spot:
            reward += 0.2

        too_easy = play_score > 0.9 and song_difficulty < 0.4
        too_hard = play_score < 0.3 and song_difficulty > 0.6

        if too_easy:
            reward -= 0.15
        if too_hard:
            reward -= 0.15

        return round(clamp(reward, -1.0, 1.5), 4)

    # ── learning update ───────────────────────────────────────────────────────

    def update(
        self,
        diff_b: int,
        perf_b: int,
        action: int,
        reward: float,
        next_diff_b: int,
        next_perf_b: int,
        n_actions: int,
    ) -> None:
        """
        Standard Q-Learning (off-policy) Bellman update:
          Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]
        """
        current_q  = self.get_q(diff_b, perf_b, action)
        future_qs  = [self.get_q(next_diff_b, next_perf_b, a) for a in range(n_actions)]
        best_future = max(future_qs)
        new_q = current_q + self.alpha * (reward + self.gamma * best_future - current_q)
        self.set_q(diff_b, perf_b, action, new_q)

        # Decay exploration rate after each real learning step
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── persistence ───────────────────────────────────────────────────────────

    def _load_qtable(self) -> None:
        if os.path.isfile(self.qtable_path):
            try:
                with open(self.qtable_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.q_table = data.get("q_table", {})
                self.epsilon = data.get("epsilon", self.epsilon)
                print(f"[QLearningAgent] Loaded Q-table from {self.qtable_path} "
                      f"({len(self.q_table)} entries, ε={self.epsilon:.3f}).")
            except (json.JSONDecodeError, KeyError):
                print(f"[QLearningAgent] Could not parse {self.qtable_path}, starting fresh.")

    def save_qtable(self) -> None:
        with open(self.qtable_path, "w", encoding="utf-8") as f:
            json.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f, indent=2)
        print(f"[QLearningAgent] Q-table saved → {self.qtable_path} "
              f"({len(self.q_table)} entries).")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RECOMMENDER  (integrates Q-Learning with original heuristic fallback)
# ══════════════════════════════════════════════════════════════════════════════

# Module-level agent instance (shared across calls in the same session)
_agent: Optional[QLearningAgent] = None


def get_agent(qtable_path: str = "qtable.json") -> QLearningAgent:
    global _agent
    if _agent is None:
        _agent = QLearningAgent(qtable_path=qtable_path)
    return _agent


def _heuristic_recommend(
    current_diff: float,
    user_score: float,
    ranked: List[Song],
    current_song_name: str,
) -> Tuple[Song, int]:
    """
    Original heuristic: pick song closest to target difficulty.
    Returns (song, action_index).
    """
    shift       = (user_score - 0.5) * 0.4
    target_diff = round(clamp(current_diff + shift), 4)
    candidates  = sorted(
        enumerate(ranked),
        key=lambda ia: (abs((ia[1].difficulty or 0.0) - target_diff), ia[1].name == current_song_name),
    )
    idx, song = candidates[0]
    if song.name == current_song_name and len(candidates) > 1:
        idx, song = candidates[1]
    return song, idx


def recommend_next_song(
    current_song_name: str,
    song_library: List[Song],
    history: List[PerformanceRecord],
    qtable_path: str = "qtable.json",
) -> Tuple[Song, Dict[str, float]]:
    """
    Main entry point — returns (recommended_song, diagnostics).

    Q-Learning is used when the agent has seen at least one prior
    session (Q-table is non-empty). Otherwise falls back to the
    original heuristic so cold-start still works.

    After calling this, you MUST call record_outcome() once the
    user finishes playing the recommended song so the agent learns.
    """
    ranked = rank_songs_with_difficulty(song_library)
    by_name = {s.name: s for s in ranked}
    if current_song_name not in by_name:
        raise ValueError(f"Song '{current_song_name}' not found in library.")

    current_song  = by_name[current_song_name]
    current_diff  = current_song.difficulty or 0.0
    user_score    = rolling_user_score(history)
    n_actions     = len(ranked)
    agent         = get_agent(qtable_path)

    diff_b = QLearningAgent.difficulty_bucket(current_diff)
    perf_b = QLearningAgent.performance_bucket(user_score)

    # Use Q-Learning if the agent has learned anything; else heuristic
    if agent.q_table:
        action = agent.select_action(diff_b, perf_b, n_actions)
        recommendation = ranked[action]
        method = "Q-Learning"
    else:
        recommendation, action = _heuristic_recommend(
            current_diff, user_score, ranked, current_song_name
        )
        method = "heuristic (cold start)"

    # Avoid recommending the same song if alternatives exist
    if recommendation.name == current_song_name and n_actions > 1:
        alt_action = (action + 1) % n_actions
        recommendation = ranked[alt_action]
        action = alt_action

    diagnostics = {
        "current_difficulty":      round(current_diff, 4),
        "user_score":              round(user_score, 4),
        "recommended_difficulty":  round(recommendation.difficulty or 0.0, 4),
        "diff_bucket":             float(diff_b),
        "perf_bucket":             float(perf_b),
        "action_index":            float(action),
        "epsilon":                 round(agent.epsilon, 4),
        "q_table_size":            float(len(agent.q_table)),
        "method":                  method,
    }

    # AI-fused difficulty level (0..5) + servo angle mapping.
    ai_level, ai_diag = estimate_ai_difficulty_level(current_diff, history, ranked)
    diagnostics["difficulty_level"] = float(ai_level)
    diagnostics["servo_angle"] = float(_level_to_servo_angle(ai_level))
    diagnostics.update(ai_diag)

    return recommendation, diagnostics


def record_outcome(
    current_song_name: str,
    recommended_song: Song,
    action_index: int,
    diff_bucket: int,
    perf_bucket: int,
    outcome_record: PerformanceRecord,
    song_library: List[Song],
    qtable_path: str = "qtable.json",
    save: bool = True,
) -> float:
    """
    Call this after the user finishes playing the recommended song.
    Updates the Q-table with the observed reward and saves to disk.

    Returns the reward value for logging.
    """
    agent      = get_agent(qtable_path)
    ranked     = rank_songs_with_difficulty(song_library)
    n_actions  = len(ranked)
    play_score = performance_score(outcome_record)
    song_diff  = recommended_song.difficulty or 0.0
    reward     = QLearningAgent.compute_reward(play_score, song_diff)

    # Next state: difficulty of the song just played + new performance
    next_diff_b = QLearningAgent.difficulty_bucket(song_diff)
    next_perf_b = QLearningAgent.performance_bucket(play_score)

    agent.update(diff_bucket, perf_bucket, action_index, reward,
                 next_diff_b, next_perf_b, n_actions)

    print(f"[QLearningAgent] Update: state=({diff_bucket},{perf_bucket}) "
          f"action={action_index} reward={reward:.3f} "
          f"→ next_state=({next_diff_b},{next_perf_b})  ε={agent.epsilon:.3f}")

    if save:
        agent.save_qtable()

    return reward


# ══════════════════════════════════════════════════════════════════════════════
# LIVE SESSION HELPER  (for ESP integration)
# ══════════════════════════════════════════════════════════════════════════════

def recommend_for_live_session(
    current_song_name: str,
    song_library: List[Song],
    past_history: List[PerformanceRecord],
    latest_accuracy: float,
    latest_total_notes: int,
    latest_mistakes: int,
    latest_elapsed_seconds: Optional[float] = None,
    latest_expected_seconds: Optional[float] = None,
    qtable_path: str = "qtable.json",
) -> Tuple[Song, Dict[str, float]]:
    """Helper for live ESP sessions after each completed melody."""
    safe_total    = max(latest_total_notes, 1)
    latest_correct = safe_total - max(0, latest_mistakes)
    latest = PerformanceRecord(
        song_name      = current_song_name,
        accuracy       = latest_accuracy,
        total_notes    = safe_total,
        correct_notes  = max(0, latest_correct),
        mistakes       = max(0, latest_mistakes),
        elapsed_seconds  = latest_elapsed_seconds,
        expected_seconds = latest_expected_seconds,
    )
    recommendation, diagnostics = recommend_next_song(
        current_song_name, song_library, [*past_history, latest], qtable_path
    )

    ranked = rank_songs_with_difficulty(song_library)
    by_name = {s.name: i for i, s in enumerate(ranked)}
    song_index = by_name.get(recommendation.name, 0)
    diagnostics["esp_diff_command"] = build_esp_difficulty_command(
        latest_accuracy=latest_accuracy,
        recommended_difficulty=recommendation.difficulty or 0.0,
        song_index=song_index,
    )
    return recommendation, diagnostics


# ══════════════════════════════════════════════════════════════════════════════
# CSV LOADER  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def load_history_from_csv(csv_path: str) -> List[PerformanceRecord]:
    rows: List[PerformanceRecord] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(PerformanceRecord(
                song_name      = row["song"],
                accuracy       = float(row["accuracy"]),
                total_notes    = int(row["total_notes"]),
                correct_notes  = int(row["correct_notes"]),
                mistakes       = int(row["mistakes"]),
                elapsed_seconds  = float(row["elapsed_seconds"]) if row.get("elapsed_seconds") else None,
                expected_seconds = float(row["expected_seconds"]) if row.get("expected_seconds") else None,
            ))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SONG LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_LIBRARY: List[Song] = [
    Song("Twinkle Twinkle",       "123456654321",       "happy"),
    Song("Mary Had a Little Lamb","32123333222455",      "neutral"),
    Song("Jingle Bells",          "333333353123",        "happy"),
    Song("Scale Sprint",          "123456123456654321",  "energetic"),
    Song("Leap Challenge",        "1616162345654321",    "energetic"),
    Song("Gentle Steps",          "1122332211",          "calm"),
]


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE-FREE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def simulate_history(
    song_name: str,
    rounds: int = 8,
    seed: int = 7,
    trend: str = "improving",
) -> List[PerformanceRecord]:
    random.seed(seed)
    history: List[PerformanceRecord] = []
    total_notes = 16

    for i in range(rounds):
        if trend == "improving":
            base_acc = 55 + (i * 5)
        elif trend == "declining":
            base_acc = 90 - (i * 5)
        else:
            base_acc = 70 + (8 * math.sin(i))

        accuracy = clamp(base_acc / 100.0) * 100
        jitter   = random.uniform(-3, 3)
        accuracy = clamp((accuracy + jitter) / 100.0) * 100
        mistakes = int(round((100 - accuracy) / 100.0 * total_notes))

        history.append(PerformanceRecord(
            song_name     = song_name,
            accuracy      = round(accuracy, 2),
            total_notes   = total_notes,
            correct_notes = total_notes - mistakes,
            mistakes      = mistakes,
            elapsed_seconds  = 48 + random.uniform(-8, 16),
            expected_seconds = 48,
        ))
    return history


def run_demo(current_song: str, trend: str, episodes: int = 10) -> None:
    """
    Simulates multiple play sessions so you can watch the Q-table grow
    and the agent shift from exploration to exploitation.
    """
    print("\n=== Q-Learning Dynamic Difficulty Demo ===")
    print(f"Song: {current_song}  |  Trend: {trend}  |  Episodes: {episodes}\n")

    history: List[PerformanceRecord] = []

    for ep in range(1, episodes + 1):
        rec, diag = recommend_next_song(current_song, DEFAULT_LIBRARY, history)

        # Simulate the user playing the recommended song
        total_notes = 16
        if trend == "improving":
            base_acc = min(50 + ep * 5, 100)
        elif trend == "declining":
            base_acc = max(90 - ep * 5, 10)
        else:
            base_acc = 70 + 15 * math.sin(ep)

        accuracy = clamp(base_acc / 100.0) * 100
        mistakes = int(round((100 - accuracy) / 100.0 * total_notes))
        outcome  = PerformanceRecord(
            song_name     = rec.name,
            accuracy      = round(accuracy, 2),
            total_notes   = total_notes,
            correct_notes = total_notes - mistakes,
            mistakes      = mistakes,
            elapsed_seconds  = 48 + random.uniform(-5, 10),
            expected_seconds = 48,
        )

        reward = record_outcome(
            current_song_name = current_song,
            recommended_song  = rec,
            action_index      = int(diag["action_index"]),
            diff_bucket       = int(diag["diff_bucket"]),
            perf_bucket       = int(diag["perf_bucket"]),
            outcome_record    = outcome,
            song_library      = DEFAULT_LIBRARY,
            save              = False,   # don't write file during demo
        )

        history.append(outcome)
        current_song = rec.name   # user is now on the recommended song

        print(f"Ep {ep:>2} | Method: {diag['method']:<22} | "
              f"Recommended: {rec.name:<25} "
              f"(diff={diag['recommended_difficulty']:.2f}) | "
              f"Level: {int(diag['difficulty_level'])} "
              f"(servo={int(diag['servo_angle'])}°) | "
              f"Accuracy: {accuracy:.1f}% | Reward: {reward:+.3f} | "
              f"ε={diag['epsilon']:.3f}")

    print("\n==========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Piano Guru — Q-Learning Dynamic Difficulty")
    parser.add_argument("--current-song", default="Twinkle Twinkle")
    parser.add_argument("--trend", default="improving",
                        choices=["improving", "declining", "inconsistent"])
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of simulated episodes for demo")
    parser.add_argument("--history-csv", default=None,
                        help="Path to real user CSV log")
    parser.add_argument("--qtable", default="qtable.json",
                        help="Path to Q-table JSON (loaded + saved automatically)")
    args = parser.parse_args()

    if args.history_csv:
        if not os.path.isfile(args.history_csv):
            print(f"\nCSV file not found: {args.history_csv}. Running demo instead.\n")
            run_demo(args.current_song, args.trend, args.episodes)
            return
        history = load_history_from_csv(args.history_csv)
        rec, diag = recommend_next_song(args.current_song, DEFAULT_LIBRARY,
                                        history, qtable_path=args.qtable)
        print("\n=== Q-Learning Recommendation From CSV ===")
        print(f"Current song  : {args.current_song}")
        print(f"User score    : {diag['user_score']:.3f}")
        print(f"Method        : {diag['method']}")
        print(f"Recommended   : {rec.name}  (difficulty={diag['recommended_difficulty']:.3f})")
        print(f"Epsilon       : {diag['epsilon']:.3f}")
        print(f"Q-table size  : {int(diag['q_table_size'])} entries")
        print("===========================================\n")
    else:
        run_demo(args.current_song, args.trend, args.episodes)


if __name__ == "__main__":
    main()