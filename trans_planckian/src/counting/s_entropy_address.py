"""
S-Entropy Address Space

Implements the three-coordinate S-entropy system (S_k, S_t, S_e) as memory addresses.
Each address represents a position in the recursive self-similar hierarchy.

The key insight: an S-address is not a location, it's a TRAJECTORY through the hierarchy.
The accumulated history of precision-by-difference values encodes this trajectory.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from collections import deque
import hashlib
import time


@dataclass
class SCoordinate:
    """
    A single S-entropy coordinate representing position in categorical space.
    
    S_k: Kinetic component (rate of state transitions)
    S_t: Thermal component (distribution breadth)  
    S_e: Entropic component (disorder/completion level)
    
    Together these form a unique address in the recursive hierarchy.
    """
    S_k: float  # Kinetic: rate of categorical completion
    S_t: float  # Thermal: breadth of accessible states
    S_e: float  # Entropic: completion/disorder measure
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    depth: int = 0  # Depth in the 3^k hierarchy
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for calculations."""
        return np.array([self.S_k, self.S_t, self.S_e])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, depth: int = 0) -> 'SCoordinate':
        """Create from numpy vector."""
        return cls(S_k=float(vec[0]), S_t=float(vec[1]), S_e=float(vec[2]), depth=depth)
    
    def distance_to(self, other: 'SCoordinate') -> float:
        """
        Calculate S-distance to another coordinate.
        This is the categorical metric, not Euclidean distance.
        """
        # S-distance accounts for the hierarchy structure
        delta = self.to_vector() - other.to_vector()
        
        # Weight by depth difference (hierarchy matters)
        depth_factor = 3 ** abs(self.depth - other.depth)
        
        return float(np.linalg.norm(delta) * depth_factor)
    
    def decompose(self) -> Tuple['SCoordinate', 'SCoordinate', 'SCoordinate']:
        """
        Decompose into three child coordinates (BMD 3^k branching).
        Each S-coordinate branches into three children, representing
        the three sub-operations of each BMD operation.
        """
        base = self.to_vector()
        
        # Three-way decomposition following BMD structure
        # Each child inherits 1/3 of parent entropy
        child1 = SCoordinate(
            S_k=base[0] / 3 + np.random.normal(0, 0.01),
            S_t=base[1] / 3,
            S_e=base[2] / 3,
            depth=self.depth + 1
        )
        child2 = SCoordinate(
            S_k=base[0] / 3,
            S_t=base[1] / 3 + np.random.normal(0, 0.01),
            S_e=base[2] / 3,
            depth=self.depth + 1
        )
        child3 = SCoordinate(
            S_k=base[0] / 3,
            S_t=base[1] / 3,
            S_e=base[2] / 3 + np.random.normal(0, 0.01),
            depth=self.depth + 1
        )
        
        return (child1, child2, child3)
    
    def __hash__(self):
        return hash((round(self.S_k, 6), round(self.S_t, 6), round(self.S_e, 6), self.depth))


class SEntropyAddress:
    """
    A complete S-entropy address, encoding a trajectory through the categorical hierarchy.
    
    The address is NOT a single point, but a SEQUENCE of precision-by-difference
    values that trace a path through the recursive structure.
    
    Key insight: The history IS the address.
    """
    
    def __init__(self, max_history: int = 64):
        """
        Initialize an S-entropy address.
        
        Args:
            max_history: Maximum trajectory length to maintain
        """
        self.trajectory: deque[SCoordinate] = deque(maxlen=max_history)
        self.precision_differences: deque[float] = deque(maxlen=max_history)
        self.creation_time = time.time()
        self.access_count = 0
        
    def record(self, coordinate: SCoordinate, precision_diff: float):
        """
        Record a new point in the trajectory.
        
        Args:
            coordinate: The S-coordinate at this step
            precision_diff: The precision-by-difference value (Delta P)
        """
        self.trajectory.append(coordinate)
        self.precision_differences.append(precision_diff)
        self.access_count += 1
        
    @property
    def current_position(self) -> Optional[SCoordinate]:
        """Get the current position in the hierarchy."""
        return self.trajectory[-1] if self.trajectory else None
    
    @property
    def trajectory_hash(self) -> str:
        """
        Generate a unique hash of the trajectory.
        This IS the memory address - the path encodes the location.
        """
        if not self.trajectory:
            return "empty"
        
        # Concatenate all precision differences into a signature
        sig = np.array(list(self.precision_differences))
        
        # Hash the trajectory
        return hashlib.sha256(sig.tobytes()).hexdigest()[:16]
    
    @property
    def hierarchy_branch(self) -> List[int]:
        """
        Convert trajectory to branch indices in the 3^k tree.
        
        Each precision-by-difference value maps to one of three branches
        at each level, creating a unique path through the hierarchy.
        """
        branches = []
        for pd in self.precision_differences:
            # Map precision difference to branch index (0, 1, or 2)
            # Using modular arithmetic on the difference
            branch = int(abs(pd * 1e9) % 3)
            branches.append(branch)
        return branches
    
    def navigate_to(self, target: 'SEntropyAddress') -> List[Tuple[int, str]]:
        """
        Compute the navigation path to reach target address.
        
        This is categorical completion - finding the optimal path
        through the hierarchy, not prediction.
        
        Returns:
            List of (branch_index, action) tuples describing the path
        """
        my_branches = self.hierarchy_branch
        target_branches = target.hierarchy_branch
        
        # Find common prefix (shared ancestry in hierarchy)
        common_depth = 0
        for i in range(min(len(my_branches), len(target_branches))):
            if my_branches[i] == target_branches[i]:
                common_depth = i + 1
            else:
                break
        
        path = []
        
        # Navigate up to common ancestor
        for i in range(len(my_branches) - 1, common_depth - 1, -1):
            path.append((my_branches[i], "ascend"))
        
        # Navigate down to target
        for i in range(common_depth, len(target_branches)):
            path.append((target_branches[i], "descend"))
            
        return path
    
    def categorical_distance(self, other: 'SEntropyAddress') -> float:
        """
        Calculate categorical distance to another address.
        
        This is the S-distance metric: the minimum path length
        through the hierarchy to reach the other address.
        """
        path = self.navigate_to(other)
        
        # Each step in the hierarchy has cost proportional to depth
        cost = 0.0
        current_depth = len(self.hierarchy_branch)
        
        for branch, action in path:
            if action == "ascend":
                cost += 1.0 / (3 ** current_depth)  # Cheaper to go up
                current_depth -= 1
            else:
                current_depth += 1
                cost += 1.0 * (3 ** (current_depth - 1))  # Expensive to go down
                
        return cost
    
    def predict_completion(self) -> Optional[SCoordinate]:
        """
        Predict the categorical completion point.
        
        NOT prediction in the statistical sense - this is extracting
        the endpoint that is ALREADY encoded in the trajectory.
        The history contains the destination.
        """
        if len(self.trajectory) < 2:
            return None
        
        # Fit trajectory to find convergence point
        coords = np.array([c.to_vector() for c in self.trajectory])
        
        # Calculate trajectory velocity
        velocities = np.diff(coords, axis=0)
        
        if len(velocities) == 0:
            return self.current_position
        
        # Project to steady state (where velocity → 0)
        avg_velocity = np.mean(velocities, axis=0)
        avg_position = np.mean(coords, axis=0)
        
        # Categorical completion point
        # The trajectory asymptotically approaches this
        decay_rate = np.linalg.norm(velocities[-1]) / (np.linalg.norm(velocities[0]) + 1e-10)
        
        if decay_rate < 1:
            # Converging trajectory - extrapolate to endpoint
            steps_to_completion = int(1 / (1 - decay_rate + 1e-10))
            completion = avg_position + avg_velocity * steps_to_completion * 0.5
            return SCoordinate.from_vector(completion, depth=len(self.trajectory))
        else:
            # Diverging - return current position
            return self.current_position
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'trajectory_hash': self.trajectory_hash,
            'hierarchy_branch': self.hierarchy_branch,
            'current_position': {
                'S_k': self.current_position.S_k,
                'S_t': self.current_position.S_t,
                'S_e': self.current_position.S_e,
            } if self.current_position else None,
            'access_count': self.access_count,
            'trajectory_length': len(self.trajectory),
        }


