"""
Categorical Memory Controller

The Maxwell Demon for memory - navigates the S-entropy hierarchy to
determine which data should be "hot" (readily available) vs "cold" (archived).

This doesn't replace existing memory - it provides intelligent caching
decisions based on categorical completion rather than statistical prediction.

Key insight: The access pattern history IS the address. 
The demon doesn't predict - it navigates to pre-determined endpoints.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Generic, TypeVar
from collections import OrderedDict
from enum import Enum
import time
import threading

from .s_entropy_address import SCoordinate, SEntropyAddress
from .hardware_oscillator import HardwareOscillatorCapture
from .precision_calculator import PrecisionByDifferenceCalculator
from .categorical_hierarchy import CategoricalHierarchy, HierarchyNode


T = TypeVar('T')


class MemoryTier(Enum):
    """Memory tiers corresponding to hierarchy depth."""
    L1_CACHE = 0  # Top of hierarchy - immediate access
    L2_CACHE = 1  # Near-top - very fast
    RAM = 2       # Middle - fast
    SSD = 3       # Lower - moderate
    ARCHIVE = 4   # Deep - slow but persistent


@dataclass
class MemoryEntry(Generic[T]):
    """An entry in the categorical memory system."""
    key: str
    data: T
    address: SEntropyAddress
    tier: MemoryTier
    size: int  # Size in bytes (estimated)
    
    # Access tracking
    creation_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    
    # Categorical metadata
    completion_probability: float = 0.0  # How likely this is the completion point
    categorical_distance: float = float('inf')  # Distance from current position


class CategoricalMemoryController(Generic[T]):
    """
    The main memory controller implementing categorical navigation.
    
    This is the "Maxwell Demon" that decides memory placement based on
    S-entropy hierarchy position rather than statistical prediction.
    
    Unlike traditional caches (LRU, LFU, etc.), this uses:
    - Precision-by-difference to determine current position
    - Categorical completion to find likely endpoints
    - Hierarchy navigation to optimize placement
    """
    
    # Default tier capacities (number of entries)
    DEFAULT_CAPACITIES = {
        MemoryTier.L1_CACHE: 64,
        MemoryTier.L2_CACHE: 256,
        MemoryTier.RAM: 4096,
        MemoryTier.SSD: 65536,
        MemoryTier.ARCHIVE: float('inf'),
    }
    
    def __init__(
        self,
        capacities: Optional[Dict[MemoryTier, int]] = None,
        oscillator: Optional[HardwareOscillatorCapture] = None
    ):
        """
        Initialize the categorical memory controller.
        
        Args:
            capacities: Capacity for each tier (defaults to DEFAULT_CAPACITIES)
            oscillator: Hardware oscillator for precision calculations
        """
        self.capacities = capacities or self.DEFAULT_CAPACITIES.copy()
        
        # Core components
        self.oscillator = oscillator or HardwareOscillatorCapture()
        self.precision_calc = PrecisionByDifferenceCalculator(self.oscillator)
        self.hierarchy = CategoricalHierarchy[MemoryEntry[T]]()
        
        # Tier storage
        self.tiers: Dict[MemoryTier, OrderedDict[str, MemoryEntry[T]]] = {
            tier: OrderedDict() for tier in MemoryTier
        }
        
        # Current position in S-entropy space
        self._current_address = self.precision_calc.create_address("__controller__")
        
        # Background maintenance
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': {tier: 0 for tier in MemoryTier},
            'misses': 0,
            'promotions': 0,
            'demotions': 0,
            'evictions': 0,
        }
    
    def _update_position(self):
        """Update our current position in S-entropy space."""
        self.precision_calc.update_address("__controller__")
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        import sys
        try:
            return sys.getsizeof(data)
        except:
            return 64  # Default estimate
    
    def _determine_tier(self, address: SEntropyAddress) -> MemoryTier:
        """
        Determine the optimal tier for an address based on hierarchy position.
        
        Uses categorical distance to current position to decide placement.
        """
        # Get depth and branch path
        depth, branches = self.precision_calc.compute_hierarchy_position(address.trajectory_hash)
        
        # Shallow depth = hot data (L1/L2 cache)
        # Deep depth = cold data (SSD/Archive)
        
        if depth <= 3:
            return MemoryTier.L1_CACHE
        elif depth <= 6:
            return MemoryTier.L2_CACHE
        elif depth <= 10:
            return MemoryTier.RAM
        elif depth <= 15:
            return MemoryTier.SSD
        else:
            return MemoryTier.ARCHIVE
    
    def _calculate_completion_probability(self, entry: MemoryEntry[T]) -> float:
        """
        Calculate how likely this entry is the categorical completion point.
        
        Based on trajectory analysis - where is the access pattern heading?
        """
        # Get predicted completion point
        completion = self._current_address.predict_completion()
        if not completion:
            return 0.0
        
        # Get entry's position
        entry_pos = entry.address.current_position
        if not entry_pos:
            return 0.0
        
        # Calculate distance to completion
        distance = completion.distance_to(entry_pos)
        
        # Convert to probability (closer = higher probability)
        return np.exp(-distance)
    
    def store(self, key: str, data: T) -> MemoryEntry[T]:
        """
        Store data in the categorical memory system.
        
        The data is placed in the hierarchy based on precision-by-difference,
        and assigned to a tier based on categorical analysis.
        
        Args:
            key: Unique identifier
            data: Data to store
            
        Returns:
            The created memory entry
        """
        with self._lock:
            # Update our position
            self._update_position()
            
            # Create address for this data
            address = self.precision_calc.create_address(key)
            self.precision_calc.update_address(key)
            
            # Determine optimal tier
            tier = self._determine_tier(address)
            
            # Create entry
            entry = MemoryEntry(
                key=key,
                data=data,
                address=address,
                tier=tier,
                size=self._estimate_size(data)
            )
            
            # Calculate completion probability
            entry.completion_probability = self._calculate_completion_probability(entry)
            
            # Store in hierarchy
            self.hierarchy.store(key, entry, address)
            
            # Store in tier (with eviction if needed)
            self._store_in_tier(entry, tier)
            
            return entry
    
    def _store_in_tier(self, entry: MemoryEntry[T], tier: MemoryTier):
        """Store entry in tier, handling eviction if needed."""
        tier_storage = self.tiers[tier]
        
        # Check capacity
        while len(tier_storage) >= self.capacities[tier]:
            # Evict based on categorical distance (not LRU!)
            evict_key = self._select_eviction_target(tier)
            if evict_key:
                evicted = tier_storage.pop(evict_key)
                # Demote to next tier
                next_tier = self._get_next_tier(tier)
                if next_tier:
                    self._store_in_tier(evicted, next_tier)
                    evicted.tier = next_tier
                    self._stats['demotions'] += 1
                else:
                    self._stats['evictions'] += 1
        
        tier_storage[entry.key] = entry
    
    def _select_eviction_target(self, tier: MemoryTier) -> Optional[str]:
        """
        Select which entry to evict from a tier.
        
        Uses categorical analysis rather than simple LRU/LFU.
        Evicts entries with lowest completion probability.
        """
        tier_storage = self.tiers[tier]
        if not tier_storage:
            return None
        
        # Find entry with lowest completion probability
        min_prob = float('inf')
        evict_key = None
        
        for key, entry in tier_storage.items():
            prob = self._calculate_completion_probability(entry)
            if prob < min_prob:
                min_prob = prob
                evict_key = key
        
        return evict_key
    
    def _get_next_tier(self, tier: MemoryTier) -> Optional[MemoryTier]:
        """Get the next tier down (for demotion)."""
        tier_order = [MemoryTier.L1_CACHE, MemoryTier.L2_CACHE, 
                      MemoryTier.RAM, MemoryTier.SSD, MemoryTier.ARCHIVE]
        try:
            idx = tier_order.index(tier)
            if idx < len(tier_order) - 1:
                return tier_order[idx + 1]
        except ValueError:
            pass
        return None
    
    def _get_prev_tier(self, tier: MemoryTier) -> Optional[MemoryTier]:
        """Get the previous tier (for promotion)."""
        tier_order = [MemoryTier.L1_CACHE, MemoryTier.L2_CACHE,
                      MemoryTier.RAM, MemoryTier.SSD, MemoryTier.ARCHIVE]
        try:
            idx = tier_order.index(tier)
            if idx > 0:
                return tier_order[idx - 1]
        except ValueError:
            pass
        return None
    
    def retrieve(self, key: str) -> Optional[T]:
        """
        Retrieve data by key.
        
        Updates categorical position and may trigger tier promotion.
        """
        with self._lock:
            # Update position
            self._update_position()
            
            # Search tiers from fastest to slowest
            for tier in MemoryTier:
                if key in self.tiers[tier]:
                    entry = self.tiers[tier][key]
                    entry.access_count += 1
                    entry.last_access = time.time()
                    
                    # Update address trajectory
                    self.precision_calc.update_address(key)
                    
                    # Record hit
                    self._stats['hits'][tier] += 1
                    
                    # Consider promotion
                    self._consider_promotion(entry)
                    
                    return entry.data
            
            # Miss
            self._stats['misses'] += 1
            return None
    
    def _consider_promotion(self, entry: MemoryEntry[T]):
        """
        Consider promoting an entry to a faster tier.
        
        Promotion happens when completion probability increases
        (i.e., we're navigating toward this data).
        """
        new_prob = self._calculate_completion_probability(entry)
        
        # Significant increase in probability triggers promotion
        if new_prob > entry.completion_probability * 1.5:
            prev_tier = self._get_prev_tier(entry.tier)
            if prev_tier:
                # Remove from current tier
                self.tiers[entry.tier].pop(entry.key, None)
                
                # Add to faster tier
                entry.tier = prev_tier
                entry.completion_probability = new_prob
                self._store_in_tier(entry, prev_tier)
                
                self._stats['promotions'] += 1
    
    def prefetch(self, n: int = 5) -> List[str]:
        """
        Prefetch data likely to be accessed soon.
        
        Uses categorical completion to identify endpoints and 
        promotes corresponding data to faster tiers.
        
        Returns:
            List of keys that were prefetched/promoted
        """
        with self._lock:
            self._update_position()
            
            # Get predictions from hierarchy
            predictions = self.hierarchy.predict_access(self._current_address)
            
            prefetched = []
            for key, prob in predictions[:n]:
                # Find entry in any tier
                for tier in reversed(list(MemoryTier)):  # Search cold first
                    if key in self.tiers[tier]:
                        entry = self.tiers[tier][key]
                        if tier != MemoryTier.L1_CACHE:
                            # Promote to L1
                            self.tiers[tier].pop(key)
                            entry.tier = MemoryTier.L1_CACHE
                            self._store_in_tier(entry, MemoryTier.L1_CACHE)
                            prefetched.append(key)
                            self._stats['promotions'] += 1
                        break
            
            return prefetched
    
    def get_categorical_position(self) -> Dict[str, Any]:
        """Get current position in the S-entropy hierarchy."""
        position = self._current_address.current_position
        depth, branches = self.precision_calc.compute_hierarchy_position("__controller__")
        
        return {
            'coordinate': {
                'S_k': position.S_k if position else 0,
                'S_t': position.S_t if position else 0,
                'S_e': position.S_e if position else 0,
            } if position else None,
            'depth': depth,
            'branch_path': branches,
            'trajectory_hash': self._current_address.trajectory_hash,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory controller statistics."""
        tier_sizes = {tier.name: len(storage) for tier, storage in self.tiers.items()}
        
        total_hits = sum(self._stats['hits'].values())
        total_ops = total_hits + self._stats['misses']
        hit_rate = total_hits / total_ops if total_ops > 0 else 0
        
        return {
            'tier_sizes': tier_sizes,
            'tier_capacities': {t.name: c for t, c in self.capacities.items()},
            'hits_by_tier': {t.name: h for t, h in self._stats['hits'].items()},
            'total_hits': total_hits,
            'misses': self._stats['misses'],
            'hit_rate': hit_rate,
            'promotions': self._stats['promotions'],
            'demotions': self._stats['demotions'],
            'evictions': self._stats['evictions'],
            'hierarchy_stats': self.hierarchy.get_branch_statistics(),
            'precision_stats': self.precision_calc.get_statistics(),
        }
    
    def demonstrate_categorical_navigation(self, n_operations: int = 100) -> Dict[str, Any]:
        """
        Demonstrate categorical memory navigation with sample operations.
        
        Returns statistics showing how the system behaves.
        """
        import random
        import string
        
        results = {
            'operations': [],
            'position_history': [],
            'tier_history': [],
        }
        
        # Generate some test data
        test_keys = [f"data_{i}" for i in range(50)]
        for key in test_keys:
            data = ''.join(random.choices(string.ascii_letters, k=100))
            self.store(key, data)
        
        # Perform random access pattern
        for i in range(n_operations):
            key = random.choice(test_keys)
            result = self.retrieve(key)
            
            pos = self.get_categorical_position()
            
            results['operations'].append({
                'step': i,
                'key': key,
                'found': result is not None,
            })
            results['position_history'].append(pos)
            
            # Periodic prefetch
            if i % 10 == 0:
                prefetched = self.prefetch(3)
                results['operations'][-1]['prefetched'] = prefetched
        
        # Final statistics
        results['final_stats'] = self.get_statistics()
        
        return results


