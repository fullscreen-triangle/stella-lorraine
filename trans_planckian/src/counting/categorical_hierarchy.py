"""
Categorical Hierarchy

Implements the recursive 3^k branching structure of S-entropy space.

Each node in the hierarchy represents a categorical state that can be:
- A memory location (data storage)
- A branch point (routing decision)
- A completion endpoint (final categorical state)

The hierarchy is self-similar: each level looks like the whole structure.
This is the "scale ambiguity" property - you can't tell if you're at the
top of a small subtree or the bottom of a large one.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Generic, TypeVar
from enum import Enum
import time
import weakref

from .s_entropy_address import SCoordinate, SEntropyAddress


T = TypeVar('T')


class NodeType(Enum):
    """Types of nodes in the hierarchy."""
    BRANCH = "branch"  # Internal node with children
    LEAF = "leaf"  # Terminal node holding data
    VIRTUAL = "virtual"  # Lazily-created placeholder


@dataclass
class HierarchyNode(Generic[T]):
    """
    A node in the categorical hierarchy.
    
    Each node can hold data and/or branch into three children,
    implementing the 3^k structure of BMD decomposition.
    """
    # Identity
    coordinate: SCoordinate
    depth: int
    branch_index: int  # Which branch from parent (0, 1, or 2)
    
    # Node type
    node_type: NodeType = NodeType.VIRTUAL
    
    # Data storage
    data: Optional[T] = None
    data_key: Optional[str] = None
    
    # Children (3-way branching)
    children: List[Optional['HierarchyNode[T]']] = field(default_factory=lambda: [None, None, None])
    
    # Parent reference (weak to avoid cycles)
    _parent: Optional[weakref.ref] = field(default=None, repr=False)
    
    # Metadata
    creation_time: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    @property
    def parent(self) -> Optional['HierarchyNode[T]']:
        return self._parent() if self._parent else None
    
    @parent.setter
    def parent(self, node: Optional['HierarchyNode[T]']):
        self._parent = weakref.ref(node) if node else None
    
    @property
    def is_leaf(self) -> bool:
        return self.node_type == NodeType.LEAF or all(c is None for c in self.children)
    
    @property
    def has_data(self) -> bool:
        return self.data is not None
    
    @property 
    def path_from_root(self) -> List[int]:
        """Get the branch path from root to this node."""
        path = []
        current = self
        while current.parent:
            path.append(current.branch_index)
            current = current.parent
        return list(reversed(path))
    
    def get_child(self, branch: int) -> Optional['HierarchyNode[T]']:
        """Get or create a child node."""
        if not 0 <= branch <= 2:
            raise ValueError(f"Branch must be 0, 1, or 2, got {branch}")
        return self.children[branch]
    
    def ensure_child(self, branch: int) -> 'HierarchyNode[T]':
        """Ensure a child exists at the given branch, creating if needed."""
        if self.children[branch] is None:
            # Create child coordinate by decomposition
            child_coords = self.coordinate.decompose()
            child = HierarchyNode(
                coordinate=child_coords[branch],
                depth=self.depth + 1,
                branch_index=branch,
                node_type=NodeType.VIRTUAL
            )
            child.parent = self
            self.children[branch] = child
            self.node_type = NodeType.BRANCH
            
        return self.children[branch]
    
    def store(self, data: T, key: Optional[str] = None):
        """Store data at this node."""
        self.data = data
        self.data_key = key
        self.node_type = NodeType.LEAF
        self.access_count += 1
        self.last_access = time.time()
    
    def retrieve(self) -> Optional[T]:
        """Retrieve data from this node."""
        self.access_count += 1
        self.last_access = time.time()
        return self.data
    
    def clear(self):
        """Clear data from this node."""
        self.data = None
        self.data_key = None
        self.node_type = NodeType.VIRTUAL if self.is_leaf else NodeType.BRANCH


class CategoricalHierarchy(Generic[T]):
    """
    The complete 3^k categorical hierarchy for memory organization.
    
    This implements the S-entropy address space as a navigable tree structure.
    Navigation uses precision-by-difference values to determine branches.
    """
    
    def __init__(self, max_depth: int = 20):
        """
        Initialize the hierarchy.
        
        Args:
            max_depth: Maximum depth of the tree (limits to 3^max_depth nodes)
        """
        self.max_depth = max_depth
        
        # Root node at origin of S-entropy space
        self.root = HierarchyNode[T](
            coordinate=SCoordinate(S_k=0, S_t=0, S_e=0),
            depth=0,
            branch_index=0,
            node_type=NodeType.BRANCH
        )
        
        # Index for fast key lookup
        self._key_index: Dict[str, HierarchyNode[T]] = {}
        
        # Statistics
        self._node_count = 1
        self._data_count = 0
        self._total_accesses = 0
        
    def _navigate_to_path(self, path: List[int], create: bool = True) -> Optional[HierarchyNode[T]]:
        """
        Navigate to a node given a branch path.
        
        Args:
            path: List of branch indices (each 0, 1, or 2)
            create: Whether to create nodes if they don't exist
            
        Returns:
            The node at the path, or None if not found and create=False
        """
        current = self.root
        
        for branch in path:
            if branch < 0 or branch > 2:
                raise ValueError(f"Invalid branch {branch}, must be 0, 1, or 2")
                
            if current.children[branch] is None:
                if create and current.depth < self.max_depth:
                    current.ensure_child(branch)
                    self._node_count += 1
                else:
                    return None
                    
            current = current.children[branch]
            
        return current
    
    def navigate_to_address(self, address: SEntropyAddress, create: bool = True) -> Optional[HierarchyNode[T]]:
        """
        Navigate to a node using an S-entropy address.
        
        The address's trajectory (precision-by-difference history) 
        determines the path through the hierarchy.
        """
        path = address.hierarchy_branch[:self.max_depth]
        return self._navigate_to_path(path, create)
    
    def store(self, key: str, data: T, address: SEntropyAddress) -> HierarchyNode[T]:
        """
        Store data at the location determined by an S-entropy address.
        
        Args:
            key: Unique identifier for this data
            data: The data to store
            address: S-entropy address determining location
            
        Returns:
            The node where data was stored
        """
        node = self.navigate_to_address(address, create=True)
        
        if node:
            node.store(data, key)
            self._key_index[key] = node
            self._data_count += 1
            
        return node
    
    def retrieve(self, address: SEntropyAddress) -> Tuple[Optional[T], Optional[HierarchyNode[T]]]:
        """
        Retrieve data from an S-entropy address.
        
        Args:
            address: The address to retrieve from
            
        Returns:
            Tuple of (data, node), both None if not found
        """
        node = self.navigate_to_address(address, create=False)
        self._total_accesses += 1
        
        if node and node.has_data:
            return node.retrieve(), node
        return None, node
    
    def retrieve_by_key(self, key: str) -> Tuple[Optional[T], Optional[HierarchyNode[T]]]:
        """
        Retrieve data by its key (fast lookup via index).
        """
        node = self._key_index.get(key)
        self._total_accesses += 1
        
        if node:
            return node.retrieve(), node
        return None, None
    
    def find_nearest(self, address: SEntropyAddress, max_distance: int = 5) -> List[Tuple[T, float, HierarchyNode[T]]]:
        """
        Find data near an address (within max_distance branches).
        
        This implements categorical proximity search.
        
        Returns:
            List of (data, distance, node) tuples, sorted by distance
        """
        results = []
        target_path = address.hierarchy_branch[:self.max_depth]
        
        # Search nodes within distance
        def search(node: HierarchyNode[T], current_path: List[int], current_distance: int):
            if current_distance > max_distance:
                return
                
            if node.has_data:
                results.append((node.data, current_distance, node))
                
            for i, child in enumerate(node.children):
                if child is not None:
                    new_distance = current_distance
                    if len(current_path) < len(target_path):
                        if target_path[len(current_path)] != i:
                            new_distance += 1
                    else:
                        new_distance += 1
                        
                    search(child, current_path + [i], new_distance)
        
        search(self.root, [], 0)
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results
    
    def predict_access(self, address: SEntropyAddress) -> List[Tuple[str, float]]:
        """
        Predict which data will likely be accessed next.
        
        Uses categorical completion to determine future access patterns.
        
        Returns:
            List of (key, probability) tuples for predicted accesses
        """
        predictions = []
        
        # Get completion point
        completion = address.predict_completion()
        if not completion:
            return predictions
        
        # Find nodes near the completion point
        # Create a temporary address pointing to completion
        temp_address = SEntropyAddress()
        temp_address.record(completion, 0.0)
        
        near_nodes = self.find_nearest(temp_address, max_distance=3)
        
        # Calculate probabilities based on distance
        if near_nodes:
            total_inverse_dist = sum(1.0 / (d + 1) for _, d, _ in near_nodes)
            for _, dist, node in near_nodes:
                if node.data_key:
                    prob = (1.0 / (dist + 1)) / total_inverse_dist
                    predictions.append((node.data_key, prob))
        
        return predictions
    
    def get_branch_statistics(self, depth: int = 3) -> Dict[str, Any]:
        """
        Get statistics about branch usage up to a certain depth.
        """
        branch_counts = [[0, 0, 0] for _ in range(depth)]
        node_counts = [0] * depth
        
        def count_nodes(node: HierarchyNode[T], d: int):
            if d >= depth:
                return
            node_counts[d] += 1
            for i, child in enumerate(node.children):
                if child is not None:
                    branch_counts[d][i] += 1
                    count_nodes(child, d + 1)
        
        count_nodes(self.root, 0)
        
        return {
            'branch_counts': branch_counts,
            'node_counts': node_counts,
            'total_nodes': self._node_count,
            'total_data': self._data_count,
            'total_accesses': self._total_accesses,
        }
    
    def compress_hierarchy(self) -> Dict[str, Any]:
        """
        Compress the hierarchy by removing unused branches.
        
        This implements the categorical filtering that BMDs perform -
        reducing many possibilities to the actually-used paths.
        
        Returns:
            Compression statistics
        """
        removed = 0
        
        def prune(node: HierarchyNode[T]) -> bool:
            """Prune empty subtrees, returns True if node should be kept."""
            if node.has_data:
                return True
                
            keep_children = []
            for i, child in enumerate(node.children):
                if child is not None:
                    if prune(child):
                        keep_children.append(i)
                    else:
                        node.children[i] = None
                        
            return len(keep_children) > 0 or node.has_data
        
        # Count before
        before = self._node_count
        
        # Prune
        prune(self.root)
        
        # Recount
        def count(node):
            c = 1
            for child in node.children:
                if child is not None:
                    c += count(child)
            return c
            
        self._node_count = count(self.root)
        removed = before - self._node_count
        
        return {
            'nodes_removed': removed,
            'nodes_remaining': self._node_count,
            'compression_ratio': before / (self._node_count + 1),
        }


