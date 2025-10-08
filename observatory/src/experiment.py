#!/usr/bin/env python3
"""
Observatory Experiment Orchestrator
==================================
This module runs experiments and manages all experimental processes.
It serves as the orchestrator that ensures correct application of methods
and manages the flow of experiments towards optimizing tangible goals.

The orchestrator is implemented as a Bayesian Network with nodes that are
updated towards optimizing specific objectives through evidence propagation
and probabilistic inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum, auto
import asyncio
from abc import ABC, abstractmethod
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, beta, gamma
import time

from experiment_config import (
    ExperimentConfig, ExperimentType, OptimizationGoal,
    TemporalPrecisionConfig, OscillatoryAnalysisConfig,
    ConsciousnessTargetingConfig, MemorialFrameworkConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the Bayesian Network"""
    PARAMETER = auto()
    OBSERVABLE = auto()
    LATENT = auto()
    GOAL = auto()
    CONSTRAINT = auto()


class DistributionType(Enum):
    """Probability distribution types for Bayesian Network nodes"""
    NORMAL = auto()
    BETA = auto()
    GAMMA = auto()
    UNIFORM = auto()
    EXPONENTIAL = auto()
    CATEGORICAL = auto()


@dataclass
class BayesianNode:
    """
    A node in the Bayesian Network representing variables in the experiment
    """
    name: str
    node_type: NodeType
    distribution_type: DistributionType
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # Distribution parameters
    prior_params: Dict[str, float] = field(default_factory=dict)
    posterior_params: Dict[str, float] = field(default_factory=dict)

    # Current values and uncertainties
    current_value: Optional[float] = None
    uncertainty: Optional[float] = None

    # Evidence and observations
    observations: List[float] = field(default_factory=list)
    evidence_weight: float = 1.0

    # Node-specific metadata
    bounds: Tuple[float, float] = (-np.inf, np.inf)
    is_discrete: bool = False
    categories: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize posterior parameters with priors"""
        if not self.posterior_params:
            self.posterior_params = self.prior_params.copy()

    def add_observation(self, value: float, weight: float = 1.0):
        """Add an observation to this node"""
        self.observations.append(value)
        self.update_posterior(value, weight)

    def update_posterior(self, observation: float, weight: float = 1.0):
        """Update posterior distribution based on observation"""
        if self.distribution_type == DistributionType.NORMAL:
            self._update_normal_posterior(observation, weight)
        elif self.distribution_type == DistributionType.BETA:
            self._update_beta_posterior(observation, weight)
        elif self.distribution_type == DistributionType.GAMMA:
            self._update_gamma_posterior(observation, weight)

        # Update current value and uncertainty
        self.current_value = self.get_posterior_mean()
        self.uncertainty = self.get_posterior_std()

    def _update_normal_posterior(self, observation: float, weight: float):
        """Update Normal distribution posterior"""
        prior_mean = self.posterior_params.get('mean', 0.0)
        prior_var = self.posterior_params.get('variance', 1.0)
        obs_var = self.posterior_params.get('obs_variance', 0.1)

        # Bayesian update for Normal-Normal conjugacy
        prior_precision = 1.0 / prior_var
        obs_precision = weight / obs_var

        posterior_precision = prior_precision + obs_precision
        posterior_mean = (prior_precision * prior_mean + obs_precision * observation) / posterior_precision
        posterior_var = 1.0 / posterior_precision

        self.posterior_params['mean'] = posterior_mean
        self.posterior_params['variance'] = posterior_var

    def _update_beta_posterior(self, observation: float, weight: float):
        """Update Beta distribution posterior"""
        alpha = self.posterior_params.get('alpha', 1.0)
        beta_param = self.posterior_params.get('beta', 1.0)

        # Beta-Binomial conjugacy (treating observation as success rate)
        if 0 <= observation <= 1:
            self.posterior_params['alpha'] = alpha + weight * observation
            self.posterior_params['beta'] = beta_param + weight * (1 - observation)

    def _update_gamma_posterior(self, observation: float, weight: float):
        """Update Gamma distribution posterior"""
        shape = self.posterior_params.get('shape', 1.0)
        rate = self.posterior_params.get('rate', 1.0)

        # Gamma-Poisson conjugacy approximation
        if observation > 0:
            self.posterior_params['shape'] = shape + weight
            self.posterior_params['rate'] = rate + weight / observation

    def get_posterior_mean(self) -> float:
        """Get the mean of the posterior distribution"""
        if self.distribution_type == DistributionType.NORMAL:
            return self.posterior_params.get('mean', 0.0)
        elif self.distribution_type == DistributionType.BETA:
            alpha = self.posterior_params.get('alpha', 1.0)
            beta_param = self.posterior_params.get('beta', 1.0)
            return alpha / (alpha + beta_param)
        elif self.distribution_type == DistributionType.GAMMA:
            shape = self.posterior_params.get('shape', 1.0)
            rate = self.posterior_params.get('rate', 1.0)
            return shape / rate
        return 0.0

    def get_posterior_std(self) -> float:
        """Get the standard deviation of the posterior distribution"""
        if self.distribution_type == DistributionType.NORMAL:
            variance = self.posterior_params.get('variance', 1.0)
            return np.sqrt(variance)
        elif self.distribution_type == DistributionType.BETA:
            alpha = self.posterior_params.get('alpha', 1.0)
            beta_param = self.posterior_params.get('beta', 1.0)
            return np.sqrt(alpha * beta_param / ((alpha + beta_param)**2 * (alpha + beta_param + 1)))
        elif self.distribution_type == DistributionType.GAMMA:
            shape = self.posterior_params.get('shape', 1.0)
            rate = self.posterior_params.get('rate', 1.0)
            return np.sqrt(shape) / rate
        return 1.0

    def sample(self, n_samples: int = 1) -> Union[float, np.ndarray]:
        """Sample from the posterior distribution"""
        if self.distribution_type == DistributionType.NORMAL:
            mean = self.posterior_params.get('mean', 0.0)
            std = np.sqrt(self.posterior_params.get('variance', 1.0))
            samples = np.random.normal(mean, std, n_samples)
        elif self.distribution_type == DistributionType.BETA:
            alpha = self.posterior_params.get('alpha', 1.0)
            beta_param = self.posterior_params.get('beta', 1.0)
            samples = np.random.beta(alpha, beta_param, n_samples)
        elif self.distribution_type == DistributionType.GAMMA:
            shape = self.posterior_params.get('shape', 1.0)
            scale = 1.0 / self.posterior_params.get('rate', 1.0)
            samples = np.random.gamma(shape, scale, n_samples)
        else:
            samples = np.random.uniform(self.bounds[0], self.bounds[1], n_samples)

        # Apply bounds
        if self.bounds != (-np.inf, np.inf):
            samples = np.clip(samples, self.bounds[0], self.bounds[1])

        return samples[0] if n_samples == 1 else samples


class ExperimentOrchestrator:
    """
    Bayesian Network-based experiment orchestrator that manages experimental
    processes and optimizes towards tangible goals through probabilistic inference.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.timestamp = datetime.now(timezone.utc)
        self.experiment_id = f"{config.experiment_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Bayesian Network structure
        self.nodes: Dict[str, BayesianNode] = {}
        self.adjacency_matrix: Optional[np.ndarray] = None

        # Experiment state
        self.is_running = False
        self.current_iteration = 0
        self.best_objective_value = -np.inf
        self.convergence_history: List[float] = []

        # Results storage
        self.results: Dict[str, Any] = {}
        self.experiment_log: List[Dict[str, Any]] = []

        # Initialize the Bayesian Network
        self._initialize_bayesian_network()

        logger.info(f"Initialized experiment orchestrator: {self.experiment_id}")

    def _initialize_bayesian_network(self):
        """Initialize the Bayesian Network structure based on experiment type"""
        logger.info("Initializing Bayesian Network structure...")

        if self.config.experiment_type == ExperimentType.PRECISION_TIMING:
            self._create_precision_timing_network()
        elif self.config.experiment_type == ExperimentType.OSCILLATORY_ANALYSIS:
            self._create_oscillatory_analysis_network()
        elif self.config.experiment_type == ExperimentType.CONSCIOUSNESS_TARGETING:
            self._create_consciousness_targeting_network()
        elif self.config.experiment_type == ExperimentType.MEMORIAL_FRAMEWORK:
            self._create_memorial_framework_network()
        else:
            self._create_generic_network()

        self._build_adjacency_matrix()
        logger.info(f"Created Bayesian Network with {len(self.nodes)} nodes")

    def _create_precision_timing_network(self):
        """Create Bayesian Network for precision timing experiments"""
        # Parameter nodes
        self.add_node("sampling_rate", NodeType.PARAMETER, DistributionType.GAMMA,
                     prior_params={'shape': 2.0, 'rate': 2e-6},
                     bounds=(1e3, 1e8))  # 1 kHz to 100 MHz

        self.add_node("oscillatory_coupling", NodeType.PARAMETER, DistributionType.BETA,
                     prior_params={'alpha': 2.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        self.add_node("quantum_enhancement", NodeType.PARAMETER, DistributionType.BETA,
                     prior_params={'alpha': 5.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        # Observable nodes
        self.add_node("measured_precision", NodeType.OBSERVABLE, DistributionType.GAMMA,
                     prior_params={'shape': 1.0, 'rate': 10.0},
                     bounds=(1e-12, 1e-6))  # picoseconds to microseconds

        self.add_node("measurement_latency", NodeType.OBSERVABLE, DistributionType.GAMMA,
                     prior_params={'shape': 2.0, 'rate': 1000.0},
                     bounds=(1e-6, 1e-2))  # microseconds to milliseconds

        # Latent nodes
        self.add_node("system_stability", NodeType.LATENT, DistributionType.BETA,
                     prior_params={'alpha': 3.0, 'beta': 1.0},
                     bounds=(0.0, 1.0))

        # Goal node
        self.add_node("precision_objective", NodeType.GOAL, DistributionType.NORMAL,
                     prior_params={'mean': 0.0, 'variance': 1.0})

        # Define dependencies (parent -> child relationships)
        self._add_edge("sampling_rate", "measured_precision")
        self._add_edge("oscillatory_coupling", "measured_precision")
        self._add_edge("quantum_enhancement", "measured_precision")
        self._add_edge("measured_precision", "precision_objective")
        self._add_edge("measurement_latency", "precision_objective")
        self._add_edge("system_stability", "precision_objective")

    def _create_oscillatory_analysis_network(self):
        """Create Bayesian Network for oscillatory analysis experiments"""
        # Parameter nodes for oscillatory framework
        self.add_node("frequency_coupling", NodeType.PARAMETER, DistributionType.BETA,
                     prior_params={'alpha': 3.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        self.add_node("oscillator_count", NodeType.PARAMETER, DistributionType.GAMMA,
                     prior_params={'shape': 3.0, 'rate': 0.003},
                     bounds=(100, 10000))

        self.add_node("convergence_threshold", NodeType.PARAMETER, DistributionType.GAMMA,
                     prior_params={'shape': 1.0, 'rate': 1e9},
                     bounds=(1e-12, 1e-6))

        # Observable nodes
        self.add_node("convergence_rate", NodeType.OBSERVABLE, DistributionType.BETA,
                     prior_params={'alpha': 2.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        self.add_node("oscillatory_stability", NodeType.OBSERVABLE, DistributionType.BETA,
                     prior_params={'alpha': 4.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        # Latent nodes
        self.add_node("multi_scale_coupling", NodeType.LATENT, DistributionType.BETA,
                     prior_params={'alpha': 3.0, 'beta': 1.5},
                     bounds=(0.0, 1.0))

        self.add_node("temporal_emergence", NodeType.LATENT, DistributionType.BETA,
                     prior_params={'alpha': 2.5, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        # Goal node
        self.add_node("oscillatory_objective", NodeType.GOAL, DistributionType.NORMAL,
                     prior_params={'mean': 0.0, 'variance': 1.0})

        # Define dependencies
        self._add_edge("frequency_coupling", "convergence_rate")
        self._add_edge("oscillator_count", "oscillatory_stability")
        self._add_edge("convergence_threshold", "convergence_rate")
        self._add_edge("frequency_coupling", "multi_scale_coupling")
        self._add_edge("multi_scale_coupling", "temporal_emergence")
        self._add_edge("convergence_rate", "oscillatory_objective")
        self._add_edge("oscillatory_stability", "oscillatory_objective")
        self._add_edge("temporal_emergence", "oscillatory_objective")

    def _create_consciousness_targeting_network(self):
        """Create Bayesian Network for consciousness targeting experiments"""
        # Parameter nodes
        self.add_node("population_size", NodeType.PARAMETER, DistributionType.GAMMA,
                     prior_params={'shape': 2.0, 'rate': 2e-4},
                     bounds=(1000, 100000))

        self.add_node("free_will_factor", NodeType.PARAMETER, DistributionType.BETA,
                     prior_params={'alpha': 2.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        self.add_node("nordic_paradox_strength", NodeType.PARAMETER, DistributionType.BETA,
                     prior_params={'alpha': 3.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        # Observable nodes
        self.add_node("targeting_accuracy", NodeType.OBSERVABLE, DistributionType.BETA,
                     prior_params={'alpha': 9.0, 'beta': 1.0},
                     bounds=(0.0, 1.0))

        self.add_node("consciousness_response", NodeType.OBSERVABLE, DistributionType.BETA,
                     prior_params={'alpha': 5.0, 'beta': 3.0},
                     bounds=(0.0, 1.0))

        # Latent nodes
        self.add_node("functional_delusion_index", NodeType.LATENT, DistributionType.BETA,
                     prior_params={'alpha': 4.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        self.add_node("death_proximity_signal", NodeType.LATENT, DistributionType.GAMMA,
                     prior_params={'shape': 1.5, 'rate': 2.0},
                     bounds=(0.0, 5.0))

        # Goal node
        self.add_node("consciousness_objective", NodeType.GOAL, DistributionType.NORMAL,
                     prior_params={'mean': 0.0, 'variance': 1.0})

        # Define dependencies
        self._add_edge("free_will_factor", "functional_delusion_index")
        self._add_edge("nordic_paradox_strength", "targeting_accuracy")
        self._add_edge("population_size", "consciousness_response")
        self._add_edge("functional_delusion_index", "targeting_accuracy")
        self._add_edge("death_proximity_signal", "consciousness_response")
        self._add_edge("targeting_accuracy", "consciousness_objective")
        self._add_edge("consciousness_response", "consciousness_objective")

    def _create_memorial_framework_network(self):
        """Create Bayesian Network for memorial framework experiments"""
        # Parameter nodes
        self.add_node("inheritance_efficiency", NodeType.PARAMETER, DistributionType.BETA,
                     prior_params={'alpha': 9.0, 'beta': 1.0},
                     bounds=(0.0, 1.0))

        self.add_node("capitalism_elimination_rate", NodeType.PARAMETER, DistributionType.BETA,
                     prior_params={'alpha': 19.0, 'beta': 1.0},
                     bounds=(0.0, 1.0))

        self.add_node("temporal_persistence", NodeType.PARAMETER, DistributionType.BETA,
                     prior_params={'alpha': 49.0, 'beta': 1.0},
                     bounds=(0.0, 1.0))

        # Observable nodes
        self.add_node("memorial_effectiveness", NodeType.OBSERVABLE, DistributionType.BETA,
                     prior_params={'alpha': 9.0, 'beta': 1.0},
                     bounds=(0.0, 1.0))

        self.add_node("cost_reduction", NodeType.OBSERVABLE, DistributionType.BETA,
                     prior_params={'alpha': 99.0, 'beta': 1.0},
                     bounds=(0.0, 1.0))

        # Latent nodes
        self.add_node("buhera_model_performance", NodeType.LATENT, DistributionType.BETA,
                     prior_params={'alpha': 9.0, 'beta': 1.0},
                     bounds=(0.0, 1.0))

        self.add_node("consciousness_transfer_rate", NodeType.LATENT, DistributionType.BETA,
                     prior_params={'alpha': 8.0, 'beta': 2.0},
                     bounds=(0.0, 1.0))

        # Goal node
        self.add_node("memorial_objective", NodeType.GOAL, DistributionType.NORMAL,
                     prior_params={'mean': 0.0, 'variance': 1.0})

        # Define dependencies
        self._add_edge("inheritance_efficiency", "consciousness_transfer_rate")
        self._add_edge("capitalism_elimination_rate", "buhera_model_performance")
        self._add_edge("temporal_persistence", "memorial_effectiveness")
        self._add_edge("consciousness_transfer_rate", "memorial_effectiveness")
        self._add_edge("buhera_model_performance", "cost_reduction")
        self._add_edge("memorial_effectiveness", "memorial_objective")
        self._add_edge("cost_reduction", "memorial_objective")

    def _create_generic_network(self):
        """Create a generic Bayesian Network for multi-objective experiments"""
        # Generic parameter nodes
        for i in range(5):
            self.add_node(f"param_{i}", NodeType.PARAMETER, DistributionType.NORMAL,
                         prior_params={'mean': 0.0, 'variance': 1.0},
                         bounds=(-5.0, 5.0))

        # Generic observable nodes
        for i in range(3):
            self.add_node(f"observable_{i}", NodeType.OBSERVABLE, DistributionType.NORMAL,
                         prior_params={'mean': 0.0, 'variance': 1.0})

        # Generic latent nodes
        for i in range(2):
            self.add_node(f"latent_{i}", NodeType.LATENT, DistributionType.NORMAL,
                         prior_params={'mean': 0.0, 'variance': 1.0})

        # Goal node
        self.add_node("generic_objective", NodeType.GOAL, DistributionType.NORMAL,
                     prior_params={'mean': 0.0, 'variance': 1.0})

        # Create some dependencies
        self._add_edge("param_0", "observable_0")
        self._add_edge("param_1", "observable_1")
        self._add_edge("param_2", "latent_0")
        self._add_edge("latent_0", "generic_objective")
        self._add_edge("observable_0", "generic_objective")

    def add_node(self, name: str, node_type: NodeType, distribution_type: DistributionType,
                 prior_params: Dict[str, float] = None, bounds: Tuple[float, float] = (-np.inf, np.inf)):
        """Add a node to the Bayesian Network"""
        if prior_params is None:
            prior_params = {'mean': 0.0, 'variance': 1.0}

        node = BayesianNode(
            name=name,
            node_type=node_type,
            distribution_type=distribution_type,
            prior_params=prior_params,
            bounds=bounds
        )
        self.nodes[name] = node

    def _add_edge(self, parent: str, child: str):
        """Add an edge (dependency) between two nodes"""
        if parent in self.nodes and child in self.nodes:
            self.nodes[parent].children.append(child)
            self.nodes[child].parents.append(parent)

    def _build_adjacency_matrix(self):
        """Build adjacency matrix for the Bayesian Network"""
        node_names = list(self.nodes.keys())
        n_nodes = len(node_names)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes))

        for i, parent_name in enumerate(node_names):
            for child_name in self.nodes[parent_name].children:
                j = node_names.index(child_name)
                self.adjacency_matrix[i, j] = 1

    def propagate_evidence(self, evidence: Dict[str, float]):
        """Propagate evidence through the Bayesian Network"""
        logger.info(f"Propagating evidence: {evidence}")

        # Add observations to nodes
        for node_name, observation in evidence.items():
            if node_name in self.nodes:
                self.nodes[node_name].add_observation(observation)

        # Perform message passing (simplified)
        self._forward_propagation(evidence)
        self._backward_propagation()

    def _forward_propagation(self, evidence: Dict[str, float]):
        """Forward message passing from parents to children"""
        node_names = list(self.nodes.keys())

        # Topological sort for proper propagation order
        for node_name in node_names:
            node = self.nodes[node_name]

            if node.parents:
                # Collect parent influences
                parent_influences = []
                for parent_name in node.parents:
                    parent_node = self.nodes[parent_name]
                    if parent_node.current_value is not None:
                        parent_influences.append(parent_node.current_value)

                if parent_influences:
                    # Simple influence combination (can be made more sophisticated)
                    combined_influence = np.mean(parent_influences)

                    # Add parent influence as pseudo-observation
                    node.add_observation(combined_influence, weight=0.5)

    def _backward_propagation(self):
        """Backward message passing from children to parents"""
        # Simplified backward pass - in practice, this would involve
        # more sophisticated message passing algorithms
        pass

    def evaluate_objective(self, parameters: Dict[str, float]) -> float:
        """Evaluate the objective function given parameter values"""
        # Set parameter values
        for param_name, value in parameters.items():
            if param_name in self.nodes:
                self.nodes[param_name].current_value = value

        # Propagate through network
        self.propagate_evidence(parameters)

        # Calculate objective based on goal nodes
        objective_value = 0.0
        goal_nodes = [node for node in self.nodes.values() if node.node_type == NodeType.GOAL]

        if goal_nodes:
            goal_values = []
            for goal_node in goal_nodes:
                if goal_node.current_value is not None:
                    goal_values.append(goal_node.current_value)
                else:
                    # Sample from posterior if no current value
                    goal_values.append(goal_node.sample())

            objective_value = np.mean(goal_values)

            # Apply optimization goal logic
            if self.config.optimization.primary_goal in [OptimizationGoal.MINIMIZE_LATENCY,
                                                       OptimizationGoal.MINIMIZE_ERROR,
                                                       OptimizationGoal.MINIMIZE_RESOURCE_USAGE]:
                objective_value = -objective_value  # Convert to maximization

        return objective_value

    async def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment with Bayesian optimization"""
        logger.info(f"Starting experiment: {self.experiment_id}")
        self.is_running = True
        start_time = time.time()

        try:
            # Initialize experiment
            await self._initialize_experiment()

            # Run optimization loop
            optimization_result = await self._run_optimization_loop()

            # Finalize experiment
            results = await self._finalize_experiment(optimization_result)

            execution_time = time.time() - start_time
            results['execution_time_seconds'] = execution_time

            logger.info(f"Experiment completed in {execution_time:.2f} seconds")
            return results

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
        finally:
            self.is_running = False

    async def _initialize_experiment(self):
        """Initialize experiment setup"""
        logger.info("Initializing experiment...")

        # Validate configuration
        is_valid, errors = self.config.validate_configuration()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")

        # Initialize results structure
        self.results = {
            'experiment_id': self.experiment_id,
            'experiment_type': self.config.experiment_type.name,
            'timestamp': self.timestamp.isoformat(),
            'configuration': self.config.to_dict(),
            'bayesian_network': {
                'nodes': len(self.nodes),
                'node_types': {nt.name: sum(1 for n in self.nodes.values() if n.node_type == nt)
                              for nt in NodeType}
            }
        }

        # Log experiment start
        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'experiment_initialized',
            'details': {'node_count': len(self.nodes)}
        })

    async def _run_optimization_loop(self) -> Dict[str, Any]:
        """Run the main optimization loop"""
        logger.info("Starting optimization loop...")

        best_params = {}
        best_objective = -np.inf

        # Get parameter bounds
        param_nodes = {name: node for name, node in self.nodes.items()
                      if node.node_type == NodeType.PARAMETER}

        # Define objective function for optimization
        def objective_function(x):
            params = {name: x[i] for i, name in enumerate(param_nodes.keys())}
            return -self.evaluate_objective(params)  # Negative for minimization

        # Set up bounds for optimization
        bounds = [node.bounds for node in param_nodes.values()]

        # Run optimization
        if len(param_nodes) > 0:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=self.config.optimization.budget_iterations,
                popsize=min(15, 3 * len(param_nodes)),
                atol=self.config.optimization.target_improvement_threshold,
                seed=self.config.validation.reproducibility_seed
            )

            best_params = {name: result.x[i] for i, name in enumerate(param_nodes.keys())}
            best_objective = -result.fun  # Convert back to maximization

            logger.info(f"Optimization completed. Best objective: {best_objective:.6f}")

        return {
            'best_parameters': best_params,
            'best_objective_value': best_objective,
            'convergence_history': self.convergence_history,
            'optimization_success': True
        }

    async def _finalize_experiment(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize experiment and prepare results"""
        logger.info("Finalizing experiment...")

        # Update results with optimization outcome
        self.results.update(optimization_result)

        # Extract final network state
        self.results['final_network_state'] = self._extract_network_state()

        # Calculate summary statistics
        self.results['summary_statistics'] = self._calculate_summary_statistics()

        # Save results if configured
        if self.config.data.save_intermediate_results:
            await self._save_results()

        # Log experiment completion
        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'experiment_completed',
            'details': {
                'best_objective': optimization_result.get('best_objective_value', 0),
                'success': optimization_result.get('optimization_success', False)
            }
        })

        self.results['experiment_log'] = self.experiment_log

        return self.results

    def _extract_network_state(self) -> Dict[str, Any]:
        """Extract current state of the Bayesian Network"""
        network_state = {}

        for node_name, node in self.nodes.items():
            network_state[node_name] = {
                'node_type': node.node_type.name,
                'distribution_type': node.distribution_type.name,
                'current_value': node.current_value,
                'uncertainty': node.uncertainty,
                'posterior_mean': node.get_posterior_mean(),
                'posterior_std': node.get_posterior_std(),
                'observation_count': len(node.observations),
                'parents': node.parents,
                'children': node.children
            }

        return network_state

    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for the experiment"""
        param_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.PARAMETER]
        observable_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.OBSERVABLE]
        goal_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.GOAL]

        return {
            'parameter_statistics': {
                'count': len(param_nodes),
                'mean_uncertainty': np.mean([n.uncertainty or 0 for n in param_nodes]),
                'parameters_converged': sum(1 for n in param_nodes if (n.uncertainty or 1) < 0.1)
            },
            'observable_statistics': {
                'count': len(observable_nodes),
                'mean_observations': np.mean([len(n.observations) for n in observable_nodes]),
                'total_observations': sum(len(n.observations) for n in observable_nodes)
            },
            'goal_statistics': {
                'count': len(goal_nodes),
                'mean_goal_value': np.mean([n.current_value or 0 for n in goal_nodes]),
                'goal_uncertainty': np.mean([n.uncertainty or 0 for n in goal_nodes])
            },
            'network_complexity': {
                'total_nodes': len(self.nodes),
                'total_edges': np.sum(self.adjacency_matrix) if self.adjacency_matrix is not None else 0,
                'avg_node_degree': np.mean(np.sum(self.adjacency_matrix, axis=1)) if self.adjacency_matrix is not None else 0
            }
        }

    async def _save_results(self):
        """Save experiment results to file"""
        output_dir = Path(self.config.data.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_dir / f"{self.experiment_id}_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {json_path}")

    def get_node_posterior_samples(self, node_name: str, n_samples: int = 1000) -> np.ndarray:
        """Get posterior samples from a specific node"""
        if node_name in self.nodes:
            return self.nodes[node_name].sample(n_samples)
        else:
            raise ValueError(f"Node '{node_name}' not found in network")

    def get_network_summary(self) -> Dict[str, Any]:
        """Get a summary of the current network state"""
        return {
            'experiment_id': self.experiment_id,
            'experiment_type': self.config.experiment_type.name,
            'network_structure': {
                'total_nodes': len(self.nodes),
                'node_types': {nt.name: sum(1 for n in self.nodes.values() if n.node_type == nt)
                              for nt in NodeType},
                'distribution_types': {dt.name: sum(1 for n in self.nodes.values() if n.distribution_type == dt)
                                     for dt in DistributionType}
            },
            'current_state': {name: {
                'value': node.current_value,
                'uncertainty': node.uncertainty,
                'observations': len(node.observations)
            } for name, node in self.nodes.items()},
            'is_running': self.is_running,
            'current_iteration': self.current_iteration,
            'best_objective': self.best_objective_value
        }


# Factory function for creating experiment orchestrators
def create_experiment_orchestrator(experiment_type: ExperimentType,
                                 config_file: Optional[str] = None) -> ExperimentOrchestrator:
    """Create an experiment orchestrator for the specified experiment type"""

    if config_file:
        config = ExperimentConfig(config_file)
    else:
        config = ExperimentConfig()
        config.experiment_type = experiment_type

        # Set appropriate presets
        if experiment_type == ExperimentType.PRECISION_TIMING:
            config.create_experiment_preset("high_precision_timing")
        elif experiment_type == ExperimentType.CONSCIOUSNESS_TARGETING:
            config.create_experiment_preset("consciousness_research")
        elif experiment_type == ExperimentType.OSCILLATORY_ANALYSIS:
            config.create_experiment_preset("oscillatory_framework")
        elif experiment_type == ExperimentType.MEMORIAL_FRAMEWORK:
            config.create_experiment_preset("memorial_optimization")
        else:
            config.create_experiment_preset("multi_objective")

    return ExperimentOrchestrator(config)


# Convenience functions for running experiments
async def run_precision_timing_experiment(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Run a precision timing experiment"""
    orchestrator = create_experiment_orchestrator(ExperimentType.PRECISION_TIMING, config_file)
    return await orchestrator.run_experiment()


async def run_consciousness_targeting_experiment(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Run a consciousness targeting experiment"""
    orchestrator = create_experiment_orchestrator(ExperimentType.CONSCIOUSNESS_TARGETING, config_file)
    return await orchestrator.run_experiment()


async def run_oscillatory_analysis_experiment(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Run an oscillatory analysis experiment"""
    orchestrator = create_experiment_orchestrator(ExperimentType.OSCILLATORY_ANALYSIS, config_file)
    return await orchestrator.run_experiment()


async def run_memorial_framework_experiment(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Run a memorial framework experiment"""
    orchestrator = create_experiment_orchestrator(ExperimentType.MEMORIAL_FRAMEWORK, config_file)
    return await orchestrator.run_experiment()


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def main():
        """Example usage of the experiment orchestrator"""

        # Create and run a precision timing experiment
        print("Running precision timing experiment...")
        orchestrator = create_experiment_orchestrator(ExperimentType.PRECISION_TIMING)

        # Print network summary
        print("\nBayesian Network Summary:")
        summary = orchestrator.get_network_summary()
        print(f"Nodes: {summary['network_structure']['total_nodes']}")
        print(f"Node types: {summary['network_structure']['node_types']}")

        # Run experiment
        results = await orchestrator.run_experiment()

        print(f"\nExperiment completed!")
        print(f"Best objective value: {results.get('best_objective_value', 'N/A'):.6f}")
        print(f"Execution time: {results.get('execution_time_seconds', 0):.2f} seconds")

        # Show best parameters
        if 'best_parameters' in results:
            print("\nBest parameters found:")
            for param, value in results['best_parameters'].items():
                print(f"  {param}: {value:.6f}")

    # Run the example
    asyncio.run(main())
