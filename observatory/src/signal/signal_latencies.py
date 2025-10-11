"""
Signal Latencies - Network and Machine Error Analysis

Determines the error in any received time signal, calculating total latency
from network effects to machine effects. Implements temporal coordination
framework with cryptographic properties and precision-by-difference calculations.

Implementation Architecture:
- Network Layer Integration (TCP/IP, UDP, HTTP)
- Temporal Coordination Layer
- Temporal Fragmentation
- Precision-by-Difference Calculator
- Preemptive State Generator
"""
import math

import numpy as np
import time
import socket
import struct
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor


class LatencyComponent(Enum):
    """Components of total signal latency"""
    PROCESSING = "processing_latency"        # L_processing
    TRANSMISSION = "transmission_latency"    # L_transmission
    PROPAGATION = "propagation_latency"      # L_propagation
    QUEUING = "queuing_latency"             # L_queuing
    NETWORK_STACK = "network_stack_latency"  # OS network stack
    HARDWARE = "hardware_latency"            # Hardware/driver latency
    TEMPORAL_COORDINATION = "temporal_coordination"  # L_temporal_coordination
    PREDICTION_ERROR = "prediction_error"    # L_prediction_error


class NetworkProtocol(Enum):
    """Network protocols for temporal coordination"""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    NTP = "ntp"
    TEMPORAL_FRAGMENT = "temporal_fragment"


class TemporalSecurity(Enum):
    """Temporal cryptographic security levels"""
    LOW = "low_security"        # < 10 temporal fragments
    MEDIUM = "medium_security"  # 10-100 temporal fragments
    HIGH = "high_security"      # 100-1000 temporal fragments
    MAXIMUM = "maximum_security"  # > 1000 temporal fragments


@dataclass
class LatencyMeasurement:
    """Single latency measurement with component breakdown"""
    measurement_id: str
    source_address: str
    destination_address: str
    protocol: NetworkProtocol
    timestamp_start: float
    timestamp_end: float
    total_latency: float
    component_latencies: Dict[LatencyComponent, float] = field(default_factory=dict)
    packet_size: int = 0
    network_conditions: Dict = field(default_factory=dict)

    def get_component_latency(self, component: LatencyComponent) -> float:
        """Get specific component latency"""
        return self.component_latencies.get(component, 0.0)

    def calculate_total_from_components(self) -> float:
        """Calculate total latency from components"""
        return sum(self.component_latencies.values())


@dataclass
class TemporalFragment:
    """Fragment of temporal message for cryptographic security"""
    fragment_id: str
    fragment_index: int
    total_fragments: int
    temporal_coordinate: float
    data_payload: bytes
    entropy_content: float
    security_level: TemporalSecurity
    creation_timestamp: float

    def calculate_reconstruction_probability(self, available_fragments: int) -> float:
        """Calculate probability of message reconstruction"""
        if available_fragments >= self.total_fragments:
            return 1.0

        # Probability bounded by (k/n)^H(M) where k=available, n=total, H=entropy
        k = available_fragments
        n = self.total_fragments
        return (k / n) ** self.entropy_content if n > 0 else 0.0


@dataclass
class PrecisionByDifferenceResult:
    """Result of precision-by-difference calculation"""
    reference_time: float
    calculated_time: float
    time_difference: float
    precision_improvement: float
    uncertainty_reduction: float
    calculation_method: str
    confidence_level: float


class SignalLatencyAnalyzer:
    """
    Signal Latency Analyzer for Network and Machine Error Analysis

    Implements comprehensive latency analysis including:
    - Traditional latency components (processing, transmission, propagation, queuing)
    - Temporal coordination layer latency
    - Precision-by-difference calculations
    - Temporal cryptographic security analysis
    - Real-time latency monitoring and prediction

    Latency Formula:
    L_traditional = L_processing + L_transmission + L_propagation + L_queuing
    L_sango = L_prediction_error + L_temporal_coordination
    """

    def __init__(self):
        self.latency_measurements: List[LatencyMeasurement] = []
        self.temporal_fragments: Dict[str, List[TemporalFragment]] = {}
        self.precision_calculations: List[PrecisionByDifferenceResult] = []

        # Network monitoring
        self.network_interfaces = self._get_network_interfaces()
        self.baseline_latencies: Dict[str, float] = {}

        # System performance monitoring
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'network_utilization': 0.0,
            'disk_io_wait': 0.0
        }

        # Threading for continuous monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

        # Latency prediction model parameters
        self.prediction_model = {
            'baseline_processing': 1e-6,    # 1 microsecond base processing
            'network_overhead': 1e-4,       # 0.1ms network overhead
            'congestion_factor': 1.0,       # Multiplier for network congestion
            'system_load_factor': 1.0       # Multiplier for system load
        }

        # Temporal coordination parameters
        self.temporal_coordination_config = {
            'fragment_size_bytes': 1024,
            'max_fragments_per_message': 1000,
            'temporal_distribution_interval': 1e-6,  # 1 microsecond intervals
            'cryptographic_entropy_threshold': 2.0
        }

    def _get_network_interfaces(self) -> Dict[str, Dict]:
        """Get available network interfaces and their properties"""
        interfaces = {}

        try:
            net_interfaces = psutil.net_if_addrs()
            net_stats = psutil.net_if_stats()

            for interface_name, addresses in net_interfaces.items():
                if interface_name in net_stats:
                    interface_info = {
                        'addresses': [addr.address for addr in addresses],
                        'is_up': net_stats[interface_name].isup,
                        'speed': net_stats[interface_name].speed,  # Mbps
                        'mtu': net_stats[interface_name].mtu
                    }
                    interfaces[interface_name] = interface_info
        except:
            # Fallback for systems without full psutil support
            interfaces['default'] = {
                'addresses': ['127.0.0.1'],
                'is_up': True,
                'speed': 1000,  # Assume 1 Gbps
                'mtu': 1500
            }

        return interfaces

    def measure_network_latency(self,
                               target_host: str,
                               target_port: int,
                               protocol: NetworkProtocol = NetworkProtocol.TCP,
                               packet_size: int = 64,
                               timeout: float = 5.0) -> LatencyMeasurement:
        """Measure comprehensive network latency with component breakdown"""

        measurement_id = f"{target_host}_{target_port}_{int(time.time() * 1000)}"
        start_time = time.time()

        # Initialize latency components
        component_latencies = {}

        try:
            # Measure different latency components
            if protocol == NetworkProtocol.TCP:
                total_latency, components = self._measure_tcp_latency(
                    target_host, target_port, packet_size, timeout
                )
            elif protocol == NetworkProtocol.UDP:
                total_latency, components = self._measure_udp_latency(
                    target_host, target_port, packet_size, timeout
                )
            elif protocol == NetworkProtocol.HTTP:
                total_latency, components = self._measure_http_latency(
                    target_host, target_port, timeout
                )
            else:
                # Generic measurement
                total_latency, components = self._measure_generic_latency(
                    target_host, target_port, timeout
                )

            component_latencies.update(components)

        except Exception as e:
            total_latency = timeout  # Maximum latency on error
            component_latencies[LatencyComponent.PROCESSING] = timeout

        end_time = time.time()

        # Add system-level latency components
        component_latencies.update(self._measure_system_latencies())

        # Get current network conditions
        network_conditions = self._assess_network_conditions()

        measurement = LatencyMeasurement(
            measurement_id=measurement_id,
            source_address=self._get_local_ip(),
            destination_address=target_host,
            protocol=protocol,
            timestamp_start=start_time,
            timestamp_end=end_time,
            total_latency=total_latency,
            component_latencies=component_latencies,
            packet_size=packet_size,
            network_conditions=network_conditions
        )

        self.latency_measurements.append(measurement)
        return measurement

    def _measure_tcp_latency(self, host: str, port: int, size: int, timeout: float) -> Tuple[float, Dict]:
        """Measure TCP-specific latency components"""

        start_time = time.perf_counter()
        components = {}

        try:
            # Create socket (processing latency)
            socket_start = time.perf_counter()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            socket_creation_time = time.perf_counter() - socket_start
            components[LatencyComponent.PROCESSING] = socket_creation_time

            # DNS resolution (if needed)
            dns_start = time.perf_counter()
            try:
                addr_info = socket.getaddrinfo(host, port)
                dns_time = time.perf_counter() - dns_start
                components[LatencyComponent.NETWORK_STACK] = dns_time
            except:
                dns_time = 0.0
                components[LatencyComponent.NETWORK_STACK] = 0.0

            # TCP connection establishment (propagation + transmission)
            connect_start = time.perf_counter()
            sock.connect((host, port))
            connect_time = time.perf_counter() - connect_start
            components[LatencyComponent.PROPAGATION] = connect_time * 0.7  # Estimate
            components[LatencyComponent.TRANSMISSION] = connect_time * 0.3  # Estimate

            # Send/receive data
            if size > 0:
                data_start = time.perf_counter()
                test_data = b'X' * size
                sock.send(test_data)
                response = sock.recv(1024)
                data_time = time.perf_counter() - data_start
                components[LatencyComponent.QUEUING] = data_time

            sock.close()

        except Exception as e:
            components[LatencyComponent.PROCESSING] = timeout

        total_time = time.perf_counter() - start_time
        return total_time, components

    def _measure_udp_latency(self, host: str, port: int, size: int, timeout: float) -> Tuple[float, Dict]:
        """Measure UDP-specific latency components"""

        start_time = time.perf_counter()
        components = {}

        try:
            # Create UDP socket
            socket_start = time.perf_counter()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            socket_creation_time = time.perf_counter() - socket_start
            components[LatencyComponent.PROCESSING] = socket_creation_time

            # Send UDP packet
            send_start = time.perf_counter()
            test_data = b'X' * size
            sock.sendto(test_data, (host, port))

            # Try to receive response (may timeout)
            try:
                response, addr = sock.recvfrom(1024)
                receive_time = time.perf_counter() - send_start
                components[LatencyComponent.TRANSMISSION] = receive_time * 0.5
                components[LatencyComponent.PROPAGATION] = receive_time * 0.5
            except socket.timeout:
                # No response, estimate one-way latency
                send_time = time.perf_counter() - send_start
                components[LatencyComponent.TRANSMISSION] = send_time
                components[LatencyComponent.PROPAGATION] = send_time

            sock.close()

        except Exception as e:
            components[LatencyComponent.PROCESSING] = timeout

        total_time = time.perf_counter() - start_time
        return total_time, components

    def _measure_http_latency(self, host: str, port: int, timeout: float) -> Tuple[float, Dict]:
        """Measure HTTP-specific latency components"""

        start_time = time.perf_counter()
        components = {}

        try:
            import urllib.request
            import urllib.error

            # DNS and connection establishment
            dns_start = time.perf_counter()
            url = f"http://{host}:{port}/" if port != 80 else f"http://{host}/"

            # Make HTTP request
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request, timeout=timeout)

            dns_connect_time = time.perf_counter() - dns_start
            components[LatencyComponent.NETWORK_STACK] = dns_connect_time * 0.2
            components[LatencyComponent.PROPAGATION] = dns_connect_time * 0.4
            components[LatencyComponent.TRANSMISSION] = dns_connect_time * 0.4

            # Read response
            read_start = time.perf_counter()
            data = response.read()
            read_time = time.perf_counter() - read_start
            components[LatencyComponent.PROCESSING] = read_time

        except Exception as e:
            components[LatencyComponent.PROCESSING] = timeout

        total_time = time.perf_counter() - start_time
        return total_time, components

    def _measure_generic_latency(self, host: str, port: int, timeout: float) -> Tuple[float, Dict]:
        """Generic latency measurement fallback"""

        start_time = time.perf_counter()

        # Simulate network round-trip
        time.sleep(0.001)  # 1ms simulated network delay

        end_time = time.perf_counter()
        total_time = end_time - start_time

        components = {
            LatencyComponent.PROCESSING: total_time * 0.1,
            LatencyComponent.TRANSMISSION: total_time * 0.3,
            LatencyComponent.PROPAGATION: total_time * 0.4,
            LatencyComponent.QUEUING: total_time * 0.2
        }

        return total_time, components

    def _measure_system_latencies(self) -> Dict[LatencyComponent, float]:
        """Measure system-level latency components"""

        components = {}

        try:
            # CPU-based processing latency
            cpu_percent = psutil.cpu_percent(interval=0.01)
            base_processing = self.prediction_model['baseline_processing']
            cpu_factor = 1.0 + (cpu_percent / 100.0)
            components[LatencyComponent.PROCESSING] = base_processing * cpu_factor

            # Memory-based latency
            memory_info = psutil.virtual_memory()
            memory_factor = 1.0 + (memory_info.percent / 100.0) * 0.5
            components[LatencyComponent.HARDWARE] = base_processing * memory_factor * 0.5

            # Network stack latency
            net_io = psutil.net_io_counters()
            if hasattr(net_io, 'packets_sent') and net_io.packets_sent > 0:
                network_load = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
                network_factor = 1.0 + min(network_load / 1000.0, 1.0)  # Cap at 2x
                components[LatencyComponent.NETWORK_STACK] = base_processing * network_factor
            else:
                components[LatencyComponent.NETWORK_STACK] = base_processing

        except Exception as e:
            # Fallback values
            components[LatencyComponent.PROCESSING] = self.prediction_model['baseline_processing']
            components[LatencyComponent.HARDWARE] = self.prediction_model['baseline_processing'] * 0.5
            components[LatencyComponent.NETWORK_STACK] = self.prediction_model['baseline_processing']

        return components

    def _assess_network_conditions(self) -> Dict:
        """Assess current network conditions"""

        conditions = {}

        try:
            # Network I/O statistics
            net_io = psutil.net_io_counters()
            conditions['bytes_sent'] = net_io.bytes_sent
            conditions['bytes_recv'] = net_io.bytes_recv
            conditions['packets_sent'] = net_io.packets_sent
            conditions['packets_recv'] = net_io.packets_recv
            conditions['errin'] = net_io.errin
            conditions['errout'] = net_io.errout
            conditions['dropin'] = net_io.dropin
            conditions['dropout'] = net_io.dropout

            # Calculate utilization
            if hasattr(net_io, 'bytes_sent'):
                total_bytes = net_io.bytes_sent + net_io.bytes_recv
                conditions['total_bytes'] = total_bytes

                # Estimate bandwidth utilization (simplified)
                max_interface_speed = max(
                    (iface.get('speed', 0) for iface in self.network_interfaces.values()),
                    default=1000  # 1 Gbps default
                )
                max_bytes_per_second = max_interface_speed * 1024 * 1024 / 8  # Convert to bytes/s
                conditions['estimated_utilization'] = min(1.0, total_bytes / max_bytes_per_second)

        except Exception as e:
            conditions['error'] = str(e)

        return conditions

    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "127.0.0.1"

    def calculate_precision_by_difference(self,
                                        reference_timestamps: List[float],
                                        test_timestamps: List[float],
                                        method: str = "statistical") -> PrecisionByDifferenceResult:
        """
        Calculate precision-by-difference for time signal validation

        Implements precision enhancement through comparative analysis rather than
        absolute time determination.
        """

        if len(reference_timestamps) != len(test_timestamps):
            raise ValueError("Reference and test timestamp arrays must have same length")

        if len(reference_timestamps) < 2:
            raise ValueError("Need at least 2 timestamps for difference calculation")

        # Calculate differences
        time_differences = [test - ref for test, ref in zip(test_timestamps, reference_timestamps)]

        if method == "statistical":
            # Statistical analysis of differences
            mean_difference = statistics.mean(time_differences)
            std_difference = statistics.stdev(time_differences) if len(time_differences) > 1 else 0.0

            # Precision improvement calculation
            reference_precision = statistics.stdev(reference_timestamps) if len(reference_timestamps) > 1 else 0.001
            test_precision = statistics.stdev(test_timestamps) if len(test_timestamps) > 1 else 0.001

            precision_improvement = reference_precision / test_precision if test_precision > 0 else 1.0
            uncertainty_reduction = 1.0 - (std_difference / reference_precision) if reference_precision > 0 else 0.0

            confidence = max(0.0, min(1.0, 1.0 - (std_difference / abs(mean_difference)) if mean_difference != 0 else 0.5))

        elif method == "kalman":
            # Kalman filter-based precision estimation
            # Simplified implementation
            process_noise = 1e-9  # 1 nanosecond process noise
            measurement_noise = np.var(time_differences) if len(time_differences) > 1 else 1e-6

            # State: [bias, drift_rate]
            state = np.array([np.mean(time_differences), 0.0])
            covariance = np.diag([1e-6, 1e-9])  # Initial uncertainty

            for i in range(1, len(time_differences)):
                dt = test_timestamps[i] - test_timestamps[i-1]

                # Predict
                F = np.array([[1.0, dt], [0.0, 1.0]])
                Q = np.diag([process_noise, process_noise/100])

                state = F @ state
                covariance = F @ covariance @ F.T + Q

                # Update
                H = np.array([[1.0, 0.0]])
                innovation = time_differences[i] - H @ state
                S = H @ covariance @ H.T + measurement_noise
                K = covariance @ H.T / S

                state = state + K * innovation
                covariance = (np.eye(2) - K @ H) @ covariance

            mean_difference = float(state[0])
            std_difference = math.sqrt(float(covariance[0, 0]))

            precision_improvement = 1.0 / std_difference if std_difference > 0 else 1.0
            uncertainty_reduction = 1.0 - std_difference
            confidence = 0.9  # High confidence for Kalman filter

        else:  # simple method
            mean_difference = np.mean(time_differences)
            std_difference = np.std(time_differences)
            precision_improvement = 1.0
            uncertainty_reduction = 0.5
            confidence = 0.7

        result = PrecisionByDifferenceResult(
            reference_time=statistics.mean(reference_timestamps),
            calculated_time=statistics.mean(test_timestamps),
            time_difference=mean_difference,
            precision_improvement=precision_improvement,
            uncertainty_reduction=max(0.0, min(1.0, uncertainty_reduction)),
            calculation_method=method,
            confidence_level=confidence
        )

        self.precision_calculations.append(result)
        return result

    def create_temporal_fragments(self,
                                message: bytes,
                                security_level: TemporalSecurity = TemporalSecurity.MEDIUM,
                                temporal_distribution_seconds: float = 1.0) -> List[TemporalFragment]:
        """
        Create temporal fragments for cryptographic security

        Implements temporal fragmentation where message security increases
        exponentially with number of fragments distributed over time.
        """

        # Determine number of fragments based on security level
        fragment_counts = {
            TemporalSecurity.LOW: 10,
            TemporalSecurity.MEDIUM: 100,
            TemporalSecurity.HIGH: 500,
            TemporalSecurity.MAXIMUM: 1000
        }

        num_fragments = fragment_counts.get(security_level, 100)
        fragment_size = max(1, len(message) // num_fragments)

        fragments = []
        base_time = time.time()

        # Calculate message entropy for reconstruction probability
        message_entropy = self._calculate_message_entropy(message)

        for i in range(num_fragments):
            start_idx = i * fragment_size
            end_idx = min((i + 1) * fragment_size, len(message))

            if start_idx >= len(message):
                break

            fragment_data = message[start_idx:end_idx]

            # Temporal coordinate for this fragment
            temporal_coordinate = base_time + (i * temporal_distribution_seconds / num_fragments)

            fragment = TemporalFragment(
                fragment_id=f"frag_{int(base_time * 1000)}_{i}",
                fragment_index=i,
                total_fragments=num_fragments,
                temporal_coordinate=temporal_coordinate,
                data_payload=fragment_data,
                entropy_content=message_entropy,
                security_level=security_level,
                creation_timestamp=time.time()
            )

            fragments.append(fragment)

        # Store fragments for reconstruction tracking
        message_id = f"msg_{int(base_time * 1000)}"
        self.temporal_fragments[message_id] = fragments

        return fragments

    def _calculate_message_entropy(self, message: bytes) -> float:
        """Calculate Shannon entropy of message"""

        if not message:
            return 0.0

        # Count byte frequencies
        byte_counts = {}
        for byte in message:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1

        # Calculate entropy
        total_bytes = len(message)
        entropy = 0.0

        for count in byte_counts.values():
            probability = count / total_bytes
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def analyze_temporal_cryptographic_security(self, message_id: str, available_fragments: int) -> Dict:
        """Analyze cryptographic security of temporal fragmentation"""

        if message_id not in self.temporal_fragments:
            return {'error': 'Message not found'}

        fragments = self.temporal_fragments[message_id]
        total_fragments = len(fragments)

        if total_fragments == 0:
            return {'error': 'No fragments found'}

        # Get representative fragment for analysis
        sample_fragment = fragments[0]

        # Calculate reconstruction probability
        reconstruction_prob = sample_fragment.calculate_reconstruction_probability(available_fragments)

        # Calculate security strength
        security_strength = -math.log2(reconstruction_prob) if reconstruction_prob > 0 else float('inf')

        # Time-based security analysis
        current_time = time.time()
        fragments_due = sum(1 for frag in fragments if current_time >= frag.temporal_coordinate)
        time_based_availability = fragments_due / total_fragments

        # Cryptographic analysis
        entropy_per_fragment = sample_fragment.entropy_content
        total_entropy = entropy_per_fragment * total_fragments

        return {
            'message_id': message_id,
            'total_fragments': total_fragments,
            'available_fragments': available_fragments,
            'fragments_due_by_time': fragments_due,
            'reconstruction_probability': reconstruction_prob,
            'security_strength_bits': security_strength,
            'security_level': sample_fragment.security_level.value,
            'time_based_availability': time_based_availability,
            'entropy_analysis': {
                'entropy_per_fragment': entropy_per_fragment,
                'total_message_entropy': total_entropy,
                'entropy_threshold': self.temporal_coordination_config['cryptographic_entropy_threshold']
            },
            'temporal_distribution': {
                'start_time': min(frag.temporal_coordinate for frag in fragments),
                'end_time': max(frag.temporal_coordinate for frag in fragments),
                'distribution_span': max(frag.temporal_coordinate for frag in fragments) - min(frag.temporal_coordinate for frag in fragments)
            }
        }

    def predict_latency(self,
                       target_host: str,
                       protocol: NetworkProtocol,
                       packet_size: int = 64) -> Dict[LatencyComponent, float]:
        """Predict latency components for target host"""

        # Get baseline latency if available
        baseline_key = f"{target_host}_{protocol.value}"
        baseline = self.baseline_latencies.get(baseline_key, self.prediction_model['baseline_processing'])

        # Adjust for current system conditions
        try:
            cpu_percent = psutil.cpu_percent(interval=0.01)
            memory_percent = psutil.virtual_memory().percent

            system_load_factor = 1.0 + (cpu_percent + memory_percent) / 200.0  # Average and scale

        except:
            system_load_factor = 1.0

        # Predict components
        predicted_latencies = {
            LatencyComponent.PROCESSING: baseline * system_load_factor,
            LatencyComponent.TRANSMISSION: baseline * 2.0 * self.prediction_model['congestion_factor'],
            LatencyComponent.PROPAGATION: baseline * 3.0,  # Usually dominated by distance
            LatencyComponent.QUEUING: baseline * 1.5 * system_load_factor,
            LatencyComponent.NETWORK_STACK: baseline * 0.5,
            LatencyComponent.HARDWARE: baseline * 0.3 * system_load_factor
        }

        # Adjust for packet size
        size_factor = 1.0 + (packet_size / 1500.0) * 0.5  # MTU-based scaling
        for component in [LatencyComponent.TRANSMISSION, LatencyComponent.PROCESSING]:
            predicted_latencies[component] *= size_factor

        return predicted_latencies

    def start_continuous_monitoring(self, monitor_targets: List[Tuple[str, int]] = None):
        """Start continuous latency monitoring"""

        if self.monitoring_active:
            return

        self.monitoring_active = True

        if monitor_targets is None:
            monitor_targets = [
                ('8.8.8.8', 53),    # Google DNS
                ('1.1.1.1', 53),    # Cloudflare DNS
                ('127.0.0.1', 80)   # Localhost
            ]

        def monitoring_loop():
            while self.monitoring_active:
                for host, port in monitor_targets:
                    try:
                        measurement = self.measure_network_latency(
                            host, port, NetworkProtocol.TCP, timeout=2.0
                        )

                        # Update baseline latencies
                        baseline_key = f"{host}_{NetworkProtocol.TCP.value}"
                        self.baseline_latencies[baseline_key] = measurement.total_latency

                    except Exception as e:
                        pass  # Continue monitoring other targets

                time.sleep(5.0)  # Monitor every 5 seconds

        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

    def get_latency_analysis_report(self) -> Dict:
        """Generate comprehensive latency analysis report"""

        if not self.latency_measurements:
            return {'error': 'No latency measurements available'}

        # Aggregate statistics by component
        component_stats = {}
        for component in LatencyComponent:
            component_latencies = []
            for measurement in self.latency_measurements:
                if component in measurement.component_latencies:
                    component_latencies.append(measurement.component_latencies[component])

            if component_latencies:
                component_stats[component.value] = {
                    'mean': statistics.mean(component_latencies),
                    'median': statistics.median(component_latencies),
                    'stdev': statistics.stdev(component_latencies) if len(component_latencies) > 1 else 0.0,
                    'min': min(component_latencies),
                    'max': max(component_latencies),
                    'sample_count': len(component_latencies)
                }

        # Protocol analysis
        protocol_stats = {}
        for protocol in NetworkProtocol:
            protocol_measurements = [m for m in self.latency_measurements if m.protocol == protocol]
            if protocol_measurements:
                latencies = [m.total_latency for m in protocol_measurements]
                protocol_stats[protocol.value] = {
                    'mean_latency': statistics.mean(latencies),
                    'measurement_count': len(protocol_measurements),
                    'reliability': 1.0 - (len([l for l in latencies if l > 1.0]) / len(latencies))
                }

        # Temporal fragmentation security analysis
        fragmentation_stats = {}
        for message_id, fragments in self.temporal_fragments.items():
            if fragments:
                security_analysis = self.analyze_temporal_cryptographic_security(message_id, len(fragments))
                fragmentation_stats[message_id] = security_analysis

        return {
            'measurement_summary': {
                'total_measurements': len(self.latency_measurements),
                'measurement_timespan': max(m.timestamp_end for m in self.latency_measurements) - min(m.timestamp_start for m in self.latency_measurements) if self.latency_measurements else 0.0,
                'unique_targets': len(set(m.destination_address for m in self.latency_measurements))
            },
            'component_analysis': component_stats,
            'protocol_analysis': protocol_stats,
            'precision_calculations': len(self.precision_calculations),
            'temporal_fragmentation': {
                'total_messages': len(self.temporal_fragments),
                'total_fragments': sum(len(fragments) for fragments in self.temporal_fragments.values()),
                'security_analysis': fragmentation_stats
            },
            'system_performance': self.system_metrics,
            'baseline_latencies': self.baseline_latencies,
            'monitoring_status': 'active' if self.monitoring_active else 'inactive'
        }


def create_signal_latency_analyzer() -> SignalLatencyAnalyzer:
    """Create signal latency analyzer for network and machine error analysis"""
    return SignalLatencyAnalyzer()
