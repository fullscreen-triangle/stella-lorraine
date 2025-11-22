"""
Precise Clock APIs - External Atomic Clock Integration

Responsible for connection and validation with external precise clocks including
atomic clocks, GPS time servers, NTP servers, and satellite timing systems.
Provides high-precision time synchronization for S-Entropy alignment validation.
"""

import numpy as np
import time
import socket
import struct
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import requests


class ClockType(Enum):
    """Types of precision clocks"""
    ATOMIC_CESIUM = "atomic_cesium"           # Cesium atomic clock
    ATOMIC_RUBIDIUM = "atomic_rubidium"       # Rubidium atomic clock
    ATOMIC_HYDROGEN = "atomic_hydrogen"       # Hydrogen maser
    GPS_SATELLITE = "gps_satellite"          # GPS satellite time
    GALILEO_SATELLITE = "galileo_satellite"   # Galileo satellite time
    NTP_STRATUM_1 = "ntp_stratum_1"          # Stratum 1 NTP server
    NTP_STRATUM_2 = "ntp_stratum_2"          # Stratum 2 NTP server
    GLONASS = "glonass"                      # GLONASS satellite time
    BEIDOU = "beidou"                        # BeiDou satellite time


class ClockPrecision(Enum):
    """Precision levels for different clock types"""
    SECOND = 1e0
    MILLISECOND = 1e-3
    MICROSECOND = 1e-6
    NANOSECOND = 1e-9
    PICOSECOND = 1e-12
    FEMTOSECOND = 1e-15
    ATTOSECOND = 1e-18


@dataclass
class ClockReading:
    """Single clock reading with precision metadata"""
    clock_id: str
    clock_type: ClockType
    timestamp: float
    precision_seconds: float
    uncertainty_seconds: float
    signal_quality: float
    latency_seconds: float
    leap_second_offset: int = 0
    raw_data: Optional[bytes] = None

    def get_precision_level(self) -> ClockPrecision:
        """Determine precision level from precision_seconds"""
        if self.precision_seconds >= 1e0:
            return ClockPrecision.SECOND
        elif self.precision_seconds >= 1e-3:
            return ClockPrecision.MILLISECOND
        elif self.precision_seconds >= 1e-6:
            return ClockPrecision.MICROSECOND
        elif self.precision_seconds >= 1e-9:
            return ClockPrecision.NANOSECOND
        elif self.precision_seconds >= 1e-12:
            return ClockPrecision.PICOSECOND
        elif self.precision_seconds >= 1e-15:
            return ClockPrecision.FEMTOSECOND
        else:
            return ClockPrecision.ATTOSECOND


@dataclass
class ClockSource:
    """External clock source configuration"""
    clock_id: str
    clock_type: ClockType
    connection_info: Dict[str, Any]  # IP, port, protocol, etc.
    expected_precision: float
    connection_timeout: float = 5.0
    polling_interval: float = 1.0
    is_active: bool = False
    last_reading: Optional[ClockReading] = None
    connection_health: float = 1.0

    def get_connection_string(self) -> str:
        """Get connection string for this clock source"""
        if 'host' in self.connection_info:
            host = self.connection_info['host']
            port = self.connection_info.get('port', 123)  # Default NTP port
            return f"{host}:{port}"
        elif 'device_path' in self.connection_info:
            return self.connection_info['device_path']
        else:
            return f"{self.clock_type.value}_{self.clock_id}"


@dataclass
class ClockValidationResult:
    """Result of clock validation process"""
    clock_id: str
    is_valid: bool
    accuracy_assessment: float
    precision_assessment: float
    stability_assessment: float
    drift_rate: float  # seconds per second
    validation_confidence: float
    validation_timestamp: float
    validation_details: Dict = field(default_factory=dict)


class PreciseClockAPIManager:
    """
    Precise Clock API Manager for External Time Source Integration

    Handles connection, validation, and synchronization with multiple types of
    precision clocks including atomic clocks, satellite timing, and NTP servers.

    Provides high-precision time references for S-Entropy alignment validation
    and strategic disagreement precision testing.
    """

    def __init__(self):
        self.clock_sources: Dict[str, ClockSource] = {}
        self.active_connections: Dict[str, Any] = {}
        self.clock_readings_history: Dict[str, List[ClockReading]] = {}
        self.validation_results: Dict[str, ClockValidationResult] = {}

        # System configuration
        self.system_reference_time = time.time()
        self.synchronization_threshold = 1e-6  # 1 microsecond
        self.validation_window = 300.0  # 5 minutes for validation
        self.max_drift_tolerance = 1e-9  # 1 nanosecond per second

        # Threading for concurrent connections
        self.thread_pool = ThreadPoolExecutor(max_workers=16)
        self.polling_active = False
        self.polling_thread = None

        # Initialize standard clock sources
        self._initialize_standard_clock_sources()

    def _initialize_standard_clock_sources(self):
        """Initialize connections to standard precision time sources"""

        # Standard NTP servers (Stratum 1 - connected to atomic clocks)
        ntp_stratum_1_servers = [
            ('time.nist.gov', 123),
            ('time.cloudflare.com', 123),
            ('pool.ntp.org', 123),
            ('time.google.com', 123)
        ]

        for i, (host, port) in enumerate(ntp_stratum_1_servers):
            clock_id = f"ntp_s1_{i}"
            self.add_clock_source(
                clock_id=clock_id,
                clock_type=ClockType.NTP_STRATUM_1,
                connection_info={'host': host, 'port': port, 'protocol': 'ntp'},
                expected_precision=1e-6  # Microsecond precision
            )

        # GPS time servers (simulated - in practice would connect to GPS receivers)
        gps_sources = [
            ('gps.nist.gov', 1234),  # Simulated GPS time server
        ]

        for i, (host, port) in enumerate(gps_sources):
            clock_id = f"gps_{i}"
            self.add_clock_source(
                clock_id=clock_id,
                clock_type=ClockType.GPS_SATELLITE,
                connection_info={'host': host, 'port': port, 'protocol': 'gps'},
                expected_precision=1e-9  # Nanosecond precision
            )

    def add_clock_source(self,
                        clock_id: str,
                        clock_type: ClockType,
                        connection_info: Dict[str, Any],
                        expected_precision: float,
                        polling_interval: float = 1.0) -> bool:
        """Add external clock source to system"""

        clock_source = ClockSource(
            clock_id=clock_id,
            clock_type=clock_type,
            connection_info=connection_info,
            expected_precision=expected_precision,
            polling_interval=polling_interval
        )

        self.clock_sources[clock_id] = clock_source
        self.clock_readings_history[clock_id] = []

        return True

    def connect_to_clock(self, clock_id: str) -> bool:
        """Establish connection to specific clock source"""

        if clock_id not in self.clock_sources:
            return False

        clock_source = self.clock_sources[clock_id]

        try:
            if clock_source.clock_type in [ClockType.NTP_STRATUM_1, ClockType.NTP_STRATUM_2]:
                connection = self._connect_ntp_clock(clock_source)
            elif clock_source.clock_type == ClockType.GPS_SATELLITE:
                connection = self._connect_gps_clock(clock_source)
            elif clock_source.clock_type in [ClockType.ATOMIC_CESIUM, ClockType.ATOMIC_RUBIDIUM]:
                connection = self._connect_atomic_clock(clock_source)
            else:
                # Generic connection
                connection = self._connect_generic_clock(clock_source)

            if connection:
                self.active_connections[clock_id] = connection
                clock_source.is_active = True
                clock_source.connection_health = 1.0
                return True
            else:
                return False

        except Exception as e:
            clock_source.connection_health = 0.0
            return False

    def _connect_ntp_clock(self, clock_source: ClockSource) -> Optional[socket.socket]:
        """Connect to NTP time server"""
        try:
            host = clock_source.connection_info['host']
            port = clock_source.connection_info.get('port', 123)

            # Create UDP socket for NTP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(clock_source.connection_timeout)

            # Test connection with NTP request
            ntp_request = self._create_ntp_request()
            sock.sendto(ntp_request, (host, port))
            response, addr = sock.recvfrom(1024)

            if len(response) >= 48:  # Valid NTP response
                return sock
            else:
                sock.close()
                return None

        except Exception as e:
            return None

    def _connect_gps_clock(self, clock_source: ClockSource) -> Optional[Dict]:
        """Connect to GPS timing source"""
        # GPS connection would typically involve serial or network GPS receivers
        # This is a simplified simulation

        try:
            host = clock_source.connection_info.get('host')
            port = clock_source.connection_info.get('port')

            if host and port:
                # Network GPS receiver
                connection = {
                    'type': 'network_gps',
                    'host': host,
                    'port': port,
                    'connected': True
                }
                return connection
            elif 'device_path' in clock_source.connection_info:
                # Serial GPS receiver
                connection = {
                    'type': 'serial_gps',
                    'device': clock_source.connection_info['device_path'],
                    'connected': True
                }
                return connection
            else:
                return None

        except Exception as e:
            return None

    def _connect_atomic_clock(self, clock_source: ClockSource) -> Optional[Dict]:
        """Connect to atomic clock (cesium, rubidium, hydrogen maser)"""
        # Atomic clock connection would typically involve specialized hardware interfaces
        # This is a simulation for the framework

        try:
            connection = {
                'type': 'atomic_clock',
                'clock_type': clock_source.clock_type.value,
                'precision': clock_source.expected_precision,
                'connected': True,
                'interface': clock_source.connection_info.get('interface', 'simulated')
            }
            return connection

        except Exception as e:
            return None

    def _connect_generic_clock(self, clock_source: ClockSource) -> Optional[Dict]:
        """Connect to generic clock source"""
        connection = {
            'type': 'generic',
            'clock_type': clock_source.clock_type.value,
            'connected': True
        }
        return connection

    def _create_ntp_request(self) -> bytes:
        """Create NTP request packet"""
        # NTP packet format (simplified)
        # First byte: version (4) and mode (3 = client)
        version = 4
        mode = 3
        li_vn_mode = (0 << 6) | (version << 3) | mode

        # Create 48-byte NTP packet
        packet = struct.pack('!B', li_vn_mode) + b'\x00' * 47

        return packet

    def read_clock(self, clock_id: str) -> Optional[ClockReading]:
        """Read current time from specific clock source"""

        if clock_id not in self.clock_sources or clock_id not in self.active_connections:
            return None

        clock_source = self.clock_sources[clock_id]
        connection = self.active_connections[clock_id]

        start_time = time.time()

        try:
            if clock_source.clock_type in [ClockType.NTP_STRATUM_1, ClockType.NTP_STRATUM_2]:
                reading = self._read_ntp_time(clock_source, connection)
            elif clock_source.clock_type == ClockType.GPS_SATELLITE:
                reading = self._read_gps_time(clock_source, connection)
            elif clock_source.clock_type in [ClockType.ATOMIC_CESIUM, ClockType.ATOMIC_RUBIDIUM, ClockType.ATOMIC_HYDROGEN]:
                reading = self._read_atomic_time(clock_source, connection)
            else:
                reading = self._read_generic_time(clock_source, connection)

            if reading:
                # Calculate latency
                reading.latency_seconds = time.time() - start_time

                # Store reading
                clock_source.last_reading = reading
                self.clock_readings_history[clock_id].append(reading)

                # Limit history size
                if len(self.clock_readings_history[clock_id]) > 1000:
                    self.clock_readings_history[clock_id] = self.clock_readings_history[clock_id][-1000:]

            return reading

        except Exception as e:
            clock_source.connection_health *= 0.9  # Degrade health on errors
            return None

    def _read_ntp_time(self, clock_source: ClockSource, connection: socket.socket) -> Optional[ClockReading]:
        """Read time from NTP server"""
        try:
            # Send NTP request
            ntp_request = self._create_ntp_request()
            host = clock_source.connection_info['host']
            port = clock_source.connection_info.get('port', 123)

            connection.sendto(ntp_request, (host, port))
            response, addr = connection.recvfrom(1024)

            if len(response) >= 48:
                # Parse NTP response (simplified)
                # Transmit timestamp is at bytes 40-47
                transmit_timestamp = struct.unpack('!Q', response[40:48])[0]

                # Convert NTP timestamp to Unix timestamp
                # NTP epoch: January 1, 1900; Unix epoch: January 1, 1970
                ntp_unix_offset = 2208988800
                timestamp = (transmit_timestamp >> 32) - ntp_unix_offset
                fraction = (transmit_timestamp & 0xFFFFFFFF) / 2**32
                precise_timestamp = timestamp + fraction

                reading = ClockReading(
                    clock_id=clock_source.clock_id,
                    clock_type=clock_source.clock_type,
                    timestamp=precise_timestamp,
                    precision_seconds=1e-6,  # Microsecond precision for NTP
                    uncertainty_seconds=1e-5,  # 10 microsecond uncertainty
                    signal_quality=0.95,
                    latency_seconds=0.0,  # Will be calculated by caller
                    raw_data=response
                )

                return reading
            else:
                return None

        except Exception as e:
            return None

    def _read_gps_time(self, clock_source: ClockSource, connection: Dict) -> Optional[ClockReading]:
        """Read time from GPS source"""
        # GPS time reading simulation

        # GPS provides nanosecond precision
        current_time = time.time()

        # Add GPS-specific corrections
        gps_precision = np.random.normal(0, 1e-9)  # 1 nanosecond standard deviation
        gps_timestamp = current_time + gps_precision

        reading = ClockReading(
            clock_id=clock_source.clock_id,
            clock_type=clock_source.clock_type,
            timestamp=gps_timestamp,
            precision_seconds=1e-9,  # Nanosecond precision
            uncertainty_seconds=1e-8,  # 10 nanosecond uncertainty
            signal_quality=0.98,
            latency_seconds=0.0
        )

        return reading

    def _read_atomic_time(self, clock_source: ClockSource, connection: Dict) -> Optional[ClockReading]:
        """Read time from atomic clock"""
        # Atomic clock reading simulation

        current_time = time.time()

        # Atomic clock precision depends on type
        precision_map = {
            ClockType.ATOMIC_CESIUM: 1e-15,    # Femtosecond precision
            ClockType.ATOMIC_RUBIDIUM: 1e-12,  # Picosecond precision
            ClockType.ATOMIC_HYDROGEN: 1e-16   # Sub-femtosecond precision
        }

        precision = precision_map.get(clock_source.clock_type, 1e-12)
        atomic_correction = np.random.normal(0, precision)
        atomic_timestamp = current_time + atomic_correction

        reading = ClockReading(
            clock_id=clock_source.clock_id,
            clock_type=clock_source.clock_type,
            timestamp=atomic_timestamp,
            precision_seconds=precision,
            uncertainty_seconds=precision * 10,
            signal_quality=0.995,
            latency_seconds=0.0
        )

        return reading

    def _read_generic_time(self, clock_source: ClockSource, connection: Dict) -> Optional[ClockReading]:
        """Read time from generic source"""
        reading = ClockReading(
            clock_id=clock_source.clock_id,
            clock_type=clock_source.clock_type,
            timestamp=time.time(),
            precision_seconds=clock_source.expected_precision,
            uncertainty_seconds=clock_source.expected_precision * 10,
            signal_quality=0.90,
            latency_seconds=0.0
        )

        return reading

    def validate_clock_accuracy(self, clock_id: str, reference_clocks: List[str] = None) -> ClockValidationResult:
        """Validate clock accuracy against reference clocks"""

        if clock_id not in self.clock_sources:
            return ClockValidationResult(
                clock_id=clock_id,
                is_valid=False,
                accuracy_assessment=0.0,
                precision_assessment=0.0,
                stability_assessment=0.0,
                drift_rate=float('inf'),
                validation_confidence=0.0,
                validation_timestamp=time.time()
            )

        clock_history = self.clock_readings_history.get(clock_id, [])
        if len(clock_history) < 10:
            # Not enough data for validation
            return ClockValidationResult(
                clock_id=clock_id,
                is_valid=False,
                accuracy_assessment=0.0,
                precision_assessment=0.0,
                stability_assessment=0.0,
                drift_rate=0.0,
                validation_confidence=0.0,
                validation_timestamp=time.time(),
                validation_details={'error': 'insufficient_data'}
            )

        # Get recent readings for analysis
        recent_readings = clock_history[-100:]  # Last 100 readings

        # Calculate accuracy against reference clocks
        accuracy_scores = []
        if reference_clocks:
            for ref_clock_id in reference_clocks:
                if ref_clock_id in self.clock_readings_history:
                    ref_history = self.clock_readings_history[ref_clock_id]
                    accuracy_score = self._calculate_accuracy_against_reference(recent_readings, ref_history)
                    accuracy_scores.append(accuracy_score)

        # Calculate precision assessment
        timestamps = [reading.timestamp for reading in recent_readings]
        expected_intervals = np.diff(timestamps)
        precision_assessment = 1.0 / (1.0 + np.std(expected_intervals)) if len(expected_intervals) > 1 else 0.5

        # Calculate stability assessment
        drift_rates = []
        if len(recent_readings) >= 20:
            # Calculate drift over time
            time_points = np.array([reading.timestamp for reading in recent_readings])
            system_times = np.array([time.time() for _ in recent_readings])  # Simulated system time

            # Linear fit to detect drift
            if len(time_points) > 1:
                drift_rate = np.polyfit(system_times - system_times[0], time_points - time_points[0], 1)[0] - 1.0
                drift_rates.append(abs(drift_rate))

        # Overall assessments
        accuracy_assessment = np.mean(accuracy_scores) if accuracy_scores else 0.7  # Default if no references
        stability_assessment = 1.0 / (1.0 + np.mean(drift_rates)) if drift_rates else 0.8
        avg_drift_rate = np.mean(drift_rates) if drift_rates else 0.0

        # Validation confidence
        validation_confidence = (accuracy_assessment + precision_assessment + stability_assessment) / 3.0

        # Determine validity
        is_valid = (accuracy_assessment >= 0.8 and
                   precision_assessment >= 0.7 and
                   stability_assessment >= 0.8 and
                   abs(avg_drift_rate) <= self.max_drift_tolerance)

        result = ClockValidationResult(
            clock_id=clock_id,
            is_valid=is_valid,
            accuracy_assessment=accuracy_assessment,
            precision_assessment=precision_assessment,
            stability_assessment=stability_assessment,
            drift_rate=avg_drift_rate,
            validation_confidence=validation_confidence,
            validation_timestamp=time.time(),
            validation_details={
                'readings_analyzed': len(recent_readings),
                'reference_clocks_used': len(reference_clocks) if reference_clocks else 0,
                'validation_criteria': {
                    'accuracy_threshold': 0.8,
                    'precision_threshold': 0.7,
                    'stability_threshold': 0.8,
                    'max_drift_tolerance': self.max_drift_tolerance
                }
            }
        )

        self.validation_results[clock_id] = result
        return result

    def _calculate_accuracy_against_reference(self, test_readings: List[ClockReading], ref_readings: List[ClockReading]) -> float:
        """Calculate accuracy of test clock against reference clock"""

        if not test_readings or not ref_readings:
            return 0.0

        # Find overlapping time periods
        test_times = [(r.timestamp, r) for r in test_readings]
        ref_times = [(r.timestamp, r) for r in ref_readings]

        # Calculate time differences for overlapping periods
        time_differences = []

        for test_time, test_reading in test_times:
            # Find closest reference reading
            closest_ref = min(ref_times, key=lambda x: abs(x[0] - test_time))
            time_diff = abs(test_time - closest_ref[0])

            if time_diff < 10.0:  # Within 10 seconds
                time_differences.append(time_diff)

        if not time_differences:
            return 0.0

        # Accuracy score based on mean time difference
        mean_diff = np.mean(time_differences)
        accuracy_score = 1.0 / (1.0 + mean_diff * 1000)  # Scale to reasonable range

        return min(1.0, max(0.0, accuracy_score))

    def start_continuous_polling(self, poll_interval: float = 1.0):
        """Start continuous polling of all active clock sources"""

        if self.polling_active:
            return

        self.polling_active = True

        def polling_loop():
            while self.polling_active:
                for clock_id in list(self.active_connections.keys()):
                    try:
                        reading = self.read_clock(clock_id)
                    except Exception as e:
                        pass  # Continue polling other clocks

                time.sleep(poll_interval)

        self.polling_thread = threading.Thread(target=polling_loop, daemon=True)
        self.polling_thread.start()

    def stop_continuous_polling(self):
        """Stop continuous polling"""
        self.polling_active = False
        if self.polling_thread:
            self.polling_thread.join(timeout=5.0)

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""

        active_clocks = sum(1 for source in self.clock_sources.values() if source.is_active)
        total_clocks = len(self.clock_sources)

        # Calculate average precision of active clocks
        active_precisions = [source.expected_precision for source in self.clock_sources.values() if source.is_active]
        avg_precision = np.mean(active_precisions) if active_precisions else 0.0
        best_precision = min(active_precisions) if active_precisions else 0.0

        # Calculate total readings
        total_readings = sum(len(readings) for readings in self.clock_readings_history.values())

        # Health assessment
        health_scores = [source.connection_health for source in self.clock_sources.values()]
        system_health = np.mean(health_scores) if health_scores else 0.0

        return {
            'total_clock_sources': total_clocks,
            'active_connections': active_clocks,
            'system_health': system_health,
            'precision_metrics': {
                'average_precision_seconds': avg_precision,
                'best_precision_seconds': best_precision,
                'worst_precision_seconds': max(active_precisions) if active_precisions else 0.0
            },
            'data_metrics': {
                'total_readings_collected': total_readings,
                'validated_clocks': len(self.validation_results),
                'valid_clocks': sum(1 for result in self.validation_results.values() if result.is_valid)
            },
            'polling_status': 'active' if self.polling_active else 'inactive',
            'synchronization_threshold': self.synchronization_threshold,
            'validation_window': self.validation_window
        }

    def get_clock_comparison_report(self) -> Dict:
        """Generate comprehensive clock comparison report"""

        report = {
            'timestamp': time.time(),
            'clock_sources': {},
            'precision_rankings': [],
            'accuracy_rankings': [],
            'stability_rankings': []
        }

        # Analyze each clock source
        for clock_id, clock_source in self.clock_sources.items():
            clock_report = {
                'clock_type': clock_source.clock_type.value,
                'is_active': clock_source.is_active,
                'expected_precision': clock_source.expected_precision,
                'connection_health': clock_source.connection_health,
                'readings_count': len(self.clock_readings_history.get(clock_id, [])),
                'last_reading': clock_source.last_reading.timestamp if clock_source.last_reading else None
            }

            # Add validation results if available
            if clock_id in self.validation_results:
                validation = self.validation_results[clock_id]
                clock_report['validation'] = {
                    'is_valid': validation.is_valid,
                    'accuracy': validation.accuracy_assessment,
                    'precision': validation.precision_assessment,
                    'stability': validation.stability_assessment,
                    'drift_rate': validation.drift_rate,
                    'confidence': validation.validation_confidence
                }

            report['clock_sources'][clock_id] = clock_report

        # Create rankings
        validated_clocks = [
            (clock_id, self.validation_results[clock_id])
            for clock_id in self.validation_results.keys()
        ]

        # Precision ranking (by expected precision)
        precision_ranking = sorted(
            [(clock_id, source.expected_precision) for clock_id, source in self.clock_sources.items()],
            key=lambda x: x[1]
        )
        report['precision_rankings'] = precision_ranking

        # Accuracy ranking
        accuracy_ranking = sorted(validated_clocks, key=lambda x: x[1].accuracy_assessment, reverse=True)
        report['accuracy_rankings'] = [(clock_id, result.accuracy_assessment) for clock_id, result in accuracy_ranking]

        # Stability ranking
        stability_ranking = sorted(validated_clocks, key=lambda x: x[1].stability_assessment, reverse=True)
        report['stability_rankings'] = [(clock_id, result.stability_assessment) for clock_id, result in stability_ranking]

        return report

    def disconnect_all_clocks(self):
        """Disconnect from all clock sources and clean up"""
        self.stop_continuous_polling()

        for clock_id, connection in self.active_connections.items():
            try:
                if isinstance(connection, socket.socket):
                    connection.close()
            except:
                pass  # Ignore errors during cleanup

        self.active_connections.clear()

        for clock_source in self.clock_sources.values():
            clock_source.is_active = False


def create_precise_clock_api_system() -> PreciseClockAPIManager:
    """Create precise clock API system for external time source integration"""
    return PreciseClockAPIManager()
