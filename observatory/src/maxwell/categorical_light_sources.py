"""
Categorical Light Sources: Information-Theoretic Light Emission
===============================================================

In categorical rendering, light sources don't emit photons - they emit
INFORMATION through S-entropy space. Pixel demons observe this information
directly without ray tracing.

Key concepts:
- Light = information source in S-space
- Intensity = information rate (bits/second)
- Color = information structure/frequency
- Propagation = categorical proximity, not physical distance

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Color:
    """RGB color representation"""
    r: float
    g: float
    b: float

    def __add__(self, other):
        return Color(self.r + other.r, self.g + other.g, self.b + other.b)

    def __mul__(self, scalar):
        return Color(self.r * scalar, self.g * scalar, self.b * scalar)

    def clamp(self):
        return Color(
            np.clip(self.r, 0, 1),
            np.clip(self.g, 0, 1),
            np.clip(self.b, 0, 1)
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.r, self.g, self.b)

    def to_array(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b])

    @classmethod
    def from_wavelength(cls, wavelength_nm: float, intensity: float = 1.0):
        """
        Convert wavelength to approximate RGB

        Visible spectrum: 380-750 nm
        """
        # Simplified wavelength to RGB conversion
        if wavelength_nm < 380 or wavelength_nm > 750:
            return cls(0, 0, 0)  # Outside visible spectrum

        if wavelength_nm < 440:
            # Violet to blue
            t = (wavelength_nm - 380) / (440 - 380)
            r = (0.5 - 0.5 * t) * intensity
            g = 0.0
            b = intensity
        elif wavelength_nm < 490:
            # Blue to cyan
            t = (wavelength_nm - 440) / (490 - 440)
            r = 0.0
            g = t * intensity
            b = intensity
        elif wavelength_nm < 510:
            # Cyan to green
            t = (wavelength_nm - 490) / (510 - 490)
            r = 0.0
            g = intensity
            b = (1 - t) * intensity
        elif wavelength_nm < 580:
            # Green to yellow
            t = (wavelength_nm - 510) / (580 - 510)
            r = t * intensity
            g = intensity
            b = 0.0
        elif wavelength_nm < 645:
            # Yellow to red
            t = (wavelength_nm - 580) / (645 - 580)
            r = intensity
            g = (1 - t) * intensity
            b = 0.0
        else:
            # Red
            r = intensity
            g = 0.0
            b = 0.0

        return cls(r, g, b)


class CategoricalLight:
    """
    Categorical light source: Emits information in S-entropy space
    """

    def __init__(
        self,
        position: np.ndarray,
        color: Color,
        intensity: float,
        light_id: Optional[str] = None
    ):
        self.position = position
        self.color = color
        self.intensity = intensity
        self.light_id = light_id or f"light_{id(self)}"

        # Categorical state
        from pixel_maxwell_demon import SEntropyCoordinates
        self.s_state = SEntropyCoordinates.from_physical_state(
            position,
            energy=intensity
        )

        # Information emission rate (bits/s)
        # Higher intensity = more information
        self.information_rate = intensity * 1e15  # Scaled to THz range

        # Categorical influence radius
        self.influence_radius = 5.0  # In S-entropy space

        logger.debug(
            f"Created CategoricalLight '{self.light_id}' at {position} "
            f"with intensity {intensity}"
        )

    def get_information_at_s_state(
        self,
        target_s_state
    ) -> Tuple[float, Color]:
        """
        Calculate information and color received at target S-entropy state

        Returns (information_bits, effective_color)
        """
        # Categorical distance
        distance = self.s_state.distance_to(target_s_state)

        # Attenuation based on categorical proximity
        if distance >= self.influence_radius:
            return 0.0, Color(0, 0, 0)

        # Information falloff (inverse square in S-space)
        attenuation = (1.0 - distance / self.influence_radius) ** 2

        # Information received
        information = self.information_rate * attenuation

        # Color modulated by attenuation
        effective_color = self.color * (self.intensity * attenuation)

        return information, effective_color

    def set_wavelength(self, wavelength_nm: float):
        """Set color from wavelength"""
        self.color = Color.from_wavelength(wavelength_nm, self.intensity)


class CategoricalPointLight(CategoricalLight):
    """Point light source (omnidirectional)"""

    def __init__(self, position: np.ndarray, color: Color, intensity: float):
        super().__init__(position, color, intensity)
        self.light_type = "point"


class CategoricalDirectionalLight(CategoricalLight):
    """Directional light (like sun - parallel rays)"""

    def __init__(
        self,
        direction: np.ndarray,
        color: Color,
        intensity: float
    ):
        # Position is irrelevant for directional light
        position = np.zeros(3)
        super().__init__(position, color, intensity)

        self.direction = direction / np.linalg.norm(direction)
        self.light_type = "directional"
        self.influence_radius = 100.0  # Large radius (affects everything)

    def get_information_at_s_state(
        self,
        target_s_state
    ) -> Tuple[float, Color]:
        """
        Directional light: constant intensity everywhere
        (But modulated by surface orientation in full renderer)
        """
        information = self.information_rate
        effective_color = self.color * self.intensity

        return information, effective_color


class CategoricalSpotLight(CategoricalLight):
    """Spot light (cone of illumination)"""

    def __init__(
        self,
        position: np.ndarray,
        direction: np.ndarray,
        color: Color,
        intensity: float,
        cone_angle_deg: float = 30.0
    ):
        super().__init__(position, color, intensity)

        self.direction = direction / np.linalg.norm(direction)
        self.cone_angle_rad = np.radians(cone_angle_deg)
        self.light_type = "spot"

    def get_information_at_s_state(
        self,
        target_s_state
    ) -> Tuple[float, Color]:
        """
        Spot light: Attenuated by angle and distance
        """
        # Base attenuation from distance
        information, effective_color = super().get_information_at_s_state(target_s_state)

        # Additional angular attenuation
        # (In full implementation, would check angle to target)
        # For now, simplified

        return information, effective_color


class CategoricalAreaLight(CategoricalLight):
    """Area light (extended source)"""

    def __init__(
        self,
        center: np.ndarray,
        normal: np.ndarray,
        width: float,
        height: float,
        color: Color,
        intensity: float
    ):
        super().__init__(center, color, intensity)

        self.normal = normal / np.linalg.norm(normal)
        self.width = width
        self.height = height
        self.area = width * height
        self.light_type = "area"

        # Area lights have softer falloff
        self.influence_radius = 10.0

    def get_information_at_s_state(
        self,
        target_s_state
    ) -> Tuple[float, Color]:
        """
        Area light: Soft shadows through multiple sample points
        """
        # Simplified: Use center point
        # In full implementation, would sample across area
        information, effective_color = super().get_information_at_s_state(target_s_state)

        # Area lights emit more total information
        information *= np.sqrt(self.area)

        return information, effective_color


class LightingEnvironment:
    """
    Collection of categorical lights forming a lighting environment
    """

    def __init__(self, name: str = "lighting_env"):
        self.name = name
        self.lights: Dict[str, CategoricalLight] = {}
        self.ambient_color = Color(0.1, 0.1, 0.1)  # Default ambient
        self.ambient_intensity = 0.1

        logger.debug(f"Created LightingEnvironment '{name}'")

    def add_light(self, light: CategoricalLight) -> str:
        """Add light to environment"""
        self.lights[light.light_id] = light
        logger.debug(f"Added {light.light_type} light '{light.light_id}'")
        return light.light_id

    def remove_light(self, light_id: str):
        """Remove light from environment"""
        if light_id in self.lights:
            del self.lights[light_id]
            logger.debug(f"Removed light '{light_id}'")

    def calculate_total_illumination_at_s_state(
        self,
        target_s_state
    ) -> Tuple[float, Color]:
        """
        Calculate total information and color at target S-state from all lights
        """
        total_information = 0.0
        total_color = Color(0, 0, 0)

        # Contribution from each light
        for light in self.lights.values():
            info, color = light.get_information_at_s_state(target_s_state)
            total_information += info
            total_color = total_color + color

        # Add ambient
        total_color = total_color + self.ambient_color * self.ambient_intensity

        return total_information, total_color.clamp()

    def set_ambient(self, color: Color, intensity: float):
        """Set ambient lighting"""
        self.ambient_color = color
        self.ambient_intensity = intensity

    def create_three_point_setup(
        self,
        target_position: np.ndarray = np.array([0, 0, 0]),
        distance: float = 5.0
    ):
        """
        Create classic three-point lighting setup
        (key, fill, back)
        """
        # Key light (main light, 45° angle, bright)
        key_pos = target_position + np.array([distance * 0.7, distance * 0.5, distance])
        key_light = CategoricalPointLight(
            position=key_pos,
            color=Color(1.0, 0.95, 0.9),  # Warm white
            intensity=2.0
        )
        self.add_light(key_light)

        # Fill light (softer, opposite side)
        fill_pos = target_position + np.array([-distance * 0.7, distance * 0.3, distance * 0.8])
        fill_light = CategoricalPointLight(
            position=fill_pos,
            color=Color(0.9, 0.95, 1.0),  # Cool white
            intensity=0.8
        )
        self.add_light(fill_light)

        # Back light (rim lighting)
        back_pos = target_position + np.array([0, distance * 0.8, -distance * 0.5])
        back_light = CategoricalPointLight(
            position=back_pos,
            color=Color(1.0, 1.0, 1.0),  # Pure white
            intensity=1.5
        )
        self.add_light(back_light)

        logger.info("Created three-point lighting setup")

    def create_natural_daylight(
        self,
        sun_direction: np.ndarray = np.array([0.3, 0.8, -0.5]),
        sun_intensity: float = 3.0
    ):
        """Create natural daylight environment"""
        # Sun (directional)
        sun = CategoricalDirectionalLight(
            direction=sun_direction,
            color=Color(1.0, 0.98, 0.95),  # Warm sunlight
            intensity=sun_intensity
        )
        self.add_light(sun)

        # Sky (ambient)
        self.set_ambient(
            color=Color(0.5, 0.7, 1.0),  # Blue sky
            intensity=0.5
        )

        logger.info("Created natural daylight environment")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of lighting environment"""
        return {
            'name': self.name,
            'num_lights': len(self.lights),
            'light_types': [light.light_type for light in self.lights.values()],
            'total_intensity': sum(light.intensity for light in self.lights.values()),
            'ambient': {
                'color': self.ambient_color.to_tuple(),
                'intensity': self.ambient_intensity
            }
        }


def create_standard_lighting_environments() -> Dict[str, LightingEnvironment]:
    """
    Create library of standard lighting environments
    """
    environments = {}

    # Studio lighting
    studio = LightingEnvironment("studio")
    studio.create_three_point_setup()
    environments['studio'] = studio

    # Outdoor daylight
    outdoor = LightingEnvironment("outdoor_day")
    outdoor.create_natural_daylight()
    environments['outdoor_day'] = outdoor

    # Indoor room
    indoor = LightingEnvironment("indoor_room")
    # Ceiling light
    ceiling = CategoricalPointLight(
        position=np.array([0, 3, 0]),
        color=Color(1.0, 0.95, 0.85),  # Warm indoor light
        intensity=1.5
    )
    indoor.add_light(ceiling)
    indoor.set_ambient(Color(0.2, 0.2, 0.2), 0.3)
    environments['indoor_room'] = indoor

    # Dramatic (single strong light)
    dramatic = LightingEnvironment("dramatic")
    key = CategoricalPointLight(
        position=np.array([3, 2, 3]),
        color=Color(1.0, 0.9, 0.7),
        intensity=3.0
    )
    dramatic.add_light(key)
    dramatic.set_ambient(Color(0.05, 0.05, 0.1), 0.05)  # Very dark ambient
    environments['dramatic'] = dramatic

    return environments
