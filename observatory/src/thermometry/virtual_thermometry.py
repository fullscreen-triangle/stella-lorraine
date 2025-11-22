"""
class VirtualThermometryStation:
    """
    Virtual thermometer created from molecular BMDs

    - No physical probe
    - No backaction
    - Trans-Planckian resolution
    - Can measure "temperature" of past/future states (via St)
    """

    def __init__(self, location: Tuple[Sk, St, Se]):
        self.location = location
        self.molecular_bmds = []

    def harvest_molecular_oscillations(self):
        """
        Create virtual spectrometer from local molecules
        Each molecule becomes a BMD navigator
        """
        molecules = get_molecules_at_location(self.location)

        for mol in molecules:
            # Hardware sync with molecular oscillations
            bmd = MolecularBMD(mol.categorical_state)
            self.molecular_bmds.append(bmd)

    def measure_temperature_via_navigation(self):
        """
        All molecular BMDs navigate to find minimum
        Temperature = distance from minimum
        """
        # Parallel navigation (all BMDs simultaneously)
        Se_current = sum(bmd.current.Se for bmd in self.molecular_bmds)
        Se_minimum = sum(bmd.navigate_to_minimum_momentum()
                        for bmd in self.molecular_bmds)

        # Temperature from categorical distance
        T = temperature_from_categorical_distance(
            Se_current, Se_minimum
        )

        return T

    def measure_past_temperature(self, delta_t_past: float):
        """
        Navigate St coordinate to access past states
        Measure temperature "retroactively"!
        """
        # Navigate to past via St
        past_location = (self.location[0],  # Sk same
                        self.location[1] - delta_t_past,  # St - Δt
                        self.location[2])  # Se will be measured

        # Measure temperature at that past categorical state
        past_system = VirtualThermometryStation(past_location)
        T_past = past_system.measure_temperature_via_navigation()

        return T_past




"""
