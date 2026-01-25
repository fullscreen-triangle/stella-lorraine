(.venv) PS C:\Users\kundai\Documents\geosciences\stella-lorraine\observatory\src\transporters> python categorical_coordinates.py
Could not find platform independent libraries <prefix>
======================================================================
MEMBRANE TRANSPORTER CONFORMATIONAL LANDSCAPE
======================================================================

Conformational States in S-Entropy Space:
----------------------------------------------------------------------

OPEN_OUTSIDE:
  Cavity volume: 5000.0 Ų
  Binding frequency: 3.80e+13 Hz
  S-coordinates: S_k=0.10, S_t=0.00, S_e=1.00
  Free energy: 0.0 kJ/mol
  ATP bound: True

OCCLUDED:
  Cavity volume: 3000.0 Ų
  Binding frequency: 4.50e+13 Hz
  S-coordinates: S_k=0.90, S_t=0.25, S_e=0.50
  Free energy: 15.0 kJ/mol
  ATP bound: True

OPEN_INSIDE:
  Cavity volume: 4500.0 Ų
  Binding frequency: 3.20e+13 Hz
  S-coordinates: S_k=0.20, S_t=0.50, S_e=0.30
  Free energy: -10.0 kJ/mol
  ATP bound: False

RESETTING:
  Cavity volume: 4000.0 Ų
  Binding frequency: 3.50e+13 Hz
  S-coordinates: S_k=0.05, S_t=0.75, S_e=0.80
  Free energy: 5.0 kJ/mol
  ATP bound: False

======================================================================
Transition Rates:
----------------------------------------------------------------------

open_outside → occluded:
  Empty: 2.96e+03 s⁻¹
  Substrate-bound: 1.44e+05 s⁻¹
  Enhancement: 48.5×

occluded → open_inside:
  Empty: 1.87e+15 s⁻¹
  Substrate-bound: 1.87e+15 s⁻¹
  Enhancement: 1.0×

open_inside → resetting:
  Empty: 2.96e+03 s⁻¹
  Substrate-bound: 2.96e+03 s⁻¹
  Enhancement: 1.0×

resetting → open_outside:
  Empty: 6.96e+06 s⁻¹
  Substrate-bound: 6.96e+06 s⁻¹
  Enhancement: 1.0×

======================================================================
S-Space Trajectory (5 cycles):
----------------------------------------------------------------------
Total trajectory points: 20
S-space distance traveled: 14.73

======================================================================
(.venv) PS C:\Users\kundai\Documents\geosciences\stella-lorraine\observatory\src\transporters> python phase_locked_selection.py
Could not find platform independent libraries <prefix>

======================================================================
VALIDATION: PHASE-LOCKED SUBSTRATE SELECTION
======================================================================

Test Substrates:
----------------------------------------------------------------------
Doxorubicin          | MW= 543.5 Da | f₀=3.50e+13 Hz | charge=+1
Verapamil            | MW= 454.6 Da | f₀=3.80e+13 Hz | charge=+1
Glucose              | MW= 180.2 Da | f₀=2.50e+13 Hz | charge=+0
Rhodamine_123        | MW= 380.8 Da | f₀=3.70e+13 Hz | charge=+1
Metformin            | MW= 129.2 Da | f₀=2.80e+13 Hz | charge=+2

INFO:__main__:======================================================================
INFO:__main__:PHASE-LOCKED SUBSTRATE SELECTION SIMULATION
INFO:__main__:======================================================================
INFO:__main__:✓ Substrate Verapamil bound (phase-lock: 0.910)
INFO:__main__:✓ Substrate Verapamil released inside
INFO:__main__:
Transported: 1/5
INFO:__main__:Efficiency: 20.0%
INFO:__main__:Selectivity: 9.10e+09

======================================================================
RESULTS
======================================================================

Phase-Lock Strengths:
----------------------------------------------------------------------
Doxorubicin          | 0.100 | ✗ REJECTED
Verapamil            | 0.910 | ✓ TRANSPORTED
Glucose              | 0.228 | ✗ REJECTED
Rhodamine_123        | 0.250 | ✗ REJECTED
Metformin            | 0.037 | ✗ REJECTED

Transport Efficiency: 20.0%
Selectivity Factor: 9.10e+09

Transporter Statistics:
  transport_events: 1
  rejected_molecules: 0
  current_state: open_outside
  current_time: 0.001337960165434672
  atp_turnover_rate: 10.0
  selectivity_factor: 1.0

======================================================================
(.venv) PS C:\Users\kundai\Documents\geosciences\stella-lorraine\observatory\src\transporters> python transplanckian_observation.py
Could not find platform independent libraries <prefix>

======================================================================
VALIDATION: TRANS-PLANCKIAN OBSERVATION
======================================================================

Observer Configuration:
----------------------------------------------------------------------
Time resolution: 1.00e-15 s (femtosecond)
S-coordinate precision: 0.01
Frequency precision: 1.00e+11 Hz

INFO:__main__:======================================================================
INFO:__main__:TRANS-PLANCKIAN MAXWELL DEMON OBSERVATION
INFO:__main__:======================================================================
INFO:__main__:
Observing Doxorubicin...
INFO:__main__:  ✗ MEASUREMENT: Not detected (phase-lock=0.100)
INFO:__main__:
Observing Verapamil...
INFO:__main__:  ✓ MEASUREMENT: Detected (phase-lock=1.000)
INFO:__main__:  ✗ FEEDBACK: No conformational change
INFO:__main__:
Observing Glucose...
INFO:__main__:  ✓ MEASUREMENT: Detected (phase-lock=0.500)
INFO:__main__:  ✗ FEEDBACK: No conformational change
INFO:__main__:
======================================================================
INFO:__main__:Total observations: 300
INFO:__main__:Measurements: 3
INFO:__main__:Feedbacks: 2
INFO:__main__:Transports: 0
INFO:__main__:Rejections: 3
INFO:__main__:Total momentum transfer: 0.00e+00 kg·m/s
INFO:__main__:Backaction per observation: 0.00e+00
INFO:__main__:======================================================================
INFO:__main__:
======================================================================
INFO:__main__:ZERO-BACKACTION VERIFICATION
INFO:__main__:======================================================================
INFO:__main__:
Observations: 300
INFO:__main__:Total momentum transfer: 0.00e+00 kg·m/s
INFO:__main__:Average per observation: 0.00e+00 kg·m/s
INFO:__main__:
Comparison:
INFO:__main__:  Heisenberg limit: 5.27e-25 kg·m/s
INFO:__main__:  Thermal momentum: 5.96e-22 kg·m/s
INFO:__main__:  Backaction/Heisenberg: 0.00e+00
INFO:__main__:  Backaction/Thermal: 0.00e+00
INFO:__main__:
✓ ZERO BACKACTION VERIFIED: True
INFO:__main__:======================================================================


✓ Validation complete
✓ Zero backaction verified: True
✓ Total observations: 300
======================================================================
ENSEMBLE TRANSPORTER DEMON VALIDATION
======================================================================

INFO:__main__:Initialized P-glycoprotein ensemble demon:
INFO:__main__:  Transporters: 5000
INFO:__main__:  Membrane area: 1000.0 μm²
INFO:__main__:  Density: 5.00 transporters/μm²

======================================================================
TEST 1: SINGLE SUBSTRATE ENSEMBLE TRANSPORT
======================================================================
INFO:__main__:
Ensemble transport of Verapamil:
INFO:__main__:  Available molecules: 10000
INFO:__main__:  Duration: 1.00 s
INFO:__main__:  Available transporters: 4250
INFO:__main__:  Ensemble transport rate: 42500.0 molecules/s
INFO:__main__:  Transported: 10000/10000 (100.0%)
INFO:__main__:  Collective phase-lock: 1.000

✓ Ensemble transported 10000 molecules in 1 second
✓ Transport rate: 42500.0 molecules/s
✓ Efficiency: 100.0%

======================================================================
TEST 2: MULTI-SUBSTRATE COMPETITION
======================================================================
INFO:__main__:======================================================================
INFO:__main__:MULTI-SUBSTRATE COMPETITION
INFO:__main__:======================================================================
INFO:__main__:
Phase-lock strengths:
INFO:__main__:  Doxorubicin         : 0.342
INFO:__main__:  Verapamil           : 1.000
INFO:__main__:  Glucose             : 1.000
INFO:__main__:  Rhodamine_123       : 1.000
INFO:__main__:  Metformin           : 0.684
INFO:__main__:
Transport Results:
INFO:__main__:----------------------------------------------------------------------
INFO:__main__:Doxorubicin         : 3611/5000 (72.2%) | prob=0.085
INFO:__main__:Verapamil           : 5000/5000 (100.0%) | prob=0.248
INFO:__main__:Glucose             : 5000/5000 (100.0%) | prob=0.248
INFO:__main__:Rhodamine_123       : 5000/5000 (100.0%) | prob=0.248
INFO:__main__:Metformin           : 5000/5000 (100.0%) | prob=0.170
INFO:__main__:
Total transported: 23611/25000 (94.4%)
INFO:__main__:======================================================================

✓ Collective selectivity: 1.00e+10
✓ Overall efficiency: 94.4%

======================================================================
ENSEMBLE STATISTICS
======================================================================
  transporter_type: P-glycoprotein
  num_transporters: 5000
  num_active: 750
  num_available: 4250
  total_transport_events: 33611
  avg_cycle_time: 0.10
  ensemble_throughput: 16805.50
  collective_selectivity: 24.20
  s_coordinate_spread: 0.10
  membrane_area_um2: 1000.00
  density_per_um2: 5.00
  current_time: 2.00

======================================================================
