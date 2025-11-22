def compare_thermometry_methods(T_true=100e-9):  # 100 nK
    """
    Test: Traditional measurement vs BMD navigation
    """

    # Setup: Rb-87 BEC at 100 nK
    system = setup_bec(T=T_true, N_atoms=10000)

    # Method 1: Traditional (measure current momentum)
    Se_current = measure_current_state(system)
    T_traditional = temperature_from_Se(Se_current)

    # Method 2: BMD Navigation (find minimum, measure distance)
    Se_minimum = navigate_to_minimum(system)
    categorical_distance = Se_current - Se_minimum
    T_navigation = temperature_from_distance(categorical_distance, Se_minimum)

    # Compare resolution
    # Traditional: Limited by thermal fluctuations (~√N)
    # Navigation: Limited by Se precision (~δt)

    results = {
        'traditional': {
            'T': T_traditional,
            'uncertainty': T_traditional / np.sqrt(system.N_atoms),  # ~√N
            'backaction': calculate_heating(system)  # Non-zero
        },
        'navigation': {
            'T': T_navigation,
            'uncertainty': extract_from_timing_precision(),  # δt limit
            'backaction': 0  # Zero! (no physical interaction)
        }
    }

    return results
