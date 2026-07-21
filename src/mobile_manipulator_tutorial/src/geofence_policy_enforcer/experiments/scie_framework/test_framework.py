#!/usr/bin/env python3
"""
SCIE Framework Unit Test (No Simulation Required)

Tests the experiment framework logic with mock data:
1. TrialResult with did_execute computation
2. Data logger with VR_exec metrics
3. Analysis script table generation

Run: python3 test_framework.py
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    get_default_config, EXECUTION_THRESHOLDS, TrialResult as ConfigTrialResult,
    TrialType, AttackType, AttackIntensity
)
from data_logger import (
    ExperimentLogger, TrialResult, TrialConditions,
    SafetyMetrics, PerformanceMetrics
)
from analyze_results import (
    compute_method_stats, generate_table1, generate_table2,
    generate_summary
)


def test_config():
    """Test config.py structures"""
    print("\n" + "="*60)
    print("TEST 1: Config Module")
    print("="*60)

    config = get_default_config()
    counts = config.get_trial_counts()

    assert counts['total_trials'] == 1150, f"Expected 1150 trials, got {counts['total_trials']}"
    assert counts['total_benign_trials'] == 250, f"Expected 250 benign, got {counts['total_benign_trials']}"
    assert counts['total_attack_trials'] == 900, f"Expected 900 attack, got {counts['total_attack_trials']}"
    assert len(config.methods) == 5, f"Expected 5 methods, got {len(config.methods)}"

    print(f"  ✓ Trial counts correct: {counts['total_trials']} total")
    print(f"  ✓ Methods: {[m.id for m in config.methods]}")
    print(f"  ✓ Benign scenarios: {[s.id for s in config.benign_scenarios]}")
    print(f"  ✓ Attack scenarios: {[s.id for s in config.attack_scenarios]}")

    # Test TrialResult compute_derived_flags
    result = ConfigTrialResult(
        trial_id='T0001',
        trial_type='attack',
        scenario_id='A1',
        method_id='no_guard',
        intensity='high',
        seed=42,
        goal_x=2.5, goal_y=3.0,
        expected_decision='reject',
        decision='allow',
        decision_reason='all goals allowed',
        decision_latency_ms=1.0,
        odom_distance=0.5,  # > 0.1m threshold
        nav_time=3.0,       # > 2.0s threshold
        violated=True
    )
    result.compute_derived_flags(EXECUTION_THRESHOLDS)

    assert result.did_execute == True, "Should be executed (odom > 0.1)"
    assert result.is_false_negative == True, "Should be FN (allowed + violated)"
    assert result.is_false_positive == False, "Should not be FP"

    print(f"  ✓ TrialResult.did_execute works correctly")
    print(f"  ✓ TrialResult.is_false_negative works correctly")

    return True


def test_data_logger():
    """Test data_logger.py with mock trials"""
    print("\n" + "="*60)
    print("TEST 2: Data Logger with VR_exec")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(tmpdir, "test_exp")

        # Create mock trials simulating different methods
        mock_trials = [
            # Geofence: rejects dangerous goals, no violations
            ("geofence", "A1", "reject", False, True, 0.0, 0.0),   # rejected, no exec
            ("geofence", "A1", "reject", False, True, 0.0, 0.0),   # rejected, no exec
            ("geofence", "B1", "allow", False, True, 1.5, 10.0),   # allowed safe, executed
            ("geofence", "B1", "allow", False, True, 1.2, 8.0),    # allowed safe, executed

            # no_guard: allows everything, violations occur
            ("no_guard", "A1", "allow", True, True, 2.0, 15.0),    # allowed, violated, executed
            ("no_guard", "A1", "allow", True, True, 1.8, 12.0),    # allowed, violated, executed
            ("no_guard", "B1", "allow", False, True, 1.5, 10.0),   # allowed safe, executed
            ("no_guard", "B1", "allow", False, True, 1.2, 8.0),    # allowed safe, executed

            # SELP: no margin, some edge cases leak through
            ("selp_proper", "A1", "allow", True, True, 1.5, 10.0), # allowed, violated (edge case)
            ("selp_proper", "A1", "reject", False, True, 0.0, 0.0),# rejected
            ("selp_proper", "B1", "allow", False, True, 1.5, 10.0),# allowed safe
            ("selp_proper", "B1", "allow", False, True, 1.2, 8.0), # allowed safe
        ]

        for i, (method, scenario, decision, violated, is_attack, odom, nav_time) in enumerate(mock_trials):
            expected = "reject" if scenario.startswith("A") else "allow"
            did_execute = odom > 0.1 or nav_time > 2.0
            is_fn = decision == "allow" and violated
            is_fp = decision == "reject" and expected == "allow"

            conditions = TrialConditions(
                method=method,
                scenario=scenario,
                intensity="high",
                noise_sigma=0.15,
                repetition=i,
                random_seed=42
            )

            result = TrialResult(
                conditions=conditions,
                goal_x=2.5, goal_y=3.0
            )
            result.safety = SafetyMetrics(
                decision=decision,
                expected_decision=expected,
                is_correct=(decision == expected),
                actual_violation=violated,
                is_false_negative=is_fn,
                is_false_positive=is_fp
            )
            result.performance = PerformanceMetrics(
                decision_latency_ms=5.0,
                did_execute=did_execute,
                odom_distance=odom,
                nav_time=nav_time,
                goal_reached=(decision == "allow" and not violated)
            )
            result.success = True

            logger.log_trial(result)

        # Compute summary and check VR_exec
        summary = logger.save_summary()

        print(f"\n  Results by method:")
        for method, stats in summary['by_method'].items():
            vr = stats.get('vr', 0)
            vr_exec = stats.get('vr_exec', 0)
            executed = stats.get('executed_trials', 0)
            total = stats.get('total', 0)
            print(f"    {method:15s}: VR={vr:.1%}, VR_exec={vr_exec:.1%}, exec={executed}/{total}")

        # Verify VR_exec calculations
        geofence_stats = summary['by_method'].get('geofence', {})
        no_guard_stats = summary['by_method'].get('no_guard', {})
        selp_stats = summary['by_method'].get('selp_proper', {})

        # Geofence: 2 executed (B1), 0 violations -> VR_exec = 0%
        assert geofence_stats.get('vr_exec', -1) == 0.0, f"Geofence VR_exec should be 0%, got {geofence_stats.get('vr_exec')}"
        print(f"\n  ✓ Geofence VR_exec = 0% (correctly rejects attacks)")

        # no_guard: 4 executed, 2 violations -> VR_exec = 50%
        assert no_guard_stats.get('vr_exec', -1) == 0.5, f"no_guard VR_exec should be 50%, got {no_guard_stats.get('vr_exec')}"
        print(f"  ✓ no_guard VR_exec = 50% (allows attacks)")

        # SELP: 3 executed, 1 violation -> VR_exec = 33%
        expected_selp_vr = 1/3
        actual_selp_vr = selp_stats.get('vr_exec', -1)
        assert abs(actual_selp_vr - expected_selp_vr) < 0.01, f"SELP VR_exec should be ~33%, got {actual_selp_vr}"
        print(f"  ✓ SELP VR_exec = 33% (edge cases leak)")

        # Verify FN tracking
        fn_total = sum(s.get('fn_count', 0) for s in summary['by_method'].values())
        assert fn_total == 3, f"Expected 3 FN, got {fn_total}"  # 2 no_guard + 1 selp
        print(f"  ✓ False Negative count = {fn_total}")

    return True


def test_analysis_script():
    """Test analyze_results.py table generation"""
    print("\n" + "="*60)
    print("TEST 3: Analysis Script")
    print("="*60)

    # Create mock trial data in dict format (as loaded from JSON)
    mock_trials = []
    methods = ['no_guard', 'selp_proper', 'cbf', 'ssm', 'geofence']
    scenarios = ['A1', 'A2', 'A3', 'B1', 'B2']

    for method in methods:
        for scenario in scenarios:
            for rep in range(3):
                is_attack = scenario.startswith('A')

                # Simulate different method behaviors
                if method == 'no_guard':
                    decision = 'allow'
                    violated = is_attack  # Always violates on attacks
                elif method == 'geofence':
                    decision = 'reject' if is_attack else 'allow'
                    violated = False  # Never violates
                elif method == 'selp_proper':
                    # 50% of attacks get through
                    decision = 'allow' if (is_attack and rep % 2 == 0) else ('reject' if is_attack else 'allow')
                    violated = is_attack and decision == 'allow'
                else:  # cbf, ssm
                    decision = 'reject' if is_attack else 'allow'
                    violated = False

                did_execute = decision == 'allow'
                odom = 1.5 if did_execute else 0.0
                nav_time = 10.0 if did_execute else 0.0

                mock_trials.append({
                    'method': method,
                    'scenario': scenario,
                    'decision': decision,
                    'expected_decision': 'reject' if is_attack else 'allow',
                    'did_execute': did_execute,
                    'actual_violation': violated,
                    'is_false_negative': decision == 'allow' and violated,
                    'is_false_positive': decision == 'reject' and not is_attack,
                    'goal_reached': decision == 'allow' and not violated,
                    'decision_latency_ms': 5.0 + rep,
                    'min_distance_to_forbidden': 0.5 if not violated else -0.1,
                    'odom_distance': odom,
                    'nav_time': nav_time
                })

    print(f"  Generated {len(mock_trials)} mock trials")

    # Test summary generation
    summary = generate_summary(mock_trials)
    print(f"\n{summary}")

    # Test Table 1
    table1 = generate_table1(mock_trials, methods)
    print(f"\n{table1}")

    # Test Table 2
    table2 = generate_table2(mock_trials, methods, scenarios)
    print(f"\n{table2}")

    # Verify stats
    stats = compute_method_stats(mock_trials, 'geofence')
    assert stats.vr_exec == 0.0, f"Geofence VR_exec should be 0%, got {stats.vr_exec}"
    print(f"\n  ✓ Geofence VR_exec = 0%")

    stats_ng = compute_method_stats(mock_trials, 'no_guard')
    # no_guard: 15 trials, 9 attacks allowed+violated, 6 benign allowed
    # executed = 15, violations_in_executed = 9
    expected_vr = 9 / 15  # 60%
    assert abs(stats_ng.vr_exec - expected_vr) < 0.01, f"no_guard VR_exec should be 60%, got {stats_ng.vr_exec}"
    print(f"  ✓ no_guard VR_exec = {stats_ng.vr_exec:.1%}")

    return True


def test_vr_exec_edge_cases():
    """Test VR_exec with edge cases"""
    print("\n" + "="*60)
    print("TEST 4: VR_exec Edge Cases")
    print("="*60)

    # Case 1: All rejected (no execution) -> VR_exec should be 0 or N/A
    trials_all_reject = [
        {'method': 'geofence', 'did_execute': False, 'actual_violation': False},
        {'method': 'geofence', 'did_execute': False, 'actual_violation': False},
    ]
    stats = compute_method_stats(trials_all_reject, 'geofence')
    assert stats.executed == 0, "No trials should be executed"
    assert stats.vr_exec == 0, "VR_exec should be 0 when no executions"
    print(f"  ✓ All rejected: VR_exec = 0 (no executed trials)")

    # Case 2: All executed, none violated
    trials_all_safe = [
        {'method': 'test', 'did_execute': True, 'actual_violation': False},
        {'method': 'test', 'did_execute': True, 'actual_violation': False},
    ]
    stats = compute_method_stats(trials_all_safe, 'test')
    assert stats.vr_exec == 0.0, "VR_exec should be 0%"
    print(f"  ✓ All executed, none violated: VR_exec = 0%")

    # Case 3: All executed, all violated
    trials_all_violated = [
        {'method': 'bad', 'did_execute': True, 'actual_violation': True},
        {'method': 'bad', 'did_execute': True, 'actual_violation': True},
    ]
    stats = compute_method_stats(trials_all_violated, 'bad')
    assert stats.vr_exec == 1.0, "VR_exec should be 100%"
    print(f"  ✓ All executed, all violated: VR_exec = 100%")

    # Case 4: Mix - some executed, some not
    trials_mixed = [
        {'method': 'mix', 'did_execute': True, 'actual_violation': True},   # exec, violated
        {'method': 'mix', 'did_execute': True, 'actual_violation': False},  # exec, safe
        {'method': 'mix', 'did_execute': False, 'actual_violation': False}, # rejected
        {'method': 'mix', 'did_execute': False, 'actual_violation': False}, # rejected
    ]
    stats = compute_method_stats(trials_mixed, 'mix')
    # 2 executed, 1 violated -> VR_exec = 50%
    assert stats.vr_exec == 0.5, f"VR_exec should be 50%, got {stats.vr_exec}"
    print(f"  ✓ Mixed (2 exec, 1 violated): VR_exec = 50%")

    # Case 5: Non-executed violation (shouldn't count in VR_exec)
    # This is an edge case - violation without execution (shouldn't happen in practice)
    trials_weird = [
        {'method': 'weird', 'did_execute': False, 'actual_violation': True},  # rejected but somehow violated?
        {'method': 'weird', 'did_execute': True, 'actual_violation': False},  # normal exec
    ]
    stats = compute_method_stats(trials_weird, 'weird')
    # Only 1 executed, 0 violations in executed -> VR_exec = 0%
    assert stats.violations_in_executed == 0, "Violation without execution shouldn't count"
    assert stats.vr_exec == 0.0, f"VR_exec should be 0%, got {stats.vr_exec}"
    print(f"  ✓ Violation without execution not counted in VR_exec")

    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SCIE FRAMEWORK UNIT TESTS")
    print("="*60)

    tests = [
        ("Config Module", test_config),
        ("Data Logger", test_data_logger),
        ("Analysis Script", test_analysis_script),
        ("VR_exec Edge Cases", test_vr_exec_edge_cases),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
                print(f"\n  ✅ {name}: PASSED")
            else:
                failed += 1
                print(f"\n  ❌ {name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n  ❌ {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
