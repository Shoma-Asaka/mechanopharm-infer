from mechanopharm_infer.synthetic import (
    SyntheticBenchmarkConfig,
    analyze_synthetic_dataset,
    generate_protected_state_endpoint,
    generate_protected_state_timecourse,
    generate_two_state_endpoint,
    generate_two_state_timecourse,
    run_synthetic_benchmark,
)


def test_two_state_generators_produce_expected_columns():
    endpoint = generate_two_state_endpoint(n_replicates=2, random_seed=1)
    timecourse = generate_two_state_timecourse(n_replicates=2, random_seed=2)
    assert {'c', 'm', 'response', 'replicate'}.issubset(endpoint.columns)
    assert {'c', 'm', 'time', 'value', 'replicate'}.issubset(timecourse.columns)


def test_protected_state_analysis_returns_supported_fingerprints():
    endpoint = generate_protected_state_endpoint(n_replicates=3, noise_sd=0.005, random_seed=3)
    timecourse = generate_protected_state_timecourse(n_replicates=3, noise_sd=0.005, random_seed=4)
    out = analyze_synthetic_dataset(endpoint, timecourse, n_boot=20, random_seed=5)
    evidence = out['evidence_df']
    assert 'fingerprint' in evidence.columns
    assert out['result']['label'] in {'protected_state_suggested', 'two_state_supported', 'inconclusive'}


def test_run_synthetic_benchmark_returns_two_rows():
    cfg = SyntheticBenchmarkConfig(n_boot=10, n_replicates=2, endpoint_noise_sd=0.005, timecourse_noise_sd=0.005, random_seed=7)
    bench = run_synthetic_benchmark(cfg)
    assert list(bench['benchmark_case']) == ['two_state', 'protected_state']
    assert {'predicted_label', 'matched_expected'}.issubset(bench.columns)
