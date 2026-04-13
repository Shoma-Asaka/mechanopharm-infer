from mechanopharm_infer.bootstrap import bootstrap_ec50_vs_m, bootstrap_mopt
from mechanopharm_infer.diagnostics import combine_diagnostics, diagnostics_messages, endpoint_diagnostics, timecourse_diagnostics
from mechanopharm_infer.fingerprints import delayed_protection_metrics, ec50_vs_m, endpoint_final_response, find_mechanical_optima, peak_metrics_by_condition
from mechanopharm_infer.io import load_endpoint_csv, load_timecourse_csv
from mechanopharm_infer.preprocess import endpoint_to_grid, split_timecourses_by_condition, summarize_endpoint, summarize_timecourse


def test_endpoint_diagnostics_flags_sparse_mechanics(tmp_path):
    csv = tmp_path / 'endpoint.csv'
    csv.write_text('c,m,response,replicate\n0,0,0.1,1\n1,0,0.4,1\n2,0,0.8,1\n0,1,0.2,1\n1,1,0.5,1\n2,1,0.9,1\n')
    df = load_endpoint_csv(csv)
    summary = summarize_endpoint(df)
    c_grid, m_grid, resp = endpoint_to_grid(summary)
    ec50 = ec50_vs_m(c_grid, m_grid, resp)
    mopt = find_mechanical_optima(c_grid, m_grid, resp)
    diag = endpoint_diagnostics(summary, ec50_df=ec50, mopt_df=mopt)
    assert 'mechanics_levels' in set(diag['item'])
    row = diag.loc[diag['item'] == 'mechanics_levels'].iloc[0]
    assert row['status'] in {'warning', 'not_assessable'}


def test_timecourse_diagnostics_reports_missing_dataset():
    diag = timecourse_diagnostics(None)
    assert diag.iloc[0]['status'] == 'not_assessable'


def test_combine_and_messages_examples():
    endpoint = endpoint_diagnostics(summary_df=load_endpoint_csv('examples/demo_endpoint.csv').pipe(summarize_endpoint))
    msgs = diagnostics_messages(endpoint, min_severity='medium')
    assert isinstance(msgs, list)
    combined = combine_diagnostics(endpoint, None)
    assert len(combined) >= len(endpoint)
