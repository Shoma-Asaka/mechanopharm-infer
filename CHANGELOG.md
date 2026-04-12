# Changelog

All notable changes to `mechanopharm-infer` will be documented in this file.

The project is currently in an early prototype stage. Version numbers below should be interpreted accordingly.

## [0.0.2] - 2026-04-12

### Added
- Packaged `mechanopharm_infer` module structure
- Endpoint and timecourse QC checks
- Reliability and warning flags for key fingerprints
- Delayed-protection metrics separated from transient-peak detection
- End-to-end CLI outputs for QC summaries, fingerprints, discrimination results, plots, and text reports
- Example datasets for smoke testing
- Basic test suite covering I/O, preprocessing, fingerprints, discrimination, and end-to-end execution

### Changed
- Architecture discrimination now accounts for QC status and fingerprint reliability
- Shift detection is now data-driven rather than assumed
- Mechanical optimum detection now includes prominence- and reliability-aware logic
- Timecourse-derived signals are handled more conservatively, with transient-peak and delayed-protection evidence separated
- Report output now includes QC summaries and delayed-protection metrics

### Fixed
- Resolved module-layout issues that could interfere with imports
- Improved robustness of EC50 and peak-related summaries for incomplete or weak datasets
- Reduced false-positive architectural calls in low-information cases by allowing inconclusive outputs

### Notes
- This release is an early research prototype intended for response-landscape analysis and architecture-class inference workflows
- The API, thresholds, and decision logic may change in future releases as the toolkit evolves toward a full methods-paper implementation

## [0.0.1] - Initial prototype

### Added
- Initial packaged CLI workflow for endpoint and timecourse analysis
- Basic response summarization and fingerprint extraction
- Initial rule-based architecture discrimination
- Minimal plots and text report generation
