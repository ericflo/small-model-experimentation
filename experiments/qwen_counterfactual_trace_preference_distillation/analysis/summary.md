# Analysis Summary

Main run: `main_counterfactual_trace_preference_s192_c1024`
Candidate surface: `246784` candidates, 45.5% prompt-level oracle, 31.7% counterfactual groups.
Preference selector fresh paired: direct 13.3%, selected 11.7%, oracle 47.7%.
Preference-selected distill fresh paired direct/search: 15.6% / 46.1%.
Answer-verified distill fresh paired direct/search: 14.8% / 53.9%.
Full-supervised ceiling fresh paired direct/search: 91.4% / 97.7%.

Figures:
- `analysis/figures/main_accuracy_by_phase.png`
- `analysis/figures/preference_selector_vs_oracle.png`
- `analysis/figures/candidate_surface.png`
- `analysis/figures/target_quality.png`
- `analysis/figures/training_and_preference_curves.png`
- `analysis/figures/selector_iteration_comparison.png`
