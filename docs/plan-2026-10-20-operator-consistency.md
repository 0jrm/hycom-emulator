# Plan: operator-consistency losses for the reanalysis emulator (draft, 2026-10-20)

Draft for discussion. Nothing here has run. The emulator line has been paused since 2026-10-13. This plan asks to reopen it for one diagnostic card (Phase 0) and one gated training card (Phase 2).

## The idea as proposed

Compute horizontal, vertical and temporal operators of the fields. Predict them with extra heads. Add loss terms that tie three things together: the operator computed on the predicted fields, the operator predicted directly, and the operator of the truth.

## What the idea buys, operator by operator

**Linear operators of one field are already in the loss.** For a linear operator L (gradient, Laplacian, finite-difference vertical derivative), L(pred) − L(truth) = L(error). An L2 penalty on it is the MSE with each wavenumber reweighted by the operator's response. That is |k|² for a gradient and |k|⁴ for a Laplacian. It is a spectral weighting of the error the model already minimizes. Two repo results bear on it.

- The track E `grad` arm put an L1 penalty on error gradients. It blurred T and doubled the basin-mean SSH offset (SSH RMSE +29.5%).
- `amse` in `rea_loss.py` already attacks the small-scale energy loss directly.

At grid scale on the stride-2 pack the weighting is also fed noise. `rea_pack` subsamples every second point (`slice(None, None, stride)`), so grid-scale differences of the truth are aliased.

**A directly predicted head of a linear operator is redundant.** The head and L(field) see the same targets. Their consistency term has a single fixed point, L(pred) = L(truth), which the field loss already targets. The head adds parameters and a second, noisier target. It earns its place only if it is used at inference. One such use is a self-consistency monitor that flags a rollout leaving the training manifold. That is a diagnostic, not a skill lever.

**Temporal operators over one step equal the main loss.** neural-lam predicts the one-day tendency, so ∂t(pred) − ∂t(truth) over one step is the error divided by one day. A temporal operator over several steps is new information, because it scores the trend across a rollout. Slow drift is the failure the conservation note measured (SSH +0.58 cm/day, T +5.1 mK/day at lead 1 on 2024).

**Cross-field and nonlinear ties are where the idea is new.** These terms couple fields that the per-channel MSE treats as independent. A long rollout breaks them first, and no reweighting of the per-channel error restores them.

| Tie | Fields coupled | Exact in truth? | Penalized quantity |
|---|---|---|---|
| Geostrophic surface velocity | ssh vs u, v at 0 m | approx., holds at ≥ 50 km | (u − u_geo(ssh))_pred − (same)_truth at a Gaussian scale |
| Thermal wind | ρ(T,S) vs ∂z u, ∂z v | approx., holds below the mixed layer away from the shelf | f ∂z v − (g/ρ0) ∂x ρ residual, pred vs truth |
| Barotropic continuity | ∂t ssh vs ∇·(H ū) | near exact for 0–24 h means, minus rivers and E−P | residual, pred vs truth |
| Static stability | T, S across levels (nonlinear σ2) | N² ≥ 0 almost everywhere | relu(−N²) on prediction |
| Multi-step trend | each Gulf-mean quantity across a rollout | exact | LSQ slope over the pushforward window, pred vs truth |

Every tie is scored against the truth's own residual, never against zero. Geostrophy and thermal wind are not exact at 0.08°, and the truth's ageostrophic part must survive.

## What NeSPReSO v2 contributes

NeSPReSO2_onTemplate (`preproc/features/operators.py`, the 19 `OP_NAMES` of the heave `ops` cache, `model/density_spice.py`) holds five ideas worth reusing.

1. **Operators at a named scale, never at grid scale.** Every derivative is a mask-normalized Gaussian derivative (`normalized_gaussian_derivative`) at `@local` and `@1.0deg`. The emulator should do the same. Use two Gaussian sigmas, 15 km (the Loop Current front) and 50 km (its eddies), as `operators.SCALES_KM`. A Gaussian derivative at scale σ weights the error by |k|² exp(−σ²k²). That is a band-pass on the mesoscale, not a high-pass that amplifies the aliased grid scale. The track E `grad` arm lacked it, and it is one candidate reason that arm failed.
2. **Mask normalization at the coast.** Smooth the mask with the same kernel and divide by it, and drop points whose weight is under 0.5. Per-level validity comes from `level_ocean`, so the operator never differences across the sea floor.
3. **Split the basin mean from the pattern.** `value_centered` removes the day's basin mean before scale-space operators. Derivatives ignore the constant offset by construction. The mean goes to its own term, which `rea_wmse` already has as term A. The geostrophic and thermal-wind ties therefore leave SSH drift alone, and term A handles it.
4. **Tendency as a 7-day LSQ slope.** `op_tendency` fits a slope through a window instead of differencing two days. In the emulator that becomes the multi-step trend tie over the 8-day pushforward window.
5. **Hard structure beats a soft penalty when the constraint is exact.** `decode_sigma0_ctrl` makes σ0 monotone in depth through softplus increments, so an inversion cannot be represented. The heave note warns that a model can win T RMSE and lose N². The emulator predicts T and S tendencies on fixed z-levels, so a monotone-density head would mean a new output parameterization. Phase 3 lists it. The first card keeps N² as a soft relu penalty, which track E measured at 8× fewer inversions for 2–3% RMSE.

One NeSPReSO lesson argues against a variant. Operators as inputs gave NeSPReSO a noise-scale gain (ops T RMSE 0.549 vs HeaveFast 0.550, D26 18.27 vs 18.46 m). That was a point model with no neighbourhood. graph_lam's message passing already learns local stencils from the state. Operator input channels are left out.

## Outside evidence

- Samudra 2 names the long-rollout failures as variance collapse and velocity patterns imprinting on deep fields. Its fix is a wider U-Net and a loss reweighted by channel error, not a physics term ([arXiv 2606.02610](https://arxiv.org/pdf/2606.02610)).
- OceanNet uses a predictor-corrector scheme plus a spectral regularizer for small-scale bias ([NSF PAR](https://par.nsf.gov/biblio/10550338-oceannet-principled-neural-operator-based-digital-twin-regional-oceans)).
- NeuralOGCM reports 90-day rollouts without drift by embedding learnable physics in the model, where a plain U-Net went unstable ([arXiv 2512.11525](https://arxiv.org/pdf/2512.11525)).
- Thermalizer stabilizes long autoregressive QG rollouts with a diffusion-model correction at inference ([arXiv 2503.18731](https://arxiv.org/pdf/2503.18731)).

One standard search found no ocean emulator that trains on a geostrophic or thermal-wind balance loss. The term is untested in this literature, not known to fail. The published stability fixes are structural (architecture, rollout training, inference-time correction), not penalty terms.

## Phase 0: measure what breaks first (no training)

The claim "unphysical real fast" has no lead time or symptom attached yet. The handoff measured 15-day free runs at about 0.7 of persistence RMSE, KE at 0.83 of truth, and SSH +2 cm. Something else must be visibly wrong in the movies. Phase 0 names it before any loss is chosen.

1. Write `src/hycom_emulator/operators.py`, a torch port of the NeSPReSO registry on the Mercator grid. It carries a per-row dx and dy from `regional.grid`, mask-normalized Gaussian derivatives at named scales, σ2-based N², and the five ties above as residual functions. It must be differentiable and batched over (batch, step, grid, feature). Test it on analytic fields (a Gaussian eddy in geostrophic balance gives a zero geostrophic residual, a linear stratification gives a constant N²) and on the land mask.
2. Measure the truth. Take each tie's residual distribution on 2001–2021 rows of `rea_s2.pack`, per scale and per depth band. A tie whose truth residual is as large as its signal is dropped here. That is the check for barotropic continuity on daily means at stride 2.
3. Measure revert-fp16. Run free rollouts from 2024 test starts to 60 days, extending the 21-day horizon to locate the break. Per lead, record:
   - each tie's residual RMS relative to truth;
   - the inversion fraction;
   - the spectra of relative vorticity ζ/f and divergence δ/f;
   - the operator-field RMS ratios (model/truth) at both scales;
   - the 7-day trend error of each Gulf-mean quantity.
4. Output a "physicality scorecard" table and plot (lead × diagnostic). Each diagnostic gets the lead at which it leaves the truth envelope, read as the 5–95% band of the truth over the same calendar days.

Cost is inference only, about 2–4 GPU-hours on one skynet GPU from 1–3, plus CPU for the truth pass. It reuses the frozen clone pattern and `rollout_rea.py`.

**Decision gate after Phase 0.** Phase 2 trains only the ties that leave the envelope early, before day 10, and have a small truth residual. If the first failure is blur or variance collapse, the lever is `amse` weight and rollout length, not new ties. If it is drift, the lever is term A and forcing. In both cases Phase 2 is replaced by that lever. If nothing leaves the envelope before 21 days, the problem lies past the 3-week horizon of interest and the card closes.

## Phase 1: the loss module

Add `operator_ties` to `rea_wmse` the same way term A and `amse` were added. Each tie has its own weight and its own scale s_tie, the train std of the truth residual, so every term is dimensionless. Weight 0 reproduces `rea_wmse` exactly. That is a unit test, as it was for term A. The term applies at every rollout step that carries a gradient, including pushforward steps after `--train_first_step`. The multi-step trend tie applies once per rollout window.

The optional directly predicted head is not built. If the Phase 0 scorecard shows that a mismatch between a head and the computed operator would forecast a rollout's failure, the head becomes a separate diagnostic card.

## Phase 2: one gated training card

- Base is revert-fp16, continuing its schedule at or below 6.25e-6 (handoff lesson 3), with 8- and 16-day stages and the same pushforward policy in every arm.
- Arms are a control (same fine-tune, all tie weights 0), plus "ties" (the Phase 0 survivors), plus "ties+N²".
- Use 3 seeds per arm (handoff lesson 2). That is 9 runs at about 4–6 GPU-hours each for an 8+16-day fine-tune, so 40–55 GPU-hours on GPUs 1–3.
- Judge on the 2024 test only (lesson 1). There are three hypotheses, fixed before training.
  - H1. The ties arm moves the Phase 0 failure lead later by at least 5 days, on the diagnostic that failed first.
  - H2. RMSE at 8–21 days is within +2% of control on every field, against a seed spread of 3–6%.
  - H3. The isotach front Hausdorff at 8–21 days is no worse than control.
- Smoke first, one 50-step run per arm (smoke-before-full-runs).

## Phase 3: only if Phase 2 passes H1

- A structural version of the winning tie. One option predicts a streamfunction and velocity potential for surface u, v, so the geostrophic part is built in. Another uses a monotone-σ2 output on the z-levels, after NeSPReSO `density_spice`.
- A stride-1 check of whether the ties transfer to 0.04°.

## What this plan does not touch

- The handoff names a no-training regime check (2024 vs 2001–2021 statistics) and heat and freshwater forcing as the levers for long-lead error. No operator loss substitutes for missing forcing. If Phase 0 shows that T and SSH drift dominate, that recommendation stands ahead of Phase 2.
- GoM-only training is not the cause of rollout instability as such. The boundary band is given from truth every step. It limits generalization to other regions, which is a different question.

## Open questions for the user

1. What did "unphysical" look like? Noise, inversions, a frozen or vanishing Loop Current, drift, boundary artifacts? At which lead? Phase 0 measures it either way, but the answer sets the first diagnostic to plot.
2. Does Phase 0 reopen the emulator line during P2–P5, or wait?
