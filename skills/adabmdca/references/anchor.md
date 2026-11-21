# Anchor selection and measured costs

bmDCA trained from scratch starts from the **anchor**, a tractable coupled model fitted to the data before training: it is sampled exactly and its log Z is known. `--anchor` picks it: `tree-low-rank` (default: Chow-Liu tree plus Gaussian hidden units), `low-rank`, `tree` or `profile`. With PTT it replaces the profile at the bottom of the ladder; with PCD it only gives the initial parameters, and exact samples of it the initial chains. eaDCA, edDCA, edgeDCA, warm starts with `-p`, `--ptt-resume` and `reintegrate` start from the profile (an explicit other `--anchor` there is an error). With PTT:

- the final ladder has fewer models (on RF00023, 6 instead of 10 at the same training log-likelihood), so
- sampling the trained model with PTT is faster, since every model of the ladder costs as much as the endpoint, and
- training quality can improve: at equal training log-likelihood the validation log-likelihood was higher.

Its fit runs before the first update and prints `Fitting the tree-low-rank anchor (...)`, then the anchor's log-likelihood per site next to the profile's.

`--anchor-rank K` (1 to 3, default 2) sets the rank of the low-rank anchors. **K = 2 is the default compromise** between what the anchor captures and the time to fit it. K = 3 gains a little (+0.007 to +0.02 log-likelihood per site over K = 2) and is a valid choice for small systems. One evaluation of the fit costs about P_K · L · q², where P_K, the number of latent lattice points, grows by a factor of 20–30 per extra rank and is larger for strongly clustered families. Measured fit times on one RTX A5000:

| Family | L | q | K = 2 | K = 3 |
| --- | --- | --- | --- | --- |
| RF00379 (RNA) | 136 | 5 | 18 s | 54 s |
| RF00023 (RNA) | 374 | 5 | 29 s | 5.4 min |
| PF00072 | 111 | 21 | 24 s | 7 min |
| Chorismate mutase | 96 | 21 | 69 s | > 15 min |
| LBD | 279 | 21 | 11 min | > 15 min (estimated hours) |

These measurements suggest considering K = 3 for RNA families (q = 5) and protein domains up to about 100–150 positions when the extra fit cost is justified; K = 2 is a practical starting point for longer proteins. These are hardware- and family-dependent observations, not universal cutoffs. A K = 2 fit that reports more than about 2,000 lattice points (the progress line shows them) signals a clustered family where K = 3 will be slow even at small L.
