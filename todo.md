TODO: Head‑only ViT→Embeddings→SAE→Gene‑Heads→CRP/RelMax

Config switch to dump embeddings

Add config flag output_embeddings: true/false.

Add config.embedding_source: {layer_name, per_token: true|false, pooled: true|false}.

Export encoder embeddings for epochs form config(encoder is frozen)

In your existing training loop, branch on output_embeddings:
forward images → model.encoder; capture pre‑pool token embeddings if per_token is desired.

Save per sample: {sample_id, tile_id, token_rc (if per_token), embedding [D], augment_seed, split_tag}.

Write shards to disk; keep an index file of shard paths.

Compute & persist normalization

One pass over shards: compute mean/std (or PCA‑whitening if configured).

Save to {embed_stats.json}; add config.embed_norm_path.

SAE training on cached embeddings

CLI/entrypoint train_sae reads shards → applies normalization → trains SAE.

SAE spec in config: {dict_size (e.g., 4–8×D), top_k, l1, signed_codes: true|false, re_norm_interval}.

If per‑token embeddings: train on token vectors; keep forward mask for top‑k.

Save sae.pt + scaler state and code “lifetime/usage” stats.

Code extraction (optional cache)

Transform shards: embeddings → SAE.codes (store per token if applicable).

Persist {sample_id, token_rc, codes[K]} or compute on‑the‑fly during head training.

Gene‑head training

Targets loader: align gene labels to {sample_id}; stratify splits by patient/site.

Pooling config if per‑token: {pool: mean|max|attention} to get tile/slide‑level code vectors.

Head spec: {loss: mse|poisson|nb, model: linear|elastic_net|ridge, multi_output: true}.

Train; log per‑gene top positive/negative codes; save heads.pt.

CRP/RelMax (SAE→head only; ViT excluded)

Add module path for SAE latent layer name (e.g., model.sae.latent).

Implement function crp_relmax(top_module=sae→gene_head, inputs=embeddings, conditions={y: gene_idx, layer: [code_ids]}).

Produce:
a) per‑code relevance scores (global & per-sample),
b) RelMax top‑k reference samples per code,
c) if per‑token: token‑level relevance maps (aggregate to patches).

Save artifacts to disk (refs/, relevance/, token_maps/).

Visualization & reports

Prototype gallery per SAE code (top RelMax samples + token maps if available).

Per‑gene report: top ± codes, coefficients, example samples, validation metrics (R²/Spearman).

CLI tasks (minimum)

embed_export --config cfg.yaml

train_sae --config cfg.yaml

train_heads --config cfg.yaml

crp_relmax --config cfg.yaml --gene gene_id --codes topN

make_reports --config cfg.yaml

Guards

If encoder fine‑tuned: invalidate cache and re‑export embeddings.

Enforce split by patient/site; store split assignment with shards.

Reference for CRP/RelMax method (concept‑conditional relevance, concept sums, RelMax selection): see Eqs. 7–10 & 17 in the CRP paper.