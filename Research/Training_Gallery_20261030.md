# Training monitor sample galleries

URL: http://127.0.0.1:55318/ (existing SSH tunnel).

Each run has a validation section above a training section, 48 fixed samples per
split and six rows per page. Columns show input, reference, generated target and
checkpoint prediction. Click an image to view it without cropping. Source labels
and pair IDs remain visible. The same IDs are used for every model; selection is
source-stratified with distinct content images within each split, seed 20261030.
These examples are for inspection, not a substitute for aggregate validation.

Predictions come from saved checkpoints, not the current in-memory training
weights or the most recent training batch. The checkpoint step is displayed.
Test images are excluded. No training jobs were restarted for this change.

The compute-node tmux session `mae-gallery-preview` uses GPU 9 independently of
the nine existing training jobs. It polls checkpoint changes every 30 seconds.
Its code, log, fixed selection, shared image assets and manifests live under
`/tmp/shihangz/photo-tone-transfer/Experiments/mae_full_20261030/` as
`gallery_code/`, `gallery_worker.log`, `gallery_selection.json`, `gallery_assets/`
and `runs/<run>/gallery.json`. It keeps two output versions per run and shares
input/reference/target thumbnails between runs. No original image is changed.

Only `mae-full-monitor` was restarted to serve the new page. Both services run on
the compute node; local disconnection does not stop them or training. Reopen the
existing local tunnel with `Dataset/Open_Training_Monitor.ps1` when necessary.

Validation: three gallery selection/pagination/path isolation tests passed;
Chrome verified nine experiment cards, separate six-row galleries, pagination,
image zoom, switching to the LoRA run and no horizontal overflow at 390px width.
