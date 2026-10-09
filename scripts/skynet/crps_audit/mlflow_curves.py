import sqlite3, sys, math
import numpy as np
db = sqlite3.connect(sys.argv[1])
runs = db.execute("select run_uuid, name from runs order by start_time").fetchall()
for uuid, name in runs:
    keys = [k for (k,) in db.execute("select distinct key from metrics where run_uuid=?", (uuid,))]
    print(f"== run {name} keys {sorted(keys)}")
    for key in ("train_loss_step", "train_loss_epoch", "val_mean_loss", "val_spread_skill", "val_loss_unroll1", "val_loss_unroll4", "lr"):
        rows = db.execute("select step, value, is_nan from metrics where run_uuid=? and key=? order by step, timestamp", (uuid, key)).fetchall()
        if not rows:
            continue
        v = np.array([float('nan') if n else x for _, x, n in rows])
        if key == "train_loss_step":
            nn = np.isnan(v) | np.isinf(v)
            fin = v[~nn]
            chunks = np.array_split(fin, min(12, len(fin))) if len(fin) else []
            print(f"  {key}: n={len(v)} nonfinite={nn.sum()} steps_nonfinite={[rows[i][0] for i in np.where(nn)[0][:10]]}")
            print(f"    chunk medians: {[round(float(np.median(c)),3) for c in chunks]}")
            print(f"    first5 {np.round(v[:5],3).tolist()} last5 {np.round(v[-5:],3).tolist()} min {np.nanmin(v):.3f}")
        else:
            print(f"  {key}: {np.round(v, 4).tolist()[:40]}")
