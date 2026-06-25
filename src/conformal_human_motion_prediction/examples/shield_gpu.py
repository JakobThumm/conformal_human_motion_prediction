"""GPU (JAX) backend for the robot-shield simulation's fine-phase checks.

Levels 1-3 of the bounding-sphere culling stay on the CPU (cheap, per-pose -- see
``pose_active_set`` in :mod:`simulate_robot_shield`). For each surviving pose this module replaces
culling levels 4-5 + the KDTree narrow phase with a dense, brute-force segment-point distance
evaluation over the level-3-active human motions, batched on the GPU.

Why this is sound / parity-exact: the over-approximation hierarchy is *conservative* -- anything it
culls provably cannot intersect. So computing the exact segment-point test over **all** active
motions (skipping levels 4-5) yields the same verified / contact / unsafe booleans the CPU path
gets by testing only the level-4/5 survivors. The CPU's ``~not_verified`` / ``~contact`` masks are
pure work-savers and do not change the final verdict. Counts therefore match
``run_pose_pure`` up to float rounding at exact tangency -- run ``--parity`` to confirm.

Layout exploited for speed: the human occupancy arrays (``pred_c/pred_r/true_c/true_r``) and the
base-frame robot capsules are pose-invariant, so they are uploaded to the device once at
construction. Per pose only the ``(rot, t)`` transform (12 floats) and the active-motion index list
cross to the GPU; only scalar counts come back (plus, when ``capture_failures`` is set, the indices
of the rare verified-but-contact pairs, for the CPU to reconstruct ``--save_failures`` geometry).

Capsule bookkeeping. Every (trajectory, interval-row, link) yields one capsule. For the predicted
check the capsule is the trajectory's own link occupancy, grown by ``addr = (tp - hz[s])*v_human``
to bridge to the exact interval time, tested at horizon step ``s = last_step_below(hz, tp)``. For
the true check the capsule is the *future* interval-0 link occupancy (``frow``), tested at
``s = nearest_step(hz, tp)``; a capsule whose link speed exceeds ``V_ROBOT_ISO`` is ``fast`` and can
flip a contact to *unsafe*. Capsules are grouped by horizon step (so a step-``s`` capsule only meets
step-``s`` human spheres) and padded to a rectangular ``[S, Cmax]`` table; the per-(traj, motion)
verdict is a scatter-max (boolean OR) over each capsule's ``traj`` id.
"""
import numpy as np

from conformal_human_motion_prediction.examples.simulate_robot_shield import (
    N_LINKS,
    last_step_below,
    nearest_step,
    pose_rt,
)


def _build_pred_tables(st):
    """Pose-invariant predicted-capsule table, grouped+padded by horizon step.

    Returns gid [S,Cmax] (flat row*N_LINKS+link gather index), effr [S,Cmax] (radius incl. the
    V_HUMAN_ISO bridge), traj [S,Cmax] (trajectory id), valid [S,Cmax] (1.0 real / 0.0 pad), Cmax.
    """
    hz = st.horizon_times
    vh = st.v_human
    S = len(hz)
    buckets = [[] for _ in range(S)]
    for tj, meta in enumerate(st.traj_meta):
        rows = meta[0]
        for row in rows:
            row = int(row)
            tp = st.tp_start[row]
            s = last_step_below(hz, tp)
            addr = (tp - hz[s]) * vh
            for a in range(N_LINKS):
                buckets[s].append((row * N_LINKS + a, float(st.rr[row, a] + addr), tj))
    Cmax = max((len(b) for b in buckets), default=1) or 1
    gid = np.zeros((S, Cmax), np.int64)
    effr = np.full((S, Cmax), -1e30, np.float64)
    traj = np.zeros((S, Cmax), np.int64)
    valid = np.zeros((S, Cmax), np.float64)
    for s, b in enumerate(buckets):
        for i, (g, e, t) in enumerate(b):
            gid[s, i], effr[s, i], traj[s, i], valid[s, i] = g, e, t, 1.0
    return gid, effr, traj, valid, Cmax


def _build_true_tables(st):
    """Pose-invariant true-capsule table (future interval-0 link occupancy), grouped by step.

    Returns gid [S,Cmax], tr [S,Cmax] (capsule radius), fast [S,Cmax] (1.0 if link speed > V_ROBOT),
    traj [S,Cmax], valid [S,Cmax], Cmax.
    """
    hz = st.horizon_times
    vr = st.v_robot
    S = len(hz)
    buckets = [[] for _ in range(S)]
    for tj, meta in enumerate(st.traj_meta):
        rows, frow_rows = meta[0], meta[1]
        for i, row in enumerate(rows):
            frow = int(frow_rows[i])
            if frow < 0:
                continue
            s = nearest_step(hz, st.tp_start[int(row)])
            for a in range(N_LINKS):
                fast = 1.0 if st.spd[frow, a] > vr else 0.0
                buckets[s].append((frow * N_LINKS + a, float(st.rr[frow, a]), fast, tj))
    Cmax = max((len(b) for b in buckets), default=1) or 1
    gid = np.zeros((S, Cmax), np.int64)
    tr = np.zeros((S, Cmax), np.float64)
    fast = np.zeros((S, Cmax), np.float64)
    traj = np.zeros((S, Cmax), np.int64)
    valid = np.zeros((S, Cmax), np.float64)
    for s, b in enumerate(buckets):
        for i, (g, r, f, t) in enumerate(b):
            gid[s, i], tr[s, i], fast[s, i], traj[s, i], valid[s, i] = g, r, f, t, 1.0
    return gid, tr, fast, traj, valid, Cmax


class GpuShieldEvaluator:
    """Dense GPU evaluator for one pose's fine-phase (levels 4-5) shield check.

    Counters returned by :meth:`eval_active` match ``run_pose_pure``'s for the same pose. The
    accounting is over *all* M motions per trajectory (inactive motions are trivially verified /
    contact-free), so ``total_pairs = n_traj * M`` and ``n_verified = n_traj*M - (not-verified
    count over active motions)``.
    """

    def __init__(self, st, n_traj, dtype="float32", a_chunk=512, capture_failures=False):
        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        self.f = jnp.float64 if dtype == "float64" else jnp.float32
        self.n_traj = int(n_traj)
        self.a_chunk = int(a_chunk)
        # When True, eval_active also returns the (trajectory, motion, unsafe) triples flagged
        # verified-but-contact, so the caller can reconstruct their full geometry on the CPU. The
        # kernel always computes the mask (cheap); only the host-side index pull is gated on this.
        self.capture_failures = bool(capture_failures)
        self.M = int(st.M)
        self.J = int(st.J)
        self.S = len(st.horizon_times)
        self.total_no_truth = int(st.total_no_truth)

        # Resident, pose-invariant device arrays.
        self.pred_c = jnp.asarray(st.pred_c, self.f)   # [M,S,J,3]
        self.pred_r = jnp.asarray(st.pred_r, self.f)   # [M,S,J]
        self.true_c = jnp.asarray(st.true_c, self.f)
        self.true_r = jnp.asarray(st.true_r, self.f)
        self.P1 = jnp.asarray(st.p1.reshape(-1, 3), self.f)   # [R*N_LINKS,3]
        self.P2 = jnp.asarray(st.p2.reshape(-1, 3), self.f)

        pg, pe, pt, pv, self.Cp = _build_pred_tables(st)
        tg, tr, tf, tt, tv, self.Ct = _build_true_tables(st)
        self.pg, self.pt = jnp.asarray(pg), jnp.asarray(pt)
        self.pe, self.pv = jnp.asarray(pe, self.f), jnp.asarray(pv, self.f)
        self.tg, self.tt = jnp.asarray(tg), jnp.asarray(tt)
        self.tr, self.tf, self.tv = (
            jnp.asarray(tr, self.f), jnp.asarray(tf, self.f), jnp.asarray(tv, self.f),
        )
        self._chunk = self._make_chunk()

    def _make_chunk(self):
        """Build the jitted per-active-chunk kernel (fixed shapes -> compiles once)."""
        jnp = self.jnp
        f = self.f
        nt, S = self.n_traj, self.S
        pred_c, pred_r = self.pred_c, self.pred_r
        true_c, true_r = self.true_c, self.true_r
        pg, pe, pt, pv = self.pg, self.pe, self.pt, self.pv
        tg, tr, tf, tt, tv = self.tg, self.tr, self.tf, self.tt, self.tv

        def seg(a, b, pts):
            """Segment(a,b)-to-point distances. a,b [C,3]; pts [A,J,3] -> [C,A,J]."""
            ab = b - a                                   # [C,3]
            denom = jnp.sum(ab * ab, axis=-1)            # [C]
            safe = jnp.maximum(denom, 1e-18)
            ap = pts[None] - a[:, None, None, :]         # [C,A,J,3]
            t = jnp.sum(ap * ab[:, None, None, :], axis=-1) / safe[:, None, None]
            t = jnp.clip(t, 0.0, 1.0)
            proj = a[:, None, None, :] + t[..., None] * ab[:, None, None, :]
            d = jnp.linalg.norm(pts[None] - proj, axis=-1)
            d_deg = jnp.linalg.norm(ap, axis=-1)         # degenerate capsule -> point distance
            return jnp.where(denom[:, None, None] <= 1e-18, d_deg, d)

        def chunk(P1r, P2r, midx, cmask):
            ac = midx.shape[0]
            pc, pr = pred_c[midx], pred_r[midx]          # [ac,S,J,3], [ac,S,J]
            tc, trd = true_c[midx], true_r[midx]
            zeros = jnp.zeros((nt, ac), f)

            not_verified = zeros
            for s in range(S):
                a_pt, b_pt = P1r[pg[s]], P2r[pg[s]]       # [Cp,3]
                d = seg(a_pt, b_pt, pc[:, s])             # [Cp,ac,J]
                thr = pe[s][:, None, None] + pr[:, s][None]
                hit = (d <= thr) & (pv[s][:, None, None] > 0)
                hit_cm = jnp.max(hit.astype(f), axis=2)   # [Cp,ac] (any joint)
                contrib = jnp.zeros((nt, ac), f).at[pt[s]].max(hit_cm)
                not_verified = jnp.maximum(not_verified, contrib)

            contact = zeros
            unsafe = zeros
            for s in range(S):
                a_pt, b_pt = P1r[tg[s]], P2r[tg[s]]
                d = seg(a_pt, b_pt, tc[:, s])
                thr = tr[s][:, None, None] + trd[:, s][None]
                hit = (d <= thr) & (tv[s][:, None, None] > 0)
                hit_cm = jnp.max(hit.astype(f), axis=2)   # [Ct,ac]
                contact = jnp.maximum(contact, jnp.zeros((nt, ac), f).at[tt[s]].max(hit_cm))
                hit_fast = hit_cm * tf[s][:, None]
                unsafe = jnp.maximum(unsafe, jnp.zeros((nt, ac), f).at[tt[s]].max(hit_fast))

            nvb = not_verified > 0.5
            cb = contact > 0.5
            ub = unsafe > 0.5
            cm = cmask[None] > 0                          # [1,ac] real-motion mask
            ver = ~nvb
            vc = ver & cb & cm                            # verified-but-contact   [nt,ac]
            vu = ver & ub & cm                            # verified-but-unsafe-contact
            NV = jnp.sum(nvb & cm)
            C = jnp.sum(cb & cm)
            U = jnp.sum(ub & cm)
            VC = jnp.sum(vc)
            VU = jnp.sum(vu)
            # vc/vu masks let the host recover which (traj, motion) pairs failed; int8 keeps the
            # (rare-failure) transfer small. The 5 scalars drive the counts as before.
            return NV, C, U, VC, VU, vc.astype(jnp.int8), vu.astype(jnp.int8)

        return self.jax.jit(chunk)

    def eval_active(self, pose, active_idx):
        """Evaluate one pose over its level-3-active motions; returns a counter dict.

        When ``capture_failures`` is set, ``res["failures"]`` lists ``(traj_idx, motion_idx,
        unsafe)`` for every verified-but-contact pair (``traj_idx`` indexes ``st.traj_meta``,
        ``motion_idx`` is the global human-sample index), so the caller can rebuild their geometry.
        """
        jnp = self.jnp
        nt, M = self.n_traj, self.M
        res = dict(
            total_pairs=nt * M, n_verified=nt * M, n_contact=0, n_unsafe=0,
            n_verified_contact=0, n_verified_unsafe=0, n_intervals_no_truth=self.total_no_truth,
            n_pred_cand=0, n_true_cand=0, n_poses_skipped=0, active=int(np.asarray(active_idx).size),
        )
        if self.capture_failures:
            res["failures"] = []
        active_idx = np.asarray(active_idx, dtype=np.int64)
        A = active_idx.size
        if A == 0:
            return res

        rot_t, tvec = pose_rt(pose)
        rt = jnp.asarray(rot_t, self.f)
        tv = jnp.asarray(tvec, self.f)
        P1r = self.P1 @ rt + tv
        P2r = self.P2 @ rt + tv

        ac = self.a_chunk
        # Accumulate the per-chunk scalar partials ON-DEVICE and pull them back once at the end.
        # Calling int() per chunk would force ~ceil(A/ac) blocking device->host syncs per pose;
        # instead each self._chunk(...) dispatches asynchronously and only the single int() barrier
        # below waits, so the chunk kernels pipeline on the GPU. (Capturing failures does add a
        # per-chunk sync to inspect the mask -- acceptable, as that path is only for debug dumps.)
        totals = None
        failures = res.get("failures")
        for c0 in range(0, A, ac):
            idx = active_idx[c0:c0 + ac]
            cm = np.ones(idx.size, np.float64)
            if idx.size < ac:                            # pad to fixed shape (compile once)
                pad = ac - idx.size
                idx = np.concatenate([idx, np.zeros(pad, np.int64)])
                cm = np.concatenate([cm, np.zeros(pad, np.float64)])
            part = self._chunk(P1r, P2r, jnp.asarray(idx), jnp.asarray(cm, self.f))
            scal = part[:5]
            totals = scal if totals is None else tuple(t + p for t, p in zip(totals, scal))
            if self.capture_failures:
                vc_np = np.asarray(part[5])               # [nt, ac] int8 verified-but-contact
                if vc_np.any():
                    vu_np = np.asarray(part[6])
                    tl, jcol = np.nonzero(vc_np)          # padded cols have cm=0 -> never flagged
                    for ti, j in zip(tl.tolist(), jcol.tolist()):
                        failures.append((ti, int(active_idx[c0 + j]), bool(vu_np[ti, j])))

        NVt, Ct, Ut, VCt, VUt = (int(x) for x in totals)   # single device->host sync per pose
        res["n_verified"] = nt * M - NVt
        res["n_contact"] = Ct
        res["n_unsafe"] = Ut
        res["n_verified_contact"] = VCt
        res["n_verified_unsafe"] = VUt
        return res
