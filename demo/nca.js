// Growing Neural Cellular Automata: one update step, in plain JavaScript.
// The math matches nca.py exactly (checked by parity_test.mjs).
// Plain script, no modules, so index.html works when opened from disk; it also
// runs under Node (parity_test.mjs imports it and reads globalThis.NCA).
globalThis.NCA = (() => {
  const H = 72, W = 72, C = 16, HID = 128, IN = 48;
  const PW = W + 2;                                  // padded row width (1-cell zero border)
  const pad = new Float32Array((H + 2) * PW * C);    // zero padding for the Sobel filters
  const next = new Float32Array(H * W * C);          // s_new, before alive masking
  const big = new Uint8Array(H * W);                 // alpha > 0.1, per cell
  const pre = new Uint8Array(H * W);                 // alive(s)
  const near = new Uint8Array(H * W);                // some neighbour is alive(s)
  const post = new Uint8Array(H * W);                // alive(s_new)
  const p = new Float64Array(IN), ds = new Float64Array(C);

  // Seed: one cell in the centre with alpha + hidden channels = 1.
  function seed() {
    const s = new Float32Array(H * W * C);
    const i = (36 * W + 36) * C;
    for (let ch = 3; ch < C; ch++) s[i + ch] = 1;
    return s;
  }

  // Weights from weights.js / the fixture -> typed arrays. W2 is stored
  // transposed (w2t[o][k]) so the MLP can skip hidden units that ReLU zeroed.
  function loadModel(m) {
    const w2t = new Float64Array(HID * C);
    for (let k = 0; k < C; k++)
      for (let o = 0; o < HID; o++) w2t[o * C + k] = m.w2[k * HID + o];
    return { w1: Float64Array.from(m.w1), b1: Float64Array.from(m.b1), w2t };
  }

  // out[i] = (3x3 max-pool of alpha around i) > 0.1, out-of-grid ignored.
  function aliveMask(s, out) {
    for (let i = 0; i < H * W; i++) big[i] = s[i * C + 3] > 0.1 ? 1 : 0;
    dilate(big, out);
  }
  function dilate(src, out) {
    for (let r = 0; r < H; r++)
      for (let c = 0; c < W; c++) {
        let v = 0;
        for (let y = Math.max(r - 1, 0); y <= Math.min(r + 1, H - 1) && !v; y++)
          for (let x = Math.max(c - 1, 0); x <= Math.min(c + 1, W - 1); x++)
            if (src[y * W + x]) { v = 1; break; }
        out[r * W + c] = v;
      }
  }

  // One step, in place on s (Float32Array, HWC).
  function step(s, model, fireRate = 0.5, angle = 0) {
    const { w1, b1, w2t } = model;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    for (let r = 0; r < H; r++) pad.set(s.subarray(r * W * C, (r + 1) * W * C), ((r + 1) * PW + 1) * C);
    aliveMask(s, pre);
    dilate(pre, near);
    next.set(s);

    for (let r = 0; r < H; r++)
      for (let c = 0; c < W; c++) {
        const cell = r * W + c;
        // Speed-up, exact: a cell with no alive(s) cell in its 3x3 is zeroed, and so
        // is every neighbour whose alive(s_new) it could affect -> its s_new is unused.
        // Cells that don't fire keep s_new = s, so they skip the MLP too.
        if (!near[cell] || !(Math.random() < fireRate)) continue;

        // 1. Perception: identity + rotated Sobel gradients, per channel.
        const b = ((r + 1) * PW + c + 1) * C, up = b - PW * C, dn = b + PW * C;
        for (let ch = 0; ch < C; ch++) {
          const nw = pad[up - C + ch], n = pad[up + ch], ne = pad[up + C + ch];
          const w = pad[b - C + ch], e = pad[b + C + ch];
          const sw = pad[dn - C + ch], so = pad[dn + ch], se = pad[dn + C + ch];
          const gx = (ne + 2 * e + se - nw - 2 * w - sw) / 8;
          const gy = (sw + 2 * so + se - nw - 2 * n - ne) / 8;
          p[ch] = pad[b + ch];
          p[16 + ch] = cos * gx - sin * gy;
          p[32 + ch] = sin * gx + cos * gy;
        }

        // 2. MLP: ds = W2 . relu(W1 . p + b1)
        ds.fill(0);
        for (let o = 0; o < HID; o++) {
          let h = b1[o];
          const wo = o * IN;
          for (let i = 0; i < IN; i++) h += w1[wo + i] * p[i];
          if (h > 0) {
            const vo = o * C;
            for (let k = 0; k < C; k++) ds[k] += h * w2t[vo + k];
          }
        }
        // 3. Stochastic update (already decided above).
        const at = cell * C;
        for (let k = 0; k < C; k++) next[at + k] = s[at + k] + ds[k];
      }

    // 4. Alive masking: keep cells alive both before and after the update.
    aliveMask(next, post);
    for (let cell = 0; cell < H * W; cell++) {
      const at = cell * C;
      if (pre[cell] && post[cell]) for (let k = 0; k < C; k++) s[at + k] = next[at + k];
      else for (let k = 0; k < C; k++) s[at + k] = 0;
    }
  }

  // Erase a disc of cells (all channels -> 0). r, c, radius in cells.
  function erase(s, r, c, radius) {
    for (let y = Math.ceil(r - radius); y <= r + radius; y++)
      for (let x = Math.ceil(c - radius); x <= c + radius; x++)
        if (y >= 0 && y < H && x >= 0 && x < W && (y - r) ** 2 + (x - c) ** 2 <= radius * radius)
          s.fill(0, (y * W + x) * C, (y * W + x + 1) * C);
  }

  return { H, W, C, seed, loadModel, step, erase };
})();
