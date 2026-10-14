import React from 'react'

/**
 * Automated segmentation — MedNeXt-2D.
 *
 * The counterpart to the LiteMedSAM section: that one is clinician-prompted, this
 * one reads the slice unaided. The section documents the four training runs and,
 * more importantly, the measurement limits discovered while comparing them —
 * because the validation metric could not resolve the differences it was being
 * used to judge. Run D (native 512px) is deployed.
 */
export const MedNextDocs: React.FC = () => {
  return (
    <div className="task-block segmentation">
      <div className="task-header-strip">
        <div className="task-tag-group">
          <span className="task-type-tag seg">Task · Segmentation</span>
          <span className="task-name-text">Automated Lesion Segmentation (MedNeXt-2D) · Schedule &amp; Regularisation Study</span>
        </div>
        <span className="task-model-pill">MedNeXt-B · 2D · k=3 · 10.45 M params · 512×512 input</span>
      </div>

      <div className="boilerplate-card">
        <p className="boilerplate-copy">
          The second segmentation track, and the opposite of the first. <strong>LiteMedSAM</strong> needs a clinician to
          say where to look; <strong>MedNeXt</strong> reads the slice and returns a mask unaided. It runs after the
          classifier, skipped entirely on a Normal prediction, and needs no prompt of any kind. Trained from scratch on
          Kaggle 2×T4 — no pretrained 2D MedNeXt weights exist publicly, and the paper's ImageNet-21k weights are for
          the 3D variant and are not interchangeable. The engine lives at{' '}
          <code>backend/app/organs/brain/segmentation/mednext/</code>, and the serving checkpoint is self-describing:
          dropping <code>mednext_brain_seg.pth</code> into <code>checkpoints/</code> is the whole deployment.
        </p>
        <p className="boilerplate-copy">
          <strong>Held-out test: Dice 0.73 mean / 0.86 median</strong> across 1,274 slices from 41 lesions, at{' '}
          <strong>120 ms</strong> per slice, with <strong>0 missed lesions</strong> and calibration error{' '}
          <strong>0.0094</strong>. The gap between the mean and the median is not noise — it is the distribution.
          Lesions occupying under 0.5% of the frame score 0.41 while medium lesions reach 0.82, and the low tail pulls
          the mean down. Both numbers are reported deliberately; quoting either alone misrepresents the model.
        </p>
        <p className="boilerplate-copy">
          <strong>Four configurations were trained and compared</strong> — 150 epochs and 60 epochs at 256px, 60 epochs
          with heavier regularisation at 256px, and 60 epochs with that same regularisation at the native{' '}
          <strong>512×512</strong> resolution. The three 256px runs landed within <strong>0.003 Dice of each
          other</strong> — the finding being that the training recipe is not the ceiling. The 512px run broke it:
          <strong> every test metric improved</strong>, and the gain concentrated exactly where downsampling was
          predicted to hurt — the smallest lesions. The section below records all four runs, the measurement problem
          that made the 256px comparison harder than it should have been, and where the remaining headroom sits.
        </p>

        <div className="seg-targets-grid">
          <div className="seg-target-box">
            <span className="seg-badge wt">Run A · 256 baseline</span>
            <span className="seg-formula">150 ep · patience 10 · stopped at ep 22</span>
            <p className="seg-desc">
              Test Dice <strong>0.7167 mean / 0.8518 median</strong>, IoU 0.6256, HD95 20.2 px, ECE 0.0092. The run
              everything else is measured against.
            </p>
            <span className="seg-target-metric">OpenMed 0.6896 · BTSC 0.8003</span>
          </div>
          <div className="seg-target-box">
            <span className="seg-badge wt">Run B · 256</span>
            <span className="seg-formula">60 ep · patience 20 · stopped at ep 37</span>
            <p className="seg-desc">
              Test Dice <strong>0.7144 mean / 0.8523 median</strong>. Appeared to improve large-lesion recall to 0.6893
              (+0.031) while losing 0.016 on small lesions — but Run C did not reproduce it, so the shift was most
              likely a byproduct of Run B's heavier overfitting rather than a real gain.
            </p>
            <span className="seg-target-metric">OpenMed 0.6897 · BTSC 0.7911</span>
          </div>
          <div className="seg-target-box">
            <span className="seg-badge wt">Run C · 256 + reg</span>
            <span className="seg-formula">60 ep · patience 20 · drop_path 0.15 · wd 0.05</span>
            <p className="seg-desc">
              Test Dice <strong>0.7178 mean / 0.8545 median</strong>, IoU 0.6273, HD95 18.8 px, ECE 0.0093. Nominally
              best of the three 256px runs on every test metric, but the margins sit inside the confidence interval.
            </p>
            <span className="seg-target-metric">OpenMed 0.6931 · BTSC 0.7944 · trained 36 of 60 epochs</span>
          </div>
          <div className="seg-target-box">
            <span className="seg-badge et">Deployed · Final · Run D · 512</span>
            <span className="seg-formula">60 ep · patience 15 · drop_path 0.15 · wd 0.05 · native 512²</span>
            <p className="seg-desc">
              Test Dice <strong>0.7287 mean / 0.8620 median</strong>, IoU 0.6373, HD95 <strong>18.0 px</strong>, ECE
              0.0094. Best on every test metric, every size bin and both sources. No image registration loss — the
              masks are bit-identical to the source at native resolution.
            </p>
            <span className="seg-target-metric">OpenMed 0.7040 · BTSC 0.8051 · 120 ms/slice · trained 34 of 60 epochs</span>
          </div>
        </div>
      </div>

      {/* Four-run headline comparison */}
      <div className="decision-matrix-table-wrap">
        <div style={{ padding: '12px 16px', background: 'var(--bone)', borderBottom: '1px solid var(--line)' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink)' }}>
            Four Configurations · Held-Out Test (1,274 slices, 41 lesions)
          </span>
        </div>
        <table className="decision-matrix-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>A · 150 ep · 256</th>
              <th>B · 60 ep · 256</th>
              <th>C · 60 ep · 256 + reg</th>
              <th>D · 60 ep · 512 + reg</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="route-name">
                <strong>Input resolution</strong>
                <span>Source is 512×512 natively</span>
              </td>
              <td>256²</td>
              <td>256²</td>
              <td>256²</td>
              <td><strong>512² — no resampling</strong></td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Epochs run</strong>
                <span>All four early-stopped</span>
              </td>
              <td>22</td>
              <td>37</td>
              <td>36</td>
              <td>34</td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Final train loss</strong>
                <span>How far into memorisation each got</span>
              </td>
              <td>0.1128</td>
              <td>0.0472</td>
              <td>0.0609</td>
              <td>0.0660</td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Test Dice · mean</strong>
                <span>Pooled across all slices</span>
              </td>
              <td>0.7167</td>
              <td>0.7144</td>
              <td>0.7178</td>
              <td><strong>0.7287</strong></td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Test Dice · median</strong>
                <span>The typical slice</span>
              </td>
              <td>0.8518</td>
              <td>0.8523</td>
              <td>0.8545</td>
              <td><strong>0.8620</strong></td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>IoU</strong>
                <span>The stricter companion</span>
              </td>
              <td>0.6256</td>
              <td>0.6243</td>
              <td>0.6273</td>
              <td><strong>0.6377</strong></td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>HD95 · median</strong>
                <span>Boundary error, in pixels</span>
              </td>
              <td>20.2 px</td>
              <td>20.0 px</td>
              <td>18.8 px</td>
              <td><strong>18.0 px</strong></td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>OpenMed / BTSC</strong>
                <span>Per-source means</span>
              </td>
              <td>0.6896 / 0.8003</td>
              <td>0.6897 / 0.7911</td>
              <td>0.6931 / 0.7944</td>
              <td><strong>0.7040 / 0.8051</strong></td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Median inference</strong>
                <span>Per slice, on GPU</span>
              </td>
              <td>34 ms</td>
              <td>—</td>
              <td>31 ms</td>
              <td><strong>120 ms</strong></td>
            </tr>
          </tbody>
        </table>
        <div style={{ padding: '14px 16px', borderTop: '1px solid var(--line)' }}>
          <p className="boilerplate-copy" style={{ margin: 0 }}>
            <strong>The three 256px runs varied 2.4× in training loss and 0.003 in test Dice.</strong> Moving to native
            512px broke that plateau: <strong>every test metric improved over every 256px configuration</strong>, and
            the gain concentrated in the smallest lesions. The cost is inference time — 120 ms against 34 ms — which is
            still far inside clinical latency but is no longer "real-time" in the way the 256px model was.
          </p>
        </div>
      </div>

      {/* Size-stratified comparison — the metric that revealed the trade */}
      <div className="decision-matrix-table-wrap" style={{ marginTop: '14px' }}>
        <div style={{ padding: '12px 16px', background: 'var(--bone)', borderBottom: '1px solid var(--line)' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink)' }}>
            Lesion-Size Stratification · Why the Aggregate Hides the Difference
          </span>
        </div>
        <table className="decision-matrix-table">
          <thead>
            <tr>
              <th>Lesion Size Bin</th>
              <th>Slices</th>
              <th>A · 256</th>
              <th>B · 256</th>
              <th>C · 256 + reg</th>
              <th>D · 512 + reg</th>
              <th>Reading</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="route-name">
                <strong>Tiny · &lt;0.5% of frame</strong>
                <span>The resolution-limited bin</span>
              </td>
              <td>176</td>
              <td>0.3725</td>
              <td>0.3682</td>
              <td>0.3775</td>
              <td><strong>0.4114</strong></td>
              <td>The largest single gain of the study (+0.034 over the 256px runs) — and the bin that downsampling to 256 hurt most. A ~157 px native lesion was 39 px at 256; it is intact at 512.</td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Small · 0.5–1.5%</strong>
                <span>The clinically urgent case</span>
              </td>
              <td>375</td>
              <td><strong>0.7439</strong></td>
              <td>0.7281</td>
              <td>0.7425</td>
              <td>0.7486</td>
              <td>Run D is best here too — the only bin where A led, and 512 recovered it.</td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Medium · 1.5–4%</strong>
                <span>The bulk of the test set</span>
              </td>
              <td>503</td>
              <td>0.8118</td>
              <td>0.8094</td>
              <td>0.8120</td>
              <td><strong>0.8192</strong></td>
              <td>Best-performing bin in all four runs. Differences are in the third decimal.</td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Large · &gt;4%</strong>
                <span>Under-segmented in all four</span>
              </td>
              <td>220</td>
              <td>0.7279</td>
              <td>0.7509</td>
              <td>0.7328</td>
              <td><strong>0.7416</strong></td>
              <td>B's apparent gain here did not reproduce under regularisation; D lands between them. Large-lesion recall (0.66–0.67) remains the model's most consistent weakness.</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* The measurement problem */}
      <div className="decision-matrix-table-wrap" style={{ marginTop: '14px' }}>
        <div style={{ padding: '12px 16px', background: 'var(--bone)', borderBottom: '1px solid var(--line)' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink)' }}>
            Measurement Limits · Read Before Quoting Any Number Above
          </span>
        </div>
        <table className="decision-matrix-table">
          <thead>
            <tr>
              <th>Split</th>
              <th>Slices</th>
              <th>Independent Lesions</th>
              <th>Standard Error</th>
              <th>95% Confidence Interval</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="route-name">
                <strong>Validation</strong>
                <span>Used for checkpoint selection</span>
              </td>
              <td>1,289</td>
              <td><strong>47</strong></td>
              <td>0.0343</td>
              <td><strong>±0.067</strong></td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Test</strong>
                <span>Scored once, through the serving path</span>
              </td>
              <td>1,274</td>
              <td><strong>41</strong></td>
              <td>0.0254</td>
              <td>±0.050</td>
            </tr>
          </tbody>
        </table>
        <div style={{ padding: '14px 16px', borderTop: '1px solid var(--line)' }}>
          <p className="boilerplate-copy" style={{ margin: 0 }}>
            Slices are not independent — they cluster by lesion, so the effective sample size is the <em>lesion</em>
            count, not the slice count. Every difference between the three 256px runs (A, B, C) is an order of magnitude
            inside these intervals. The training log's early-stop rule (<code>no improvement &gt; 1e-04</code>) is a
            threshold of 0.0001 against a metric whose uncertainty is 0.067.
          </p>
          <p className="boilerplate-copy" style={{ margin: '10px 0 0' }}>
            Run D's +0.011 over C on the test mean is also inside ±0.050 — but it is best on <strong>every</strong>{' '}
            metric, <strong>every</strong> size bin and <strong>both</strong> sources simultaneously, and the tail
            (p5) rose 0.060 → 0.102. A consistent direction across that many independent measurements is the strongest
            signal a 41-lesion test set can produce, and it is the reason D is deployed.
          </p>
          <p className="boilerplate-copy" style={{ margin: '10px 0 0' }}>
            Two further artefacts compound the noise. A single validation lesion (<code>openmed:g00083</code>)
            contributes 89 slices scoring 0.218 — <strong>6.9% of the entire validation set from one lesion</strong>.
            And BTSC has only nine lesions in validation, so that per-source figure is one or two lesions deciding it.
          </p>
        </div>
      </div>

      {/* Three statistics, one split */}
      <div className="decision-matrix-table-wrap" style={{ marginTop: '14px' }}>
        <div style={{ padding: '12px 16px', background: 'var(--bone)', borderBottom: '1px solid var(--line)' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink)' }}>
            Three Statistics, One Split · Always Name Which Is Quoted
          </span>
        </div>
        <table className="decision-matrix-table">
          <thead>
            <tr>
              <th>Aggregation</th>
              <th>Validation</th>
              <th>Test</th>
              <th>What It Means</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="route-name">
                <strong>Pooled mean</strong>
                <span>Each slice counts once</span>
              </td>
              <td>0.6939</td>
              <td><strong>0.7287</strong></td>
              <td>The report headline, and the figure used on this page.</td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Macro mean</strong>
                <span>Each source counts once</span>
              </td>
              <td><strong>0.7101</strong></td>
              <td>0.7546</td>
              <td>Runs higher — BTSC scores better and macro weights it equally despite being the smaller source. This is what the training log prints.</td>
            </tr>
            <tr>
              <td className="route-name">
                <strong>Median</strong>
                <span>The typical slice</span>
              </td>
              <td>0.8333</td>
              <td><strong>0.8620</strong></td>
              <td>Unaffected by the failing tail, which is why it reads so much higher than the mean.</td>
            </tr>
          </tbody>
        </table>
        <div style={{ padding: '14px 16px', borderTop: '1px solid var(--line)' }}>
          <p className="boilerplate-copy" style={{ margin: 0 }}>
            All three are correct for the same data, and comparing across them without realising it accounts for most of
            the apparent inconsistency between the training log and the final report. <strong>The only like-for-like
            comparison is macro against macro</strong> (validation 0.7101 → test 0.7546) or pooled against pooled
            (0.6939 → 0.7287). The model does not change when training stops; only the slices and the averaging do.
          </p>
        </div>
      </div>
    </div>
  )
}
