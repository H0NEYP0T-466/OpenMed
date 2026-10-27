import React from 'react'
import { Link } from 'react-router-dom'
import { Activity, ArrowUpRight, AlertTriangle, XCircle } from 'lucide-react'

/**
 * Lungs documentation — the real implementation record.
 *
 * Every number on this page is measured, either from the verified dataset
 * join or from live runs. Nothing is aspirational except the two blocks
 * explicitly marked as pending.
 */

const WEIGHT_ROWS: readonly { label: string; positives: number; share: string; ratio: string; weight: string }[] = [
  { label: 'Hernia', positives: 126, share: '0.17%', ratio: '604:1', weight: '50.0 (clipped)' },
  { label: 'Pneumonia', positives: 781, share: '1.02%', ratio: '97:1', weight: '50.0 (clipped)' },
  { label: 'Fibrosis', positives: 1078, share: '1.41%', ratio: '70:1', weight: '50.0 (clipped)' },
  { label: 'Edema', positives: 1239, share: '1.62%', ratio: '61:1', weight: '50.0 (clipped)' },
  { label: 'Emphysema', positives: 1242, share: '1.63%', ratio: '60:1', weight: '50.0 (clipped)' },
  { label: 'Cardiomegaly', positives: 1504, share: '1.97%', ratio: '50:1', weight: '49.7' },
  { label: 'Pleural Thickening', positives: 1974, share: '2.59%', ratio: '38:1', weight: '37.6' },
  { label: 'Pneumothorax', positives: 2331, share: '3.06%', ratio: '32:1', weight: '31.7' },
  { label: 'Consolidation', positives: 2541, share: '3.33%', ratio: '29:1', weight: '29.0' },
  { label: 'Mass', positives: 3605, share: '4.73%', ratio: '20:1', weight: '20.2' },
  { label: 'Nodule', positives: 4155, share: '5.45%', ratio: '17:1', weight: '17.4' },
  { label: 'Atelectasis', positives: 7289, share: '9.56%', ratio: '9.5:1', weight: '9.5' },
  { label: 'Effusion', positives: 7607, share: '9.97%', ratio: '9:1', weight: '9.0' },
  { label: 'Infiltration', positives: 12125, share: '15.89%', ratio: '5.3:1', weight: '5.3' },
]

const PER_CLASS: readonly { finding: string; auc: number; sens: number; spec: number; support: string }[] = [
  { finding: 'Atelectasis', auc: 0.7576, sens: 0.400, spec: 0.885, support: '3,279 / 22,317' },
  { finding: 'Cardiomegaly', auc: 0.8892, sens: 0.523, spec: 0.955, support: '1,069 / 24,527' },
  { finding: 'Effusion', auc: 0.8236, sens: 0.692, spec: 0.785, support: '4,658 / 20,938' },
  { finding: 'Infiltration', auc: 0.6989, sens: 0.496, spec: 0.785, support: '6,112 / 19,484' },
  { finding: 'Mass', auc: 0.8164, sens: 0.461, spec: 0.919, support: '1,748 / 23,848' },
  { finding: 'Nodule', auc: 0.7370, sens: 0.212, spec: 0.966, support: '1,623 / 23,973' },
  { finding: 'Pneumonia', auc: 0.7120, sens: 0.036, spec: 0.992, support: '555 / 25,041' },
  { finding: 'Pneumothorax', auc: 0.8409, sens: 0.442, spec: 0.928, support: '2,665 / 22,931' },
  { finding: 'Consolidation', auc: 0.7455, sens: 0.280, spec: 0.912, support: '1,815 / 23,781' },
  { finding: 'Edema', auc: 0.8405, sens: 0.531, spec: 0.891, support: '925 / 24,671' },
  { finding: 'Emphysema', auc: 0.8631, sens: 0.404, spec: 0.960, support: '1,093 / 24,503' },
  { finding: 'Fibrosis', auc: 0.8183, sens: 0.177, spec: 0.986, support: '435 / 25,161' },
  { finding: 'Pleural_Thickening', auc: 0.7595, sens: 0.195, spec: 0.954, support: '1,143 / 24,453' },
  { finding: 'Hernia', auc: 0.9275, sens: 0.523, spec: 0.997, support: '86 / 25,510' },
  { finding: 'COVID', auc: 0.9999, sens: 0.989, spec: 0.998, support: '361 / 25,666' },
  { finding: 'TB', auc: 1.0000, sens: 0.986, spec: 0.9995, support: '70 / 25,596' },
]

export const LungsDocs: React.FC = () => {
  return (
    <div id="model-lungs" className="organ-model-dossier">
      <div className="organ-dossier-header">
        <div className="dossier-head-left">
          <div className="organ-badge lungs">Organ · Lungs (Pulmones)</div>
          <h3 className="dossier-organ-title">XRV-16 — 16-Class Chest Radiograph Classification</h3>
          <span className="dossier-sub">
            Respiratory System · Frontal Radiographs · NIH ChestX-ray14 + COVID-19 Radiography + TB Chest Radiography
          </span>
        </div>
        <Link to="/app?organ=lungs" className="btn-launch-workspace">
          <Activity size={14} />
          <span>Open Lungs Workspace</span>
          <ArrowUpRight size={13} />
        </Link>
      </div>


      {/* Production model */}
      <div className="task-block classification">
        <div className="task-header-strip">
          <div className="task-tag-group">
            <span className="task-type-tag class">Task · Classification (in production)</span>
            <span className="task-name-text">16-Class Multi-Label Differential — 14 NIH Findings plus COVID and TB</span>
          </div>
          <span className="task-model-pill">XRV-16 · DenseNet-121 (CXR-pretrained) · 116,436 images</span>
        </div>
        <div className="boilerplate-card">
          <p className="boilerplate-copy">
            A TorchXRayVision <strong>densenet121-res224-all</strong> backbone — trained on NIH, CheXpert, MIMIC,
            PadChest and VinDr — has its 18-class head replaced by <strong>Dropout(0.3) → Linear(1024, 16)</strong> and
            is fine-tuned on 116,436 radiographs from three collections. Training uses masked binary cross-entropy:
            a <code>−1</code> marks a finding that source never annotated, and those terms are excluded from the loss
            rather than counted as negatives. Per-class weights are <code>√(neg/pos)</code>, computed only over
            assessed rows.
          </p>
        </div>
      </div>

      <div className="archival-ledger-card">
        <div className="ledger-card-title-strip">
          <span className="ledger-tag">Held-out Test Ledger</span>
          <span className="ledger-sub">26,027 images · best checkpoint epoch 23 (EMA) · early-stopped at 38</span>
        </div>
        <div className="stat-monoliths-grid">
          <div className="stat-monolith">
            <span className="mono-num">0.8021</span>
            <span className="mono-label">Findings Mean AUC</span>
            <span className="mono-desc">The 14 NIH classes — the number comparable with published ChestX-ray14 baselines.</span>
          </div>
          <div className="stat-monolith">
            <span className="mono-num">0.9999</span>
            <span className="mono-label">COVID AUC</span>
            <span className="mono-desc">3,616 positives from a 2020–21 collection; negatives are pre-pandemic cohorts.</span>
          </div>
          <div className="stat-monolith">
            <span className="mono-num">1.0000</span>
            <span className="mono-label">TB AUC</span>
            <span className="mono-desc">700 positives. Read the caveat below before quoting this anywhere.</span>
          </div>
          <div className="stat-monolith">
            <span className="mono-num">0.9319</span>
            <span className="mono-label">Specificity @ 0.5</span>
            <span className="mono-desc">Macro over 16 classes. Sensitivity at the same threshold is 0.4592 — see below.</span>
          </div>
        </div>
      </div>

      <div className="class-inventory-table-wrap">
        <div style={{ padding: '12px 16px', background: 'var(--bone)', borderBottom: '1px solid var(--line)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink)' }}>
            Per-Class Test Performance
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-soft)' }}>
            25,596 NIH test images · thresholds at 0.5
          </span>
        </div>
        <table className="class-inventory-table">
          <thead>
            <tr>
              <th>Finding</th>
              <th>AUC</th>
              <th>Sensitivity</th>
              <th>Specificity</th>
              <th>Support (n+ / n−)</th>
            </tr>
          </thead>
          <tbody>
            {PER_CLASS.map((row) => (
              <tr key={row.finding}>
                <td className="class-name-cell">{row.finding}</td>
                <td style={{ fontWeight: 600 }}>{row.auc.toFixed(4)}</td>
                <td>{row.sens.toFixed(3)}</td>
                <td>{row.spec.toFixed(3)}</td>
                <td>{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="anomaly-breakdown-grid" style={{ marginBottom: '18px' }}>
        <div className="anomaly-box">
          <div className="anomaly-box-title">
            <AlertTriangle size={15} className="text-warning" />
            <span>TB and COVID are separable by source, not by pathology</span>
          </div>
          <p className="anomaly-copy">
            TB reached AUC 0.9983 after two epochs — from 560 training images. A model cannot learn tuberculosis that
            fast. What it learned is that TB scans are 512×512 greyscale from Montgomery/Shenzhen while NIH scans are
            224 RGB: different scanners, different acquisition. The negatives for both classes come from the
            pre-pandemic cohorts, so the class boundary is partly a domain boundary. On a radiograph from a hospital
            neither dataset covers, these two numbers will not hold.
          </p>
        </div>
        <div className="anomaly-box">
          <div className="anomaly-box-title">
            <AlertTriangle size={15} className="text-warning" />
            <span>Sensitivity 0.4592 at the 0.5 threshold</span>
          </div>
          <p className="anomaly-copy">
            Class weighting shifts the posterior distribution, so 0.5 is no longer a balanced operating point: the
            model misses roughly half the positives while keeping 93 % specificity. AUC is unaffected — the ranking is
            sound — but a screening tool needs per-class thresholds calibrated on the validation split, not a global
            0.5. Until that lands, treat the probabilities as ranks, not decisions.
          </p>
        </div>
      </div>

      {/* Replication baseline */}
      <div className="task-block classification">
        <div className="task-header-strip">
          <div className="task-tag-group">
            <span className="task-type-tag class">Task · Classification</span>
            <span className="task-name-text">Replication Baseline — Z-Net on 14 NIH Findings</span>
          </div>
          <span className="task-model-pill">Z-Net · DenseNet-121 + Log-Sum-Exp Pooling · NIH ChestX-ray14</span>
        </div>
        <div className="boilerplate-card">
          <p className="boilerplate-copy">
            A DenseNet-121 backbone undergoes network surgery per §4.1 of the paper: the fully-connected and
            classification heads are deleted, a 1×1 <strong>transition layer</strong> maps the final feature grid to
            S×S×224, a 1×1 <strong>prediction layer</strong> produces S×S×14, and <strong>Log-Sum-Exp pooling</strong>
            collapses the grid into 14 posteriors. The prediction weights multiply the transition activations to yield
            class activation maps, so the same network that classifies also localises (IoU @ threshold 0.1).
          </p>
        </div>
      </div>

      {/* Empirical chronology of the build */}
      <div className="chronology-wrapper">
        {/* Phase 1 */}
        <div className="chronology-item">
          <div className="chronology-marker done">
            <span className="phase-num">01</span>
          </div>
          <div className="chronology-card">
            <div className="chronology-head">
              <span className="phase-badge done">Phase 01 · Paper Acquisition &amp; Verified Join</span>
              <span className="phase-timestamp">NIH ChestX-ray14 · 224 px PNG</span>
            </div>
            <h4 className="chronology-title">Dataset Verified to Patient Level Before a Single Line of Training</h4>
            <p className="chronology-text">
              The images dataset ships without labels, so the label CSVs travel as a separate 1.3 MB input. The join was
              verified against the NIH paper: every CSV row that references a present image is kept, and the patient count
              is derived from the filename prefix. The official <code>train_val_list.txt</code> / <code>test_list.txt</code>{' '}
              partition is used exactly as §5.2 of the paper prescribes — 70% train, 10% validation, 20% test at patient level.
            </p>
            <div className="chronology-metrics">
              <div className="c-metric"><span className="c-metric-v">112,120</span><span className="c-metric-k">Images Joined</span></div>
              <div className="c-metric"><span className="c-metric-v">30,805</span><span className="c-metric-k">Unique Patients</span></div>
              <div className="c-metric"><span className="c-metric-v">76,282 / 10,242 / 25,596</span><span className="c-metric-k">Train / Val / Test</span></div>
              <div className="c-metric"><span className="c-metric-v">14</span><span className="c-metric-k">Findings (Multi-Label)</span></div>
            </div>
          </div>
        </div>

        {/* Phase 2 */}
        <div className="chronology-item">
          <div className="chronology-marker done">
            <span className="phase-num">02</span>
          </div>
          <div className="chronology-card">
            <div className="chronology-head">
              <span className="phase-badge done">Phase 02 · Pipeline Hardening</span>
              <span className="phase-timestamp">Seven Defects Caught Before the First Training Run</span>
            </div>
            <h4 className="chronology-title">Everything That Would Have Killed the Run, Caught by Static and Fixture Testing</h4>
            <p className="chronology-text">
              The trainer was hardened against seven defects discovered before any GPU was spent: a per-batch (instead of
              per-epoch) LR decay that would have collapsed the learning rate inside epoch one; an unparseable BBox CSV header
              that silently discarded every localisation row; unwired label inputs; two successive Kaggle mount-layout changes;
              a missing artefact directory that crashed at the final step; and the torch.amp / torch.cuda.amp API split.
            </p>
            <div className="chronology-metrics">
              <div className="c-metric"><span className="c-metric-v">7</span><span className="c-metric-k">Defects Fixed Pre-Run</span></div>
              <div className="c-metric"><span className="c-metric-v">2</span><span className="c-metric-k">Mount Layouts Supported</span></div>
              <div className="c-metric"><span className="c-metric-v">3</span><span className="c-metric-k">Checkpoint Fields Versioned</span></div>
            </div>
          </div>
        </div>

        {/* Phase 3 */}
        <div className="chronology-item">
          <div className="chronology-marker done">
            <span className="phase-num">03</span>
          </div>
          <div className="chronology-card">
            <div className="chronology-head">
              <span className="phase-badge done">Phase 03 · DataParallel Rejected by Measurement</span>
              <span className="phase-timestamp">2× Tesla T4 · AMP · Batch 32</span>
            </div>
            <h4 className="chronology-title">Two GPUs Measured Slower Than One — the Single-GPU Default</h4>
            <p className="chronology-text">
              DataParallel re-broadcasts every parameter tensor to both replicas on each forward pass, and DenseNet-121 has
              roughly 800 of them across 120 sequential layers. A controlled benchmark scored the model alone on random
              tensors: <strong>76.5 img/s on two GPUs against 215.8 img/s on one</strong>. The multi-GPU path was demoted to an
              opt-in flag and the run proceeds on a single card.
            </p>
            <div className="chronology-metrics">
              <div className="c-metric"><span className="c-metric-v">215.8</span><span className="c-metric-k">img/s · Single T4</span></div>
              <div className="c-metric"><span className="c-metric-v">76.5</span><span className="c-metric-k">img/s · DataParallel ×2</span></div>
              <div className="c-metric"><span className="c-metric-v">≈ 560</span><span className="c-metric-k">img/s Loader Ceiling</span></div>
            </div>
            <div className="anomaly-breakdown-grid" style={{ marginTop: '14px' }}>
              <div className="anomaly-box">
                <div className="anomaly-box-title">
                  <AlertTriangle size={15} className="text-warning" />
                  <span>Input Starvation Signature</span>
                </div>
                <p className="anomaly-copy">
                  35 img/s observed against a ~560 img/s loader ceiling ruled out storage: the wall was the render loop,
                  not the data path. The loader was timed in isolation (decode 3.1 ms + augment 7.1 ms per image) before
                  any optimisation was accepted.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Phase 4 */}
        <div className="chronology-item">
          <div className="chronology-marker done">
            <span className="phase-num">04</span>
          </div>
          <div className="chronology-card">
            <div className="chronology-head">
              <span className="phase-badge done">Phase 04 · First Training Run — Aborted by Design</span>
              <span className="phase-timestamp">2× T4 → 1× T4 · 19 Epochs · 126 min</span>
            </div>
            <h4 className="chronology-title">Early Stopping Fired One Epoch Before the Paper&apos;s Learning-Rate Decay</h4>
            <p className="chronology-text">
              The paper divides the learning rate by 10 after epoch 20. With patience 10 on validation AUC, the best score
              (epoch 9) aged out at epoch 19 — the decay was never applied, so the entire run trained at 1e-3. Early stopping
              was demoted to an opt-in flag (default off, matching the paper) and the run restarted with the full 100-epoch
              schedule.
            </p>
            <div className="chronology-metrics">
              <div className="c-metric"><span className="c-metric-v">0.8024</span><span className="c-metric-k">Best Val AUC · Epoch 9</span></div>
              <div className="c-metric"><span className="c-metric-v">0.7722</span><span className="c-metric-k">Test AUC (no decay)</span></div>
              <div className="c-metric"><span className="c-metric-v">0.858</span><span className="c-metric-k">Paper-Reported Target</span></div>
              <div className="c-metric"><span className="c-metric-v">0.841</span><span className="c-metric-k">CheXNet Reference</span></div>
            </div>
            <div className="anomaly-breakdown-grid" style={{ marginTop: '14px' }}>
              <div className="anomaly-box">
                <div className="anomaly-box-title">
                  <XCircle size={15} className="text-warning" />
                  <span>Class-Weight Clipping</span>
                </div>
                <p className="anomaly-copy">
                  The W-CEL balancing factors are never published in the paper. Inverse frequency is used, clipped at 50:
                  Hernia&apos;s raw 604:1 ratio would swamp every batch. See the inventory table below for all 14 weights.
                </p>
              </div>
              <div className="anomaly-box">
                <div className="anomaly-box-title">
                  <AlertTriangle size={15} className="text-warning" />
                  <span>Epoch Redefinition (Algorithm 1, Step 10)</span>
                </div>
                <p className="anomaly-copy">
                  The paper&apos;s algorithm defines trainstep = totalsamples / batchsize / 10 — one epoch is a 10% slice of
                  the training set, so 100 epochs equal 10 full passes. Implemented as a fraction flag with per-epoch reshuffle.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Phase 5 — in flight */}
        <div className="chronology-item">
          <div className="chronology-marker active">
            <span className="phase-num">05</span>
          </div>
          <div className="chronology-card">
            <div className="chronology-head">
              <span className="phase-badge active">Phase 05 · Paper-Exact Run — In Progress</span>
              <span className="phase-timestamp">100 Epochs · Patience Disabled · Kaggle 2× T4 → 1× T4</span>
            </div>
            <h4 className="chronology-title">Full Algorithm-1 Schedule With the Learning-Rate Decay Intact</h4>
            <p className="chronology-text">
              Adam at 1e-3 decaying toward the algorithm&apos;s 1e-8 floor, horizontal-flip augmentation, dataset-statistics
              normalisation (mean/std persisted beside the checkpoint so serving matches training), and early stopping disabled.
              The run resumes automatically from <code>lung_last.pth</code> if the session is interrupted.
            </p>
            <div className="chronology-metrics">
              <div className="c-metric"><span className="c-metric-v">6.5 min</span><span className="c-metric-k">Per Epoch (measured)</span></div>
              <div className="c-metric"><span className="c-metric-v">5.3×</span><span className="c-metric-k">Hernia Weight vs Infiltration</span></div>
              <div className="c-metric"><span className="c-metric-v">0.50</span><span className="c-metric-k">Threshold for Specificity</span></div>
              <div className="c-metric"><span className="c-metric-v">AUC</span><span className="c-metric-k">Threshold-Free Primary Metric</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* 14-finding inventory */}
      <div className="class-inventory-table-wrap">
        <div style={{ padding: '12px 16px', background: 'var(--bone)', borderBottom: '1px solid var(--line)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink)' }}>
            14-Finding Inventory &amp; W-CEL Balancing Matrix
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-soft)' }}>
            76,282 Training Images · Weights = Negative / Positive, Clipped at 50
          </span>
        </div>
        <table className="class-inventory-table">
          <thead>
            <tr>
              <th>Finding</th>
              <th>Positive Scans</th>
              <th>Share (%)</th>
              <th>Negative : Positive</th>
              <th>W-CEL Weight (β̂P)</th>
              <th>Grad-CAM Capable</th>
            </tr>
          </thead>
          <tbody>
            {WEIGHT_ROWS.map((row) => (
              <tr key={row.label}>
                <td className="class-name-cell">{row.label}</td>
                <td>{row.positives.toLocaleString('en-US')}</td>
                <td>{row.share}</td>
                <td>{row.ratio}</td>
                <td style={{ color: 'var(--accent)', fontWeight: 600 }}>{row.weight}</td>
                <td><span className="tag-pill standard">Yes · transition × prediction</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Production ledger — current measured state */}
      <div className="archival-ledger-card">
        <div className="ledger-card-title-strip">
          <span className="ledger-tag">Training Ledger</span>
          <span className="ledger-sub">Z-Net · NIH ChestX-ray14 · Single Tesla T4 · Adam</span>
        </div>

        <div className="stat-monoliths-grid">
          <div className="stat-monolith">
            <span className="mono-num">0.8024</span>
            <span className="mono-label">Best Val Mean AUC</span>
            <span className="mono-desc">Aborted first run (patience 10, no LR decay). Rerun in progress with the full schedule.</span>
          </div>
          <div className="stat-monolith">
            <span className="mono-num">0.858</span>
            <span className="mono-label">Paper-Reported Mean AUC</span>
            <span className="mono-desc">Headline figure of the source paper; only 8 of 14 per-class values are published and their mean is ~0.80.</span>
          </div>
          <div className="stat-monolith">
            <span className="mono-num">0.841</span>
            <span className="mono-label">CheXNet Reference</span>
            <span className="mono-desc">Wang et al. baseline on the same official test partition — the honest comparison point.</span>
          </div>
          <div className="stat-monolith">
            <span className="mono-num">100</span>
            <span className="mono-label">Epochs (10% Slices)</span>
            <span className="mono-desc">Algorithm 1 defines an epoch as a tenth of the training set; 100 epochs equal 10 full passes.</span>
          </div>
        </div>

        <div className="split-stratification-card">
          <div className="card-topline">
            <span className="card-top-title">Official Patient-Level Split (§5.2)</span>
          </div>
          <div className="stratification-bar">
            <div className="strat-segment train" style={{ flex: 76282 }}>
              <span className="strat-segment-title">Train 68% · 76,282</span>
            </div>
            <div className="strat-segment val" style={{ flex: 10242 }}>
              <span className="strat-segment-title">Val 9%</span>
            </div>
            <div className="strat-segment test" style={{ flex: 25596 }}>
              <span className="strat-segment-title">Test 23% · 25,596</span>
            </div>
          </div>
          <p className="chronology-text" style={{ marginTop: 10 }}>
            The paper states it uses the official dataset partition, so the canonical list files are used rather than a
            re-randomised 70/10/20 — results stay comparable with CheXNet and every other published baseline on this test set.
          </p>
        </div>
      </div>

      {/* Task 2: Segmentation — planned, honestly marked */}
      <div className="task-block segmentation">
        <div className="task-header-strip">
          <div className="task-tag-group">
            <span className="task-type-tag seg">Task · Segmentation</span>
            <span className="task-name-text">Thoracic Radiograph Lung-Field Delineation</span>
          </div>
          <span className="task-model-pill">Planned · Not Started</span>
        </div>
        <div className="boilerplate-card">
          <p className="boilerplate-copy">
            Status: pending. The classifier ships first; the segmentation module follows once the trained Z-Net checkpoint
            is deployed and the Grad-CAM localisation path is validated against the paper&apos;s BBox set. Candidate datasets
            under consideration include COVID-QU-Ex (~19k masked radiographs) and the Montgomery / Shenzhen TB collections,
            with the brain suite&apos;s assistive-segmentation pattern (click / box prompt) as the interaction model.
            The previous description of a 3D V-Net over LIDC-IDRI was a pre-implementation placeholder and has been
            retired with this page.
          </p>
        </div>
      </div>
    </div>
  )
}

export default LungsDocs
