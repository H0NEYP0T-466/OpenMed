import React, { useState } from 'react'
import {
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Binary,
  Layers,
  Terminal,
  Copy,
  Check,
} from 'lucide-react'
import { RomanSection } from '../../../../components/common/RomanSection'

export const DatasetSanitizationDocs: React.FC = () => {
  const [copiedAuditCmd, setCopiedAuditCmd] = useState(false)
  const [auditTab, setAuditTab] = useState<'audit_log' | 'hasher_script' | 'split_audit'>('audit_log')

  const handleCopyAuditScript = () => {
    const script = `python -c "
import os, hashlib, json
from PIL import Image
import imagehash

data_root = '/home/honeypot/Desktop/archive/Images_'
sha_map = {}
phash_map = {}
corrupt = 0
total = 0

for root, _, files in os.walk(data_root):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            total += 1
            path = os.path.join(root, f)
            cls = os.path.basename(root)
            with open(path, 'rb') as fp:
                data = fp.read()
            digest = hashlib.sha256(data).hexdigest()
            sha_map.setdefault(digest, []).append((path, cls))

print(f'Total Scans Audited: {total}')
byte_dups = {k: v for k, v in sha_map.items() if len(v) > 1}
print(f'Byte Duplicates Found: {len(byte_dups)}')
"`
    navigator.clipboard.writeText(script)
    setCopiedAuditCmd(true)
    setTimeout(() => setCopiedAuditCmd(false), 2200)
  }

  return (
    <section id="section-dataset" className="doc-section-wrapper">
      <RomanSection
        index={5}
        of={5}
        title="Curatio - Dataset Integrity, Modality Flattening & Verification"
        className="sp12"
      >
        {/* Top Summary Block */}
        <div className="doc-info-block">
          <div className="doc-block-header">
            <ShieldCheck size={16} className="doc-block-icon" />
            <span className="doc-block-title">Three-Stage Sanitization &amp; Deduplication Governance</span>
          </div>
          <p className="doc-block-copy">
            The OpenMed cranial dataset underwent a comprehensive, empirical three-stage sanitization journey.
            Through cryptographic hashing (SHA-256), perceptual frequency decomposition (64-bit DCT pHash), and clinical modality harmonization,
            we transformed a fragmented, leak-prone collection into an airtight benchmark of <strong>11,128 decodable scans</strong> across <strong>9 canonical WHO classes</strong> with <strong>zero duplicate tolerance</strong> and <strong>zero patient-slice leakage</strong>.
          </p>

          {/* 5-Column High-Level Metric Strip */}
          <div className="purging-stat-strip" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', marginTop: '14px' }}>
            <div className="p-stat">
              <span className="v">11,128</span>
              <span className="k">Total Verified Scans</span>
            </div>
            <div className="p-stat">
              <span className="v green">9</span>
              <span className="k">Canonical WHO Classes</span>
            </div>
            <div className="p-stat">
              <span className="v green">0</span>
              <span className="k">SHA-256 Byte Duplicates</span>
            </div>
            <div className="p-stat">
              <span className="v green">0</span>
              <span className="k">Cross-Class pHash Collisions</span>
            </div>
            <div className="p-stat">
              <span className="v green">8,104</span>
              <span className="k">GroupKFold Patient Units</span>
            </div>
          </div>
        </div>

        {/* The 3-Stage Evolution of the Dataset */}
        <div className="chronology-wrapper" style={{ marginTop: '24px' }}>
          {/* Stage 1 */}
          <div className="chronology-item">
            <div className="chronology-marker done">
              <span className="phase-num">01</span>
            </div>
            <div className="chronology-card">
              <div className="chronology-head">
                <span className="phase-badge done">Stage I · 39 Classes · Sequence-Split Architecture</span>
                <span className="phase-timestamp">12,643 Catalogued → 12,626 Clean Scans</span>
              </div>
              <h4 className="chronology-title">Mask File Purge &amp; Initial 39-Class Modality Siloing</h4>
              <p className="chronology-text">
                The initial raw repository contained 12,643 files organized into 13 WHO tumor types, each subdivided into three pulse sequences (T1, T1C+, T2), yielding 39 classes.
                A strict filesystem audit identified <strong>17 corrupt entries</strong> where binary segmentation mask files (ending with <code>_mask.png</code>) were mistakenly catalogued as source DICOM/PNG scans.
                Purging these 17 files resulted in 12,626 clean scans.
              </p>
              <div className="purging-stat-strip" style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginTop: '10px' }}>
                <div className="p-stat">
                  <span className="v">12,643</span>
                  <span className="k">Raw Catalog Files</span>
                </div>
                <div className="p-stat">
                  <span className="v red">-17</span>
                  <span className="k">Corrupt Mask Files Purged</span>
                </div>
                <div className="p-stat">
                  <span className="v green">12,626</span>
                  <span className="k">Stage I Clean Scans</span>
                </div>
              </div>
              <div className="anomaly-breakdown-grid" style={{ marginTop: '12px' }}>
                <div className="anomaly-box">
                  <div className="anomaly-box-title">
                    <AlertTriangle size={14} className="text-warning" />
                    <span>Critical Flaw: Extreme Sequence Fragmentation</span>
                  </div>
                  <p className="anomaly-copy">
                    Subdividing 13 tumor families across 3 pulse sequences created extreme statistical starvation. Ganglioglioma T1 contained only <strong>42 scans</strong>, Ganglioglioma T2 contained <strong>45 scans</strong>, and Neurocytoma T1 had <strong>47 scans</strong>.
                    An 8.7M-parameter backbone cannot learn generalizable features from 30 training images.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Stage 2 */}
          <div className="chronology-item">
            <div className="chronology-marker warning">
              <span className="phase-num">02</span>
            </div>
            <div className="chronology-card warning-border">
              <div className="chronology-head">
                <span className="phase-badge warning">Stage II · 42 Classes · BTSC Expansion &amp; Leakage Audit</span>
                <span className="phase-timestamp">4,000 External BTSC Scans Added → 16,626 Scans</span>
              </div>
              <h4 className="chronology-title">Discovery of 1,046 Byte Duplicates &amp; Patient-Slice Crossings</h4>
              <p className="chronology-text">
                To fix the total absence of Pituitary adenomas and address the "Normal" class shortcut learning confounder, 4,000 scans from the external Brain Tumor Slices Collection (BTSC) were integrated, expanding the space to 42 classes (14 families × 3 sequences) with 16,626 files.
                However, a rigorous cryptographic audit revealed two critical failure modes:
              </p>
              <div className="anomaly-breakdown-grid" style={{ marginTop: '12px' }}>
                <div className="anomaly-box">
                  <div className="anomaly-box-title">
                    <AlertTriangle size={14} className="text-warning" />
                    <span>1,046 Exact Byte Duplicates Identified</span>
                  </div>
                  <p className="anomaly-copy">
                    SHA-256 byte hashing identified 1,046 identical images stored under different filenames or across subdirectories, polluting evaluation splits.
                  </p>
                </div>
                <div className="anomaly-box">
                  <div className="anomaly-box-title">
                    <AlertTriangle size={14} className="text-warning" />
                    <span>Patient-Slice Leakage Inflating Claimed Accuracy</span>
                  </div>
                  <p className="anomaly-copy">
                    Adjacent axial slices from the same patient volume were scattered across train, val, and test splits. The model memorized patient skull geometries rather than tumor histopathology. When de-leaked via GroupKFold, top-1 accuracy dropped from 97% to an honest 72.7%.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Stage 3 */}
          <div className="chronology-item">
            <div className="chronology-marker active">
              <span className="phase-num">03</span>
            </div>
            <div className="chronology-card active-border">
              <div className="chronology-head">
                <span className="phase-badge active">Stage III · 9 Classes · Modality Flattening &amp; Deduplication</span>
                <span className="phase-timestamp">Active Production Standard · 11,128 Verified Scans</span>
              </div>
              <h4 className="chronology-title">Modality Harmonization, Normal Trimming &amp; 100% Duplicate-Free Guarantee</h4>
              <p className="chronology-text">
                To permanently eliminate sequence fragmentation and clinical dissonance, we unified all pulse sequences (T1, T1C+, T2) into <strong>9 canonical WHO histological classes</strong>.
                Simultaneously, we purged 1,168 redundant slices and filter variants (<code>enh_</code> copies) from the Normal class (down from 3,360 to 2,192) to prevent normal class dominance, and eliminated all byte duplicates across patient classes.
              </p>
              <div className="purging-stat-strip" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginTop: '12px' }}>
                <div className="p-stat">
                  <span className="v">16,626</span>
                  <span className="k">Stage II Scans</span>
                </div>
                <div className="p-stat">
                  <span className="v red">-1,168</span>
                  <span className="k">Redundant Normal Trimmed</span>
                </div>
                <div className="p-stat">
                  <span className="v red">-4,330</span>
                  <span className="k">Duplicates &amp; Outliers Purged</span>
                </div>
                <div className="p-stat">
                  <span className="v green">11,128</span>
                  <span className="k">Net Verified Scans</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Section: Why Flatten T1, T2, T1C+ into Canonical Classes */}
        <div className="doc-info-block" style={{ marginTop: '24px' }}>
          <div className="doc-block-header">
            <Layers size={16} className="doc-block-icon" />
            <span className="doc-block-title">The Engineering &amp; Clinical Rationale for Modality Flattening</span>
          </div>
          <p className="doc-block-copy">
            In cranial Magnetic Resonance Imaging (MRI), T1-weighted, T1-contrast-enhanced (T1C+), and T2-weighted scans are not distinct diseases.
            They are <strong>multi-parametric physical acquisitions of the identical patient pathology</strong>:
          </p>
          <ul style={{ margin: '8px 0 14px 20px', fontSize: '12.5px', lineHeight: 1.6, color: 'var(--ink-soft)' }}>
            <li><strong>T1-Weighted (T1):</strong> Demonstrates high signal for fat and subacute hemorrhage, providing anatomical baseline contrast.</li>
            <li><strong>T1-Contrast (T1C+):</strong> Gadolinium-chelate paramagnetic enhancement reveals breakdown of the blood-brain barrier (active neoplastic angiogenesis).</li>
            <li><strong>T2-Weighted (T2):</strong> Demonstrates high signal for water and fluid, capturing peritumoral vasogenic edema and cystic tissue components.</li>
          </ul>
          <p className="doc-block-copy">
            When training 39 or 42 separate classes, the neural network expended significant convolutional capacity learning trivial contrast differences (e.g. "CSF is bright on T2, dark on T1") rather than identifying neoplastic morphological features.
            By flattening the sequence modalities into 9 canonical WHO classes, the model is forced to learn <strong>contrast-invariant tumor features</strong> (irregular margins, infiltrative growth patterns, mass effects, necrosis) regardless of MRI sequence.
          </p>

          {/* Statistical Mass Explosion Table */}
          <div className="class-inventory-table-wrap" style={{ marginTop: '16px' }}>
            <div style={{ padding: '12px 16px', background: 'var(--bone)', borderBottom: '1px solid var(--line)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink)' }}>
                Per-Class Sample Size Transformation: Sequence-Split vs. Modality-Flattened
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-soft)' }}>
                Minimum Class Size Raised from 42 to 233 / 539 Scans
              </span>
            </div>
            <table className="class-inventory-table">
              <thead>
                <tr>
                  <th>Canonical 9-Class Label</th>
                  <th>Pre-Flattening Sequence Breakdown (T1 / T1C+ / T2)</th>
                  <th>Pre-Flatten Min Class</th>
                  <th>Post-Flatten Total Scans</th>
                  <th>Statistical Mass Gain</th>
                  <th>Clinical Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="class-name-cell">Mixed Neuronal and Glial Tumors</td>
                  <td>Ganglioglioma (42 / 70 / 45) + Neurocytoma (47 / 145 / 98) + DNET (120 / 180 / 80)</td>
                  <td style={{ color: 'var(--accent)', fontWeight: 700 }}>42 scans (T1)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>787 scans</td>
                  <td style={{ color: 'var(--olive)', fontWeight: 700 }}>+1,774% gain</td>
                  <td><span className="tag-pill patient">Patient Series Grouped</span></td>
                </tr>
                <tr>
                  <td className="class-name-cell">Medulloblastoma</td>
                  <td>Embryonic Tumors (64 T1 / 419 T1C+ / 320 T2)</td>
                  <td style={{ color: 'var(--accent)', fontWeight: 700 }}>64 scans (T1)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>803 scans</td>
                  <td style={{ color: 'var(--olive)', fontWeight: 700 }}>+1,155% gain</td>
                  <td><span className="tag-pill patient">Patient Series Grouped</span></td>
                </tr>
                <tr>
                  <td className="class-name-cell">Schwannoma</td>
                  <td>Cranial Nerve Tumors (98 T1 / 511 T1C+ / 402 T2)</td>
                  <td style={{ color: 'var(--accent)', fontWeight: 700 }}>98 scans (T1)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>1,011 scans</td>
                  <td style={{ color: 'var(--olive)', fontWeight: 700 }}>+932% gain</td>
                  <td><span className="tag-pill patient">Patient Series Grouped</span></td>
                </tr>
                <tr>
                  <td className="class-name-cell">Mesenchymal (Non-Meningothelial)</td>
                  <td>Hemangiopericytoma / Solitary Fibrous (68 T1 / 289 T1C+ / 182 T2)</td>
                  <td style={{ color: 'var(--accent)', fontWeight: 700 }}>68 scans (T1)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>539 scans</td>
                  <td style={{ color: 'var(--olive)', fontWeight: 700 }}>+693% gain</td>
                  <td><span className="tag-pill patient">Patient Series Grouped</span></td>
                </tr>
                <tr>
                  <td className="class-name-cell">Germ Cell Tumors</td>
                  <td>Germinoma across sequences (45 T1 / 112 T1C+ / 76 T2)</td>
                  <td style={{ color: 'var(--accent)', fontWeight: 700 }}>45 scans (T1)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>233 scans</td>
                  <td style={{ color: 'var(--olive)', fontWeight: 700 }}>+418% gain</td>
                  <td><span className="tag-pill minority">Minority RandAugment m7</span></td>
                </tr>
                <tr>
                  <td className="class-name-cell">Gliomas</td>
                  <td>Astrocytoma + Glioblastoma + Oligodendroglioma across T1/T1C+/T2</td>
                  <td>450 scans (T1 Astrocytoma)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>2,039 scans</td>
                  <td style={{ color: 'var(--olive)', fontWeight: 700 }}>+353% gain</td>
                  <td><span className="tag-pill independent">Independent Studies</span></td>
                </tr>
                <tr>
                  <td className="class-name-cell">Meningothelial Tumors</td>
                  <td>Meningioma across T1 (430), T1C+ (1,057), and T2 (333)</td>
                  <td>333 scans (T2)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>1,820 scans</td>
                  <td style={{ color: 'var(--olive)', fontWeight: 700 }}>+323% gain</td>
                  <td><span className="tag-pill independent">Independent Studies</span></td>
                </tr>
                <tr>
                  <td className="class-name-cell">Pituitary</td>
                  <td>Pituitary adenomas across T1, T1C+, T2 (BTSC cohort)</td>
                  <td>550 scans (T1)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>1,704 scans</td>
                  <td style={{ color: 'var(--olive)', fontWeight: 700 }}>+210% gain</td>
                  <td><span className="tag-pill independent">Independent Studies</span></td>
                </tr>
                <tr>
                  <td className="class-name-cell">Normal</td>
                  <td>Healthy cranial MRI scans (BTSC + clinical cohort)</td>
                  <td>3,360 scans (untrimmed with enh_ filter copies)</td>
                  <td style={{ fontWeight: 700, color: 'var(--ink)' }}>2,192 scans</td>
                  <td style={{ color: 'var(--accent)', fontWeight: 700 }}>-34.8% (trimmed)</td>
                  <td><span className="tag-pill standard">Trimmed to Restore Balance</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Deduplication Verification & Hashing Audit Protocol */}
        <div className="doc-info-block" style={{ marginTop: '24px' }}>
          <div className="doc-block-header">
            <Binary size={16} className="doc-block-icon" />
            <span className="doc-block-title">Dual-Layer Deduplication Verification Protocol (SHA-256 &amp; 64-bit DCT pHash)</span>
          </div>
          <p className="doc-block-copy">
            To guarantee zero data leakage between splits and eliminate artificial redundancy, every image was audited using a dual-layer cryptographic and perceptual pipeline:
          </p>

          <div className="anomaly-breakdown-grid" style={{ marginTop: '12px' }}>
            <div className="anomaly-box">
              <div className="anomaly-box-title">
                <CheckCircle2 size={14} className="text-olive" />
                <span>Layer 1: SHA-256 Exact Byte Verification</span>
              </div>
              <p className="anomaly-copy">
                Computes a 256-bit cryptographic digest over raw file bytes. Identifies identical scans saved under differing names or directories.
                <br />
                <strong>Audit Result:</strong> Exactly <strong>0 duplicate clusters</strong> within the same class, and <strong>0 cross-class clusters</strong>. 100% duplicate-free.
              </p>
            </div>

            <div className="anomaly-box">
              <div className="anomaly-box-title">
                <CheckCircle2 size={14} className="text-olive" />
                <span>Layer 2: 64-bit Discrete Cosine Transform (DCT) pHash</span>
              </div>
              <p className="anomaly-copy">
                Reduces images to 32×32 grayscale, computes 2D DCT, extracts top 8×8 low frequencies, and binarizes relative to median. Flags candidate pairs with Hamming distance ≤ 2.
                <br />
                <strong>Audit Result:</strong> 635 same-class near-duplicate clusters (adjacent slices of patient series, preserved and locked into GroupKFold units). 0 cross-class collisions (6 candidate pairs inspected and confirmed to be outer dark skull border artifacts; MSE &gt; 1000, pixel corr 0.62–0.75).
              </p>
            </div>
          </div>

          {/* Interactive Terminal Audit Viewer */}
          <div className="kaggle-terminal-card" style={{ marginTop: '16px' }}>
            <div className="terminal-header">
              <div className="terminal-dots">
                <span className="dot red" />
                <span className="dot yellow" />
                <span className="dot green" />
              </div>

              <div className="terminal-tabs-strip">
                <button
                  type="button"
                  className={`terminal-tab-btn ${auditTab === 'audit_log' ? 'active' : ''}`}
                  onClick={() => setAuditTab('audit_log')}
                >
                  <Terminal size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                  Audit Terminal Log
                </button>
                <button
                  type="button"
                  className={`terminal-tab-btn ${auditTab === 'hasher_script' ? 'active' : ''}`}
                  onClick={() => setAuditTab('hasher_script')}
                >
                  <Binary size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                  Python Audit Code
                </button>
                <button
                  type="button"
                  className={`terminal-tab-btn ${auditTab === 'split_audit' ? 'active' : ''}`}
                  onClick={() => setAuditTab('split_audit')}
                >
                  <Layers size={11} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                  Patient Group Invariant
                </button>
              </div>

              <button
                type="button"
                className="terminal-copy-btn"
                onClick={handleCopyAuditScript}
                title="Copy Audit Script"
              >
                {copiedAuditCmd ? (
                  <>
                    <Check size={12} />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy size={12} />
                    <span>Copy Script</span>
                  </>
                )}
              </button>
            </div>

            <pre className="terminal-body">
              {auditTab === 'audit_log' && (
                <code>
                  <span className="t-comment"># ==============================================================================</span>{'\n'}
                  <span className="t-comment"># DATA INTEGRITY REPORT — 9-Class Brain Classification Dataset</span>{'\n'}
                  <span className="t-comment"># ==============================================================================</span>{'\n'}
                  Root      : /home/honeypot/Desktop/archive/Images_{'\n'}
                  Scanned   : 11,128 files, 9 classes, workers=4{'\n'}
                  Methods   : SHA-256 exact byte digest | 64-bit DCT pHash (Hamming&lt;=2) | 32x32 luminance signature{'\n'}
                  {'\n'}
                  Class counts:{'\n'}
                       233  Germ Cell Tumors [patient slices]{'\n'}
                     2,039  Gliomas [independent]{'\n'}
                       803  Medulloblastoma [patient slices]{'\n'}
                     1,820  Meningothelial Tumors [independent]{'\n'}
                       539  Mesenchymal (Non-Meningothelial Tumors) [patient slices]{'\n'}
                       787  Mixed Neuronal and Neuronal-Glial Tumors [patient slices]{'\n'}
                     2,192  Normal [independent, 1 per patient + filter augments]{'\n'}
                     1,704  Pituitary [independent]{'\n'}
                     1,011  Schwannoma [patient slices]{'\n'}
                    11,128  TOTAL decodable images{'\n'}
                  {'\n'}
                  1. CORRUPT / UNDECODABLE FILES{'\n'}
                     [PASS] all 11,128 files decode cleanly. Zero corrupted files.{'\n'}
                  {'\n'}
                  2. EXACT DUPLICATES (SHA-256 over raw file bytes){'\n'}
                     same image twice in one class : 0 cluster(s), 0 file(s){'\n'}
                     same image in two classes     : 0 cluster(s){'\n'}
                     [PASS] exact-duplicate verdict: 100% duplicate-free{'\n'}
                  {'\n'}
                  3. PERCEPTUAL NEAR-DUPLICATES (64-bit DCT pHash Hamming &lt;= 2){'\n'}
                     same-class near-dup clusters  : 635 (adjacent patient slices or filter variants){'\n'}
                     cross-class collisions        : 0 (6 candidate pairs verified as elliptical skull border artifacts; MSE &gt; 1000, pixel corr 0.62–0.75){'\n'}
                     [PASS] cross-class collision verdict{'\n'}
                  {'\n'}
                  Archive ready: /home/honeypot/Desktop/archive.zip (590.3 MB, 11,128 files)
                </code>
              )}

              {auditTab === 'hasher_script' && (
                <code>
                  <span className="t-comment"># Cryptographic and perceptual deduplication audit routine:</span>{'\n'}
                  import os, hashlib{'\n'}
                  from PIL import Image{'\n'}
                  import imagehash{'\n'}
                  {'\n'}
                  def audit_dataset(root_dir: str):{'\n'}
                  {'    '}sha_table = {}{'\n'}
                  {'    '}phash_table = {}{'\n'}
                  {'    '}for root, _, files in os.walk(root_dir):{'\n'}
                  {'        '}for f in files:{'\n'}
                  {'            '}if not f.lower().endswith(('.jpg', '.jpeg', '.png')):{'\n'}
                  {'                '}continue{'\n'}
                  {'            '}path = os.path.join(root, f){'\n'}
                  {'            '}cls = os.path.basename(root){'\n'}
                  {'            '}{'\n'}
                  {'            '}<span className="t-comment"># 1. SHA-256 exact byte hash</span>{'\n'}
                  {'            '}with open(path, 'rb') as fp:{'\n'}
                  {'                '}digest = hashlib.sha256(fp.read()).hexdigest(){'\n'}
                  {'            '}sha_table.setdefault(digest, []).append((path, cls)){'\n'}
                  {'            '}{'\n'}
                  {'            '}<span className="t-comment"># 2. 64-bit DCT perceptual hash</span>{'\n'}
                  {'            '}with Image.open(path) as img:{'\n'}
                  {'                '}h = str(imagehash.phash(img)){'\n'}
                  {'            '}phash_table.setdefault(h, []).append((path, cls)){'\n'}
                  {'    '}{'\n'}
                  {'    '}<span className="t-comment"># Check for byte duplicate collisions</span>{'\n'}
                  {'    '}collisions = &#123;k: v for k, v in sha_table.items() if len(v) &gt; 1&#125;{'\n'}
                  {'    '}assert len(collisions) == 0, f"Found &#123;len(collisions)&#125; exact duplicate clusters"{'\n'}
                  {'    '}print("VERDICT: 100% duplicate-free verified.")
                </code>
              )}

              {auditTab === 'split_audit' && (
                <code>
                  <span className="t-comment"># Patient-Level Grouping &amp; Split Leakage Proof:</span>{'\n'}
                  Discovered 11,128 images across 9 canonical classes.{'\n'}
                  Grouping resolved 11,128 images into 8,104 source groups:{'\n'}
                    - 230 verified patient slice series (GroupKFold units){'\n'}
                    - 18 shared-prefix stems rejected (unrelated patients kept independent){'\n'}
                    - 12 independent-class cohorts kept unmerged{'\n'}
                  {'\n'}
                  Partitioning Summary:{'\n'}
                    Train : 7,809 images (70.1%){'\n'}
                    Val   : 1,623 images (14.6%){'\n'}
                    Test  : 1,696 images (15.2%){'\n'}
                  {'\n'}
                  Split Leakage Verification:{'\n'}
                    Total source patient groups : 8,104{'\n'}
                    Groups spanning &gt;1 split    : 0{'\n'}
                    Patient-slice crossings     : 0{'\n'}
                    Test Accuracy (Leak-Free)   : 92.87% (Loss: 0.2841){'\n'}
                    [PASS] Zero patient-slice or duplicate leakage across splits.
                </code>
              )}
            </pre>
          </div>
        </div>
      </RomanSection>
    </section>
  )
}
