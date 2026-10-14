import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

/**
 * Scorecard displayed at the top of each retriever column showing real-time
 * benchmark evaluation metrics for the active claim.
 */
function ModelScorecard({ modelKey, modelTitle, metrics, isKnownClaim, k }) {
  if (!isKnownClaim) {
    return (
      <div className="model-scorecard scorecard-custom">
        <div className="custom-status-row">
          <span className="scorecard-tag">{modelTitle}</span>
          <span className="badge-adhoc">Ad-hoc Ranking</span>
        </div>
        <span className="scorecard-hint">
          Ranked by similarity score. (Hit/Miss requires human gold labels).
        </span>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="model-scorecard scorecard-pending">
        <span className="scorecard-hint">No evaluation data for this model</span>
      </div>
    );
  }

  const isHit = metrics.first_gold_rank !== null;

  return (
    <div className={`model-scorecard ${isHit ? 'scorecard-hit' : 'scorecard-miss'}`}>
      <div className="scorecard-header-row">
        <div className={`scorecard-status-badge ${isHit ? 'badge-hit' : 'badge-miss'}`}>
          {isHit ? `⭐ First Gold: Rank #${metrics.first_gold_rank}` : `⚠️ Missed in Top-${k}`}
        </div>
        <span className="scorecard-tag">{modelTitle}</span>
      </div>

      <div className="scorecard-metrics-grid">
        <div className="metric-cell" title="Recall@K: Fraction of human-annotated gold evidence retrieved">
          <span className="metric-label">Recall@{k}</span>
          <span className={`metric-value ${metrics.recall > 0 ? 'text-highlight' : ''}`}>
            {(metrics.recall * 100).toFixed(0)}%
          </span>
        </div>

        <div className="metric-cell" title="Precision@K: Fraction of retrieved documents that are gold evidence">
          <span className="metric-label">Prec@{k}</span>
          <span className="metric-value">
            {(metrics.precision * 100).toFixed(0)}%
          </span>
        </div>

        <div className="metric-cell" title="Average Precision (AP) at depth K">
          <span className="metric-label">Avg Prec</span>
          <span className="metric-value">
            {metrics.average_precision.toFixed(2)}
          </span>
        </div>

        <div className="metric-cell" title="Reciprocal Rank (1 / rank of first gold document)">
          <span className="metric-label">MRR</span>
          <span className={`metric-value ${isHit ? 'text-highlight' : ''}`}>
            {metrics.reciprocal_rank.toFixed(2)}
          </span>
        </div>

        <div className="metric-cell" title="Multi-Hop Evidence Set Coverage@K">
          <span className="metric-label">Set Cov</span>
          <span className={`metric-value ${metrics.set_coverage > 0 ? 'text-highlight' : ''}`}>
            {(metrics.set_coverage * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState('');
  const [k, setK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [claimBenchmark, setClaimBenchmark] = useState(null);
  const [globalBenchmarks, setGlobalBenchmarks] = useState([]);

  // Modals & Panels
  const [showConfig, setShowConfig] = useState(false);
  const [showBenchmarkModal, setShowBenchmarkModal] = useState(false);
  const [pineconeKey, setPineconeKey] = useState('');
  const [pineconeIndex, setPineconeIndex] = useState('fever-evidence-index');
  const [configMessage, setConfigMessage] = useState('');

  // Initial fetch on mount
  useEffect(() => {
    fetchStatus();
    fetchSamples();
    fetchBenchmarks();
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.warn('Backend not yet reachable on http://localhost:8000:', err);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/sample-claims`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data);
      }
    } catch (err) {
      console.warn('Could not fetch sample claims:', err);
    }
  };

  const fetchBenchmarks = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/benchmarks`);
      if (res.ok) {
        const data = await res.json();
        setGlobalBenchmarks(data);
      }
    } catch (err) {
      console.warn('Could not fetch global benchmarks:', err);
    }
  };

  const handleSearch = async (overrideQuery = null, claimMeta = null) => {
    const q = overrideQuery || query;
    if (!q.trim()) return;

    if (claimMeta) {
      setSelectedClaim(claimMeta);
    }

    setLoading(true);
    try {
      const payload = {
        query: q,
        k: parseInt(k, 10),
        qid: claimMeta?.id || selectedClaim?.id || null,
        gold_doc_ids: claimMeta?.gold_doc_ids || selectedClaim?.gold_doc_ids || null,
        label: claimMeta?.label || selectedClaim?.label || null,
      };

      const res = await fetch(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        setResults(data.results);
        setClaimBenchmark(data.claim_benchmark);
        if (data.global_benchmarks?.length) {
          setGlobalBenchmarks(data.global_benchmarks);
        }
        if (data.status) {
          setStatus(data.status);
        }
      }
    } catch (err) {
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClick = (sample) => {
    setQuery(sample.claim);
    setSelectedClaim(sample);
    handleSearch(sample.claim, sample);
  };

  const handleSavePinecone = async () => {
    if (!pineconeKey.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/api/pinecone-config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: pineconeKey.trim(),
          index_name: pineconeIndex.trim() || 'fever-evidence-index',
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setConfigMessage(data.dense_ready ? 'Connected to Pinecone!' : 'Configured (verify index on Colab).');
        fetchStatus();
        setTimeout(() => setConfigMessage(''), 4000);
      }
    } catch (err) {
      setConfigMessage('Error contacting backend.');
    }
  };

  const isGoldHit = (docId) => {
    if (claimBenchmark?.gold_doc_ids?.includes(docId)) return true;
    if (selectedClaim?.gold_doc_ids?.includes(docId)) return true;
    return false;
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header glass-panel">
        <div className="brand-section">
          <div className="brand-icon">🔬</div>
          <div>
            <h1 className="brand-title">FEVERScope</h1>
            <p className="brand-subtitle">
              Evidence Retrieval Benchmark: TF-IDF vs BM25 vs Semantic Transformer vs RRF Hybrid
            </p>
          </div>
        </div>

        <div className="status-bar">
          <div className={`status-pill ${status?.tfidf?.ready ? 'ready' : 'pending'}`}>
            <span className="status-dot"></span>
            TF-IDF: {status?.tfidf?.ready ? `${status.tfidf.num_docs.toLocaleString()} docs` : 'Ready'}
          </div>

          <div className={`status-pill ${status?.bm25?.ready ? 'ready' : 'pending'}`}>
            <span className="status-dot"></span>
            BM25: {status?.bm25?.ready ? `${status.bm25.num_docs.toLocaleString()} docs` : 'Ready'}
          </div>

          <div className={`status-pill ${status?.dense?.ready ? 'ready' : 'pending'}`}>
            <span className="status-dot"></span>
            Semantic Transformer: {status?.dense?.ready ? 'Active' : 'Connecting...'}
          </div>

          <div className="status-pill hybrid">
            <span className="status-dot"></span>
            RRF Hybrid: k={status?.rrf?.k || 60}
          </div>

          <button className="btn-benchmark-modal" onClick={() => setShowBenchmarkModal(true)}>
            📊 Global Benchmark
          </button>

          <button className="btn-config" onClick={() => setShowConfig(!showConfig)}>
            ⚙️ Vector Index Config
          </button>
        </div>
      </header>

      {/* Vector Index Config Drawer */}
      {showConfig && (
        <section className="config-drawer glass-panel">
          <input
            type="password"
            placeholder="Vector DB API Key (pcsk_...)"
            value={pineconeKey}
            onChange={(e) => setPineconeKey(e.target.value)}
          />
          <input
            type="text"
            placeholder="Vector Index Name (default: fever-evidence-index)"
            value={pineconeIndex}
            onChange={(e) => setPineconeIndex(e.target.value)}
          />
          <button className="btn-save-key" onClick={handleSavePinecone}>
            Save & Connect
          </button>
          {configMessage && <span style={{ color: '#10b981', fontSize: '0.85rem' }}>{configMessage}</span>}
        </section>
      )}

      {/* Query & Search Console */}
      <section className="search-console glass-panel">
        <div className="search-bar-wrapper">
          <input
            type="text"
            className="search-input"
            placeholder="Enter a factual claim or select a benchmark sample below..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedClaim(null);
            }}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />

          <select className="k-selector" value={k} onChange={(e) => setK(e.target.value)}>
            <option value="5">Top 5</option>
            <option value="10">Top 10</option>
            <option value="20">Top 20</option>
          </select>

          <button className="btn-search" onClick={() => handleSearch()} disabled={loading || !query.trim()}>
            {loading ? 'Searching & Benchmarking...' : 'Retrieve Evidence'}
          </button>
        </div>

        {/* Sample Claims Pills from Real Evaluation Set */}
        {samples.length > 0 && (
          <div className="sample-claims-section">
            <span className="sample-label">Benchmark Samples:</span>
            <div className="sample-chips">
              {samples.map((s) => (
                <button
                  key={s.id}
                  className={`sample-chip ${selectedClaim?.id === s.id ? 'active' : ''}`}
                  onClick={() => handleSampleClick(s)}
                >
                  <span className={s.label === 'SUPPORTS' ? 'badge-supports' : 'badge-refutes'}>
                    {s.label}
                  </span>
                  <span>{s.claim}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Claim Benchmark Status Banner */}
      {claimBenchmark && (
        <section className="claim-benchmark-banner glass-panel">
          <div className="banner-left">
            <div className="banner-tags">
              {claimBenchmark.is_known_claim ? (
                <>
                  <span className={claimBenchmark.label === 'SUPPORTS' ? 'badge-supports-lg' : 'badge-refutes-lg'}>
                    {claimBenchmark.label}
                  </span>
                  <span className="banner-verified-pill">
                    ✓ FEVER Ground Truth Benchmark
                  </span>
                </>
              ) : (
                <span className="badge-custom-lg">
                  💡 Custom User Claim (Ad-hoc)
                </span>
              )}
            </div>

            <div className="banner-query-text">
              "{query}"
            </div>

            {claimBenchmark.is_known_claim && claimBenchmark.gold_doc_ids?.length > 0 && (
              <div className="banner-gold-row">
                <span className="gold-label">Human-Annotated Gold Sentences:</span>
                <div className="gold-pills-list">
                  {claimBenchmark.gold_doc_ids.map((id) => (
                    <span key={id} className="gold-doc-pill">
                      ⭐ <code>{id}</code>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="banner-right">
            {claimBenchmark.is_known_claim ? (
              <div className="multi-model-summary">
                <span className="summary-title">First Gold Evidence Hit Rank:</span>
                <div className="model-summary-badges">
                  {[
                    { key: 'tfidf', label: 'TF-IDF' },
                    { key: 'bm25', label: 'BM25' },
                    { key: 'dense', label: 'Semantic Transformer' },
                    { key: 'rrf', label: 'RRF Hybrid' },
                  ].map(({ key, label }) => {
                    const m = claimBenchmark.metrics_per_model?.[key];
                    const hasHit = m?.first_gold_rank !== null && m?.first_gold_rank !== undefined;
                    return (
                      <div key={key} className={`summary-pill ${hasHit ? 'hit' : 'miss'}`}>
                        <span className="pill-name">{label}</span>
                        <span className="pill-rank">
                          {hasHit ? `Rank #${m.first_gold_rank}` : 'Miss'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="custom-banner-status">
                <span className="status-title">⚡ Ad-Hoc Retrieval Mode</span>
                <p className="status-desc">
                  Ranked by similarity scores. Ground-truth Hit/Miss evaluation requires human gold labels (try sample claims above).
                </p>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Multi-Method Comparison Grid with Real-Time Scorecards */}
      <main className="results-grid">
        {/* Method 1: TF-IDF */}
        <section className="method-column glass-panel">
          <div className="column-header">
            <div>
              <h2 className="method-title">
                <span>TF-IDF</span>
                <span className="method-tag">Lexical</span>
              </h2>
              <p className="method-desc">Sparse Term Frequency & Cosine Similarity</p>
            </div>
          </div>

          {/* Benchmark Scorecard for TF-IDF */}
          <ModelScorecard
            modelKey="tfidf"
            modelTitle="TF-IDF Benchmark"
            metrics={claimBenchmark?.metrics_per_model?.tfidf}
            isKnownClaim={claimBenchmark?.is_known_claim}
            k={k}
          />

          <div className="hits-list">
            {results?.tfidf?.length > 0 ? (
              results.tfidf.map((hit) => {
                const isGold = isGoldHit(hit.doc_id);
                return (
                  <article key={hit.rank} className={`hit-card ${isGold ? 'hit-card-gold' : ''}`}>
                    <div className="hit-card-top">
                      <div className="rank-group">
                        <span className="rank-pill">#{hit.rank}</span>
                        {isGold && <span className="badge-gold-hit">⭐ GOLD EVIDENCE</span>}
                      </div>
                      <span className="score-badge">score: {hit.score}</span>
                    </div>
                    <div className="hit-doc-id">
                      📄 {hit.page_title} (sent {hit.sent_id})
                    </div>
                    <p className="hit-text">{hit.text}</p>
                  </article>
                );
              })
            ) : (
              <div className="empty-state">
                <span className="empty-icon">📊</span>
                <p className="empty-text">Enter a claim or select a benchmark sample above to retrieve documents.</p>
              </div>
            )}
          </div>
        </section>

        {/* Method 2: BM25 */}
        <section className="method-column glass-panel">
          <div className="column-header">
            <div>
              <h2 className="method-title">
                <span>BM25</span>
                <span className="method-tag">Okapi</span>
              </h2>
              <p className="method-desc">Probabilistic BM25 with length normalization</p>
            </div>
          </div>

          {/* Benchmark Scorecard for BM25 */}
          <ModelScorecard
            modelKey="bm25"
            modelTitle="BM25 Benchmark"
            metrics={claimBenchmark?.metrics_per_model?.bm25}
            isKnownClaim={claimBenchmark?.is_known_claim}
            k={k}
          />

          <div className="hits-list">
            {results?.bm25?.length > 0 ? (
              results.bm25.map((hit) => {
                const isGold = isGoldHit(hit.doc_id);
                return (
                  <article key={hit.rank} className={`hit-card ${isGold ? 'hit-card-gold' : ''}`}>
                    <div className="hit-card-top">
                      <div className="rank-group">
                        <span className="rank-pill">#{hit.rank}</span>
                        {isGold && <span className="badge-gold-hit">⭐ GOLD EVIDENCE</span>}
                      </div>
                      <span className="score-badge">score: {hit.score}</span>
                    </div>
                    <div className="hit-doc-id">
                      📄 {hit.page_title} (sent {hit.sent_id})
                    </div>
                    <p className="hit-text">{hit.text}</p>
                  </article>
                );
              })
            ) : (
              <div className="empty-state">
                <span className="empty-icon">📈</span>
                <p className="empty-text">Enter a claim to retrieve BM25 probabilistic rankings.</p>
              </div>
            )}
          </div>
        </section>

        {/* Method 3: Semantic Transformer */}
        <section className="method-column glass-panel">
          <div className="column-header">
            <div>
              <h2 className="method-title">
                <span>Semantic Transformer</span>
                <span className="method-tag">Dense Bi-Encoder</span>
              </h2>
              <p className="method-desc">Sentence-Transformers (all-MiniLM-L6-v2 · 384d)</p>
            </div>
          </div>

          {/* Benchmark Scorecard for Semantic Transformer */}
          <ModelScorecard
            modelKey="dense"
            modelTitle="Semantic Transformer"
            metrics={claimBenchmark?.metrics_per_model?.dense}
            isKnownClaim={claimBenchmark?.is_known_claim}
            k={k}
          />

          <div className="hits-list">
            {results?.dense?.length > 0 ? (
              results.dense.map((hit) => {
                const isGold = isGoldHit(hit.doc_id);
                return (
                  <article key={hit.rank} className={`hit-card ${isGold ? 'hit-card-gold' : ''}`}>
                    <div className="hit-card-top">
                      <div className="rank-group">
                        <span className="rank-pill">#{hit.rank}</span>
                        {isGold && <span className="badge-gold-hit">⭐ GOLD EVIDENCE</span>}
                      </div>
                      <span className="score-badge">cosine: {hit.score}</span>
                    </div>
                    <div className="hit-doc-id">
                      📄 {hit.page_title} (sent {hit.sent_id})
                    </div>
                    <p className="hit-text">{hit.text}</p>
                  </article>
                );
              })
            ) : (
              <div className="empty-state">
                <span className="empty-icon">🧠</span>
                <p className="empty-text">
                  {status?.dense?.ready
                    ? 'Enter a claim to retrieve Semantic Transformer dense representations.'
                    : 'Semantic Transformer embeddings powered by all-MiniLM-L6-v2. Configure vector index above.'}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Method 4: RRF Hybrid */}
        <section className="method-column hybrid-column glass-panel">
          <div className="column-header">
            <div>
              <h2 className="method-title">
                <span>RRF Hybrid</span>
                <span className="method-tag" style={{ background: 'rgba(99, 102, 241, 0.25)', color: '#c7d2fe' }}>
                  Fusion
                </span>
              </h2>
              <p className="method-desc">Reciprocal Rank Fusion across active systems</p>
            </div>
          </div>

          {/* Benchmark Scorecard for RRF */}
          <ModelScorecard
            modelKey="rrf"
            modelTitle="RRF Hybrid Benchmark"
            metrics={claimBenchmark?.metrics_per_model?.rrf}
            isKnownClaim={claimBenchmark?.is_known_claim}
            k={k}
          />

          <div className="hits-list">
            {results?.rrf?.length > 0 ? (
              results.rrf.map((hit) => {
                const isGold = isGoldHit(hit.doc_id);
                return (
                  <article key={hit.rank} className={`hit-card ${isGold ? 'hit-card-gold' : ''}`}>
                    <div className="hit-card-top">
                      <div className="rank-group">
                        <span className="rank-pill" style={{ color: '#a5b4fc', background: 'rgba(99, 102, 241, 0.2)' }}>
                          #{hit.rank}
                        </span>
                        {isGold && <span className="badge-gold-hit">⭐ GOLD EVIDENCE</span>}
                      </div>
                      <span className="score-badge">rrf: {hit.score}</span>
                    </div>
                    <div className="hit-doc-id">
                      📄 {hit.page_title} (sent {hit.sent_id})
                    </div>
                    <p className="hit-text">{hit.text}</p>
                  </article>
                );
              })
            ) : (
              <div className="empty-state">
                <span className="empty-icon">⚡</span>
                <p className="empty-text">RRF merges rankings from all active retrievers above.</p>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Global Benchmark Modal */}
      {showBenchmarkModal && (
        <div className="modal-backdrop" onClick={() => setShowBenchmarkModal(false)}>
          <div className="modal-container glass-panel" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h2 className="modal-title">📊 FEVER Evidence Retrieval Scientific Benchmark</h2>
                <p className="modal-subtitle">
                  Evaluation across 1,482 test queries over 52,051 Wikipedia documents
                </p>
              </div>
              <button className="btn-close-modal" onClick={() => setShowBenchmarkModal(false)}>
                ✕
              </button>
            </div>

            <div className="modal-body">
              {/* Comparative Metrics Table */}
              <div className="table-wrapper">
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Retriever Model</th>
                      <th>P@1</th>
                      <th>P@5</th>
                      <th>P@10</th>
                      <th>Recall@5</th>
                      <th>Recall@10</th>
                      <th>Recall@20</th>
                      <th>MAP@100</th>
                      <th>nDCG@10</th>
                      <th>MRR</th>
                      <th>SetCov@10</th>
                    </tr>
                  </thead>
                  <tbody>
                    {globalBenchmarks.map((row, idx) => {
                      const isRRF = row.Model?.toLowerCase().includes('rrf');
                      return (
                        <tr key={idx} className={isRRF ? 'row-highlight' : ''}>
                          <td className="model-name-cell">
                            <strong>{row.Model}</strong>
                          </td>
                          <td className={row['p@1'] > 0.7 ? 'metric-lead' : ''}>
                            {(row['p@1'] * 100).toFixed(1)}%
                          </td>
                          <td>{(row['p@5'] * 100).toFixed(1)}%</td>
                          <td>{(row['p@10'] * 100).toFixed(1)}%</td>
                          <td>{(row['r@5'] * 100).toFixed(1)}%</td>
                          <td>{(row['r@10'] * 100).toFixed(1)}%</td>
                          <td className={row['r@20'] > 0.93 ? 'metric-lead' : ''}>
                            <strong>{(row['r@20'] * 100).toFixed(1)}%</strong>
                          </td>
                          <td>{row['map@100']?.toFixed(3) || '-'}</td>
                          <td>{row['ndcg@10']?.toFixed(3) || '-'}</td>
                          <td className={row.mrr > 0.8 ? 'metric-lead' : ''}>
                            <strong>{row.mrr?.toFixed(3) || '-'}</strong>
                          </td>
                          <td className={row['set_cov@10'] > 0.899 ? 'metric-lead' : ''}>
                            <strong>{(row['set_cov@10'] * 100).toFixed(1)}%</strong>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Benchmark Scientific Insights */}
              <div className="benchmark-cards-grid">
                <div className="insight-card">
                  <div className="insight-header">
                    <span className="insight-icon">🎯</span>
                    <h3>BM25 Okapi Dominance in Precision</h3>
                  </div>
                  <p>
                    BM25 achieves the highest single-hit Precision@1 (<strong>72.06%</strong>) and MRR (<strong>0.804</strong>) because Wikipedia article titles and key entity terms create strong inverse document frequency signals.
                  </p>
                </div>

                <div className="insight-card">
                  <div className="insight-header">
                    <span className="insight-icon">🧠</span>
                    <h3>Semantic Transformer (Dense Bi-Encoder)</h3>
                  </div>
                  <p>
                    Sentence-Transformers (<code>all-MiniLM-L6-v2</code>) 384-dimensional dense vectors capture conceptual synonyms and paraphrased claims that fail lexical string matching.
                  </p>
                </div>

                <div className="insight-card">
                  <div className="insight-header">
                    <span className="insight-icon">⚡</span>
                    <h3>RRF Hybrid Ensembling Synergy</h3>
                  </div>
                  <p>
                    Reciprocal Rank Fusion achieves peak Recall@20 (<strong>93.42%</strong>) and multi-hop Set Coverage (<strong>90.01%</strong>), combining sparse lexical precision with dense semantic recall.
                  </p>
                </div>
              </div>
            </div>

            <div className="modal-footer">
              <span className="footer-note">
                Evaluated against the official FEVER task dev split with human ground-truth evidence sets (§6).
              </span>
              <button className="btn-close-footer" onClick={() => setShowBenchmarkModal(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
