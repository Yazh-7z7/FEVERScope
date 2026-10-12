import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [query, setQuery] = useState('');
  const [k, setK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedClaim, setSelectedClaim] = useState(null);

  // Pinecone drawer toggle and inputs
  const [showConfig, setShowConfig] = useState(false);
  const [pineconeKey, setPineconeKey] = useState('');
  const [pineconeIndex, setPineconeIndex] = useState('fever-evidence-index');
  const [configMessage, setConfigMessage] = useState('');

  // Initial fetch on mount
  useEffect(() => {
    fetchStatus();
    fetchSamples();
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

  const handleSearch = async (overrideQuery = null, claimMeta = null) => {
    const q = overrideQuery || query;
    if (!q.trim()) return;

    if (claimMeta) {
      setSelectedClaim(claimMeta);
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, k: parseInt(k, 10) }),
      });

      if (res.ok) {
        const data = await res.json();
        setResults(data.results);
        if (data.status) setStatus(data.status);
      }
    } catch (err) {
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClick = (sample) => {
    setQuery(sample.claim);
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
    return selectedClaim?.gold_doc_ids?.includes(docId);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header glass-panel">
        <div className="brand-section">
          <div className="brand-icon">🔬</div>
          <div>
            <h1 className="brand-title">FEVERScope</h1>
            <p className="brand-subtitle">Evidence Retrieval Benchmark: TF-IDF vs BM25 vs Dense vs RRF Hybrid</p>
          </div>
        </div>

        <div className="status-bar">
          <div className={`status-pill ${status?.tfidf?.ready ? 'ready' : 'pending'}`}>
            <span className="status-dot"></span>
            TF-IDF: {status?.tfidf?.ready ? `${status.tfidf.num_docs} docs` : 'Ready'}
          </div>

          <div className={`status-pill ${status?.bm25?.ready ? 'ready' : 'pending'}`}>
            <span className="status-dot"></span>
            BM25: {status?.bm25?.ready ? `${status.bm25.num_docs} docs` : 'Ready'}
          </div>

          <div className={`status-pill ${status?.dense?.ready ? 'ready' : 'pending'}`}>
            <span className="status-dot"></span>
            Dense: {status?.dense?.ready ? 'Pinecone Active' : 'Pinecone (Colab)'}
          </div>

          <div className="status-pill hybrid">
            <span className="status-dot"></span>
            RRF Hybrid: k={status?.rrf?.k || 60}
          </div>

          <button className="btn-config" onClick={() => setShowConfig(!showConfig)}>
            ⚙️ Pinecone Config
          </button>
        </div>
      </header>

      {/* Pinecone Config Drawer */}
      {showConfig && (
        <section className="config-drawer glass-panel">
          <input
            type="password"
            placeholder="Paste Pinecone API Key (pcsk_...)"
            value={pineconeKey}
            onChange={(e) => setPineconeKey(e.target.value)}
          />
          <input
            type="text"
            placeholder="Pinecone Index Name (default: fever-evidence-index)"
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
            placeholder="Enter a factual claim to retrieve evidence (e.g. 'Roman Empire was one of the largest empires in history')..."
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
            {loading ? 'Searching...' : 'Retrieve Evidence'}
          </button>
        </div>

        {/* Sample Claims Pills */}
        {samples.length > 0 && (
          <div className="sample-claims-section">
            <span className="sample-label">Try a sample claim:</span>
            <div className="sample-chips">
              {samples.map((s) => (
                <button key={s.id} className="sample-chip" onClick={() => handleSampleClick(s)}>
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

      {/* Multi-Method Comparison Grid */}
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

          <div className="hits-list">
            {results?.tfidf?.length > 0 ? (
              results.tfidf.map((hit) => (
                <article key={hit.rank} className="hit-card">
                  <div className="hit-card-top">
                    <span className="rank-pill">#{hit.rank}</span>
                    <span className="score-badge">score: {hit.score}</span>
                  </div>
                  <div className="hit-doc-id">
                    📄 {hit.page_title} (sent {hit.sent_id})
                    {isGoldHit(hit.doc_id) && <span className="badge-supports">⭐ GOLD</span>}
                  </div>
                  <p className="hit-text">{hit.text}</p>
                </article>
              ))
            ) : (
              <div className="empty-state">
                <span className="empty-icon">📊</span>
                <p className="empty-text">Enter a claim or select a sample above to view TF-IDF rankings.</p>
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

          <div className="hits-list">
            {results?.bm25?.length > 0 ? (
              results.bm25.map((hit) => (
                <article key={hit.rank} className="hit-card">
                  <div className="hit-card-top">
                    <span className="rank-pill">#{hit.rank}</span>
                    <span className="score-badge">score: {hit.score}</span>
                  </div>
                  <div className="hit-doc-id">
                    📄 {hit.page_title} (sent {hit.sent_id})
                    {isGoldHit(hit.doc_id) && <span className="badge-supports">⭐ GOLD</span>}
                  </div>
                  <p className="hit-text">{hit.text}</p>
                </article>
              ))
            ) : (
              <div className="empty-state">
                <span className="empty-icon">📈</span>
                <p className="empty-text">Enter a claim to view BM25 lexical ranking results.</p>
              </div>
            )}
          </div>
        </section>

        {/* Method 3: Dense Semantic (Pinecone) */}
        <section className="method-column glass-panel">
          <div className="column-header">
            <div>
              <h2 className="method-title">
                <span>Dense Semantic</span>
                <span className="method-tag">Pinecone</span>
              </h2>
              <p className="method-desc">Sentence-Transformers (all-MiniLM-L6-v2)</p>
            </div>
          </div>

          <div className="hits-list">
            {results?.dense?.length > 0 ? (
              results.dense.map((hit) => (
                <article key={hit.rank} className="hit-card">
                  <div className="hit-card-top">
                    <span className="rank-pill">#{hit.rank}</span>
                    <span className="score-badge">cosine: {hit.score}</span>
                  </div>
                  <div className="hit-doc-id">
                    📄 {hit.page_title} (sent {hit.sent_id})
                    {isGoldHit(hit.doc_id) && <span className="badge-supports">⭐ GOLD</span>}
                  </div>
                  <p className="hit-text">{hit.text}</p>
                </article>
              ))
            ) : (
              <div className="empty-state">
                <span className="empty-icon">🧠</span>
                <p className="empty-text">
                  {status?.dense?.ready
                    ? 'Enter a claim to query Pinecone vectors.'
                    : 'Dense runs on Pinecone. Use notebooks/fever_pinecone_colab.ipynb and enter key in Config above.'}
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
                <span className="method-tag" style={{ background: 'rgba(99, 102, 241, 0.25)', color: '#c7d2fe' }}>Fusion</span>
              </h2>
              <p className="method-desc">Reciprocal Rank Fusion across active systems</p>
            </div>
          </div>

          <div className="hits-list">
            {results?.rrf?.length > 0 ? (
              results.rrf.map((hit) => (
                <article key={hit.rank} className="hit-card">
                  <div className="hit-card-top">
                    <span className="rank-pill" style={{ color: '#a5b4fc', background: 'rgba(99, 102, 241, 0.2)' }}>
                      #{hit.rank}
                    </span>
                    <span className="score-badge">rrf: {hit.score}</span>
                  </div>
                  <div className="hit-doc-id">
                    📄 {hit.page_title} (sent {hit.sent_id})
                    {isGoldHit(hit.doc_id) && <span className="badge-supports">⭐ GOLD</span>}
                  </div>
                  <p className="hit-text">{hit.text}</p>
                </article>
              ))
            ) : (
              <div className="empty-state">
                <span className="empty-icon">⚡</span>
                <p className="empty-text">RRF merges rankings from all active retrievers above.</p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
