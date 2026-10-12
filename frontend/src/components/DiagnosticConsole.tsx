import React, { useState } from 'react';
import { DiagnosticRequest, DiagnosticResult, apiClient } from '../api/client';

export const DiagnosticConsole: React.FC = () => {
  const [tfVersion, setTfVersion] = useState('4.41.0');
  const [torchVersion, setTorchVersion] = useState('2.3.0');
  const [hasCuda, setHasCuda] = useState(true);
  const [deviceType, setDeviceType] = useState('cuda');
  const [vramGb, setVramGb] = useState(4.0);
  const [errorLog, setErrorLog] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [advisory, setAdvisory] = useState<string | null>(null);

  const handleRunDiagnostic = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setAdvisory(null);
    try {
      const req: DiagnosticRequest = {
        transformers_version: tfVersion,
        torch_version: torchVersion,
        has_cuda: hasCuda,
        device_type: deviceType,
        vram_gb: Number(vramGb),
        error_log: errorLog.trim() || undefined
      };
      const res = await apiClient.runDiagnostic(req);
      setResult(res);

      const adv = await apiClient.getAdvisory({
        context_type: 'diagnostic',
        context_data: res as unknown as Record<string, unknown>
      });
      setAdvisory(adv.advisory_text);
    } catch (err: any) {
      setError(err.message || 'Diagnostic execution failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Workflow 3: Runtime Compatibility & Error Diagnostic</h2>
      <p className="description">
        Scans runtime dependencies and execution traces for known breaking changes (e.g. GenerationMixin _is_stateful, DynamicCache indexing, ROCm DirectML incompatibilities).
      </p>

      <form onSubmit={handleRunDiagnostic} className="form-grid">
        <div className="form-group">
          <label>Transformers Version</label>
          <input
            type="text"
            value={tfVersion}
            onChange={(e) => setTfVersion(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>PyTorch Version</label>
          <input
            type="text"
            value={torchVersion}
            onChange={(e) => setTorchVersion(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>Execution Backend</label>
          <select value={deviceType} onChange={(e) => setDeviceType(e.target.value)}>
            <option value="cuda">NVIDIA CUDA</option>
            <option value="rocm">AMD ROCm</option>
            <option value="directml">Windows DirectML</option>
            <option value="cpu">CPU Fallback</option>
          </select>
        </div>

        <div className="form-group">
          <label>Target VRAM (GB)</label>
          <input
            type="number"
            step="0.5"
            value={vramGb}
            onChange={(e) => setVramGb(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>CUDA Enabled Flag</label>
          <select value={hasCuda ? 'true' : 'false'} onChange={(e) => setHasCuda(e.target.value === 'true')}>
            <option value="true">True (CUDA compiled)</option>
            <option value="false">False (CPU-only build)</option>
          </select>
        </div>

        <div className="form-group full-width">
          <label>Error Trace / Console Log (Optional)</label>
          <textarea
            rows={3}
            placeholder="Paste stack traces like AttributeError: 'AirLLMLlama2' has no attribute '_is_stateful' or 'LlamaAttention' object has no attribute 'rotary_emb'..."
            value={errorLog}
            onChange={(e) => setErrorLog(e.target.value)}
          />
        </div>

        <div className="form-actions">
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? 'Analyzing...' : 'Run Compatibility Diagnostic'}
          </button>
        </div>
      </form>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="results-container">
          <div className="metric-badges">
            <div className={`badge ${result.is_compatible ? 'badge-success' : 'badge-danger'}`}>
              Compatibility: {result.is_compatible ? 'PASS' : 'FAIL'}
            </div>
            <div className="badge badge-info">
              Health Score: {result.compatibility_score} / 100
            </div>
            <div className="badge badge-warning">
              Issues Flagged: {result.issues_found.length}
            </div>
          </div>

          {advisory && (
            <div className="advisory-box">
              <h4>Advisory Telemetry Insight</h4>
              <p>{advisory}</p>
            </div>
          )}

          <h3>Identified Issues & Resolution Playbook</h3>
          {result.issues_found.length === 0 ? (
            <p className="empty-message">No known breaking incompatibilities detected in environment.</p>
          ) : (
            <div className="issues-list">
              {result.issues_found.map((issue, idx) => (
                <div key={idx} className={`issue-card issue-${issue.severity}`}>
                  <div className="issue-header">
                    <span className="issue-severity">{issue.severity.toUpperCase()}</span>
                    <span className="issue-category">{issue.category}</span>
                  </div>
                  <h4>{issue.summary}</h4>
                  <p><strong>Root Cause:</strong> {issue.root_cause}</p>
                  <p className="fix-instruction"><strong>Recommended Fix:</strong> {issue.recommended_fix}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
