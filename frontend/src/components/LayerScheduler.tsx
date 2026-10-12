import React, { useState } from 'react';
import { ScheduleRequest, ScheduleResult, apiClient } from '../api/client';

export const LayerScheduler: React.FC = () => {
  const [layers, setLayers] = useState(80);
  const [layerMb, setLayerMb] = useState(450.0);
  const [availableVram, setAvailableVram] = useState(3800.0);
  const [pcieBw, setPcieBw] = useState(12.0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScheduleResult | null>(null);
  const [advisory, setAdvisory] = useState<string | null>(null);

  const handleOptimize = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setAdvisory(null);
    try {
      const req: ScheduleRequest = {
        total_layers: Number(layers),
        single_layer_mb: Number(layerMb),
        available_vram_mb: Number(availableVram),
        pcie_bandwidth_gbps: Number(pcieBw)
      };
      const res = await apiClient.createSchedule(req);
      setResult(res);

      const adv = await apiClient.getAdvisory({
        context_type: 'schedule',
        context_data: res as unknown as Record<string, unknown>
      });
      setAdvisory(adv.advisory_text);
    } catch (err: any) {
      setError(err.message || 'Error scheduling batches');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Workflow 2: Layer Batching Schedule Optimizer</h2>
      <p className="description">
        Solves the VRAM under-utilization problem by calculating the optimal multi-layer chunk batch size to maximize GPU memory and minimize PCIe transfer latency.
      </p>

      <form onSubmit={handleOptimize} className="form-grid">
        <div className="form-group">
          <label>Total Model Layers</label>
          <input
            type="number"
            value={layers}
            onChange={(e) => setLayers(parseInt(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Single Layer Weight Size (MB)</label>
          <input
            type="number"
            step="10"
            value={layerMb}
            onChange={(e) => setLayerMb(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Available GPU VRAM (MB)</label>
          <input
            type="number"
            step="100"
            value={availableVram}
            onChange={(e) => setAvailableVram(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>PCIe Bus Bandwidth (GB/s)</label>
          <input
            type="number"
            step="1"
            value={pcieBw}
            onChange={(e) => setPcieBw(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-actions">
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? 'Optimizing Schedule...' : 'Compute Optimal Batch Schedule'}
          </button>
        </div>
      </form>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="results-container">
          <div className="metric-badges">
            <div className="badge badge-success">
              Optimal Layers / Batch: {result.optimal_layers_per_batch}
            </div>
            <div className="badge badge-info">
              Total Batches: {result.total_batches}
            </div>
            <div className="badge badge-info">
              VRAM Saturation: {result.vram_utilization_pct}%
            </div>
            <div className="badge badge-success">
              Theoretical Speedup: {result.speedup_vs_single_layer}x
            </div>
          </div>

          {advisory && (
            <div className="advisory-box">
              <h4>Advisory Telemetry Insight</h4>
              <p>{advisory}</p>
            </div>
          )}

          <h3>Execution Batch Chunks</h3>
          <div className="table-responsive">
            <table>
              <thead>
                <tr>
                  <th>Batch #</th>
                  <th>Layer Range</th>
                  <th>Layer Count</th>
                  <th>Batch VRAM (MB)</th>
                  <th>Transfer Time (ms)</th>
                </tr>
              </thead>
              <tbody>
                {result.chunks.map((c) => (
                  <tr key={c.chunk_index}>
                    <td>Batch {c.chunk_index + 1}</td>
                    <td>Layer {c.start_layer} → {c.end_layer}</td>
                    <td>{c.layers_count}</td>
                    <td>{c.vram_usage_mb} MB</td>
                    <td>{c.transfer_time_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
