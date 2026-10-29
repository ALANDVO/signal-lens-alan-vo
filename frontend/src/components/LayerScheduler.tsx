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
        total_layers: layers,
        single_layer_mb: layerMb,
        available_vram_mb: availableVram,
        pcie_bandwidth_gbps: pcieBw
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

  const handleExport = (format: 'json' | 'csv') => {
    if (!result) return;
    const data = {
      ...result,
      advisory: advisory || undefined,
      request: {
        total_layers: layers,
        single_layer_mb: layerMb,
        available_vram_mb: availableVram,
        pcie_bandwidth_gbps: pcieBw
      }
    };
    if (format === 'json') {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `schedule-${layers}layers-${availableVram}mb-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else {
      const chunksCsv = result.chunks.map(c => `${c.chunk_index},${c.start_layer},${c.end_layer},${c.layers_count},${c.vram_usage_mb},${c.transfer_time_ms}`).join('\n');
      const csv = `Total_Layers,Single_Layer_MB,Available_VRAM_MB,PCIE_Bandwidth_GBPS,Optimal_Layers_Per_Batch,Total_Batches,VRAM_Utilization_Pct,Total_Transfer_Time_Seconds,Speedup_Vs_Single_Layer\n${layers},${layerMb},${availableVram},${pcieBw},${result.optimal_layers_per_batch},${result.total_batches},${result.vram_utilization_pct},${result.total_transfer_time_seconds},${result.speedup_vs_single_layer}\n\nChunk_Index,Start_Layer,End_Layer,Layers_Count,VRAM_Usage_MB,Transfer_Time_MS\n${chunksCsv}`;
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `schedule-${layers}layers-${availableVram}mb-${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
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
          <label>Total Transformer Layers</label>
          <input
            type="number"
            value={layers}
            onChange={(e) => setLayers(parseInt(e.target.value))}
            min={1}
            max={200}
            required
          />
        </div>

        <div className="form-group">
          <label>Single Layer VRAM (MB)</label>
          <input
            type="number"
            step="10"
            value={layerMb}
            onChange={(e) => setLayerMb(parseFloat(e.target.value))}
            min={50}
            required
          />
        </div>

        <div className="form-group">
          <label>Available GPU VRAM (MB)</label>
          <input
            type="number"
            step="50"
            value={availableVram}
            onChange={(e) => setAvailableVram(parseFloat(e.target.value))}
            min={500}
            required
          />
        </div>

        <div className="form-group">
          <label>PCIe Bandwidth (GB/s)</label>
          <input
            type="number"
            step="1"
            value={pcieBw}
            onChange={(e) => setPcieBw(parseFloat(e.target.value))}
            min={1}
            max={64}
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
            <div className="badge badge-info">
              Optimal Layers/Batch: {result.optimal_layers_per_batch}
            </div>
            <div className="badge badge-info">
              Total Batches: {result.total_batches}
            </div>
            <div className="badge badge-success">
              VRAM Utilization: {result.vram_utilization_pct}%
            </div>
            <div className="badge badge-warning">
              Est. Transfer Time: {result.total_transfer_time_seconds}s
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

          <h3>Batch Schedule (All {result.chunks.length} chunks)</h3>
          <div className="table-responsive">
            <table>
              <thead>
                <tr>
                  <th>Chunk</th>
                  <th>Start Layer</th>
                  <th>End Layer</th>
                  <th>Layers</th>
                  <th>VRAM (MB)</th>
                  <th>Transfer (ms)</th>
                </tr>
              </thead>
              <tbody>
                {result.chunks.map((c) => (
                  <tr key={c.chunk_index}>
                    <td>{c.chunk_index + 1}</td>
                    <td>{c.start_layer}</td>
                    <td>{c.end_layer}</td>
                    <td>{c.layers_count}</td>
                    <td>{c.vram_usage_mb}</td>
                    <td>{c.transfer_time_ms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="export-actions">
            <button type="button" onClick={() => handleExport('json')} className="btn-secondary">Export JSON</button>
            <button type="button" onClick={() => handleExport('csv')} className="btn-secondary">Export CSV</button>
          </div>
        </div>
      )}
    </div>
  );
};
