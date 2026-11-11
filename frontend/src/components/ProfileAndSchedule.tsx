import React, { useState } from 'react';
import { ProfileAndScheduleRequest, ProfileAndScheduleResult, apiClient } from '../api/client';

export const ProfileAndSchedule: React.FC = () => {
  const [modelName, setModelName] = useState('Llama-3-70B-bnb-4bit');
  const [paramsB, setParamsB] = useState(70.0);
  const [bits, setBits] = useState(4);
  const [layers, setLayers] = useState(80);
  const [contextLen, setContextLen] = useState(2048);
  const [targetVram, setTargetVram] = useState(4.0);
  const [pcieBw, setPcieBw] = useState(12.0);
  const [availableVramMb, setAvailableVramMb] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProfileAndScheduleResult | null>(null);
  const [advisory, setAdvisory] = useState<string | null>(null);

  const handleRunCombined = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setAdvisory(null);
    try {
      const req: ProfileAndScheduleRequest = {
        model_name: modelName,
        parameter_count_billions: Number(paramsB),
        precision_bits: Number(bits),
        total_layers: Number(layers),
        context_length: Number(contextLen),
        target_vram_gb: Number(targetVram),
        pcie_bandwidth_gbps: Number(pcieBw),
        available_vram_mb: availableVramMb ? Number(availableVramMb) : undefined
      };
      const res = await apiClient.createProfileAndSchedule(req);
      setResult(res);

      // Fetch advisory for the combined result
      const adv = await apiClient.getAdvisory({
        context_type: 'profile',
        context_data: {
          model_name: res.profile.model_name,
          fits_in_target_vram: res.profile.fits_in_target_vram,
          single_layer_vram_mb: res.profile.single_layer_vram_mb,
          full_model_vram_gb: res.profile.full_model_vram_gb,
          vram_headroom_mb: res.profile.vram_headroom_mb,
          optimal_layers_per_batch: res.schedule.optimal_layers_per_batch,
          total_batches: res.schedule.total_batches,
          vram_utilization_pct: res.schedule.vram_utilization_pct,
          speedup_vs_single_layer: res.schedule.speedup_vs_single_layer
        }
      });
      setAdvisory(adv.advisory_text);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Error executing combined profiling and scheduling';
      setError(message);
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
        model_name: modelName,
        parameter_count_billions: paramsB,
        precision_bits: bits,
        total_layers: layers,
        context_length: contextLen,
        target_vram_gb: targetVram,
        pcie_bandwidth_gbps: pcieBw,
        available_vram_mb: availableVramMb ? Number(availableVramMb) : undefined
      }
    };
    if (format === 'json') {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `profile-schedule-${modelName.replace(/\s+/g, '-')}-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else {
      const profileCsv = `Profile_Parameter,Value\nModel,${result.profile.model_name}\nParameters_B,${result.profile.parameter_count_billions}\nPrecision_Bits,${result.profile.precision_bits}\nTotal_Layers,${result.profile.total_layers}\nTarget_VRAM_GB,${result.profile.target_vram_gb}\nSingle_Layer_VRAM_MB,${result.profile.single_layer_vram_mb}\nFull_Model_VRAM_GB,${result.profile.full_model_vram_gb}\nFits_In_Target_VRAM,${result.profile.fits_in_target_vram}\nVRAM_Headroom_MB,${result.profile.vram_headroom_mb}\n`;
      const scheduleCsv = `\nSchedule_Parameter,Value\nOptimal_Layers_Per_Batch,${result.schedule.optimal_layers_per_batch}\nTotal_Batches,${result.schedule.total_batches}\nVRAM_Utilization_Pct,${result.schedule.vram_utilization_pct}\nTotal_Transfer_Time_Sec,${result.schedule.total_transfer_time_seconds}\nSpeedup_vs_Single_Layer,${result.schedule.speedup_vs_single_layer}\n`;
      const chunksCsv = `\nChunk_Index,Start_Layer,End_Layer,Layers_Count,VRAM_Usage_MB,Transfer_Time_MS\n${result.schedule.chunks.map(c => `${c.chunk_index},${c.start_layer},${c.end_layer},${c.layers_count},${c.vram_usage_mb},${c.transfer_time_ms}`).join('\n')}`;
      const csv = profileCsv + scheduleCsv + chunksCsv;
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `profile-schedule-${modelName.replace(/\s+/g, '-')}-${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="card">
      <h2>Workflow 4: Combined Profile + Auto-Schedule</h2>
      <p className="description">
        One-click workflow that runs memory profiling and automatically computes the optimal layer batching schedule
        using the profiler's single-layer VRAM calculation. Maximizes GPU utilization while preventing OOM crashes.
      </p>

      <form onSubmit={handleRunCombined} className="form-grid">
        <div className="form-group">
          <label>Model Name</label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>Parameter Count (Billions)</label>
          <input
            type="number"
            step="0.5"
            value={paramsB}
            onChange={(e) => setParamsB(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Quantization (bits)</label>
          <select value={bits} onChange={(e) => setBits(Number(e.target.value))}>
            <option value={4}>4-bit (INT4)</option>
            <option value={8}>8-bit (INT8)</option>
            <option value={16}>16-bit (FP16/BF16)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Total Layers</label>
          <input
            type="number"
            step="1"
            value={layers}
            onChange={(e) => setLayers(Number(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Context Length</label>
          <input
            type="number"
            step="128"
            value={contextLen}
            onChange={(e) => setContextLen(Number(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Target VRAM (GB)</label>
          <input
            type="number"
            step="0.5"
            value={targetVram}
            onChange={(e) => setTargetVram(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>PCIe Bandwidth (GB/s)</label>
          <input
            type="number"
            step="1.0"
            value={pcieBw}
            onChange={(e) => setPcieBw(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Available VRAM (MB) - Optional</label>
          <input
            type="number"
            step="100"
            value={availableVramMb}
            onChange={(e) => setAvailableVramMb(e.target.value)}
            placeholder="Auto-calculated from target VRAM"
          />
          <small className="form-hint">Defaults to 95% of target VRAM in MB</small>
        </div>

        <div className="form-actions">
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? 'Profiling & Scheduling...' : 'Run Combined Profile + Auto-Schedule'}
          </button>
        </div>
      </form>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="results-container">
          <div className="metric-badges">
            <div className={`badge ${result.profile.fits_in_target_vram ? 'badge-success' : 'badge-danger'}`}>
              Profile: {result.profile.fits_in_target_vram ? 'FITS IN VRAM' : 'EXCEEDS VRAM'}
            </div>
            <div className="badge badge-info">
              Single Layer: {result.profile.single_layer_vram_mb.toFixed(1)} MB
            </div>
            <div className="badge badge-warning">
              Full Model: {result.profile.full_model_vram_gb.toFixed(2)} GB
            </div>
            <div className="badge badge-success">
              VRAM Headroom: {result.profile.vram_headroom_mb.toFixed(1)} MB
            </div>
            <div className="badge badge-primary">
              Optimal Batch: {result.schedule.optimal_layers_per_batch} layers
            </div>
            <div className="badge badge-info">
              Total Batches: {result.schedule.total_batches}
            </div>
            <div className="badge badge-warning">
              VRAM Utilization: {result.schedule.vram_utilization_pct.toFixed(1)}%
            </div>
            <div className="badge badge-success">
              Speedup: {result.schedule.speedup_vs_single_layer.toFixed(2)}x
            </div>
          </div>

          {advisory && (
            <div className="advisory-box">
              <h4>Advisory Telemetry Insight</h4>
              <p>{advisory}</p>
            </div>
          )}

          <div className="results-sections">
            <div className="result-section">
              <h3>Memory Profile Details</h3>
              <div className="profile-summary">
                <p><strong>Model:</strong> {result.profile.model_name}</p>
                <p><strong>Parameters:</strong> {result.profile.parameter_count_billions}B ({result.profile.precision_bits}-bit)</p>
                <p><strong>Layers:</strong> {result.profile.total_layers}</p>
                <p><strong>Context Length:</strong> {result.profile.total_layers}</p>
                <p><strong>Target VRAM:</strong> {result.profile.target_vram_gb} GB</p>
              </div>
            </div>

            <div className="result-section">
              <h3>Layer Schedule Optimization</h3>
              <table>
                <thead>
                  <tr>
                    <th>Chunk</th>
                    <th>Layers</th>
                    <th>Range</th>
                    <th>VRAM (MB)</th>
                    <th>Transfer (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.schedule.chunks.map((chunk) => (
                    <tr key={chunk.chunk_index}>
                      <td>#{chunk.chunk_index + 1}</td>
                      <td>{chunk.layers_count}</td>
                      <td>{chunk.start_layer}–{chunk.end_layer}</td>
                      <td>{chunk.vram_usage_mb.toFixed(1)}</td>
                      <td>{chunk.transfer_time_ms.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
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

export default ProfileAndSchedule;
