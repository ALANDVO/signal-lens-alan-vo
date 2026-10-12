import React, { useState } from 'react';
import { ProfileRequest, ProfileResult, apiClient } from '../api/client';

export const MemoryProfiler: React.FC = () => {
  const [modelName, setModelName] = useState('Llama-3-70B-bnb-4bit');
  const [paramsB, setParamsB] = useState(70.0);
  const [bits, setBits] = useState(4);
  const [layers, setLayers] = useState(80);
  const [contextLen, setContextLen] = useState(2048);
  const [targetVram, setTargetVram] = useState(4.0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProfileResult | null>(null);
  const [advisory, setAdvisory] = useState<string | null>(null);

  const handleSimulate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setAdvisory(null);
    try {
      const req: ProfileRequest = {
        model_name: modelName,
        parameter_count_billions: Number(paramsB),
        precision_bits: Number(bits),
        total_layers: Number(layers),
        context_length: Number(contextLen),
        target_vram_gb: Number(targetVram)
      };
      const res = await apiClient.createProfile(req);
      setResult(res);

      // Fetch grounded advisory
      const adv = await apiClient.getAdvisory({
        context_type: 'profile',
        context_data: res as unknown as Record<string, unknown>
      });
      setAdvisory(adv.advisory_text);
    } catch (err: any) {
      setError(err.message || 'Error executing simulation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Workflow 1: Layer-Pipelined Memory Profiler</h2>
      <p className="description">
        Simulates AirLLM layer-by-layer parameter loading and execution buffers for massive LLMs on low-VRAM target GPUs.
      </p>

      <form onSubmit={handleSimulate} className="form-grid">
        <div className="form-group">
          <label>Model Architecture / Name</label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>Parameters (Billions)</label>
          <input
            type="number"
            step="0.5"
            value={paramsB}
            onChange={(e) => setParamsB(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Quantization (Bits)</label>
          <select value={bits} onChange={(e) => setBits(parseInt(e.target.value))}>
            <option value={4}>4-bit (NF4 / AWQ / GPTQ)</option>
            <option value={8}>8-bit (int8 / FP8)</option>
            <option value={16}>16-bit (FP16 / BF16)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Total Layers</label>
          <input
            type="number"
            value={layers}
            onChange={(e) => setLayers(parseInt(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Context Length (Tokens)</label>
          <input
            type="number"
            value={contextLen}
            onChange={(e) => setContextLen(parseInt(e.target.value))}
            required
          />
        </div>

        <div className="form-group">
          <label>Target GPU VRAM (GB)</label>
          <input
            type="number"
            step="0.5"
            value={targetVram}
            onChange={(e) => setTargetVram(parseFloat(e.target.value))}
            required
          />
        </div>

        <div className="form-actions">
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? 'Profiling Pipeline...' : 'Run Memory Simulation'}
          </button>
        </div>
      </form>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="results-container">
          <div className="metric-badges">
            <div className={`badge ${result.fits_in_target_vram ? 'badge-success' : 'badge-danger'}`}>
              Status: {result.fits_in_target_vram ? 'FITS IN TARGET VRAM' : 'EXCEEDS VRAM'}
            </div>
            <div className="badge badge-info">
              Single Layer VRAM: {result.single_layer_vram_mb} MB
            </div>
            <div className="badge badge-info">
              Full Model VRAM: {result.full_model_vram_gb} GB
            </div>
            <div className="badge badge-info">
              VRAM Headroom: {result.vram_headroom_mb} MB
            </div>
          </div>

          {advisory && (
            <div className="advisory-box">
              <h4>Advisory Telemetry Insight</h4>
              <p>{advisory}</p>
            </div>
          )}

          <h3>Layer Breakdown (First 5 of {result.layers.length} layers)</h3>
          <div className="table-responsive">
            <table>
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Param Size (MB)</th>
                  <th>Activation (MB)</th>
                  <th>KV Cache (MB)</th>
                  <th>Total Active VRAM (MB)</th>
                </tr>
              </thead>
              <tbody>
                {result.layers.slice(0, 5).map((l) => (
                  <tr key={l.layer_index}>
                    <td>{l.name}</td>
                    <td>{l.param_size_mb}</td>
                    <td>{l.activation_size_mb}</td>
                    <td>{l.kv_cache_size_mb}</td>
                    <td>{l.cumulative_vram_mb}</td>
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
