import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';
import { apiClient } from './api/client';

describe('SignalLens Frontend Integration Tests', () => {
  it('renders application header and tab navigation', () => {
    render(<App />);
    expect(screen.getByText(/SignalLens — AI & Machine Learning/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Memory Profiler/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Layer Scheduler/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Runtime Diagnostics/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Audit Trail/i })).toBeInTheDocument();
  });

  it('runs Memory Profiler simulation workflow and displays results', async () => {
    const mockProfileResult = {
      id: 1,
      model_name: 'Llama-3-70B-bnb-4bit',
      parameter_count_billions: 70.0,
      precision_bits: 4,
      total_layers: 80,
      target_vram_gb: 4.0,
      single_layer_vram_mb: 432.5,
      full_model_vram_gb: 32.5,
      fits_in_target_vram: true,
      vram_headroom_mb: 3663.5,
      layers: [
        {
          layer_index: 0,
          name: 'model.layers.0',
          param_size_mb: 393.75,
          activation_size_mb: 32.0,
          kv_cache_size_mb: 8.0,
          cumulative_vram_mb: 432.5
        }
      ]
    };

    vi.spyOn(apiClient, 'createProfile').mockResolvedValueOnce(mockProfileResult);
    vi.spyOn(apiClient, 'getAdvisory').mockResolvedValueOnce({
      advisory_text: 'Simulated advisory for Llama-3-70B',
      provider: 'openai-compatible',
      model: 'gpt-4o-mini',
      is_simulated: true
    });

    render(<App />);
    const submitBtn = screen.getByRole('button', { name: /Run Memory Simulation/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/FITS IN TARGET VRAM/i)).toBeInTheDocument();
      expect(screen.getByText(/Single Layer VRAM: 432.5 MB/i)).toBeInTheDocument();
      expect(screen.getByText(/Simulated advisory for Llama-3-70B/i)).toBeInTheDocument();
    });
  });

  it('switches to Layer Scheduler and executes batch optimization', async () => {
    const mockSchedule = {
      id: 1,
      optimal_layers_per_batch: 7,
      total_batches: 12,
      vram_utilization_pct: 82.89,
      total_transfer_time_seconds: 23.4,
      speedup_vs_single_layer: 1.33,
      chunks: [
        {
          chunk_index: 0,
          start_layer: 0,
          end_layer: 6,
          layers_count: 7,
          vram_usage_mb: 3150.0,
          transfer_time_ms: 2052.0
        }
      ]
    };

    vi.spyOn(apiClient, 'createSchedule').mockResolvedValueOnce(mockSchedule);
    vi.spyOn(apiClient, 'getAdvisory').mockResolvedValueOnce({
      advisory_text: 'Batch 7 layers for optimal PCIe throughput',
      provider: 'openai-compatible',
      model: 'gpt-4o-mini',
      is_simulated: true
    });

    render(<App />);
    const schedTab = screen.getByRole('button', { name: /Layer Scheduler/i });
    fireEvent.click(schedTab);

    const submitBtn = screen.getByRole('button', { name: /Compute Optimal Batch Schedule/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/Optimal Layers.*Batch: 7/i)).toBeInTheDocument();
      expect(screen.getByText(/Theoretical Speedup: 1.33x/i)).toBeInTheDocument();
    });
  });
});
