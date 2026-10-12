export interface ProfileRequest {
  model_name: string;
  parameter_count_billions: number;
  precision_bits: number;
  total_layers: number;
  context_length: number;
  target_vram_gb: number;
}

export interface LayerProfileItem {
  layer_index: number;
  name: string;
  param_size_mb: number;
  activation_size_mb: number;
  kv_cache_size_mb: number;
  cumulative_vram_mb: number;
}

export interface ProfileResult {
  id?: number;
  model_name: string;
  parameter_count_billions: number;
  precision_bits: number;
  total_layers: number;
  target_vram_gb: number;
  single_layer_vram_mb: number;
  full_model_vram_gb: number;
  fits_in_target_vram: boolean;
  vram_headroom_mb: number;
  layers: LayerProfileItem[];
  created_at?: string;
}

export interface ScheduleRequest {
  total_layers: number;
  single_layer_mb: number;
  available_vram_mb: number;
  pcie_bandwidth_gbps: number;
}

export interface ScheduleBatchChunk {
  chunk_index: number;
  start_layer: number;
  end_layer: number;
  layers_count: number;
  vram_usage_mb: number;
  transfer_time_ms: number;
}

export interface ScheduleResult {
  id?: number;
  optimal_layers_per_batch: number;
  total_batches: number;
  vram_utilization_pct: number;
  total_transfer_time_seconds: number;
  chunks: ScheduleBatchChunk[];
  speedup_vs_single_layer: number;
  created_at?: string;
}

export interface DiagnosticRequest {
  transformers_version: string;
  torch_version: string;
  has_cuda: boolean;
  device_type: string;
  vram_gb: number;
  error_log?: string;
}

export interface DiagnosticIssue {
  category: string;
  severity: string;
  summary: string;
  root_cause: string;
  recommended_fix: string;
}

export interface DiagnosticResult {
  id?: number;
  issues_found: DiagnosticIssue[];
  is_compatible: boolean;
  compatibility_score: number;
  created_at?: string;
}

export interface AdvisoryRequest {
  context_type: string;
  context_data: Record<string, unknown>;
  user_prompt?: string;
}

export interface AdvisoryResponse {
  advisory_text: string;
  provider: string;
  model: string;
  is_simulated: boolean;
  grounded_record_id?: string;
}

export interface AuditLogItem {
  id: number;
  timestamp: string;
  action: string;
  user_sub: string;
  details: string;
}

const API_BASE = '/api';

export const apiClient = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
  },

  async createProfile(req: ProfileRequest): Promise<ProfileResult> {
    const res = await fetch(`${API_BASE}/profiles`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    });
    if (!res.ok) throw new Error(`Failed to create profile: ${res.status}`);
    return res.json();
  },

  async listProfiles(): Promise<ProfileResult[]> {
    const res = await fetch(`${API_BASE}/profiles`);
    if (!res.ok) throw new Error(`Failed to list profiles: ${res.status}`);
    return res.json();
  },

  async createSchedule(req: ScheduleRequest): Promise<ScheduleResult> {
    const res = await fetch(`${API_BASE}/schedules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    });
    if (!res.ok) throw new Error(`Failed to create schedule: ${res.status}`);
    return res.json();
  },

  async listSchedules(): Promise<ScheduleResult[]> {
    const res = await fetch(`${API_BASE}/schedules`);
    if (!res.ok) throw new Error(`Failed to list schedules: ${res.status}`);
    return res.json();
  },

  async runDiagnostic(req: DiagnosticRequest): Promise<DiagnosticResult> {
    const res = await fetch(`${API_BASE}/diagnostics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    });
    if (!res.ok) throw new Error(`Failed to run diagnostic: ${res.status}`);
    return res.json();
  },

  async listDiagnostics(): Promise<DiagnosticResult[]> {
    const res = await fetch(`${API_BASE}/diagnostics`);
    if (!res.ok) throw new Error(`Failed to list diagnostics: ${res.status}`);
    return res.json();
  },

  async getAdvisory(req: AdvisoryRequest): Promise<AdvisoryResponse> {
    const res = await fetch(`${API_BASE}/advisory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    });
    if (!res.ok) throw new Error(`Failed to fetch advisory: ${res.status}`);
    return res.json();
  },

  async getAuditLogs(): Promise<AuditLogItem[]> {
    const res = await fetch(`${API_BASE}/audit-logs`);
    if (!res.ok) throw new Error(`Failed to fetch audit logs: ${res.status}`);
    return res.json();
  }
};
