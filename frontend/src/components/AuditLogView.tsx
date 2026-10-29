import React, { useState, useEffect } from 'react';
import { AuditLogItem, apiClient } from '../api/client';

export const AuditLogView: React.FC = () => {
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getAuditLogs();
      setLogs(data);
    } catch (err: any) {
      setError(err.message || 'Failed to retrieve audit log trail');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = (format: 'json' | 'csv') => {
    if (logs.length === 0) return;
    if (format === 'json') {
      const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audit-log-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else {
      const csv = `ID,Timestamp,Action,Operator_Sub,Details\n${logs.map(l => `${l.id},${new Date(l.timestamp).toISOString()},${l.action},${l.user_sub},"${l.details}"`).join('\n')}`;
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audit-log-${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  return (
    <div className="card">
      <div className="header-actions">
        <h2>System Audit Trail & Telemetry Records</h2>
        <div className="header-buttons">
          <button onClick={fetchLogs} disabled={loading} className="btn-secondary">
            {loading ? 'Refreshing...' : 'Refresh Logs'}
          </button>
          <button onClick={() => handleExport('json')} className="btn-secondary" disabled={logs.length === 0}>Export JSON</button>
          <button onClick={() => handleExport('csv')} className="btn-secondary" disabled={logs.length === 0}>Export CSV</button>
        </div>
      </div>
      <p className="description">
        Immutable record of all profiling simulations, schedule optimizations, and diagnostics executed against the SignalLens backend.
      </p>

      {error && <div className="alert alert-error">{error}</div>}

      {logs.length === 0 && !loading ? (
        <p className="empty-message">No audit entries recorded yet.</p>
      ) : (
        <div className="table-responsive">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Timestamp (UTC)</th>
                <th>Action</th>
                <th>Operator Sub</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={log.id}>
                  <td>#{log.id}</td>
                  <td>{new Date(log.timestamp).toLocaleString()}</td>
                  <td><span className="badge badge-info">{log.action}</span></td>
                  <td>{log.user_sub}</td>
                  <td>{log.details}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
