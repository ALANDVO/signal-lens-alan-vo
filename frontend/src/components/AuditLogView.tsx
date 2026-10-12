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

  useEffect(() => {
    fetchLogs();
  }, []);

  return (
    <div className="card">
      <div className="header-actions">
        <h2>System Audit Trail & Telemetry Records</h2>
        <button onClick={fetchLogs} disabled={loading} className="btn-secondary">
          {loading ? 'Refreshing...' : 'Refresh Logs'}
        </button>
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
