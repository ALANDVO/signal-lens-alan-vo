import React, { useState } from 'react';
import { MemoryProfiler } from './components/MemoryProfiler';
import { LayerScheduler } from './components/LayerScheduler';
import { DiagnosticConsole } from './components/DiagnosticConsole';
import { AuditLogView } from './components/AuditLogView';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'profile' | 'schedule' | 'diagnostic' | 'audit'>('profile');

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <div className="brand">
            <span className="brand-logo">⚡</span>
            <div>
              <h1>SignalLens — AI & Machine Learning</h1>
              <p className="subtitle">AirLLM 70B Layer-Pipelined Profiler, Scheduler & Diagnostic Workbench</p>
            </div>
          </div>
          <div className="user-badge">
            <span className="dot dot-online"></span>
            <span>Local Demo Mode (Operator)</span>
          </div>
        </div>

        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'profile' ? 'active' : ''}`}
            onClick={() => setActiveTab('profile')}
          >
            Memory Profiler
          </button>
          <button
            className={`tab-btn ${activeTab === 'schedule' ? 'active' : ''}`}
            onClick={() => setActiveTab('schedule')}
          >
            Layer Scheduler
          </button>
          <button
            className={`tab-btn ${activeTab === 'diagnostic' ? 'active' : ''}`}
            onClick={() => setActiveTab('diagnostic')}
          >
            Runtime Diagnostics
          </button>
          <button
            className={`tab-btn ${activeTab === 'audit' ? 'active' : ''}`}
            onClick={() => setActiveTab('audit')}
          >
            Audit Trail
          </button>
        </nav>
      </header>

      <main className="main-content">
        {activeTab === 'profile' && <MemoryProfiler />}
        {activeTab === 'schedule' && <LayerScheduler />}
        {activeTab === 'diagnostic' && <DiagnosticConsole />}
        {activeTab === 'audit' && <AuditLogView />}
      </main>

      <footer className="app-footer">
        <p>SignalLens | Maintained by Alan Vo &lt;alanvo@gmail.com&gt;</p>
      </footer>
    </div>
  );
};

export default App;
