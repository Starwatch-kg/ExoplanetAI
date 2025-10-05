import React, { useState, useEffect } from 'react';
import { Play, Square, RefreshCw, TrendingUp, AlertCircle, CheckCircle, Clock, Cpu, HardDrive, Activity } from 'lucide-react';

interface DiscoveryStats {
  total_processed: number;
  total_candidates: number;
  high_confidence_candidates: number;
  last_check_time: string | null;
  is_running: boolean;
  confidence_threshold: number;
}

interface Candidate {
  target_name: string;
  tic_id: string | null;
  mission: string;
  confidence: number;
  predicted_class: string;
  period: number | null;
  depth: number | null;
  snr: number | null;
  discovery_time: string;
}

interface SystemHealth {
  status: string;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  error_rate: number;
  last_check: string;
}

interface DashboardData {
  realtime: {
    targets_processed: number;
    candidates_found: number;
    high_confidence: number;
    avg_processing_time: number;
    error_rate: number;
  };
  system_health: SystemHealth;
}

const AutoDiscoveryPage: React.FC = () => {
  const [stats, setStats] = useState<DiscoveryStats | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.85);
  const [checkInterval, setCheckInterval] = useState(6);

  useEffect(() => {
    fetchStats();
    fetchCandidates();
    fetchDashboard();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchStats();
      fetchCandidates();
      fetchDashboard();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/v1/auto-discovery/status');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  };

  const fetchCandidates = async () => {
    try {
      const response = await fetch('/api/v1/auto-discovery/candidates/top?limit=10');
      if (response.ok) {
        const data = await response.json();
        setCandidates(data);
      }
    } catch (err) {
      console.error('Error fetching candidates:', err);
    }
  };

  const fetchDashboard = async () => {
    try {
      const response = await fetch('/api/v1/monitoring/dashboard');
      if (response.ok) {
        const data = await response.json();
        setDashboard(data);
      }
    } catch (err) {
      console.error('Error fetching dashboard:', err);
    }
  };

  const startDiscovery = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/auto-discovery/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          confidence_threshold: confidenceThreshold,
          check_interval_hours: checkInterval,
          max_concurrent_tasks: 5
        })
      });
      
      if (response.ok) {
        await fetchStats();
      } else {
        const data = await response.json();
        setError(data.detail || 'Failed to start discovery');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  const stopDiscovery = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/auto-discovery/stop', {
        method: 'POST'
      });
      
      if (response.ok) {
        await fetchStats();
      } else {
        const data = await response.json();
        setError(data.detail || 'Failed to stop discovery');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'degraded': return 'text-yellow-500';
      case 'unhealthy': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getClassColor = (predictedClass: string) => {
    switch (predictedClass) {
      case 'Confirmed': return 'text-green-500';
      case 'Candidate': return 'text-blue-500';
      case 'False Positive': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">🤖 Automated Discovery System</h1>
          <p className="text-gray-300">Continuous monitoring for new exoplanet candidates</p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 bg-red-500/20 border border-red-500 rounded-lg p-4 flex items-center gap-3">
            <AlertCircle className="text-red-500" size={24} />
            <span className="text-red-300">{error}</span>
          </div>
        )}

        {/* Control Panel */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6 border border-white/20">
          <h2 className="text-2xl font-bold text-white mb-4">Control Panel</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Confidence Threshold
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
                disabled={stats?.is_running}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Check Interval (hours)
              </label>
              <input
                type="number"
                min="1"
                max="24"
                value={checkInterval}
                onChange={(e) => setCheckInterval(parseInt(e.target.value))}
                className="w-full px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
                disabled={stats?.is_running}
              />
            </div>
            
            <div className="flex items-end gap-2">
              {!stats?.is_running ? (
                <button
                  onClick={startDiscovery}
                  disabled={loading}
                  className="flex-1 bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                >
                  <Play size={20} />
                  Start Discovery
                </button>
              ) : (
                <button
                  onClick={stopDiscovery}
                  disabled={loading}
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                >
                  <Square size={20} />
                  Stop Discovery
                </button>
              )}
              
              <button
                onClick={() => {
                  fetchStats();
                  fetchCandidates();
                  fetchDashboard();
                }}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
              >
                <RefreshCw size={20} />
              </button>
            </div>
          </div>

          {/* Status */}
          {stats && (
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                {stats.is_running ? (
                  <CheckCircle className="text-green-500" size={20} />
                ) : (
                  <AlertCircle className="text-gray-500" size={20} />
                )}
                <span className="text-gray-300">
                  Status: {stats.is_running ? 'Running' : 'Stopped'}
                </span>
              </div>
              
              {stats.last_check_time && (
                <div className="flex items-center gap-2">
                  <Clock className="text-blue-500" size={20} />
                  <span className="text-gray-300">
                    Last check: {new Date(stats.last_check_time).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Statistics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 backdrop-blur-md rounded-xl p-6 border border-blue-500/30">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-blue-300">Total Processed</h3>
              <TrendingUp className="text-blue-400" size={20} />
            </div>
            <p className="text-3xl font-bold text-white">{stats?.total_processed || 0}</p>
          </div>

          <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 backdrop-blur-md rounded-xl p-6 border border-green-500/30">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-green-300">Total Candidates</h3>
              <Activity className="text-green-400" size={20} />
            </div>
            <p className="text-3xl font-bold text-white">{stats?.total_candidates || 0}</p>
          </div>

          <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/20 backdrop-blur-md rounded-xl p-6 border border-purple-500/30">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-purple-300">High Confidence</h3>
              <CheckCircle className="text-purple-400" size={20} />
            </div>
            <p className="text-3xl font-bold text-white">{stats?.high_confidence_candidates || 0}</p>
          </div>

          <div className="bg-gradient-to-br from-orange-500/20 to-orange-600/20 backdrop-blur-md rounded-xl p-6 border border-orange-500/30">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-orange-300">Threshold</h3>
              <TrendingUp className="text-orange-400" size={20} />
            </div>
            <p className="text-3xl font-bold text-white">{((stats?.confidence_threshold || 0) * 100).toFixed(0)}%</p>
          </div>
        </div>

        {/* System Health */}
        {dashboard?.system_health && (
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6 border border-white/20">
            <h2 className="text-2xl font-bold text-white mb-4">System Health</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Status</span>
                  <span className={`text-sm font-bold ${getHealthColor(dashboard.system_health.status)}`}>
                    {dashboard.system_health.status.toUpperCase()}
                  </span>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">CPU</span>
                  <Cpu className="text-blue-400" size={16} />
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: `${dashboard.system_health.cpu_usage}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400">{dashboard.system_health.cpu_usage.toFixed(1)}%</span>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Memory</span>
                  <Activity className="text-green-400" size={16} />
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all"
                    style={{ width: `${dashboard.system_health.memory_usage}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400">{dashboard.system_health.memory_usage.toFixed(1)}%</span>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">Disk</span>
                  <HardDrive className="text-purple-400" size={16} />
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full transition-all"
                    style={{ width: `${dashboard.system_health.disk_usage}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400">{dashboard.system_health.disk_usage.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Recent Candidates */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
          <h2 className="text-2xl font-bold text-white mb-4">Top Candidates</h2>
          
          {candidates.length === 0 ? (
            <p className="text-gray-400 text-center py-8">No candidates found yet</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/20">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Target</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Mission</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Class</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Confidence</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Period</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">SNR</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Discovered</th>
                  </tr>
                </thead>
                <tbody>
                  {candidates.map((candidate, idx) => (
                    <tr key={idx} className="border-b border-white/10 hover:bg-white/5 transition-colors">
                      <td className="py-3 px-4 text-white font-medium">{candidate.target_name}</td>
                      <td className="py-3 px-4 text-gray-300">{candidate.mission}</td>
                      <td className={`py-3 px-4 font-medium ${getClassColor(candidate.predicted_class)}`}>
                        {candidate.predicted_class}
                      </td>
                      <td className="py-3 px-4 text-white">
                        {(candidate.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-gray-300">
                        {candidate.period ? `${candidate.period.toFixed(2)}d` : 'N/A'}
                      </td>
                      <td className="py-3 px-4 text-gray-300">
                        {candidate.snr ? candidate.snr.toFixed(1) : 'N/A'}
                      </td>
                      <td className="py-3 px-4 text-gray-400 text-sm">
                        {new Date(candidate.discovery_time).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AutoDiscoveryPage;
