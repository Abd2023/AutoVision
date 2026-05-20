import { useState, useRef, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Upload, Sparkles, ImageIcon, Loader2, CheckCircle2, AlertCircle, Car, RotateCcw } from 'lucide-react';
import './App.css';

const CLASS_COLORS = {
  F1: '#ef4444',
  HATCHBACK: '#f97316',
  MICRO: '#eab308',
  PICK_UP: '#22c55e',
  SEDAN: '#06b6d4',
  STATION_WAGON: '#6366f1',
  SUV: '#a855f7',
  VAN: '#ec4899',
};

const CLASS_LABELS_TR = {
  F1: 'F1 (Açık Tekerlekli)',
  HATCHBACK: 'Hatchback',
  MICRO: 'Micro',
  PICK_UP: 'Pick-Up',
  SEDAN: 'Sedan',
  STATION_WAGON: 'Station Wagon',
  SUV: 'SUV',
  VAN: 'Van',
};

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // ─── Image Selection ───────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Lütfen bir görüntü dosyası seçin (PNG, JPG, WEBP).');
      return;
    }
    setError(null);
    setPredictions(null);
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer?.files?.[0];
    handleFile(file);
  }, [handleFile]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
  }, []);

  // ─── Prediction ────────────────────────────────────────
  const handlePredict = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (data.error) {
        setError(data.error);
      } else if (data.predictions) {
        setPredictions(data.predictions);
      } else {
        setError('Sunucudan beklenmeyen bir yanıt alındı.');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(`Sunucuya bağlanılamadı. Backend çalışıyor mu? (python main.py)`);
    } finally {
      setLoading(false);
    }
  };

  // ─── Reset ─────────────────────────────────────────────
  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setPredictions(null);
    setError(null);
    setLoading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ─── Chart data ────────────────────────────────────────
  const chartData = predictions
    ? Object.entries(predictions)
        .map(([name, value]) => ({
          name,
          label: CLASS_LABELS_TR[name] || name,
          probability: +(value * 100).toFixed(2),
          color: CLASS_COLORS[name] || '#6366f1',
        }))
        .sort((a, b) => b.probability - a.probability)
    : [];

  const topPrediction = chartData.length > 0 ? chartData[0] : null;

  // ─── Render ────────────────────────────────────────────
  return (
    <div className="app-wrapper">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-group">
            <div className="logo-icon">
              <Car size={28} />
            </div>
            <div>
              <h1 className="app-title">
                Auto<span className="gradient-text">Vision</span>
              </h1>
              <p className="app-subtitle">Araba Gövde Tipi Sınıflandırma</p>
            </div>
          </div>
          <div className="header-badge glass">
            <Sparkles size={14} />
            <span>AI-Powered</span>
          </div>
        </div>
      </header>

      <main className="app-main">
        {/* ── Upload Card ── */}
        <section className="upload-section animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
          <div className="section-header">
            <h2><ImageIcon size={20} /> Görüntü Yükle</h2>
            <p>Sınıflandırmak istediğiniz araç görüntüsünü sürükleyin veya seçin</p>
          </div>

          <div
            id="dropzone"
            className={`dropzone ${dragActive ? 'dropzone--active' : ''} ${imagePreview ? 'dropzone--has-image' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              id="file-input"
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => handleFile(e.target.files?.[0])}
            />

            {imagePreview ? (
              <div className="preview-container">
                <img src={imagePreview} alt="Yüklenen araç" className="preview-image" />
                <div className="preview-overlay">
                  <span>Değiştirmek için tıklayın</span>
                </div>
              </div>
            ) : (
              <div className="dropzone-placeholder">
                <div className="dropzone-icon">
                  <Upload size={32} />
                </div>
                <p className="dropzone-text">Görüntüyü buraya sürükleyin</p>
                <p className="dropzone-hint">veya dosya seçmek için tıklayın</p>
                <span className="dropzone-formats">PNG, JPG, WEBP</span>
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="action-buttons">
            <button
              id="predict-btn"
              className={`btn btn-primary ${loading ? 'btn--loading' : ''}`}
              onClick={handlePredict}
              disabled={!imageFile || loading}
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="spin" />
                  Analiz ediliyor…
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Tahmin Et
                </>
              )}
            </button>
            {imageFile && (
              <button id="reset-btn" className="btn btn-ghost" onClick={handleReset}>
                <RotateCcw size={16} />
                Sıfırla
              </button>
            )}
          </div>
        </section>

        {/* ── Error ── */}
        {error && (
          <div className="error-banner animate-fade-in">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* ── Results ── */}
        {predictions && (
          <section className="results-section animate-fade-in-up" style={{ animationDelay: '0.15s' }}>
            {/* Top prediction card */}
            <div className="top-prediction glass">
              <div className="top-prediction-badge" style={{ background: topPrediction?.color }}>
                <CheckCircle2 size={20} />
              </div>
              <div className="top-prediction-info">
                <span className="top-prediction-label">Tahmin Edilen Sınıf</span>
                <h2 className="top-prediction-class" style={{ color: topPrediction?.color }}>
                  {topPrediction?.label}
                </h2>
              </div>
              <div className="top-prediction-score">
                <span className="score-value">{topPrediction?.probability.toFixed(1)}%</span>
                <span className="score-label">Güven Skoru</span>
              </div>
            </div>

            {/* Chart */}
            <div className="chart-card glass">
              <h3 className="chart-title">Olasılık Dağılımı</h3>
              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                    <XAxis
                      type="number"
                      domain={[0, 100]}
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      tickFormatter={(v) => `${v}%`}
                      axisLine={{ stroke: 'rgba(148,163,184,0.2)' }}
                    />
                    <YAxis
                      type="category"
                      dataKey="label"
                      width={130}
                      tick={{ fill: '#e2e8f0', fontSize: 13, fontWeight: 500 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: '#1e293b',
                        border: '1px solid rgba(99,102,241,0.3)',
                        borderRadius: '12px',
                        color: '#f1f5f9',
                        fontSize: 13,
                      }}
                      formatter={(value) => [`${value.toFixed(2)}%`, 'Olasılık']}
                      cursor={{ fill: 'rgba(99,102,241,0.08)' }}
                    />
                    <Bar dataKey="probability" radius={[0, 6, 6, 0]} barSize={28}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.85} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Probabilities grid */}
            <div className="prob-grid">
              {chartData.map((item, i) => (
                <div key={item.name} className="prob-card glass" style={{ animationDelay: `${i * 0.05}s` }}>
                  <div className="prob-color-dot" style={{ background: item.color }} />
                  <div className="prob-info">
                    <span className="prob-name">{item.label}</span>
                    <div className="prob-bar-track">
                      <div
                        className="prob-bar-fill"
                        style={{ width: `${item.probability}%`, background: item.color }}
                      />
                    </div>
                  </div>
                  <span className="prob-value" style={{ color: item.color }}>
                    {item.probability.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <p>
          Kocaeli Üniversitesi — Yazılım Laboratuvarı II — Proje III
        </p>
      </footer>
    </div>
  );
}

export default App;
