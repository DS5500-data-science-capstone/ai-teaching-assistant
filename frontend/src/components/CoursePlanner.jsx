import { useState } from 'react';
import { Plus, Trash2, Loader2, Download, ChevronDown, ChevronUp } from 'lucide-react';

const API = '/api';

const DIFFICULTIES = ['Easy', 'Medium', 'Hard'];
const EMPTY_WEEK = (n) => ({ week: n, topic: '', difficulty: 'Medium', description: '' });

const DEFAULT_PLAN = Array.from({ length: 16 }, (_, i) => EMPTY_WEEK(i + 1));

export default function CoursePlanner() {
  const [plan, setPlan] = useState(DEFAULT_PLAN);
  const [selectedWeek, setSelectedWeek] = useState(null);
  const [slides, setSlides] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');
  const [expandedWeek, setExpandedWeek] = useState(null);

  const updateWeek = (i, field, val) => {
    const updated = [...plan];
    updated[i] = { ...updated[i], [field]: val };
    setPlan(updated);
  };

  const addWeek = () => setPlan([...plan, EMPTY_WEEK(plan.length + 1)]);
  const removeWeek = (i) => setPlan(plan.filter((_, idx) => idx !== i).map((w, idx) => ({ ...w, week: idx + 1 })));

  const handleGenerate = async (week) => {
    if (!week.topic.trim()) { setError('Please enter a topic for this week.'); return; }
    setError('');
    setGenerating(true);
    setSelectedWeek(week);
    setSlides(null);
    try {
      const res = await fetch(`${API}/generate-slides`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          week: week.week,
          topic: week.topic,
          difficulty: week.difficulty,
          description: week.description,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Generation failed');
      setSlides(data.slides);
    } catch (e) {
      setError(e.message);
    } finally {
      setGenerating(false);
    }
  };

  const handleDownload = async (fmt) => {
    if (!slides || !selectedWeek) return;
    setDownloading(true);
    try {
      const res = await fetch(`${API}/download-slides`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          week: selectedWeek.week,
          topic: selectedWeek.topic,
          difficulty: selectedWeek.difficulty,
          slides,
          format: fmt,
        }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const mime = fmt === 'pptx'
        ? 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        : 'application/pdf';
      const blob = new Blob([await res.arrayBuffer()], { type: mime });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `Week${selectedWeek.week}_${selectedWeek.topic.replace(/\s+/g, '_')}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      setError('Download failed: ' + e.message);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Course Plan Builder */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Course Plan</h2>
            <p className="text-sm text-gray-500 mt-1">CS 5200 — Database Management Systems • Prof. Cristiano</p>
          </div>
          <button onClick={addWeek}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-medium">
            <Plus className="w-4 h-4" /> Add Week
          </button>
        </div>

        <div className="space-y-2">
          {plan.map((w, i) => (
            <div key={i} className="border border-gray-200 rounded-lg overflow-hidden">
              {/* Week header */}
              <div
                className="flex items-center gap-3 p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
                onClick={() => setExpandedWeek(expandedWeek === i ? null : i)}
              >
                <span className="w-16 text-xs font-bold text-red-700 bg-red-50 px-2 py-1 rounded text-center">
                  Week {w.week}
                </span>
                <span className="flex-1 text-sm text-gray-800 font-medium truncate">
                  {w.topic || <span className="text-gray-400 italic">No topic set</span>}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                  w.difficulty === 'Easy' ? 'bg-green-100 text-green-700' :
                  w.difficulty === 'Hard' ? 'bg-red-100 text-red-700' :
                  'bg-yellow-100 text-yellow-700'
                }`}>{w.difficulty}</span>
                {expandedWeek === i ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
              </div>

              {/* Week editor */}
              {expandedWeek === i && (
                <div className="p-4 space-y-3 border-t border-gray-100">
                  <div className="grid grid-cols-3 gap-3">
                    <div className="col-span-2">
                      <label className="text-xs font-medium text-gray-600 mb-1 block">Topic</label>
                      <input
                        type="text"
                        value={w.topic}
                        onChange={e => updateWeek(i, 'topic', e.target.value)}
                        placeholder="e.g. Relational Model & Algebra"
                        className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1 block">Difficulty</label>
                      <select
                        value={w.difficulty}
                        onChange={e => updateWeek(i, 'difficulty', e.target.value)}
                        className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500"
                      >
                        {DIFFICULTIES.map(d => <option key={d}>{d}</option>)}
                      </select>
                    </div>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-600 mb-1 block">Description (optional)</label>
                    <input
                      type="text"
                      value={w.description}
                      onChange={e => updateWeek(i, 'description', e.target.value)}
                      placeholder="Brief description of what will be covered..."
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500"
                    />
                  </div>
                  <div className="flex justify-between items-center pt-1">
                    <button onClick={() => removeWeek(i)}
                      className="flex items-center gap-1 text-xs text-red-500 hover:text-red-700">
                      <Trash2 className="w-3 h-3" /> Remove week
                    </button>
                    <button
                      onClick={() => handleGenerate(w)}
                      disabled={generating || !w.topic.trim()}
                      className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-medium disabled:opacity-50"
                    >
                      {generating && selectedWeek?.week === w.week
                        ? <><Loader2 className="w-4 h-4 animate-spin" /> Generating...</>
                        : '🎯 Generate Slides'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">{error}</p>}

      {/* Slide Preview */}
      {slides && selectedWeek && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-bold text-gray-900">
                Week {selectedWeek.week} — {selectedWeek.topic}
              </h3>
              <p className="text-sm text-gray-500">{slides.length} slides generated</p>
            </div>
            <div className="flex gap-2">
              <button onClick={() => handleDownload('pptx')} disabled={downloading}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-medium disabled:opacity-50">
                <Download className="w-4 h-4" />
                {downloading ? 'Downloading...' : 'Download PPTX'}
              </button>
              <button onClick={() => handleDownload('pdf')} disabled={downloading}
                className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-800 text-sm font-medium disabled:opacity-50">
                <Download className="w-4 h-4" /> Download PDF
              </button>
            </div>
          </div>

          <div className="space-y-3">
            {slides.map((slide, i) => (
              <div key={i} className={`rounded-lg border p-4 ${
                slide.type === 'title' ? 'bg-red-700 text-white border-red-800' :
                slide.type === 'agenda' ? 'bg-red-50 border-red-200' :
                slide.type === 'references' ? 'bg-gray-50 border-gray-200' :
                'bg-white border-gray-200'
              }`}>
                <div className="flex items-start gap-3">
                  <span className={`text-xs font-bold px-2 py-0.5 rounded flex-shrink-0 ${
                    slide.type === 'title' ? 'bg-white text-red-700' :
                    'bg-red-100 text-red-700'
                  }`}>
                    {i + 1}
                  </span>
                  <div className="flex-1">
                    <p className={`font-semibold text-sm ${slide.type === 'title' ? 'text-white' : 'text-gray-900'}`}>
                      {slide.title}
                    </p>
                    {slide.bullets?.length > 0 && (
                      <ul className={`mt-2 space-y-1 ${slide.type === 'title' ? 'text-red-100' : 'text-gray-600'}`}>
                        {slide.bullets.map((b, j) => (
                          <li key={j} className="text-xs flex items-start gap-1">
                            <span className="mt-1 w-1.5 h-1.5 rounded-full bg-red-400 flex-shrink-0" />
                            {b}
                          </li>
                        ))}
                      </ul>
                    )}
                    {slide.definitions?.length > 0 && (
                      <div className="mt-2 space-y-1">
                        {slide.definitions.map((d, j) => (
                          <div key={j} className="text-xs bg-yellow-50 border border-yellow-200 rounded px-2 py-1">
                            <span className="font-semibold text-yellow-800">{d.term}:</span>{' '}
                            <span className="text-gray-700">{d.definition}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {slide.diagram_placeholder && (
                      <div className="mt-2 h-12 bg-gray-100 border-2 border-dashed border-gray-300 rounded flex items-center justify-center text-xs text-gray-400">
                        📊 {slide.diagram_placeholder}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}