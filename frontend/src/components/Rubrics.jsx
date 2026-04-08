import { useState } from 'react';
import { Plus, Trash2, Loader2, Download, Sparkles } from 'lucide-react';
import jsPDF from 'jspdf';

const API = '/api';

const TYPES   = ['Homework', 'Project', 'Quiz'];
const LEVELS  = ['Excellent', 'Good', 'Satisfactory', 'Needs Improvement'];
const LEVEL_COLORS = {
  'Excellent':          'bg-green-50 border-green-200',
  'Good':               'bg-blue-50 border-blue-200',
  'Satisfactory':       'bg-yellow-50 border-yellow-200',
  'Needs Improvement':  'bg-red-50 border-red-200',
};

const EMPTY_CRITERION = (id) => ({
  id,
  name:        '',
  description: '',
  points:      10,
  levels: LEVELS.map(l => ({ level: l, description: '' })),
});

export default function Rubrics() {
  const [title,       setTitle]       = useState('');
  const [type,        setType]        = useState('Homework');
  const [totalPoints, setTotalPoints] = useState(100);
  const [criteria,    setCriteria]    = useState([EMPTY_CRITERION(1)]);
  const [generating,  setGenerating]  = useState(null); // criterion id being generated
  const [error,       setError]       = useState('');

  const addCriterion = () =>
    setCriteria([...criteria, EMPTY_CRITERION(Date.now())]);

  const removeCriterion = id =>
    setCriteria(criteria.filter(c => c.id !== id));

  const updateCriterion = (id, field, val) =>
    setCriteria(criteria.map(c => c.id === id ? { ...c, [field]: val } : c));

  const updateLevel = (cid, level, val) =>
    setCriteria(criteria.map(c =>
      c.id === cid
        ? { ...c, levels: c.levels.map(l => l.level === level ? { ...l, description: val } : l) }
        : c
    ));

  const suggestCriteria = async (criterion) => {
    if (!criterion.name.trim()) {
      setError('Enter a criterion name first.');
      return;
    }
    setError('');
    setGenerating(criterion.id);
    try {
      const res  = await fetch(`${API}/suggest-rubric-criteria`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assignment_title: title || `${type} Assignment`,
          assignment_type:  type,
          criterion_name:   criterion.name,
          criterion_description: criterion.description,
          total_points:     criterion.points,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      setCriteria(criteria.map(c =>
        c.id === criterion.id
          ? { ...c, levels: data.levels }
          : c
      ));
    } catch (e) {
      setError(e.message);
    } finally {
      setGenerating(null);
    }
  };

  const downloadPDF = () => {
    const doc  = new jsPDF();
    const RED  = [204, 0, 0];
    const W    = doc.internal.pageSize.getWidth();
    let y      = 0;

    const checkPage = (needed = 10) => {
      if (y + needed > 280) { doc.addPage(); y = 15; }
    };

    // Header
    doc.setFillColor(...RED);
    doc.rect(0, 0, W, 28, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text(title || `${type} Rubric`, 14, 12);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`${type}  •  Total: ${totalPoints} pts  •  CS 5200 — Prof. Cristiano`, 14, 22);

    y = 36;

    criteria.forEach((c, ci) => {
      checkPage(30);

      // Criterion header
      doc.setFillColor(245, 245, 245);
      doc.rect(10, y, W - 20, 10, 'F');
      doc.setTextColor(...RED);
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text(`${ci + 1}. ${c.name}  (${c.points} pts)`, 14, y + 7);
      y += 14;

      if (c.description) {
        doc.setTextColor(80, 80, 80);
        doc.setFontSize(9);
        doc.setFont('helvetica', 'italic');
        doc.text(c.description, 14, y);
        y += 7;
      }

      // Level rows
      c.levels.forEach(l => {
        checkPage(16);
        doc.setTextColor(50, 50, 50);
        doc.setFontSize(10);
        doc.setFont('helvetica', 'bold');
        doc.text(`${l.level}:`, 14, y);
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9);
        const lines = doc.splitTextToSize(l.description || '—', W - 60);
        doc.text(lines, 50, y);
        y += lines.length * 5 + 4;
      });

      y += 4;
    });

    // Footer
    doc.setDrawColor(...RED);
    doc.setLineWidth(0.5);
    doc.line(10, 287, W - 10, 287);
    doc.setTextColor(150, 150, 150);
    doc.setFontSize(8);
    doc.text('Northeastern University  •  Khoury College of Computer Sciences', 14, 292);

    doc.save(`${(title || type).replace(/\s+/g, '_')}_rubric.pdf`);
  };

  const totalAllocated = criteria.reduce((s, c) => s + (parseInt(c.points) || 0), 0);

  return (
    <div className="space-y-6">
      {/* Header form */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">Rubric Builder</h2>

        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="col-span-2">
            <label className="text-sm font-medium text-gray-700 mb-1 block">Assignment Title</label>
            <input
              type="text" value={title} onChange={e => setTitle(e.target.value)}
              placeholder="e.g. Homework 2 — Relational Algebra"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-red-500"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700 mb-1 block">Type</label>
            <select value={type} onChange={e => setType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-red-500">
              {TYPES.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="w-40">
            <label className="text-sm font-medium text-gray-700 mb-1 block">Total Points</label>
            <input type="number" min={1} value={totalPoints}
              onChange={e => setTotalPoints(parseInt(e.target.value) || 100)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-red-500"
            />
          </div>
          <div className="flex-1 mt-5">
            <div className={`text-sm font-medium px-3 py-2 rounded-lg ${
              totalAllocated === totalPoints
                ? 'bg-green-50 text-green-700'
                : 'bg-yellow-50 text-yellow-700'
            }`}>
              Allocated: {totalAllocated} / {totalPoints} pts
              {totalAllocated !== totalPoints && ' — adjust criterion points to match total'}
            </div>
          </div>
        </div>
      </div>

      {/* Criteria */}
      {criteria.map((c, ci) => (
        <div key={c.id} className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-gray-900">Criterion {ci + 1}</h3>
            {criteria.length > 1 && (
              <button onClick={() => removeCriterion(c.id)}
                className="text-red-400 hover:text-red-600 p-1">
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>

          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="col-span-2">
              <label className="text-xs font-medium text-gray-600 mb-1 block">Criterion Name</label>
              <input type="text" value={c.name}
                onChange={e => updateCriterion(c.id, 'name', e.target.value)}
                placeholder="e.g. Correctness, Code Quality, Analysis..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-red-500"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-600 mb-1 block">Points</label>
              <input type="number" min={1} value={c.points}
                onChange={e => updateCriterion(c.id, 'points', parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-red-500"
              />
            </div>
          </div>

          <div className="mb-4">
            <label className="text-xs font-medium text-gray-600 mb-1 block">Description (optional)</label>
            <input type="text" value={c.description}
              onChange={e => updateCriterion(c.id, 'description', e.target.value)}
              placeholder="Brief description of what this criterion evaluates..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-red-500"
            />
          </div>

          {/* AI suggest button */}
          <div className="flex justify-end mb-4">
            <button
              onClick={() => suggestCriteria(c)}
              disabled={generating === c.id}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-medium disabled:opacity-50"
            >
              {generating === c.id
                ? <><Loader2 className="w-4 h-4 animate-spin" /> Suggesting...</>
                : <><Sparkles className="w-4 h-4" /> AI Suggest Levels</>}
            </button>
          </div>

          {/* Level descriptors */}
          <div className="grid grid-cols-2 gap-3">
            {c.levels.map(l => (
              <div key={l.level} className={`border rounded-lg p-3 ${LEVEL_COLORS[l.level]}`}>
                <p className="text-xs font-bold text-gray-700 mb-1">{l.level}</p>
                <textarea
                  value={l.description}
                  onChange={e => updateLevel(c.id, l.level, e.target.value)}
                  placeholder={`Describe ${l.level.toLowerCase()} performance...`}
                  rows={3}
                  className="w-full text-xs border-0 bg-transparent resize-none focus:outline-none text-gray-700 placeholder-gray-400"
                />
              </div>
            ))}
          </div>
        </div>
      ))}

      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">{error}</p>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button onClick={addCriterion}
          className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 text-sm font-medium">
          <Plus className="w-4 h-4" /> Add Criterion
        </button>
        <button onClick={downloadPDF}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-medium">
          <Download className="w-4 h-4" /> Download PDF
        </button>
      </div>
    </div>
  );
}