import { useState, useEffect, useRef } from 'react';
import { Brain, Download, Loader, ChevronDown } from 'lucide-react';
import jsPDF from 'jspdf';

const API = 'http://localhost:8000';
const DIFFICULTIES = ['easy', 'medium', 'hard'];
const STYLES = ['conceptual', 'scenario', 'definition'];
const QUESTION_TYPES = [
  { value: 'mcq',         label: 'Multiple Choice' },
  { value: 'fill_blank',  label: 'Fill in the Blank' },
  { value: 'long_answer', label: 'Long Answer' },
  { value: 'true_false',  label: 'True / False' },
];

function QuestionCard({ q, idx }) {
  return (
    <div className="p-4 border border-gray-200 rounded-lg">
      <div className="flex items-start gap-3">
        <span className="w-7 h-7 bg-purple-100 text-purple-700 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0">
          {idx + 1}
        </span>
        <div className="flex-1">
          <p className="font-medium text-gray-900 mb-3">{q.question || q.statement}</p>

          {/* MCQ */}
          {q.type === 'mcq' && q.options && (
            <div className="space-y-2">
              {q.options.map((opt, j) => (
                <div key={j} className="px-3 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-700">
                  {String.fromCharCode(65 + j)}. {opt}
                </div>
              ))}
            </div>
          )}

          {/* True/False */}
          {q.type === 'true_false' && (
            <div className="flex gap-3">
              <div className="px-6 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-700">True</div>
              <div className="px-6 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-700">False</div>
            </div>
          )}

          {/* Fill blank */}
          {q.type === 'fill_blank' && (
            <div className="px-3 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-400 italic">
              Answer: _______________
            </div>
          )}

          {/* Long answer */}
          {q.type === 'long_answer' && (
            <div className="px-3 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-400 italic h-16">
              Student answer here...
            </div>
          )}

          <div className="flex gap-2 mt-3">
            <span className="text-xs px-2 py-1 bg-purple-50 text-purple-700 rounded">{q.topic}</span>
            {q.marks && <span className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded">{q.marks} mark{q.marks > 1 ? 's' : ''}</span>}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Quiz() {
  const [topics, setTopics] = useState([]);
  const [form, setForm] = useState({
    topic: '', num_questions: 3, difficulty: 'medium',
    style: 'conceptual', question_type: 'mcq', num_options: 4, source_filter: '',
  });
  const [quiz, setQuiz] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/quiz-topics`).then(r => r.json()).then(d => setTopics(d.topics || [])).catch(() => {});
  }, []);

  useEffect(() => {
    const handleClick = (e) => { if (menuRef.current && !menuRef.current.contains(e.target)) setShowDownloadMenu(false); };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const handleGenerate = async () => {
    if (!form.topic.trim()) { setError('Please select or enter a topic.'); return; }
    setError(''); setLoading(true); setQuiz(null);
    try {
      const res = await fetch(`${API}/generate-quiz`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, source_filter: form.source_filter || null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Generation failed');
      setQuiz(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (type) => {
    if (!quiz) return;
    const doc = new jsPDF();
    const margin = 15;
    const pageW = doc.internal.pageSize.getWidth();
    const maxW = pageW - margin * 2;
    let y = 20;

    const addText = (text, size = 11, bold = false, color = [0, 0, 0]) => {
      doc.setFontSize(size);
      doc.setFont('helvetica', bold ? 'bold' : 'normal');
      doc.setTextColor(...color);
      const lines = doc.splitTextToSize(String(text), maxW);
      lines.forEach(line => {
        if (y > 275) { doc.addPage(); y = 20; }
        doc.text(line, margin, y);
        y += size * 0.5;
      });
      y += 2;
    };

    const addLine = () => {
      doc.setDrawColor(220, 220, 220);
      doc.line(margin, y, pageW - margin, y);
      y += 5;
    };

    const date = new Date().toISOString().split('T')[0];
    const topicSlug = form.topic.replace(/[^a-z0-9]/gi, '_').toLowerCase();

    // Header
    addText(form.topic, 15, true);
    addText(`${quiz.difficulty.charAt(0).toUpperCase() + quiz.difficulty.slice(1)} • ${quiz.style} • ${quiz.questions?.length} questions • ${date}`, 10, false, [120, 120, 120]);
    if (type === 'answers') addText('ANSWER KEY', 12, true, [100, 50, 180]);
    y += 4;

    quiz.questions?.forEach((q, i) => {
      addLine();
      addText(`Q${i + 1}. ${q.question || q.statement}`, 11, true);
      addText(`Topic: ${q.topic}${q.marks ? `  |  ${q.marks} mark${q.marks > 1 ? 's' : ''}` : ''}`, 9, false, [140, 140, 140]);
      y += 2;

      if (type === 'questions') {
        if (q.options) q.options.forEach((opt, j) => addText(`   ${String.fromCharCode(65 + j)}. ${opt}`, 10));
        if (q.type === 'true_false') { addText('   A. True', 10); addText('   B. False', 10); }
        if (q.type === 'fill_blank') addText('   Answer: _______________', 10);
        if (q.type === 'long_answer') { addText('   Answer:', 10); y += 20; }
      }

      if (type === 'answers') {
        if (q.options) {
          q.options.forEach((opt, j) => {
            const isCorrect = opt === q.answer;
            // green for correct, red for incorrect
            const color = isCorrect ? [34, 139, 34] : [180, 0, 0];
            addText(`   ${String.fromCharCode(65 + j)}. ${opt}${isCorrect ? '  ✓' : '  ✗'}`, 10, isCorrect, color);
          });
        }
        if (q.type === 'true_false') {
          ['True', 'False'].forEach(opt => {
            const isCorrect = opt === q.answer;
            addText(`   ${opt}${isCorrect ? '  ✓' : '  ✗'}`, 10, isCorrect, isCorrect ? [34, 139, 34] : [180, 0, 0]);
          });
        }
        if (q.type === 'fill_blank') addText(`   Answer: ${q.answer}`, 10, true, [34, 139, 34]);
        if (q.type === 'long_answer') {
          addText(`   Model Answer: ${q.model_answer}`, 10);
          if (q.key_points?.length) {
            addText('   Key points:', 10, true);
            q.key_points.forEach(pt => addText(`     • ${pt}`, 10));
          }
        }
      }

      y += 4;
    });

    doc.save(`${topicSlug}_${type}_${date}.pdf`);
    setShowDownloadMenu(false);
  };

  return (
    <div className="space-y-6">
      {/* Config */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Brain className="w-6 h-6 text-purple-500" /> Quiz Builder
        </h2>
        <div className="grid grid-cols-2 gap-4">
          <div className="col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">Topic</label>
            {topics.length > 0 ? (
              <select value={form.topic} onChange={e => setForm({ ...form, topic: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500">
                <option value="">Select a topic...</option>
                {topics.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            ) : (
              <input type="text" value={form.topic} onChange={e => setForm({ ...form, topic: e.target.value })}
                placeholder="e.g. Database indexing, SQL joins..."
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" />
            )}
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Question Type</label>
            <select value={form.question_type} onChange={e => setForm({ ...form, question_type: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500">
              {QUESTION_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Number of Questions</label>
            <input type="number" min={1} max={20} value={form.num_questions}
              onChange={e => setForm({ ...form, num_questions: parseInt(e.target.value) })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Difficulty</label>
            <select value={form.difficulty} onChange={e => setForm({ ...form, difficulty: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500">
              {DIFFICULTIES.map(d => <option key={d} value={d}>{d.charAt(0).toUpperCase() + d.slice(1)}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Style</label>
            <select value={form.style} onChange={e => setForm({ ...form, style: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500">
              {STYLES.map(s => <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>)}
            </select>
          </div>
          {form.question_type === 'mcq' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Number of Options</label>
              <select value={form.num_options} onChange={e => setForm({ ...form, num_options: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500">
                {[3, 4, 5].map(n => <option key={n} value={n}>{n} options</option>)}
              </select>
            </div>
          )}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Source Filter (optional)</label>
            <input type="text" value={form.source_filter} onChange={e => setForm({ ...form, source_filter: e.target.value })}
              placeholder="e.g. lectures.pdf"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" />
          </div>
        </div>
        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
        <button onClick={handleGenerate} disabled={loading}
          className="mt-6 w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-semibold flex items-center justify-center gap-2 disabled:opacity-50">
          {loading
            ? <><Loader className="w-5 h-5 animate-spin" /> Generating quiz...</>
            : <><Brain className="w-5 h-5" /> Generate Quiz</>}
        </button>
      </div>

      {/* Quiz display */}
      {quiz && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-bold text-gray-900">{form.topic}</h3>
              <p className="text-sm text-gray-500">
                {quiz.questions?.length} questions • {quiz.difficulty} • {quiz.style}
              </p>
            </div>

            {/* Download dropdown — questions only + answer key */}
            <div className="relative" ref={menuRef}>
              <button onClick={() => setShowDownloadMenu(!showDownloadMenu)}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center gap-2">
                <Download className="w-4 h-4" /> Download <ChevronDown className="w-4 h-4" />
              </button>
              {showDownloadMenu && (
                <div className="absolute right-0 mt-1 w-52 bg-white border border-gray-200 rounded-lg shadow-lg z-10">
                  <button onClick={() => handleDownload('questions')}
                    className="w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 border-b border-gray-100">
                    Questions only
                    <p className="text-xs text-gray-400 mt-0.5">No answers — for students</p>
                  </button>
                  <button onClick={() => handleDownload('answers')}
                    className="w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-gray-50">
                    Answer key
                    <p className="text-xs text-gray-400 mt-0.5">Green = correct, Red = wrong</p>
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-4">
            {quiz.questions?.map((q, i) => <QuestionCard key={q.id || i} q={q} idx={i} />)}
          </div>
        </div>
      )}
    </div>
  );
}