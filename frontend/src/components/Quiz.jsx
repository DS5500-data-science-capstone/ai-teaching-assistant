import { useState, useEffect, useRef } from 'react';
import { Brain, Download, Loader, ChevronDown, Plus, Trash2 } from 'lucide-react';
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

// Default marks per question type
const DEFAULT_MARKS = { mcq: 1, true_false: 1, fill_blank: 2, long_answer: 5 };

function QuestionCard({ q, idx }) {
  return (
    <div className="p-4 border border-gray-200 rounded-lg">
      <div className="flex items-start gap-3">
        <span className="w-7 h-7 bg-purple-100 text-purple-700 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0">
          {idx + 1}
        </span>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-600 rounded capitalize">{q.type?.replace('_', ' ')}</span>
            <span className="text-xs px-2 py-0.5 bg-purple-50 text-purple-700 rounded">{q.topic}</span>
            <span className="text-xs px-2 py-0.5 bg-yellow-50 text-yellow-700 rounded font-medium">{q.marks} pt{q.marks !== 1 ? 's' : ''}</span>
          </div>
          <p className="font-medium text-gray-900 mb-3">{q.question || q.statement}</p>

          {q.type === 'mcq' && q.options && (
            <div className="space-y-2">
              {q.options.map((opt, j) => (
                <div key={j} className="px-3 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-700">
                  {String.fromCharCode(65 + j)}. {opt}
                </div>
              ))}
            </div>
          )}
          {q.type === 'true_false' && (
            <div className="flex gap-3">
              <div className="px-6 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-700">True</div>
              <div className="px-6 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-700">False</div>
            </div>
          )}
          {q.type === 'fill_blank' && (
            <div className="px-3 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-400 italic">Answer: _______________</div>
          )}
          {q.type === 'long_answer' && (
            <div className="px-3 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-400 italic h-16">Student answer here...</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Quiz() {
  const [availableTopics, setAvailableTopics] = useState([]);
  const [topics, setTopics] = useState([{ topic: '', weight: 100 }]);
  const [questionTypes, setQuestionTypes] = useState(['mcq', 'true_false']);
  const [totalQuestions, setTotalQuestions] = useState(5);
  const [marksPerType, setMarksPerType] = useState({ ...DEFAULT_MARKS });
  const [form, setForm] = useState({ difficulty: 'medium', style: 'conceptual', num_options: 4, source_filter: '' });
  const [quiz, setQuiz] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/quiz-topics`).then(r => r.json()).then(d => setAvailableTopics(d.topics || [])).catch(() => {});
  }, []);

  useEffect(() => {
    const handleClick = (e) => { if (menuRef.current && !menuRef.current.contains(e.target)) setShowDownloadMenu(false); };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const totalWeight = topics.reduce((sum, t) => sum + (parseInt(t.weight) || 0), 0);

  const addTopic = () => setTopics([...topics, { topic: '', weight: Math.max(100 - totalWeight, 0) }]);
  const removeTopic = (i) => setTopics(topics.filter((_, idx) => idx !== i));
  const updateTopic = (i, field, value) => {
    const updated = [...topics];
    updated[i] = { ...updated[i], [field]: value };
    setTopics(updated);
  };
  const toggleType = (type) => setQuestionTypes(prev => prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]);

  // Compute total possible marks based on distribution
  const computeTotalMarks = (questions) => questions.reduce((sum, q) => sum + (q.marks || 0), 0);

  const handleGenerate = async () => {
    if (topics.some(t => !t.topic.trim())) { setError('Please select a topic for each row.'); return; }
    if (questionTypes.length === 0) { setError('Please select at least one question type.'); return; }
    if (totalWeight !== 100) { setError(`Total weight must equal 100%. Currently: ${totalWeight}%`); return; }
    setError(''); setLoading(true); setQuiz(null);

    try {
      const combinations = topics.flatMap(t => questionTypes.map(qt => ({ topic: t, qtype: qt })));
      const perCombo = Math.max(1, Math.floor(totalQuestions / combinations.length));
      const remainder = totalQuestions - perCombo * combinations.length;

      const allQuestions = [];
      for (let i = 0; i < combinations.length; i++) {
        const { topic: topicCfg, qtype } = combinations[i];
        const numQ = perCombo + (i < remainder ? 1 : 0);
        const res = await fetch(`${API}/generate-quiz`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic: topicCfg.topic,
            num_questions: numQ,
            difficulty: form.difficulty,
            style: form.style,
            question_type: qtype,
            num_options: form.num_options,
            source_filter: form.source_filter || null,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Generation failed');
        const qs = (data.questions || []).map(q => ({
          ...q,
          topic_weight: topicCfg.weight,
          marks: marksPerType[qtype] ?? 1,
        }));
        allQuestions.push(...qs);
      }

      const totalMarks = computeTotalMarks(allQuestions);
      setQuiz({
        quiz_id: `quiz_${Date.now()}`,
        difficulty: form.difficulty,
        style: form.style,
        questions: allQuestions,
        total_marks: totalMarks,
      });
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
      doc.setFontSize(size); doc.setFont('helvetica', bold ? 'bold' : 'normal'); doc.setTextColor(...color);
      const lines = doc.splitTextToSize(String(text), maxW);
      lines.forEach(line => { if (y > 275) { doc.addPage(); y = 20; } doc.text(line, margin, y); y += size * 0.5; });
      y += 2;
    };
    const addLine = () => { doc.setDrawColor(220,220,220); doc.line(margin, y, pageW-margin, y); y += 5; };

    const date = new Date().toISOString().split('T')[0];
    const topicSlug = topics.map(t => t.topic).join('_').replace(/[^a-z0-9]/gi, '_').toLowerCase().slice(0, 40);

    addText(topics.map(t => t.topic).join(', '), 15, true);
    addText(`${quiz.difficulty} • ${quiz.style} • ${quiz.questions?.length} questions • Total: ${quiz.total_marks} marks • ${date}`, 10, false, [120,120,120]);
    if (type === 'answers') addText('ANSWER KEY', 12, true, [100,50,180]);
    y += 4;

    quiz.questions?.forEach((q, i) => {
      addLine();
      addText(`Q${i+1}. [${(q.type||'').replace('_',' ').toUpperCase()}] ${q.question || q.statement}`, 11, true);
      addText(`Topic: ${q.topic}  |  Weight: ${q.topic_weight}%  |  ${q.marks} pt${q.marks !== 1 ? 's' : ''}`, 9, false, [140,140,140]);
      y += 2;

      if (type === 'questions') {
        if (q.options) q.options.forEach((opt, j) => addText(`   ${String.fromCharCode(65+j)}. ${opt}`, 10));
        if (q.type === 'true_false') { addText('   A. True', 10); addText('   B. False', 10); }
        if (q.type === 'fill_blank') addText('   Answer: _______________', 10);
        if (q.type === 'long_answer') { addText('   Answer:', 10); y += 20; }
      }
      if (type === 'answers') {
        if (q.options) q.options.forEach((opt, j) => {
          const ok = opt === q.answer;
          addText(`   ${String.fromCharCode(65+j)}. ${opt}${ok?'  ✓':'  ✗'}`, 10, ok, ok?[34,139,34]:[180,0,0]);
        });
        if (q.type === 'true_false') ['True','False'].forEach(opt => {
          const ok = opt === q.answer;
          addText(`   ${opt}${ok?'  ✓':'  ✗'}`, 10, ok, ok?[34,139,34]:[180,0,0]);
        });
        if (q.type === 'fill_blank') addText(`   Answer: ${q.answer}`, 10, true, [34,139,34]);
        if (q.type === 'long_answer') { addText(`   Model Answer: ${q.model_answer}`, 10); if (q.key_points?.length) q.key_points.forEach(pt => addText(`     • ${pt}`, 10)); }
      }
      y += 4;
    });

    doc.save(`${topicSlug}_${type}_${date}.pdf`);
    setShowDownloadMenu(false);
  };

  const totalPossibleMarks = quiz?.total_marks || 0;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Brain className="w-6 h-6 text-purple-500" /> Quiz Builder
        </h2>

        {/* Total questions */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-1">Total Number of Questions</label>
          <input type="number" min={1} max={50} value={totalQuestions}
            onChange={e => setTotalQuestions(parseInt(e.target.value) || 1)}
            className="w-40 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" />
          <p className="text-xs text-gray-400 mt-1">
            ~{Math.floor(totalQuestions / Math.max(1, topics.length * questionTypes.length))} per topic × type combination
          </p>
        </div>

        {/* Topics */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-gray-700">Topics & Weightage</label>
            <span className={`text-xs font-medium px-2 py-1 rounded ${totalWeight === 100 ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
              {totalWeight}% / 100%
            </span>
          </div>
          <div className="space-y-3">
            {topics.map((t, i) => (
              <div key={i} className="flex gap-2 items-center">
                <div className="flex-1">
                  {availableTopics.length > 0 ? (
                    <select value={t.topic} onChange={e => updateTopic(i, 'topic', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500">
                      <option value="">Select topic...</option>
                      {availableTopics.map(tp => <option key={tp} value={tp}>{tp}</option>)}
                    </select>
                  ) : (
                    <input type="text" value={t.topic} onChange={e => updateTopic(i, 'topic', e.target.value)}
                      placeholder="Topic name..."
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500" />
                  )}
                </div>
                <div className="w-28 relative">
                  <input type="number" min={1} max={100} value={t.weight}
                    onChange={e => updateTopic(i, 'weight', parseInt(e.target.value) || 0)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 pr-8" />
                  <span className="absolute right-3 top-2.5 text-gray-400 text-xs">% wt</span>
                </div>
                {topics.length > 1 && (
                  <button onClick={() => removeTopic(i)} className="p-2 text-red-400 hover:text-red-600 hover:bg-red-50 rounded-lg">
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}
          </div>
          <button onClick={addTopic} className="flex items-center gap-1 text-sm text-purple-600 hover:text-purple-800 font-medium mt-2">
            <Plus className="w-4 h-4" /> Add topic
          </button>
        </div>

        {/* Question Types + Marks */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Question Types & Marks per Question</label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {QUESTION_TYPES.map(t => (
              <div key={t.value} className={`rounded-lg border transition-all ${
                questionTypes.includes(t.value) ? 'bg-purple-50 border-purple-400' : 'bg-gray-50 border-gray-200'
              }`}>
                <label className="flex items-center gap-2 px-3 py-2 cursor-pointer">
                  <input type="checkbox" checked={questionTypes.includes(t.value)} onChange={() => toggleType(t.value)} className="w-4 h-4 accent-purple-600" />
                  <span className={`text-sm ${questionTypes.includes(t.value) ? 'text-purple-700' : 'text-gray-600'}`}>{t.label}</span>
                </label>
                <div className="px-3 pb-2 flex items-center gap-1">
                  <input type="number" min={1} max={20} value={marksPerType[t.value]}
                    onChange={e => setMarksPerType(prev => ({ ...prev, [t.value]: parseInt(e.target.value) || 1 }))}
                    disabled={!questionTypes.includes(t.value)}
                    className="w-14 px-2 py-1 border border-gray-300 rounded text-xs focus:ring-1 focus:ring-purple-500 disabled:opacity-40" />
                  <span className="text-xs text-gray-400">pts</span>
                </div>
              </div>
            ))}
          </div>
          {questionTypes.length > 0 && (
            <p className="text-xs text-gray-400 mt-2">
              Max score: {questionTypes.reduce((sum, qt) => {
                const perCombo = Math.max(1, Math.floor(totalQuestions / Math.max(1, topics.length * questionTypes.length)));
                return sum + perCombo * topics.length * (marksPerType[qt] || 1);
              }, 0)} pts
            </p>
          )}
        </div>

        {/* Other settings */}
        <div className="grid grid-cols-2 gap-4">
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
          {questionTypes.includes('mcq') && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">MCQ Options</label>
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
          {loading ? <><Loader className="w-5 h-5 animate-spin" /> Generating quiz...</> : <><Brain className="w-5 h-5" /> Generate Quiz</>}
        </button>
      </div>

      {/* Quiz display */}
      {quiz && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-bold text-gray-900">{topics.map(t => t.topic).join(', ')}</h3>
              <p className="text-sm text-gray-500">{quiz.questions?.length} questions • {quiz.difficulty}</p>
            </div>
            <div className="flex items-center gap-3">
              {/* Score summary */}
              <div className="text-right">
                <p className="text-2xl font-bold text-purple-700">{totalPossibleMarks} pts</p>
                <p className="text-xs text-gray-400">Total marks</p>
              </div>
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
          </div>

          {/* Marks breakdown */}
          <div className="flex gap-3 mb-4 flex-wrap">
            {questionTypes.map(qt => {
              const count = quiz.questions.filter(q => q.type === qt).length;
              const pts = marksPerType[qt] || 1;
              return count > 0 ? (
                <div key={qt} className="px-3 py-1.5 bg-gray-50 border border-gray-200 rounded-lg text-xs text-gray-600">
                  <span className="font-medium capitalize">{qt.replace('_', ' ')}</span>: {count} × {pts}pt = <span className="font-medium text-purple-700">{count * pts}pts</span>
                </div>
              ) : null;
            })}
          </div>

          <div className="space-y-4">
            {quiz.questions?.map((q, i) => <QuestionCard key={q.id || i} q={q} idx={i} />)}
          </div>
        </div>
      )}
    </div>
  );
}