import { useState, useEffect, useRef } from 'react';
import { Brain, Download, Loader, ChevronDown, ChevronUp, Plus, Trash2, Send, BookOpen } from 'lucide-react';
import jsPDF from 'jspdf';
import { pushQuiz } from '../store';

const API          = '/api';
const DIFFICULTIES = ['easy', 'medium', 'hard'];
const STYLES       = ['conceptual', 'scenario', 'definition'];
const QUESTION_TYPES = [
  { value: 'mcq',         label: 'Multiple Choice' },
  { value: 'fill_blank',  label: 'Fill in the Blank' },
  { value: 'long_answer', label: 'Long Answer' },
  { value: 'true_false',  label: 'True / False' },
];
const DEFAULT_MARKS = { mcq: 1, true_false: 1, fill_blank: 2, long_answer: 5 };

// ── Source badge ──────────────────────────────────────────────────────────────
function SourceBadge({ sources, chunks }) {
  const [open, setOpen] = useState(false);
  if (!sources?.length) return null;

  const getChunk = ref => {
    if (!chunks?.length) return null;
    const num = ref.replace(/[^0-9]/g, '');
    return chunks.find(c => String(c.chunk_index ?? c.index ?? '').replace(/[^0-9]/g, '') === num);
  };

  return (
    <div className="relative inline-block">
      <button onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 px-2 py-0.5 bg-blue-50 rounded border border-blue-200">
        <BookOpen className="w-3 h-3" />
        {sources.length} source{sources.length > 1 ? 's' : ''}
        {open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      </button>
      {open && (
        <div className="absolute left-0 top-7 z-20 w-80 bg-white border border-gray-200 rounded-lg shadow-lg p-3 space-y-2">
          {sources.map((src, i) => {
            const chunk = getChunk(src);
            return (
              <div key={i} className="text-xs">
                <p className="font-semibold text-gray-700 mb-1">
                  {chunk ? `${chunk.source || src} — page ${(chunk.page ?? 0) + 1}` : src}
                </p>
                {chunk?.content
                  ? <p className="text-gray-500 italic border-l-2 border-blue-200 pl-2 line-clamp-3">
                      {chunk.content.slice(0, 200)}{chunk.content.length > 200 ? '...' : ''}
                    </p>
                  : <p className="text-gray-400 italic">Source passage not available</p>
                }
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Question card ─────────────────────────────────────────────────────────────
function QuestionCard({ q, idx, chunks }) {
  const [showExp, setShowExp] = useState(false);

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
            <SourceBadge sources={q.sources} chunks={chunks} />
          </div>
          <p className="font-medium text-gray-900 mb-3">{q.question || q.statement}</p>

          {q.type === 'mcq' && q.options && (
            <div className="space-y-2 mb-3">
              {q.options.map((opt, j) => (
                <div key={j} className={`px-3 py-2 rounded-lg text-sm border ${
                  opt === q.answer ? 'border-green-400 bg-green-50 text-green-800 font-medium' : 'border-gray-200 bg-gray-50 text-gray-700'
                }`}>
                  {String.fromCharCode(65 + j)}. {opt}
                  {opt === q.answer && <span className="ml-2 text-green-600 font-bold">(correct)</span>}
                </div>
              ))}
            </div>
          )}
          {q.type === 'true_false' && (
            <div className="flex gap-3 mb-3">
              {['True', 'False'].map(v => (
                <div key={v} className={`px-6 py-2 rounded-lg text-sm border ${
                  v === q.answer ? 'border-green-400 bg-green-50 text-green-800 font-medium' : 'border-gray-200 bg-gray-50 text-gray-700'
                }`}>
                  {v}{v === q.answer && <span className="ml-2 text-green-600 font-bold">(correct)</span>}
                </div>
              ))}
            </div>
          )}
          {q.type === 'fill_blank' && (
            <div className="mb-3">
              <div className="px-3 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-400 italic">Answer: _______________</div>
              {q.answer && <p className="text-xs text-green-700 mt-1 font-medium">Answer: {q.answer}</p>}
            </div>
          )}
          {q.type === 'long_answer' && (
            <div className="mb-3">
              <div className="px-3 py-2 rounded-lg text-sm border border-gray-200 bg-gray-50 text-gray-400 italic h-16">Student answer here...</div>
              {q.model_answer && <p className="text-xs text-blue-700 mt-1">Model: {q.model_answer}</p>}
            </div>
          )}

          {q.explanation && (
            <div className="mt-2">
              <button onClick={() => setShowExp(!showExp)}
                className="text-xs text-purple-600 hover:text-purple-800 flex items-center gap-1">
                {showExp ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                {showExp ? 'Hide' : 'Show'} explanation
              </button>
              {showExp && (
                <div className="mt-2 space-y-2">
                  <div className="text-xs bg-purple-50 border border-purple-200 rounded p-2 text-gray-700">
                    <p className="font-semibold text-purple-700 mb-1">Why this answer is correct:</p>
                    {q.explanation}
                  </div>
                  {q.incorrect_answers?.map((d, i) => (
                    <div key={i} className="text-xs bg-red-50 border border-red-200 rounded p-2 text-gray-700">
                      <p className="font-semibold text-red-700 mb-1">Why "{d.answer}" is wrong:</p>
                      {d.explanation}
                      {d.sources?.length > 0 && <div className="mt-1"><SourceBadge sources={d.sources} chunks={chunks} /></div>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Quiz Builder ─────────────────────────────────────────────────────────
export default function Quiz() {
  const [topics,      setTopics]      = useState([{ topic: '', weight: 100 }]);
  const [types,       setTypes]       = useState(['mcq']);
  const [numQ,        setNumQ]        = useState(5);
  const [marksTarget, setMarksTarget] = useState(20);
  const [marksMode,   setMarksMode]   = useState('equal'); // 'equal' | 'per_type'
  const [marks,       setMarks]       = useState({ ...DEFAULT_MARKS });
  const [settings,    setSettings]    = useState({ difficulty: 'medium', style: 'conceptual', num_options: 4, source_filter: '' });
  const [quiz,        setQuiz]        = useState(null);
  const [generating,  setGenerating]  = useState(false);
  const [error,       setError]       = useState('');
  const [pushed,      setPushed]      = useState(false);
  const [availTopics, setAvailTopics] = useState([]);
  const [pdfOpen,     setPdfOpen]     = useState(false);
  const pdfRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/quiz-topics`).then(r => r.json()).then(d => setAvailTopics(d.topics || [])).catch(() => {});
  }, []);

  useEffect(() => {
    const h = e => { if (pdfRef.current && !pdfRef.current.contains(e.target)) setPdfOpen(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  // Auto-compute equal marks per question when in equal mode
  const equalMarksPerQ = marksMode === 'equal' && numQ > 0 ? Math.round(marksTarget / numQ) || 1 : null;

  // Computed total from per-type marks
  const computedTotal = marksMode === 'per_type'
    ? types.reduce((s, t) => {
        const qCount = Math.max(1, Math.floor(numQ / types.length));
        return s + qCount * (marks[t] || 1);
      }, 0)
    : marksTarget;

  const marksOk     = marksMode === 'equal'
    ? marksTarget % 5 === 0
    : computedTotal === marksTarget;
  const totalWeight = topics.reduce((s, t) => s + (parseInt(t.weight) || 0), 0);

  const updateTopic = (i, f, v) => { const t = [...topics]; t[i] = { ...t[i], [f]: v }; setTopics(t); };
  const removeTopic = i => setTopics(topics.filter((_, j) => j !== i));
  const toggleType  = v => setTypes(ts => ts.includes(v) ? ts.filter(t => t !== v) : [...ts, v]);

  const handleGenerate = async () => {
    if (topics.some(t => !t.topic.trim())) { setError('Please select a topic for each row.'); return; }
    if (!types.length)                      { setError('Select at least one question type.'); return; }
    if (totalWeight !== 100)                { setError(`Weights must sum to 100%. Currently: ${totalWeight}%`); return; }
    if (marksTarget % 5 !== 0)              { setError('Total marks must be a multiple of 5.'); return; }
    setError(''); setGenerating(true); setQuiz(null); setPushed(false);

    // Resolve marks per type
    const resolvedMarks = { ...marks };
    if (marksMode === 'equal') {
      types.forEach(t => { resolvedMarks[t] = equalMarksPerQ; });
    }

    try {
      const combos    = topics.flatMap(t => types.map(qt => ({ topic: t, qtype: qt })));
      const perCombo  = Math.max(1, Math.floor(numQ / combos.length));
      const extra     = numQ - perCombo * combos.length;
      const allQ      = [];
      const allChunks = [];

      for (let i = 0; i < combos.length; i++) {
        const { topic, qtype } = combos[i];
        const res  = await fetch(`${API}/generate-quiz`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic: topic.topic, num_questions: perCombo + (i < extra ? 1 : 0),
            difficulty: settings.difficulty, style: settings.style,
            question_type: qtype, num_options: settings.num_options,
            source_filter: settings.source_filter || null,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Generation failed');
        const qs = (data.questions || []).map(q => ({ ...q, topic_weight: topic.weight, marks: resolvedMarks[qtype] ?? 1 }));
        allQ.push(...qs);
        if (data.run_metadata?.retrieved_chunks) allChunks.push(...data.run_metadata.retrieved_chunks);
      }

      const total = allQ.reduce((s, q) => s + (q.marks || 0), 0);
      setQuiz({ quiz_id: `quiz_${Date.now()}`, difficulty: settings.difficulty, style: settings.style,
        questions: allQ, total_marks: total, marks_target: marksTarget, chunks: allChunks });
    } catch (e) {
      setError(e.message);
    } finally {
      setGenerating(false);
    }
  };

  const handlePush = () => {
    if (!quiz) return;
    const date = new Date().toISOString().split('T')[0];
    pushQuiz({ id: quiz.quiz_id, title: `${topics.map(t => t.topic).join(', ')} — ${date}`,
      date, questions: quiz.questions, totalMarks: quiz.total_marks });
    setPushed(true);
  };

  const downloadPDF = (mode) => {
    if (!quiz) return;
    const doc = new jsPDF(); const lm = 15; const pw = doc.internal.pageSize.getWidth() - lm * 2;
    let y = 20;
    const line = (txt, sz = 11, bold = false, color = [0,0,0]) => {
      doc.setFontSize(sz); doc.setFont('helvetica', bold ? 'bold' : 'normal'); doc.setTextColor(...color);
      doc.splitTextToSize(String(txt), pw).forEach(l => { if (y > 275) { doc.addPage(); y = 20; } doc.text(l, lm, y); y += sz * 0.5; }); y += 2;
    };
    line(`${topics.map(t => t.topic).join(', ')}`, 15, true);
    line(`${quiz.difficulty} • ${quiz.questions.length} questions • ${totalMarks} pts • CS 5200 — Prof. Cristiano`, 10, false, [120,120,120]);
    if (mode === 'answers') line('ANSWER KEY', 12, true, [100,50,180]);
    quiz.questions.forEach((q, i) => {
      y += 3; doc.setDrawColor(220,220,220); doc.line(lm, y, lm + pw, y); y += 5;
      line(`Q${i+1}. [${(q.type||'').replace('_',' ').toUpperCase()}] ${q.question||q.statement}`, 11, true);
      line(`Topic: ${q.topic}  |  ${q.marks} pt${q.marks!==1?'s':''}`, 9, false, [140,140,140]);
      if (mode === 'questions') {
        q.options?.forEach((o, j) => line(`   ${String.fromCharCode(65+j)}. ${o}`, 10));
        if (q.type === 'true_false') { line('   A. True', 10); line('   B. False', 10); }
        if (q.type === 'fill_blank') line('   Answer: _______________', 10);
        if (q.type === 'long_answer') { line('   Answer:', 10); y += 20; }
      }
      if (mode === 'answers') {
        q.options?.forEach((o, j) => {
          const ok = o === q.answer;
          line(`   ${String.fromCharCode(65+j)}. ${o}${ok?' (correct)':''}`, 10, ok, ok?[34,139,34]:[80,80,80]);
        });
        if (q.type === 'true_false') ['True','False'].forEach(v => { const ok = v===q.answer; line(`   ${v}${ok?' (correct)':''}`, 10, ok, ok?[34,139,34]:[80,80,80]); });
        if (q.type === 'fill_blank') line(`   Answer: ${q.answer}`, 10, true, [34,139,34]);
        if (q.type === 'long_answer' && q.model_answer) line(`   Model: ${q.model_answer}`, 10);
        if (q.explanation) line(`   Explanation: ${q.explanation}`, 9, false, [100,100,100]);
      }
      y += 4;
    });
    const date = new Date().toISOString().split('T')[0];
    const slug = topics.map(t=>t.topic).join('_').replace(/[^a-z0-9]/gi,'_').slice(0,40).toLowerCase();
    doc.save(`${slug}_${mode}_${date}.pdf`);
    setPdfOpen(false);
  };

  const totalMarks = quiz?.questions?.reduce((s, q) => s + (q.marks || 0), 0) || 0;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Brain className="w-6 h-6 text-purple-500" /> Quiz Builder
        </h2>

        {/* Step 0 — Total marks */}
        <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Step 1 — Total Marks Target</h3>
          <div className="flex items-center gap-4">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Total marks (multiple of 5)</label>
              <input type="number" min={5} step={5} value={marksTarget}
                onChange={e => setMarksTarget(parseInt(e.target.value) || 5)}
                className={`w-32 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-purple-500 ${marksTarget % 5 !== 0 ? 'border-red-400' : 'border-gray-300'}`} />
              {marksTarget % 5 !== 0 && <p className="text-xs text-red-500 mt-1">Must be a multiple of 5</p>}
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Marks distribution</label>
              <div className="flex gap-2">
                {[
                  { val: 'equal',    label: 'Equal per question' },
                  { val: 'per_type', label: 'Different per type' },
                ].map(m => (
                  <button key={m.val} onClick={() => setMarksMode(m.val)}
                    className={`px-3 py-2 text-xs rounded-lg border font-medium transition-all ${
                      marksMode === m.val ? 'bg-purple-600 text-white border-purple-600' : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
                    }`}>
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
            {marksMode === 'equal' && numQ > 0 && (
              <div className="text-sm text-gray-600">
                <span className="font-medium">{equalMarksPerQ} pt</span> per question
              </div>
            )}
          </div>
        </div>

        {/* Step 1 — Total questions */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-1">Step 2 — Total Number of Questions</label>
          <input type="number" min={1} max={50} value={numQ} onChange={e => setNumQ(parseInt(e.target.value)||1)}
            className="w-40 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" />
        </div>

        {/* Step 2 — Question types + marks */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Step 3 — Question Types{marksMode === 'per_type' ? ' & Marks per Type' : ''}</label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {QUESTION_TYPES.map(qt => (
              <div key={qt.value} className={`rounded-lg border transition-all ${types.includes(qt.value)?'bg-purple-50 border-purple-400':'bg-gray-50 border-gray-200'}`}>
                <label className="flex items-center gap-2 px-3 py-2 cursor-pointer">
                  <input type="checkbox" checked={types.includes(qt.value)} onChange={() => toggleType(qt.value)} className="w-4 h-4 accent-purple-600" />
                  <span className={`text-sm ${types.includes(qt.value)?'text-purple-700':'text-gray-600'}`}>{qt.label}</span>
                </label>
                {marksMode === 'per_type' && (
                  <div className="px-3 pb-2 flex items-center gap-1">
                    <input type="number" min={1} max={20} value={marks[qt.value]}
                      onChange={e => setMarks(m => ({ ...m, [qt.value]: parseInt(e.target.value)||1 }))}
                      disabled={!types.includes(qt.value)}
                      className="w-14 px-2 py-1 border border-gray-300 rounded text-xs focus:ring-1 focus:ring-purple-500 disabled:opacity-40" />
                    <span className="text-xs text-gray-400">pts</span>
                  </div>
                )}
              </div>
            ))}
          </div>
          {marksMode === 'per_type' && types.length > 0 && (
            <div className={`mt-2 text-xs px-3 py-2 rounded-lg font-medium ${computedTotal === marksTarget ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'}`}>
              Computed total: {computedTotal} / {marksTarget} pts
              {computedTotal !== marksTarget && ' — adjust marks per type to match target'}
            </div>
          )}
        </div>

        {/* Step 3 — Topics */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-gray-700">Step 4 — Topics & Weightage</label>
            <span className={`text-xs font-medium px-2 py-1 rounded ${totalWeight===100?'bg-green-50 text-green-700':'bg-red-50 text-red-600'}`}>
              {totalWeight}% / 100%
            </span>
          </div>
          <div className="space-y-3">
            {topics.map((t, i) => (
              <div key={i} className="flex gap-2 items-center">
                <div className="flex-1">
                  {availTopics.length > 0
                    ? <select value={t.topic} onChange={e => updateTopic(i, 'topic', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500">
                        <option value="">Select topic...</option>
                        {availTopics.map(at => <option key={at} value={at}>{at}</option>)}
                      </select>
                    : <input type="text" value={t.topic} onChange={e => updateTopic(i, 'topic', e.target.value)}
                        placeholder="Topic name..." className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500" />
                  }
                </div>
                <div className="w-28 relative">
                  <input type="number" min={1} max={100} value={t.weight} onChange={e => updateTopic(i, 'weight', parseInt(e.target.value)||0)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 pr-8" />
                  <span className="absolute right-3 top-2.5 text-gray-400 text-xs">%</span>
                </div>
                {topics.length > 1 && (
                  <button onClick={() => removeTopic(i)} className="p-2 text-red-400 hover:text-red-600 hover:bg-red-50 rounded-lg">
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}
          </div>
          <button onClick={() => setTopics([...topics, { topic: '', weight: 0 }])}
            className="flex items-center gap-1 text-sm text-purple-600 hover:text-purple-800 font-medium mt-2">
            <Plus className="w-4 h-4" /> Add topic
          </button>
        </div>

        {/* Step 4 — Settings */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Step 5 — Difficulty & Style</label>
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: 'Difficulty', key: 'difficulty', opts: DIFFICULTIES },
              { label: 'Style',      key: 'style',      opts: STYLES },
            ].map(({ label, key, opts }) => (
              <div key={key}>
                <label className="block text-xs text-gray-500 mb-1">{label}</label>
                <select value={settings[key]} onChange={e => setSettings(s => ({ ...s, [key]: e.target.value }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500">
                  {opts.map(o => <option key={o} value={o}>{o.charAt(0).toUpperCase()+o.slice(1)}</option>)}
                </select>
              </div>
            ))}
            {types.includes('mcq') && (
              <div>
                <label className="block text-xs text-gray-500 mb-1">MCQ Options</label>
                <select value={settings.num_options} onChange={e => setSettings(s => ({ ...s, num_options: parseInt(e.target.value) }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500">
                  {[3,4,5].map(n => <option key={n} value={n}>{n} options</option>)}
                </select>
              </div>
            )}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Source Filter (optional)</label>
              <input type="text" value={settings.source_filter} onChange={e => setSettings(s => ({ ...s, source_filter: e.target.value }))}
                placeholder="e.g. lectures.pdf" className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" />
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="mb-4 p-4 bg-gray-50 rounded-lg border border-gray-200 text-sm text-gray-700 space-y-1">
          <p className="font-semibold text-gray-900 mb-2">Quiz Summary</p>
          <p>Topics: {topics.filter(t=>t.topic).map(t=>`${t.topic} (${t.weight}%)`).join(', ') || '—'}</p>
          <p>Types: {types.join(', ') || '—'}</p>
          <p>Questions: {numQ} | Difficulty: {settings.difficulty} | Style: {settings.style}</p>
          <p>
            Marks: {marksMode === 'equal'
              ? `${equalMarksPerQ} pt x ${numQ} = ${marksTarget} pts target`
              : `${types.map(t=>`${t}: ${marks[t]}pt`).join(', ')} = ${computedTotal} / ${marksTarget} pts target`}
            {' '}
            <span className={marksOk ? 'text-green-600 font-medium' : 'text-red-500 font-medium'}>
              {marksOk ? '(target met)' : '(target not met)'}
            </span>
          </p>
        </div>

        {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

        <button onClick={handleGenerate} disabled={generating}
          className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-semibold flex items-center justify-center gap-2 disabled:opacity-50">
          {generating
            ? <><Loader className="w-5 h-5 animate-spin" /> Generating...</>
            : <><Brain className="w-5 h-5" /> Generate Quiz</>}
        </button>
      </div>

      {/* Quiz preview */}
      {quiz && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-bold text-gray-900">{topics.map(t => t.topic).join(', ')}</h3>
              <p className="text-sm text-gray-500">{quiz.questions?.length} questions • {quiz.difficulty}</p>
            </div>
            <div className="flex items-center gap-2">
              <div className="text-right mr-2">
                <p className="text-2xl font-bold text-purple-700">{totalMarks} pts</p>
                {quiz.marks_target && (
                  <p className={`text-xs font-medium ${totalMarks === quiz.marks_target ? 'text-green-600' : 'text-red-500'}`}>
                    target: {quiz.marks_target} pts {totalMarks === quiz.marks_target ? '(met)' : '(not met)'}
                  </p>
                )}
              </div>
              <button onClick={handlePush} disabled={pushed}
                className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 ${pushed?'bg-green-100 text-green-700 cursor-default':'bg-purple-600 text-white hover:bg-purple-700'}`}>
                <Send className="w-4 h-4" /> {pushed ? 'Pushed' : 'Push to Students'}
              </button>
              <div className="relative" ref={pdfRef}>
                <button onClick={() => setPdfOpen(!pdfOpen)}
                  className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center gap-2">
                  <Download className="w-4 h-4" /> <ChevronDown className="w-4 h-4" />
                </button>
                {pdfOpen && (
                  <div className="absolute right-0 mt-1 w-52 bg-white border border-gray-200 rounded-lg shadow-lg z-10">
                    <button onClick={() => downloadPDF('questions')} className="w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 border-b border-gray-100">
                      Questions only
                      <p className="text-xs text-gray-400 mt-0.5">No answers — for students</p>
                    </button>
                    <button onClick={() => downloadPDF('answers')} className="w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-gray-50">
                      Answer key
                      <p className="text-xs text-gray-400 mt-0.5">With explanations</p>
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            {quiz.questions?.map((q, i) => (
              <QuestionCard key={q.id || i} q={q} idx={i} chunks={quiz.chunks} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}