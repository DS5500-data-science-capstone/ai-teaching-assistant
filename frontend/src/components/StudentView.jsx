import { useState, useEffect } from 'react';
import { LogOut, BookOpen, MessageCircle, ClipboardList, TrendingUp, CheckCircle, XCircle } from 'lucide-react';
import Discussion from './Discussion';
import { STUDENTS, pushedQuizzes, subscribeQuizzes, submitQuizResult, getStudentResults, quizResults, subscribeResults } from '../store';

const TABS = [
  { id: 'dashboard',  label: 'Dashboard',  icon: TrendingUp },
  { id: 'discussion', label: 'Q&A', icon: MessageCircle },
  { id: 'quizzes',    label: 'My Quizzes', icon: ClipboardList },
];

// ── Quiz Taking Component ─────────────────────────────────────────────────────
function TakeQuiz({ quiz, studentName, onDone, existingResult }) {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(!!existingResult);
  const [result, setResult] = useState(existingResult || null);

  const handleSelect = (qId, value) => {
    if (submitted) return;
    setAnswers(prev => ({ ...prev, [qId]: value }));
  };

  const handleSubmit = () => {
    let score = 0;
    quiz.questions.forEach(q => {
      const ans = answers[q.id || q.question];
      const correct = q.answer || q.statement;
      if (ans === correct) score += (q.marks || 1);
    });
    const r = { studentName, quizId: quiz.id, answers, score, total: quiz.totalMarks };
    submitQuizResult(r);
    setResult(r);
    setSubmitted(true);
  };

  const pct = result ? Math.round((result.score / result.total) * 100) : 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-bold text-gray-900">{quiz.title}</h3>
          <p className="text-sm text-gray-500">{quiz.questions.length} questions • {quiz.totalMarks} marks total</p>
        </div>
        <button onClick={onDone} className="text-sm text-gray-500 hover:text-gray-700">Back</button>
      </div>

      {submitted && result && (
        <div className={`p-4 rounded-lg border text-center ${pct >= 50 ? 'bg-green-50 border-green-300' : 'bg-red-50 border-red-300'}`}>
          <p className="text-2xl font-bold">{result.score} / {result.total}</p>
          <p className="text-sm mt-1">{pct}% — {pct >= 80 ? 'Excellent' : pct >= 60 ? 'Good' : pct >= 40 ? 'Needs improvement' : 'Keep practicing'}</p>
        </div>
      )}

      <div className="space-y-4">
        {quiz.questions.map((q, i) => {
          const qId = q.id || q.question;
          const selected = answers[qId];
          const correct = q.answer;

          return (
            <div key={i} className="p-4 border border-gray-200 rounded-lg">
              <div className="flex items-start gap-2 mb-3">
                <span className="w-6 h-6 bg-purple-100 text-purple-700 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">{i+1}</span>
                <div className="flex-1">
                  <p className="font-medium text-gray-900 text-sm">{q.question || q.statement}</p>
                  <span className="text-xs text-yellow-700 bg-yellow-50 px-2 py-0.5 rounded mt-1 inline-block">{q.marks || 1} pt{(q.marks || 1) > 1 ? 's' : ''}</span>
                </div>
              </div>

              {/* MCQ */}
              {q.type === 'mcq' && q.options && (
                <div className="space-y-2 ml-8">
                  {q.options.map((opt, j) => {
                    let cls = 'border-gray-200 bg-gray-50 text-gray-700';
                    if (submitted) {
                      if (opt === correct) cls = 'border-green-400 bg-green-50 text-green-800 font-medium';
                      else if (opt === selected && opt !== correct) cls = 'border-red-400 bg-red-50 text-red-800';
                    } else if (opt === selected) cls = 'border-purple-400 bg-purple-50 text-purple-800';
                    return (
                      <div key={j} onClick={() => handleSelect(qId, opt)}
                        className={`px-3 py-2 rounded-lg text-sm border transition-all flex items-center justify-between ${cls} ${!submitted ? 'cursor-pointer' : ''}`}>
                        <span>{String.fromCharCode(65+j)}. {opt}</span>
                        {submitted && opt === correct && <CheckCircle className="w-4 h-4 text-green-600" />}
                        {submitted && opt === selected && opt !== correct && <XCircle className="w-4 h-4 text-red-600" />}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* True/False */}
              {q.type === 'true_false' && (
                <div className="flex gap-3 ml-8">
                  {['True', 'False'].map(opt => {
                    let cls = 'border-gray-200 bg-gray-50 text-gray-700';
                    if (submitted) {
                      if (opt === correct) cls = 'border-green-400 bg-green-50 text-green-800 font-medium';
                      else if (opt === selected && opt !== correct) cls = 'border-red-400 bg-red-50 text-red-800';
                    } else if (opt === selected) cls = 'border-purple-400 bg-purple-50 text-purple-800';
                    return (
                      <div key={opt} onClick={() => handleSelect(qId, opt)}
                        className={`px-6 py-2 rounded-lg text-sm border transition-all ${cls} ${!submitted ? 'cursor-pointer' : ''}`}>
                        {opt}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Fill blank */}
              {q.type === 'fill_blank' && (
                <div className="ml-8">
                  <input type="text" value={answers[qId] || ''} disabled={submitted}
                    onChange={e => handleSelect(qId, e.target.value)}
                    placeholder="Type your answer..."
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" />
                  {submitted && <p className="text-xs text-green-700 mt-1 font-medium">Answer: {correct}</p>}
                </div>
              )}

              {/* Long answer */}
              {q.type === 'long_answer' && (
                <div className="ml-8">
                  <textarea rows={3} value={answers[qId] || ''} disabled={submitted}
                    onChange={e => handleSelect(qId, e.target.value)}
                    placeholder="Write your answer..."
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" />
                  {submitted && <p className="text-xs text-blue-700 mt-1">Model answer: {q.model_answer}</p>}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {!submitted && (
        <button onClick={handleSubmit}
          disabled={Object.keys(answers).length === 0}
          className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-semibold disabled:opacity-50">
          Submit Quiz
        </button>
      )}
    </div>
  );
}

// ── Main StudentView ──────────────────────────────────────────────────────────
export default function StudentView({ onBack }) {
  const [selectedStudent, setSelectedStudent] = useState('');
  const [loggedIn, setLoggedIn] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [quizzes, setQuizzes] = useState(pushedQuizzes);
  const [results, setResults] = useState([]);
  const [activeQuiz, setActiveQuiz] = useState(null);

  useEffect(() => {
    const unsub1 = subscribeQuizzes(q => setQuizzes([...q]));
    const unsub2 = subscribeResults(() => setResults(getStudentResults(selectedStudent)));
    return () => { unsub1(); unsub2(); };
  }, [selectedStudent]);

  useEffect(() => {
    if (selectedStudent) setResults(getStudentResults(selectedStudent));
  }, [selectedStudent]);

  const student = STUDENTS.find(s => s.name === selectedStudent);

  if (!loggedIn) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-500 to-blue-600 px-4">
        <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <BookOpen className="w-8 h-8 text-blue-600" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900 mb-1">Student Hub</h1>
            <p className="text-gray-500 text-sm">CS 5200 — Database Management Systems</p>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Select your name</label>
              <select value={selectedStudent} onChange={e => setSelectedStudent(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                <option value="">Choose student...</option>
                {STUDENTS.map(s => <option key={s.id} value={s.name}>{s.name}</option>)}
              </select>
            </div>
            <button onClick={() => selectedStudent && setLoggedIn(true)} disabled={!selectedStudent}
              className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold disabled:opacity-50">
              Enter Student Hub
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b sticky top-0 z-20">
        <div className="max-w-3xl mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Student Hub</h1>
            <p className="text-xs text-gray-500">CS 5200 — Database Management Systems</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-semibold">
              {selectedStudent.split(' ').map(n => n[0]).join('')}
            </div>
            <span className="text-sm text-gray-700 hidden sm:block">{selectedStudent}</span>
            <button onClick={() => setLoggedIn(false)} className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg">
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="max-w-3xl mx-auto px-4">
          <nav className="flex gap-1">
            {TABS.map(tab => (
              <button key={tab.id} onClick={() => { setActiveTab(tab.id); setActiveQuiz(null); }}
                className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors text-sm ${
                  activeTab === tab.id ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-600 hover:text-gray-900'
                }`}>
                <tab.icon className="w-4 h-4" />
                {tab.label}
                {tab.id === 'quizzes' && quizzes.length > 0 && (
                  <span className="w-5 h-5 bg-purple-600 text-white rounded-full text-xs flex items-center justify-center">{quizzes.length}</span>
                )}
              </button>
            ))}
          </nav>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-4 py-6">

        {/* Dashboard */}
        {activeTab === 'dashboard' && student && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-bold text-gray-900 mb-4">My Overview</h2>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <p className="text-2xl font-bold text-blue-700">{student.grade}%</p>
                  <p className="text-xs text-gray-500 mt-1">Current Grade</p>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <p className="text-2xl font-bold text-green-700">{student.attendance}%</p>
                  <p className="text-xs text-gray-500 mt-1">Attendance</p>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <p className="text-2xl font-bold text-purple-700">{student.assignments}/8</p>
                  <p className="text-xs text-gray-500 mt-1">Assignments</p>
                </div>
              </div>
            </div>

            {/* Quiz results */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-bold text-gray-900 mb-4">My Quiz Results</h2>
              {results.length === 0 ? (
                <p className="text-sm text-gray-500">No quizzes submitted yet.</p>
              ) : (
                <div className="space-y-3">
                  {results.map((r, i) => {
                    const quiz = quizzes.find(q => q.id === r.quizId);
                    const pct = Math.round((r.score / r.total) * 100);
                    return (
                      <div key={i} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                        <div>
                          <p className="font-medium text-gray-900 text-sm">{quiz?.title || r.quizId}</p>
                          <p className="text-xs text-gray-500">{quiz?.date}</p>
                        </div>
                        <div className="text-right">
                          <p className="font-bold text-lg">{r.score}/{r.total}</p>
                          <p className={`text-xs font-medium ${pct >= 60 ? 'text-green-600' : 'text-red-600'}`}>{pct}%</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Discussion */}
        {activeTab === 'discussion' && (
          <Discussion role="student" studentName={selectedStudent} />
        )}

        {/* Quizzes */}
        {activeTab === 'quizzes' && (
          <div>
            {activeQuiz ? (
              <TakeQuiz
                quiz={activeQuiz}
                studentName={selectedStudent}
                onDone={() => setActiveQuiz(null)}
                existingResult={results.find(r => r.quizId === activeQuiz.id)}
              />
            ) : (
              <div className="space-y-4">
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <h2 className="text-lg font-bold text-gray-900 mb-4">Available Quizzes</h2>
                  {quizzes.length === 0 ? (
                    <p className="text-sm text-gray-500">No quizzes have been pushed yet. Check back later.</p>
                  ) : (
                    <div className="space-y-3">
                      {quizzes.map(quiz => {
                        const result = results.find(r => r.quizId === quiz.id);
                        const pct = result ? Math.round((result.score / result.total) * 100) : null;
                        return (
                          <div key={quiz.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50">
                            <div>
                              <p className="font-medium text-gray-900">{quiz.title}</p>
                              <p className="text-xs text-gray-500">{quiz.date} • {quiz.questions.length} questions • {quiz.totalMarks} marks</p>
                            </div>
                            <div className="flex items-center gap-3">
                              {result && (
                                <span className={`text-sm font-bold ${pct >= 60 ? 'text-green-600' : 'text-red-600'}`}>
                                  {result.score}/{result.total}
                                </span>
                              )}
                              <button onClick={() => setActiveQuiz(quiz)}
                                className={`px-4 py-2 rounded-lg text-sm font-medium ${
                                  result ? 'bg-gray-100 text-gray-700 hover:bg-gray-200' : 'bg-purple-600 text-white hover:bg-purple-700'
                                }`}>
                                {result ? 'Review' : 'Take Quiz'}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}