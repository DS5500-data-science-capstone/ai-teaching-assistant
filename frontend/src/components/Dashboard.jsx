import { useState, useEffect } from 'react';
import { AlertCircle, Book, Clock, AlertTriangle } from 'lucide-react';

const API = '/api';

const courseData = {
  courseNumber: 'CS 5200',
  courseName:   'Database Management Systems',
  semester:     'Spring 2025',
  schedule:     'Mon/Wed 6:00 PM - 7:30 PM',
  room:         'Snell Library 453',
};

const LEVEL_STYLES = {
  high:   { badge: 'bg-red-100 text-red-700',    bar: 'bg-red-500',    label: 'High Risk' },
  medium: { badge: 'bg-orange-100 text-orange-700', bar: 'bg-orange-400', label: 'Medium Risk' },
  low:    { badge: 'bg-green-100 text-green-700', bar: 'bg-green-500',  label: 'Low Risk' },
};

export default function Dashboard({ students, documents, onContactStudent }) {
  const [riskData, setRiskData] = useState({});
  const [loading, setLoading]   = useState(true);
  const [expanded, setExpanded] = useState(null);

  const atRisk   = students.filter(s => s.needsAttention);
  const avgGrade = Math.round(students.reduce((a, s) => a + s.grade, 0) / students.length);

  useEffect(() => { fetchRisk(); }, []);

  const fetchRisk = async () => {
    setLoading(true);
    try {
      const res  = await fetch(`${API}/predict-risk-batch`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(students.map(s => ({
          name:        s.name,
          grade:       s.grade,
          attendance:  s.attendance,
          assignments: s.assignments,
          quiz_avg:    s.grade, // fallback until real quiz data
        }))),
      });
      const data = await res.json();
      const map  = {};
      data.results.forEach(r => { map[r.name] = r; });
      setRiskData(map);
    } catch (e) {
      console.error('Risk fetch failed:', e);
    } finally {
      setLoading(false);
    }
  };

  const highRiskCount = Object.values(riskData).filter(r => r.level === 'high').length;

  return (
    <div className="space-y-6">
      {/* Course Card */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{courseData.courseNumber}</h2>
            <p className="text-lg text-gray-700 mt-1">{courseData.courseName}</p>
          </div>
          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
            {courseData.semester}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div className="flex items-center gap-3 text-gray-600">
            <Clock className="w-5 h-5" />
            <span>{courseData.schedule}</span>
          </div>
          <div className="flex items-center gap-3 text-gray-600">
            <Book className="w-5 h-5" />
            <span>{courseData.room}</span>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm p-4">
          <p className="text-sm text-gray-600">Total Students</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{students.length}</p>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <p className="text-sm text-gray-600">Need Attention</p>
          <p className="text-2xl font-bold text-red-600 mt-1">{atRisk.length}</p>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <p className="text-sm text-gray-600">Avg Grade</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{avgGrade}%</p>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <p className="text-sm text-gray-600">Documents</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{documents.length}</p>
        </div>
      </div>

      {/* High risk alert banner */}
      {highRiskCount > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <p className="text-sm text-red-700">
            <strong>{highRiskCount} student{highRiskCount > 1 ? 's' : ''}</strong> flagged as high risk by the ML model. Faculty notified via email.
          </p>
        </div>
      )}

      {/* At-Risk Students (existing) */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-red-500" />
          Students Needing Attention
        </h3>
        <div className="space-y-3">
          {atRisk.map(student => (
            <div key={student.id} className="flex items-center justify-between p-3 bg-red-50 border border-red-200 rounded-lg">
              <div>
                <p className="font-medium text-gray-900">{student.name}</p>
                <p className="text-sm text-gray-600">Grade: {student.grade}% • Last active: {student.lastActive}</p>
              </div>
              <button
                onClick={() => onContactStudent(student)}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Contact
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* ML Risk Assessment */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-500" />
            ML Risk Assessment
          </h3>
          <button onClick={fetchRisk} className="text-xs text-blue-600 hover:text-blue-800 font-medium">
            Refresh
          </button>
        </div>

        {loading ? (
          <p className="text-sm text-gray-400 text-center py-6">Running risk model...</p>
        ) : (
          <div className="space-y-2">
            {students.map(s => {
              const risk   = riskData[s.name];
              const style  = risk ? LEVEL_STYLES[risk.level] : LEVEL_STYLES.low;
              const isOpen = expanded === s.id;
              return (
                <div key={s.id} className={`border rounded-lg overflow-hidden ${risk?.at_risk ? 'border-red-200' : 'border-gray-200'}`}>
                  <div
                    className="flex items-center gap-3 p-3 cursor-pointer hover:bg-gray-50"
                    onClick={() => setExpanded(isOpen ? null : s.id)}
                  >
                    <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-gray-700 font-semibold text-xs flex-shrink-0">
                      {s.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-900 text-sm">{s.name}</p>
                    </div>
                    {risk && (
                      <div className="hidden sm:flex items-center gap-2 w-28">
                        <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${style.bar}`} style={{ width: `${risk.risk_score * 100}%` }} />
                        </div>
                        <span className="text-xs text-gray-500 w-8">{Math.round(risk.risk_score * 100)}%</span>
                      </div>
                    )}
                    {risk && (
                      <span className={`text-xs px-2 py-0.5 rounded font-medium ${style.badge}`}>
                        {style.label}
                      </span>
                    )}
                  </div>
                  {isOpen && risk?.at_risk && (
                    <div className="border-t border-gray-100 bg-gray-50 px-4 py-3">
                      <div className="grid grid-cols-4 gap-2 mb-2 text-center">
                        {[
                          { label: 'Grade',       value: `${s.grade}%` },
                          { label: 'Attendance',  value: `${s.attendance}%` },
                          { label: 'Assignments', value: `${s.assignments}/8` },
                          { label: 'Risk Score',  value: `${Math.round(risk.risk_score * 100)}%` },
                        ].map(m => (
                          <div key={m.label}>
                            <p className="text-sm font-bold text-gray-900">{m.value}</p>
                            <p className="text-xs text-gray-500">{m.label}</p>
                          </div>
                        ))}
                      </div>
                      <p className="text-xs font-semibold text-red-700 mb-1">Risk Factors:</p>
                      <ul className="space-y-0.5">
                        {risk.reasons.map((r, i) => (
                          <li key={i} className="text-xs text-red-600 flex items-center gap-1">
                            <span className="w-1.5 h-1.5 bg-red-400 rounded-full flex-shrink-0" />
                            {r}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}