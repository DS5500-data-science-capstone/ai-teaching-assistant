import { useState } from 'react';
import { AlertCircle, Mail, Send } from 'lucide-react';

export default function Students({ students }) {
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [emailMessage, setEmailMessage] = useState('');
  const [aiAssistEnabled, setAiAssistEnabled] = useState(false);

  const handleSendEmail = () => {
    if (!selectedStudent || !emailMessage) return;
    let msg = emailMessage;
    if (aiAssistEnabled) {
      msg = `Dear ${selectedStudent.name},\n\nI hope this message finds you well. I've noticed some areas where additional support might be beneficial:\n\n${emailMessage}\n\nI'm here to help you succeed. Please don't hesitate to reach out.\n\nBest regards,\nProfessor`;
    }
    alert(`Email sent to ${selectedStudent.email}\n\n${msg}`);
    setEmailMessage('');
    setSelectedStudent(null);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Student List</h2>
        <div className="space-y-2">
          {students.map(student => (
            <div
              key={student.id}
              className={`p-4 rounded-lg border-2 transition-all ${
                student.needsAttention ? 'bg-red-50 border-red-300' : 'bg-gray-50 border-gray-200'
              } ${selectedStudent?.id === student.id ? 'ring-2 ring-blue-500' : ''}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <h3 className="font-semibold text-gray-900">{student.name}</h3>
                    {student.needsAttention && (
                      <span className="flex items-center gap-1 px-2 py-1 bg-red-100 text-red-700 rounded text-xs font-medium">
                        <AlertCircle className="w-3 h-3" /> Needs Attention
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{student.email}</p>
                  <div className="flex gap-4 mt-2 text-sm text-gray-600">
                    <span>Grade: <strong className={student.grade < 60 ? 'text-red-600' : 'text-gray-900'}>{student.grade}%</strong></span>
                    <span>Assignments: {student.assignments}/8</span>
                    <span>Attendance: {student.attendance}%</span>
                    <span>Last active: {student.lastActive}</span>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedStudent(student)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
                >
                  <Mail className="w-4 h-4" /> Email
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {selectedStudent && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Email to {selectedStudent.name}</h3>
            <button onClick={() => setSelectedStudent(null)} className="text-gray-500 hover:text-gray-700">✕</button>
          </div>
          <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-4">
            <input
              type="checkbox"
              checked={aiAssistEnabled}
              onChange={e => setAiAssistEnabled(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded"
            />
            Enable AI Enhancement
          </label>
          <textarea
            value={emailMessage}
            onChange={e => setEmailMessage(e.target.value)}
            placeholder="Type your message here..."
            className="w-full h-32 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
          <div className="flex justify-end gap-3 mt-4">
            <button onClick={() => setSelectedStudent(null)} className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50">
              Cancel
            </button>
            <button onClick={handleSendEmail} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <Send className="w-4 h-4" /> Send Email
            </button>
          </div>
        </div>
      )}
    </div>
  );
}