import React, { useState } from 'react';
import { Upload, Send, MessageCircle, AlertCircle, Book, Users, Clock, Mail, FileText, Download, LogOut, Search, Filter } from 'lucide-react';

const AITeachingAssistant = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [emailMessage, setEmailMessage] = useState('');
  const [discussionMessage, setDiscussionMessage] = useState('');
  const [aiAssistEnabled, setAiAssistEnabled] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');

  // Mock data - replace with actual API calls
  const courseData = {
    courseNumber: 'CS 5200',
    courseName: 'Database Management Systems',
    semester: 'Spring 2025',
    schedule: 'Mon/Wed 6:00 PM - 7:30 PM',
    room: 'Snell Library 453'
  };

  const students = [
    { id: 1, name: 'Sarah Johnson', email: 'johnson.s@northeastern.edu', grade: 45, needsAttention: true, lastActive: '2 days ago', assignments: 3, attendance: 60 },
    { id: 2, name: 'Michael Chen', email: 'chen.m@northeastern.edu', grade: 52, needsAttention: true, lastActive: '5 hours ago', assignments: 4, attendance: 75 },
    { id: 3, name: 'Emily Rodriguez', email: 'rodriguez.e@northeastern.edu', grade: 67, needsAttention: false, lastActive: '1 hour ago', assignments: 6, attendance: 85 },
    { id: 4, name: 'David Kim', email: 'kim.d@northeastern.edu', grade: 78, needsAttention: false, lastActive: '30 min ago', assignments: 7, attendance: 90 },
    { id: 5, name: 'Aisha Patel', email: 'patel.a@northeastern.edu', grade: 85, needsAttention: false, lastActive: '15 min ago', assignments: 8, attendance: 95 },
    { id: 6, name: 'James Wilson', email: 'wilson.j@northeastern.edu', grade: 92, needsAttention: false, lastActive: '10 min ago', assignments: 8, attendance: 100 },
  ];

  const documents = [
    { id: 1, name: 'Lecture 5 - SQL Joins.pdf', uploadDate: '2025-02-01', size: '2.3 MB' },
    { id: 2, name: 'Assignment 3 - Normalization.docx', uploadDate: '2025-01-28', size: '156 KB' },
    { id: 3, name: 'Midterm Study Guide.pdf', uploadDate: '2025-01-25', size: '1.8 MB' },
    ...uploadedFiles
  ];

  const discussions = [
    { id: 1, author: 'Aisha Patel', role: 'student', message: 'Can someone explain the difference between INNER JOIN and LEFT JOIN?', time: '2 hours ago', replies: 3 },
    { id: 2, author: 'You', role: 'faculty', message: 'Great question! INNER JOIN returns only matching rows, while LEFT JOIN returns all rows from the left table...', time: '1 hour ago', replies: 0 },
    { id: 3, author: 'David Kim', role: 'student', message: 'Is the assignment due tonight or tomorrow night?', time: '30 min ago', replies: 1 },
  ];

  const handleLogin = (e) => {
    e.preventDefault();
    // Simple mock authentication
    if (loginEmail && loginPassword) {
      setIsLoggedIn(true);
    } else {
      alert('Please enter email and password');
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setLoginEmail('');
    setLoginPassword('');
    setActiveTab('dashboard');
  };

  const handleSendEmail = async () => {
    if (!selectedStudent || !emailMessage) return;

    let finalMessage = emailMessage;

    if (aiAssistEnabled) {
      // Simulate AI enhancement
      finalMessage = `Dear ${selectedStudent.name},

I hope this message finds you well. I've noticed some areas where additional support might be beneficial:

${emailMessage}

I'm here to help you succeed in this course. Please don't hesitate to reach out during office hours or schedule a meeting.

Best regards,
Professor`;
    }

    alert(`Email sent to ${selectedStudent.email}\n\nMessage:\n${finalMessage}`);
    setEmailMessage('');
    setSelectedStudent(null);
  };

  const handleUploadDocument = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    if (!file.name.endsWith('.pdf')) {
      alert('Only PDF files are allowed.');
      return;
    }

    setUploadStatus('uploading');
    setUploadMessage(`Uploading ${file.name}...`);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        const newDoc = {
          id: Date.now(),
          name: file.name,
          uploadDate: new Date().toISOString().split('T')[0],
          size: (file.size / 1024).toFixed(2) + ' KB',
        };
        setUploadedFiles([...uploadedFiles, newDoc]);
        setUploadStatus('success');
        setUploadMessage(`✅ "${file.name}" uploaded! Watcher will process it automatically.`);
      } else {
        setUploadStatus('error');
        setUploadMessage(`❌ Upload failed: ${data.detail}`);
      }
    } catch (err) {
      setUploadStatus('error');
      setUploadMessage(`❌ Could not reach upload server. Is FastAPI running on port 8000?`);
    }

    e.target.value = '';
    setTimeout(() => setUploadStatus(null), 5000);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {!isLoggedIn ? (
        // Login Screen
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600">
          <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md">
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Teaching Assistant</h1>
              <p className="text-gray-600">Northeastern University</p>
            </div>

            <form onSubmit={handleLogin} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Email Address
                </label>
                <input
                  type="email"
                  value={loginEmail}
                  onChange={(e) => setLoginEmail(e.target.value)}
                  placeholder="faculty@northeastern.edu"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Password
                </label>
                <input
                  type="password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <button
                type="submit"
                className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors font-semibold"
              >
                Sign In
              </button>

              <p className="text-center text-sm text-gray-600">
                Demo: Use any email and password
              </p>
            </form>
          </div>
        </div>
      ) : (
        // Main Application
        <>
          {/* Header */}
          <div className="bg-white shadow-sm border-b">
            <div className="max-w-7xl mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">AI Teaching Assistant</h1>
                  <p className="text-sm text-gray-600">Northeastern University</p>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-gray-600">Prof. Sarah Martinez</span>
                  <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-semibold">
                    SM
                  </div>
                  <button
                    onClick={handleLogout}
                    className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Logout"
                  >
                    <LogOut className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="bg-white border-b">
            <div className="max-w-7xl mx-auto px-4">
              <nav className="flex gap-1">
                {[
                  { id: 'dashboard', label: 'Dashboard', icon: Book },
                  { id: 'students', label: 'Students', icon: Users },
                  { id: 'documents', label: 'Documents', icon: FileText },
                  { id: 'discussion', label: 'Discussion', icon: MessageCircle },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-600 hover:text-gray-900'
                      }`}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="max-w-7xl mx-auto px-4 py-6">
            {/* Dashboard Tab */}
            {activeTab === 'dashboard' && (
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
                    <p className="text-2xl font-bold text-red-600 mt-1">
                      {students.filter(s => s.needsAttention).length}
                    </p>
                  </div>
                  <div className="bg-white rounded-lg shadow-sm p-4">
                    <p className="text-sm text-gray-600">Avg Grade</p>
                    <p className="text-2xl font-bold text-gray-900 mt-1">
                      {Math.round(students.reduce((acc, s) => acc + s.grade, 0) / students.length)}%
                    </p>
                  </div>
                  <div className="bg-white rounded-lg shadow-sm p-4">
                    <p className="text-sm text-gray-600">Documents</p>
                    <p className="text-2xl font-bold text-gray-900 mt-1">{documents.length}</p>
                  </div>
                </div>

                {/* At-Risk Students */}
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-red-500" />
                    Students Needing Attention
                  </h3>
                  <div className="space-y-3">
                    {students.filter(s => s.needsAttention).map(student => (
                      <div key={student.id} className="flex items-center justify-between p-3 bg-red-50 border border-red-200 rounded-lg">
                        <div>
                          <p className="font-medium text-gray-900">{student.name}</p>
                          <p className="text-sm text-gray-600">Grade: {student.grade}% • Last active: {student.lastActive}</p>
                        </div>
                        <button
                          onClick={() => {
                            setSelectedStudent(student);
                            setActiveTab('students');
                          }}
                          className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                        >
                          Contact
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Students Tab */}
            {activeTab === 'students' && (
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-4">Student List (Sorted by Grade)</h2>
                  <div className="space-y-2">
                    {students.map(student => (
                      <div
                        key={student.id}
                        className={`p-4 rounded-lg border-2 transition-all ${student.needsAttention
                            ? 'bg-red-50 border-red-300'
                            : 'bg-gray-50 border-gray-200'
                          } ${selectedStudent?.id === student.id ? 'ring-2 ring-blue-500' : ''}`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3">
                              <h3 className="font-semibold text-gray-900">{student.name}</h3>
                              {student.needsAttention && (
                                <span className="flex items-center gap-1 px-2 py-1 bg-red-100 text-red-700 rounded text-xs font-medium">
                                  <AlertCircle className="w-3 h-3" />
                                  Needs Attention
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
                            <Mail className="w-4 h-4" />
                            Email
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Email Panel */}
                {selectedStudent && (
                  <div className="bg-white rounded-lg shadow-sm p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">
                        Email to {selectedStudent.name}
                      </h3>
                      <button
                        onClick={() => setSelectedStudent(null)}
                        className="text-gray-500 hover:text-gray-700"
                      >
                        ✕
                      </button>
                    </div>

                    <div className="mb-4">
                      <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
                        <input
                          type="checkbox"
                          checked={aiAssistEnabled}
                          onChange={(e) => setAiAssistEnabled(e.target.checked)}
                          className="w-4 h-4 text-blue-600 rounded"
                        />
                        Enable AI Enhancement (improves tone and adds context)
                      </label>
                    </div>

                    <textarea
                      value={emailMessage}
                      onChange={(e) => setEmailMessage(e.target.value)}
                      placeholder="Type your message here..."
                      className="w-full h-32 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <div className="flex justify-end gap-3 mt-4">
                      <button
                        onClick={() => setSelectedStudent(null)}
                        className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleSendEmail}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
                      >
                        <Send className="w-4 h-4" />
                        Send Email
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Documents Tab */}
            {activeTab === 'documents' && (
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold text-gray-900">Course Documents</h2>
                    <label className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 cursor-pointer">
                      <Upload className="w-4 h-4" />
                      Upload Document
                      <input
                        type="file"
                        onChange={handleUploadDocument}
                        className="hidden"
                        accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.xlsx,.xls"
                      />
                    </label>
                  </div>

                  <div className="space-y-3">
                    {documents.map(doc => (
                      <div key={doc.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                        <div className="flex items-center gap-3">
                          <FileText className="w-8 h-8 text-blue-500" />
                          <div>
                            <p className="font-medium text-gray-900">{doc.name}</p>
                            <p className="text-sm text-gray-600">Uploaded: {doc.uploadDate} • {doc.size}</p>
                          </div>
                        </div>
                        <button className="p-2 text-gray-600 hover:text-gray-900 rounded-lg hover:bg-gray-100">
                          <Download className="w-5 h-5" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>{uploadStatus && (
                  <div className={`p-3 rounded-lg text-sm font-medium ${uploadStatus === 'uploading' ? 'bg-blue-50 text-blue-700' :
                      uploadStatus === 'success' ? 'bg-green-50 text-green-700' :
                        'bg-red-50 text-red-700'
                    }`}>
                    {uploadMessage}
                  </div>
                )}

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-sm text-blue-800">
                    <strong>Note:</strong> When you upload a document, all enrolled students will receive an email notification automatically.
                  </p>
                </div>
              </div>
            )}

            {/* Discussion Tab */}
            {activeTab === 'discussion' && (
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-6">Course Discussion</h2>

                  {/* Post Message */}
                  <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                    <textarea
                      value={discussionMessage}
                      onChange={(e) => setDiscussionMessage(e.target.value)}
                      placeholder="Post a message to the class..."
                      className="w-full h-24 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <div className="flex justify-end mt-3">
                      <button
                        onClick={handlePostDiscussion}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
                      >
                        <Send className="w-4 h-4" />
                        Post
                      </button>
                    </div>
                  </div>

                  {/* Discussion Threads */}
                  <div className="space-y-4">
                    {discussions.map(thread => (
                      <div key={thread.id} className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                        <div className="flex items-start gap-3">
                          <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center text-gray-700 font-semibold">
                            {thread.author.split(' ').map(n => n[0]).join('')}
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-semibold text-gray-900">{thread.author}</span>
                              <span className={`px-2 py-0.5 rounded text-xs font-medium ${thread.role === 'faculty' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
                                }`}>
                                {thread.role}
                              </span>
                              <span className="text-sm text-gray-500">• {thread.time}</span>
                            </div>
                            <p className="text-gray-700">{thread.message}</p>
                            {thread.replies > 0 && (
                              <button className="mt-2 text-sm text-blue-600 hover:text-blue-700 font-medium">
                                {thread.replies} {thread.replies === 1 ? 'reply' : 'replies'}
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default AITeachingAssistant;