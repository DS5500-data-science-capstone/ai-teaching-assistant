import { useState } from 'react';
import { Book, Users, FileText, MessageCircle, LogOut } from 'lucide-react';
import Dashboard from './components/Dashboard';
import Students from './components/Students';
import Lectures from './components/Lectures';
import Discussion from './components/Discussion';

const students = [
  { id: 1, name: 'Sarah Johnson', email: 'johnson.s@northeastern.edu', grade: 45, needsAttention: true, lastActive: '2 days ago', assignments: 3, attendance: 60 },
  { id: 2, name: 'Michael Chen', email: 'chen.m@northeastern.edu', grade: 52, needsAttention: true, lastActive: '5 hours ago', assignments: 4, attendance: 75 },
  { id: 3, name: 'Emily Rodriguez', email: 'rodriguez.e@northeastern.edu', grade: 67, needsAttention: false, lastActive: '1 hour ago', assignments: 6, attendance: 85 },
  { id: 4, name: 'David Kim', email: 'kim.d@northeastern.edu', grade: 78, needsAttention: false, lastActive: '30 min ago', assignments: 7, attendance: 90 },
  { id: 5, name: 'Aisha Patel', email: 'patel.a@northeastern.edu', grade: 85, needsAttention: false, lastActive: '15 min ago', assignments: 8, attendance: 95 },
  { id: 6, name: 'James Wilson', email: 'wilson.j@northeastern.edu', grade: 92, needsAttention: false, lastActive: '10 min ago', assignments: 8, attendance: 100 },
];

const tabs = [
  { id: 'dashboard', label: 'Dashboard', icon: Book },
  { id: 'students',  label: 'Students',  icon: Users },
  { id: 'lectures',  label: 'Lectures',  icon: FileText },
  { id: 'discussion',label: 'Discussion',icon: MessageCircle },
];

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [activeTab, setActiveTab] = useState('dashboard');

  const handleLogin = (e) => {
    e.preventDefault();
    if (loginEmail && loginPassword) setIsLoggedIn(true);
  };

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600">
        <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Teaching Assistant</h1>
            <p className="text-gray-600">Northeastern University</p>
          </div>
          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
              <input
                type="email"
                value={loginEmail}
                onChange={e => setLoginEmail(e.target.value)}
                placeholder="faculty@northeastern.edu"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
              <input
                type="password"
                value={loginPassword}
                onChange={e => setLoginPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            <button type="submit" className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 font-semibold">
              Sign In
            </button>
            <p className="text-center text-sm text-gray-600">Demo: Use any email and password</p>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">AI Teaching Assistant</h1>
            <p className="text-sm text-gray-600">Northeastern University</p>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">Prof. Sarah Martinez</span>
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-semibold">SM</div>
            <button onClick={() => setIsLoggedIn(false)} className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg" title="Logout">
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Nav */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4">
          <nav className="flex gap-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                  activeTab === tab.id ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-600 hover:text-gray-900'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'dashboard'  && <Dashboard students={students} documents={[]} onContactStudent={() => setActiveTab('students')} />}
        {activeTab === 'students'   && <Students students={students} />}
        {activeTab === 'lectures'   && <Lectures />}
        {activeTab === 'discussion' && <Discussion />}
      </div>
    </div>
  );
}