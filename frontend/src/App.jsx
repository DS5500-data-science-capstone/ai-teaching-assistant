import { useState } from 'react';
import { Book, Users, FileText, MessageCircle, LogOut, Brain, Menu, X } from 'lucide-react';
import Dashboard from './components/Dashboard';
import Students from './components/Students';
import Lectures from './components/Lectures';
import Discussion from './components/Discussion';
import Quiz from './components/Quiz';
import StudentView from './components/StudentView';
import CoursePlanner from './components/CoursePlanner';
import { Presentation } from 'lucide-react'; 
import Rubrics from './components/Rubrics';
import { ClipboardList } from 'lucide-react';

const students = [
  { id: 1, name: 'Mathesh Ramesh', email: 'rames.m@northeastern.edu', grade: 45, needsAttention: true, lastActive: '2 days ago', assignments: 3, attendance: 60 },
  { id: 2, name: 'Kaviarasu Annadurai', email: 'annadurai.k@northeastern.edu', grade: 52, needsAttention: true, lastActive: '5 hours ago', assignments: 4, attendance: 75 },
  { id: 3, name: 'Anjana Deivasigamani', email: 'deivasigamani.a@northeastern.edu', grade: 67, needsAttention: false, lastActive: '1 hour ago', assignments: 6, attendance: 85 },
  { id: 4, name: 'Raghu Ram Baskaran', email: 'baskaran.r@northeastern.edu', grade: 78, needsAttention: false, lastActive: '30 min ago', assignments: 7, attendance: 90 },
  { id: 5, name: 'Priyadharshan Sengutuvan', email: 'sengutuvan.p@northeastern.edu', grade: 85, needsAttention: false, lastActive: '15 min ago', assignments: 8, attendance: 95 },
];

const tabs = [
  { id: 'dashboard',  label: 'Dashboard',    icon: Book },
  { id: 'students',   label: 'Students',     icon: Users },
  { id: 'lectures',   label: 'Lectures',     icon: FileText },
  { id: 'planner', label: 'Course Planner', icon: Presentation },
  { id: 'discussion', label: 'Q&A', icon: MessageCircle },
  { id: 'quiz',       label: 'Quiz Builder', icon: Brain },
  { id: 'rubrics', label: 'Rubrics', icon: ClipboardList },
];

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(() => sessionStorage.getItem('loggedIn') === 'true');
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [menuOpen, setMenuOpen] = useState(false);

  const handleLogin = (e) => {
    e.preventDefault();
    if (loginEmail && loginPassword) {
      sessionStorage.setItem('loggedIn', 'true');
      setIsLoggedIn(true);
    }
  };

  const handleLogout = () => {
    sessionStorage.removeItem('loggedIn');
    setIsLoggedIn(false);
  };

  const handleTabChange = (id) => {
    setActiveTab(id);
    setMenuOpen(false);
  };

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600 px-4">
        <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md">
          <div className="text-center mb-8">
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-2">AI Teaching Assistant</h1>
            <p className="text-gray-600">Northeastern University</p>
          </div>
          <form onSubmit={handleLogin} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
              <input type="email" value={loginEmail} onChange={e => setLoginEmail(e.target.value)}
                placeholder="faculty@northeastern.edu"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
              <input type="password" value={loginPassword} onChange={e => setLoginPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500" required />
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
      <div className="bg-white shadow-sm border-b sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg sm:text-2xl font-bold text-gray-900 leading-tight">AI Teaching Assistant</h1>
            <p className="text-xs sm:text-sm text-gray-600">Northeastern University</p>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <span className="hidden sm:block text-sm text-gray-600">Prof. Priyadharshan</span>
            <div className="w-9 h-9 bg-blue-500 rounded-full flex items-center justify-center text-white font-semibold text-sm">PS</div>
            <button onClick={handleLogout} className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg" title="Logout">
              <LogOut className="w-5 h-5" />
            </button>
            <button onClick={() => setMenuOpen(!menuOpen)} className="sm:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-lg">
              {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Desktop Nav */}
        <div className="hidden sm:block bg-white border-t">
          <div className="max-w-7xl mx-auto px-4">
            <nav className="flex gap-1 overflow-x-auto">
              {tabs.map(tab => (
                <button key={tab.id} onClick={() => handleTabChange(tab.id)}
                  className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors whitespace-nowrap ${
                    activeTab === tab.id ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-600 hover:text-gray-900'
                  }`}>
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Mobile Nav */}
        {menuOpen && (
          <div className="sm:hidden border-t bg-white shadow-lg">
            {tabs.map(tab => (
              <button key={tab.id} onClick={() => handleTabChange(tab.id)}
                className={`w-full flex items-center gap-3 px-5 py-3 text-sm transition-colors ${
                  activeTab === tab.id ? 'bg-blue-50 text-blue-600 font-medium' : 'text-gray-700 hover:bg-gray-50'
                }`}>
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
            <button onClick={() => setStudentView(true)}
              className="w-full flex items-center gap-3 px-5 py-3 text-sm text-blue-600 hover:bg-blue-50">
              <MessageCircle className="w-4 h-4" /> Student View
            </button>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-3 sm:px-4 py-4 sm:py-6">
        {activeTab === 'dashboard'  && <Dashboard students={students} documents={[]} onContactStudent={() => setActiveTab('students')} />}
        {activeTab === 'students'   && <Students students={students} />}
        {activeTab === 'lectures'   && <Lectures />}
        {activeTab === 'planner' && <CoursePlanner />}
        {activeTab === 'discussion' && <Discussion role="faculty" />}
        {activeTab === 'quiz'       && <Quiz />}
        {activeTab === 'rubrics' && <Rubrics />}
      </div>
    </div>
  );
}