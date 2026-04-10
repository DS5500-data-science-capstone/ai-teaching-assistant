import { useState, useEffect } from 'react';
import { Send, Bot, ChevronDown, ChevronUp } from 'lucide-react';

const API = '/api';
const POLL_MS = 3000;

function ThreadCard({ thread, role, authorName, onThreadsUpdate }) {
  const [open, setOpen] = useState(false);
  const [replyText, setReplyText] = useState('');
  const [aiLoading, setAiLoading] = useState(false);

  const handleReply = async (text, author, replyRole) => {
    if (!text.trim()) return;
    await fetch(`${API}/discussion/${thread.id}/reply`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ author, role: replyRole, message: text }),
    });
    onThreadsUpdate();
    setReplyText('');
  };

  const handleAIReply = async () => {
    setAiLoading(true);
    try {
      const res = await fetch(`${API}/discussion/ai-reply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: thread.id, question: thread.message }),
      });
      const data = await res.json();
      console.log('AI reply response:', res.status, data);
      onThreadsUpdate();
    } catch (err) {
      console.error('AI reply error:', err);
    }
    setAiLoading(false);
  };

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <div className="p-4 hover:bg-gray-50 transition-colors">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center text-gray-700 font-semibold text-sm flex-shrink-0">
            {thread.author.split(' ').map(n => n[0]).join('')}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <span className="font-semibold text-gray-900">{thread.author}</span>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                thread.role === 'faculty' ? 'bg-purple-100 text-purple-700' :
                thread.role === 'ai'      ? 'bg-green-100 text-green-700' :
                                            'bg-blue-100 text-blue-700'
              }`}>{thread.role}</span>
              <span className="text-xs text-gray-400">• {thread.time}</span>
            </div>
            <p className="text-gray-700 text-sm">{thread.message}</p>
            <button
              onClick={() => setOpen(!open)}
              className="mt-2 flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium"
            >
              {open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {thread.replies.length > 0
                ? `${thread.replies.length} ${thread.replies.length === 1 ? 'reply' : 'replies'}`
                : 'Reply'}
            </button>
          </div>
        </div>
      </div>

      {open && (
        <div className="border-t border-gray-100 bg-gray-50 px-4 py-3 space-y-3">
          {thread.replies.map(r => (
            <div key={r.id} className="flex items-start gap-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-semibold flex-shrink-0 ${
                r.role === 'faculty' ? 'bg-purple-500' :
                r.role === 'ai'      ? 'bg-green-500' :
                                       'bg-blue-500'
              }`}>
                {r.role === 'ai' ? <Bot className="w-4 h-4" /> : r.author.split(' ').map(n => n[0]).join('')}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-sm font-semibold text-gray-900">{r.author}</span>
                  <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                    r.role === 'faculty' ? 'bg-purple-100 text-purple-700' :
                    r.role === 'ai'      ? 'bg-green-100 text-green-700' :
                                           'bg-blue-100 text-blue-700'
                  }`}>{r.role}</span>
                  <span className="text-xs text-gray-400">• {r.time}</span>
                </div>
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{r.message}</p>
              </div>
            </div>
          ))}

          <div className="flex gap-2 pt-1">
            <input
              type="text"
              value={replyText}
              onChange={e => setReplyText(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleReply(replyText, authorName, role)}
              placeholder="Write a reply..."
              className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={() => handleReply(replyText, authorName, role)}
              className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm flex items-center gap-1"
            >
              <Send className="w-3 h-3" /> Reply
            </button>
            {role === 'faculty' && (
              <button
                onClick={handleAIReply}
                disabled={aiLoading}
                className="px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm flex items-center gap-1 disabled:opacity-50"
              >
                <Bot className="w-3 h-3" /> {aiLoading ? 'Thinking...' : 'AI Reply'}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Discussion({ role = 'faculty', studentName = '' }) {
  const [threads, setThreads] = useState([]);
  const [message, setMessage] = useState('');

  const authorName = role === 'faculty' ? 'Prof. Priyadharshan' : studentName;

  const fetchThreads = async () => {
    try {
      const res = await fetch(`${API}/discussion`);
      const data = await res.json();
      setThreads(data.threads || []);
    } catch {
      // backend unreachable — keep current state
    }
  };

  // initial fetch + polling
  useEffect(() => {
    fetchThreads();
    const id = setInterval(fetchThreads, POLL_MS);
    return () => clearInterval(id);
  }, []);

  const handlePost = async () => {
    if (!message.trim()) return;
    await fetch(`${API}/discussion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ author: authorName, role, message }),
    });
    setMessage('');
    fetchThreads();
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-900">Course Discussion</h2>
        <span className={`px-3 py-1 rounded-full text-xs font-medium ${
          role === 'faculty' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
        }`}>
          {role === 'faculty' ? 'Faculty View' : `Student View — ${studentName}`}
        </span>
      </div>

      <div className="p-4 bg-gray-50 rounded-lg">
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder={role === 'faculty' ? 'Post an announcement to the class...' : 'Ask a question...'}
          className="w-full h-20 px-4 py-3 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        />
        <div className="flex justify-end mt-2">
          <button
            onClick={handlePost}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 text-sm"
          >
            <Send className="w-4 h-4" /> {role === 'faculty' ? 'Post' : 'Ask'}
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {threads.length === 0
          ? <p className="text-sm text-gray-400 text-center py-4">No posts yet.</p>
          : threads.map(t => (
              <ThreadCard
                key={t.id}
                thread={t}
                role={role}
                authorName={authorName}
                onThreadsUpdate={fetchThreads}
              />
            ))
        }
      </div>
    </div>
  );
}