import { useState, useRef, useEffect } from 'react';
import { Send, Bot, ChevronDown, ChevronUp } from 'lucide-react';

const STUDENTS = [
  'Mathesh Rames',
  'Kaviarasu Annadurai',
  'Anjana Deivasigamani',
  'Raghu Ram Baskaran',
  'Priyadharshan Sengutuvan',
];

const initialThreads = [
  {
    id: 1,
    author: 'Mathesh Rames',
    role: 'student',
    message: 'Can someone explain the difference between INNER JOIN and LEFT JOIN?',
    time: '2 hours ago',
    replies: [
      {
        id: 11,
        author: 'Prof. Priyadharshan',
        role: 'faculty',
        message: 'INNER JOIN returns only matching rows from both tables. LEFT JOIN returns all rows from the left table and matching rows from the right — unmatched rows get NULL.',
        time: '1 hour ago',
      },
    ],
  },
  {
    id: 2,
    author: 'Kaviarasu Annadurai',
    role: 'student',
    message: 'Is the assignment due tonight or tomorrow night?',
    time: '30 min ago',
    replies: [],
  },
];

// Shared thread state (simulates a backend — both views use this)
let sharedThreads = initialThreads;
const listeners = new Set();

function useSharedThreads() {
  const [threads, setThreads] = useState(sharedThreads);
  useEffect(() => {
    const update = (t) => setThreads([...t]);
    listeners.add(update);
    return () => listeners.delete(update);
  }, []);
  return [threads, (t) => { sharedThreads = t; listeners.forEach(fn => fn(t)); }];
}

async function getAIReply(question) {
  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1000,
        system: 'You are a helpful teaching assistant for a Database Systems course at Northeastern University. Answer student questions concisely and clearly based on database systems concepts.',
        messages: [{ role: 'user', content: question }],
      }),
    });
    const data = await res.json();
    return data.content?.[0]?.text || 'Sorry, I could not generate a reply.';
  } catch {
    return 'AI assistant is currently unavailable.';
  }
}

function ThreadCard({ thread, onReply, role, authorName }) {
  const [open, setOpen] = useState(false);
  const [replyText, setReplyText] = useState('');
  const [aiLoading, setAiLoading] = useState(false);

  const handleReply = () => {
    if (!replyText.trim()) return;
    onReply(thread.id, replyText, authorName, role);
    setReplyText('');
  };

  const handleAIReply = async () => {
    setAiLoading(true);
    const answer = await getAIReply(thread.message);
    onReply(thread.id, answer, 'AI Assistant', 'ai');
    setAiLoading(false);
  };

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Main post */}
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
                thread.role === 'ai' ? 'bg-green-100 text-green-700' :
                'bg-blue-100 text-blue-700'
              }`}>{thread.role}</span>
              <span className="text-xs text-gray-400">• {thread.time}</span>
            </div>
            <p className="text-gray-700 text-sm">{thread.message}</p>
            <button
              onClick={() => setOpen(!open)}
              className="mt-2 flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium"
            >
              {thread.replies.length > 0 ? (
                <>{open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />} {thread.replies.length} {thread.replies.length === 1 ? 'reply' : 'replies'}</>
              ) : (
                <>{open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />} Reply</>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Replies + reply box */}
      {open && (
        <div className="border-t border-gray-100 bg-gray-50 px-4 py-3 space-y-3">
          {thread.replies.map(r => (
            <div key={r.id} className="flex items-start gap-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-semibold flex-shrink-0 ${
                r.role === 'faculty' ? 'bg-purple-500' :
                r.role === 'ai' ? 'bg-green-500' :
                'bg-blue-500'
              }`}>
                {r.role === 'ai' ? <Bot className="w-4 h-4" /> : r.author.split(' ').map(n => n[0]).join('')}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-sm font-semibold text-gray-900">{r.author}</span>
                  <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                    r.role === 'faculty' ? 'bg-purple-100 text-purple-700' :
                    r.role === 'ai' ? 'bg-green-100 text-green-700' :
                    'bg-blue-100 text-blue-700'
                  }`}>{r.role}</span>
                  <span className="text-xs text-gray-400">• {r.time}</span>
                </div>
                <p className="text-sm text-gray-700">{r.message}</p>
              </div>
            </div>
          ))}

          {/* Reply input */}
          <div className="flex gap-2 pt-1">
            <input
              type="text"
              value={replyText}
              onChange={e => setReplyText(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleReply()}
              placeholder="Write a reply..."
              className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
            <button onClick={handleReply}
              className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm flex items-center gap-1">
              <Send className="w-3 h-3" /> Reply
            </button>
            {role === 'faculty' && (
              <button onClick={handleAIReply} disabled={aiLoading}
                className="px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm flex items-center gap-1 disabled:opacity-50">
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
  const [threads, setThreads] = useSharedThreads();
  const [message, setMessage] = useState('');

  const authorName = role === 'faculty' ? 'Prof. Priyadharshan' : studentName;

  const handlePost = () => {
    if (!message.trim()) return;
    const newThread = {
      id: Date.now(),
      author: authorName,
      role,
      message,
      time: 'just now',
      replies: [],
    };
    setThreads([newThread, ...threads]);
    setMessage('');
  };

  const handleReply = (threadId, text, author, replyRole) => {
    const updated = threads.map(t =>
      t.id === threadId
        ? { ...t, replies: [...t.replies, { id: Date.now(), author, role: replyRole, message: text, time: 'just now' }] }
        : t
    );
    setThreads(updated);
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

      {/* Post box */}
      <div className="p-4 bg-gray-50 rounded-lg">
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder={role === 'faculty' ? 'Post an announcement to the class...' : 'Ask a question...'}
          className="w-full h-20 px-4 py-3 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        />
        <div className="flex justify-end mt-2">
          <button onClick={handlePost}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 text-sm">
            <Send className="w-4 h-4" /> {role === 'faculty' ? 'Post' : 'Ask'}
          </button>
        </div>
      </div>

      {/* Threads */}
      <div className="space-y-3">
        {threads.map(t => (
          <ThreadCard key={t.id} thread={t} onReply={handleReply} role={role} authorName={authorName} />
        ))}
      </div>
    </div>
  );
}