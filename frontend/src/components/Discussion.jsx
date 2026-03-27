import { useState } from 'react';
import { Send } from 'lucide-react';

const initialThreads = [
  { id: 1, author: 'Aisha Patel', role: 'student', message: 'Can someone explain the difference between INNER JOIN and LEFT JOIN?', time: '2 hours ago', replies: 3 },
  { id: 2, author: 'You', role: 'faculty', message: 'Great question! INNER JOIN returns only matching rows, while LEFT JOIN returns all rows from the left table...', time: '1 hour ago', replies: 0 },
  { id: 3, author: 'David Kim', role: 'student', message: 'Is the assignment due tonight or tomorrow night?', time: '30 min ago', replies: 1 },
];

export default function Discussion() {
  const [threads, setThreads] = useState(initialThreads);
  const [message, setMessage] = useState('');

  const handlePost = () => {
    if (!message.trim()) return;
    setThreads([...threads, {
      id: Date.now(),
      author: 'You',
      role: 'faculty',
      message,
      time: 'just now',
      replies: 0,
    }]);
    setMessage('');
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 space-y-6">
      <h2 className="text-xl font-bold text-gray-900">Course Discussion</h2>

      {/* Post Box */}
      <div className="p-4 bg-gray-50 rounded-lg">
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="Post a message to the class..."
          className="w-full h-24 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        />
        <div className="flex justify-end mt-3">
          <button
            onClick={handlePost}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
          >
            <Send className="w-4 h-4" /> Post
          </button>
        </div>
      </div>

      {/* Threads */}
      <div className="space-y-4">
        {threads.map(thread => (
          <div key={thread.id} className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center text-gray-700 font-semibold">
                {thread.author.split(' ').map(n => n[0]).join('')}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-gray-900">{thread.author}</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    thread.role === 'faculty' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
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
  );
}