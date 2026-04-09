import { useState, useEffect } from 'react';
import { Upload, FileText, Download, Send } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export default function Lectures() {
  const [documents, setDocuments] = useState([]);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [asking, setAsking] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API}/documents`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch {
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.name.endsWith('.pdf')) {
      alert('Only PDF files are allowed.');
      return;
    }
    setUploadStatus('uploading');
    setUploadMessage(`Uploading ${file.name}...`);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API}/upload`, { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus('success');
        setUploadMessage(`"${file.name}" uploaded to GCS. RAG pipeline will auto-process it.`);
        fetchDocuments();
      } else {
        setUploadStatus('error');
        setUploadMessage(`Upload failed: ${data.detail}`);
      }
    } catch {
      setUploadStatus('error');
      setUploadMessage('Could not reach backend. Is FastAPI running on port 8000?');
    }
    e.target.value = '';
    setTimeout(() => setUploadStatus(null), 6000);
  };

  const handleAskRAG = async () => {
    if (!question.trim()) return;
    setAsking(true);
    setAnswer('');
    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setAnswer(data.answer || 'No answer returned.');
    } catch {
      setAnswer('Could not reach RAG backend.');
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload + Document List */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-gray-900">Course Documents (GCS)</h2>
          <label className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 cursor-pointer">
            <Upload className="w-4 h-4" />
            Upload PDF
            <input type="file" onChange={handleUpload} className="hidden" accept=".pdf" />
          </label>
        </div>

        {uploadStatus && (
          <div className={`mb-4 p-3 rounded-lg text-sm font-medium ${
            uploadStatus === 'uploading' ? 'bg-blue-50 text-blue-700' :
            uploadStatus === 'success' ? 'bg-green-50 text-green-700' :
            'bg-red-50 text-red-700'
          }`}>
            {uploadMessage}
          </div>
        )}

        {loading ? (
          <p className="text-gray-500 text-sm">Loading documents from GCS...</p>
        ) : documents.length === 0 ? (
          <p className="text-gray-500 text-sm">No documents uploaded yet.</p>
        ) : (
          <div className="space-y-3">
            {documents.map((doc, i) => (
              <div key={i} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center gap-3">
                  <FileText className="w-8 h-8 text-blue-500" />
                  <div>
                    <p className="font-medium text-gray-900">{doc.name}</p>
                    <p className="text-sm text-gray-600">Uploaded: {doc.uploadDate} • {doc.size}</p>
                  </div>
                </div>
                <a href={doc.url} target="_blank" rel="noreferrer" className="p-2 text-gray-600 hover:text-gray-900 rounded-lg hover:bg-gray-100">
                  <Download className="w-5 h-5" />
                </a>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* RAG Q&A */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-1">Query Course Knowledge Base</h2>
        <p className="text-sm text-gray-500 mb-4">
          Ask questions about uploaded course materials to quickly find content, prepare lecture answers, or verify what is in the knowledge base.
        </p>
        <div className="flex gap-3">
          <input
            type="text"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAskRAG()}
            placeholder="e.g. What does the syllabus say about late submissions?"
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleAskRAG}
            disabled={asking}
            className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
            {asking ? 'Asking...' : 'Ask'}
          </button>
        </div>
        {answer && (
          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg text-gray-800 text-sm whitespace-pre-wrap">
            {answer}
          </div>
        )}
      </div>
    </div>
  );
}