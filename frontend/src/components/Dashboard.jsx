import { AlertCircle, Book, Clock } from 'lucide-react';

const courseData = {
  courseNumber: 'CS 5200',
  courseName: 'Database Management Systems',
  semester: 'Spring 2025',
  schedule: 'Mon/Wed 6:00 PM - 7:30 PM',
  room: 'Snell Library 453'
};

export default function Dashboard({ students, documents, onContactStudent }) {
  const atRisk = students.filter(s => s.needsAttention);
  const avgGrade = Math.round(students.reduce((a, s) => a + s.grade, 0) / students.length);

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

      {/* At-Risk Students */}
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
    </div>
  );
}