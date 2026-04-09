// Shared store using localStorage so faculty and student tabs stay in sync

const QUIZ_KEY   = 'ait_pushed_quizzes';
const RESULT_KEY = 'ait_quiz_results';

export const STUDENTS = [
  { id: 1, name: 'Mathesh Rames',           email: 'rames.m@northeastern.edu',         grade: 45, attendance: 60, assignments: 3 },
  { id: 2, name: 'Kaviarasu Annadurai',      email: 'annadurai.k@northeastern.edu',     grade: 52, attendance: 75, assignments: 4 },
  { id: 3, name: 'Anjana Deivasigamani',     email: 'deivasigamani.a@northeastern.edu', grade: 67, attendance: 85, assignments: 6 },
  { id: 4, name: 'Raghu Ram Baskaran',       email: 'baskaran.r@northeastern.edu',      grade: 78, attendance: 90, assignments: 7 },
  { id: 5, name: 'Priyadharshan Sengutuvan', email: 'sengutuvan.p@northeastern.edu',    grade: 85, attendance: 95, assignments: 8 },
];

function load(key) {
  try { return JSON.parse(localStorage.getItem(key) || '[]'); } catch { return []; }
}

function save(key, data) {
  localStorage.setItem(key, JSON.stringify(data));
  // Notify other tabs
  window.dispatchEvent(new StorageEvent('storage', { key }));
}

export let pushedQuizzes = load(QUIZ_KEY);
export let quizResults   = load(RESULT_KEY);

const quizListeners   = new Set();
const resultListeners = new Set();

// Listen for changes from other tabs
window.addEventListener('storage', (e) => {
  if (e.key === QUIZ_KEY) {
    pushedQuizzes = load(QUIZ_KEY);
    quizListeners.forEach(fn => fn(pushedQuizzes));
  }
  if (e.key === RESULT_KEY) {
    quizResults = load(RESULT_KEY);
    resultListeners.forEach(fn => fn(quizResults));
  }
});

export function subscribeQuizzes(fn)  { quizListeners.add(fn);   return () => quizListeners.delete(fn); }
export function subscribeResults(fn)  { resultListeners.add(fn); return () => resultListeners.delete(fn); }

export function pushQuiz(quiz) {
  pushedQuizzes = [...load(QUIZ_KEY), quiz];
  save(QUIZ_KEY, pushedQuizzes);
  quizListeners.forEach(fn => fn(pushedQuizzes));
}

export function submitQuizResult(result) {
  const existing = load(RESULT_KEY);
  quizResults = [
    ...existing.filter(r => !(r.studentName === result.studentName && r.quizId === result.quizId)),
    result,
  ];
  save(RESULT_KEY, quizResults);
  resultListeners.forEach(fn => fn(quizResults));
}

export function getStudentResults(studentName) {
  return load(RESULT_KEY).filter(r => r.studentName === studentName);
}

export function getQuizResults(quizId) {
  return load(RESULT_KEY).filter(r => r.quizId === quizId);
}