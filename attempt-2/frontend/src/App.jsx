import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, Briefcase, GraduationCap, Sparkles, ChevronRight, CheckCircle2 } from 'lucide-react';
import './index.css';

const API_URL = 'http://127.0.0.1:5000/api';

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [quizData, setQuizData] = useState(null);
  const [formData, setFormData] = useState({
    skills: [],
    personalities: [],
    expected_salary: ''
  });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch quiz data in the background
    fetch(`${API_URL}/quiz`)
      .then(res => res.json())
      .then(data => setQuizData(data))
      .catch(err => console.error("Error fetching quiz data:", err));
  }, []);

  const handleStart = () => setCurrentPage('quiz');
  
  const handleQuizSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          skills: formData.skills,
          personalities: formData.personalities,
          expected_salary: formData.expected_salary || 0
        })
      });
      if (!res.ok) throw new Error('Failed to get prediction');
      const data = await res.json();
      setResult(data);
      setCurrentPage('result');
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const pageVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
    exit: { opacity: 0, y: -20, transition: { duration: 0.3 } }
  };

  return (
    <div className="app-container" style={{ minHeight: '100vh', padding: '2rem' }}>
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '3rem' }}>
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          style={{ display: 'flex', alignItems: 'center', gap: '10px' }}
        >
          <div style={{ background: 'var(--primary)', padding: '12px', borderRadius: '14px', color: 'white' }}>
            <GraduationCap size={28} />
          </div>
          <h1 style={{ fontSize: '2rem', margin: 0 }}>
            Career<span className="gradient-text">Craft</span>
          </h1>
        </motion.div>
      </header>

      <main style={{ maxWidth: '900px', margin: '0 auto' }}>
        <AnimatePresence mode="wait">
          {currentPage === 'home' && (
            <motion.div 
              key="home"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="glass-card"
              style={{ padding: '4rem 2rem', textAlign: 'center' }}
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
              >
                <Sparkles size={48} color="var(--primary)" style={{ marginBottom: '1.5rem' }} />
              </motion.div>
              <h2 style={{ fontSize: '3rem', marginBottom: '1.5rem', lineHeight: 1.2 }}>
                Discover your true <br/>
                <span className="gradient-text">Professional Calling</span>
              </h2>
              <p style={{ fontSize: '1.2rem', color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto 3rem', lineHeight: 1.6 }}>
                Leverage AI-driven insights to match your unique skills and personality with the perfect career trajectory.
              </p>
              <button className="btn-primary" onClick={handleStart} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', fontSize: '1.2rem', padding: '16px 36px' }}>
                Take the Quiz <ArrowRight size={20} />
              </button>
            </motion.div>
          )}

          {currentPage === 'quiz' && (
            <motion.div 
              key="quiz"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              {!quizData ? (
                <div className="loader-container">
                  <div className="spinner"></div>
                </div>
              ) : (
                <div className="glass-card" style={{ padding: '3rem' }}>
                  <h2 style={{ fontSize: '2.2rem', marginBottom: '2rem', textAlign: 'center' }}>Tell us about yourself</h2>
                  <form onSubmit={handleQuizSubmit}>
                    
                    {/* Skills Section */}
                    <div style={{ marginBottom: '2.5rem' }}>
                      <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Briefcase size={20} className="gradient-text" /> 
                        Select your skills (up to 10)
                      </h3>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                        {quizData.skills.map((skill) => (
                          <div key={skill}>
                            <input 
                              type="checkbox" 
                              id={`skill-${skill}`} 
                              className="skill-checkbox"
                              checked={formData.skills.includes(skill)}
                              onChange={(e) => {
                                const checked = e.target.checked;
                                setFormData(prev => {
                                  if (checked && prev.skills.length < 10) {
                                    return { ...prev, skills: [...prev.skills, skill] };
                                  } else if (!checked) {
                                    return { ...prev, skills: prev.skills.filter(s => s !== skill) };
                                  }
                                  return prev;
                                });
                              }}
                            />
                            <label htmlFor={`skill-${skill}`} className="skill-label">
                              {skill}
                            </label>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Personality Section */}
                    <div style={{ marginBottom: '2.5rem' }}>
                      <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Sparkles size={20} className="gradient-text" /> 
                        Personality Traits (up to 3)
                      </h3>
                      <div style={{ background: 'rgba(255,255,255,0.5)', padding: '15px', borderRadius: '12px', marginBottom: '15px', fontSize: '0.9rem', color: 'var(--text-muted)', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
                        {Object.entries(quizData.personality_meanings).map(([key, value]) => (
                          <div key={key}><strong>{key}</strong>: {value}</div>
                        ))}
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                        {quizData.personalities.map((p) => (
                          <div key={p}>
                            <input 
                              type="checkbox" 
                              id={`p-${p}`} 
                              className="skill-checkbox"
                              checked={formData.personalities.includes(p)}
                              onChange={(e) => {
                                const checked = e.target.checked;
                                setFormData(prev => {
                                  if (checked && prev.personalities.length < 3) {
                                    return { ...prev, personalities: [...prev.personalities, p] };
                                  } else if (!checked) {
                                    return { ...prev, personalities: prev.personalities.filter(item => item !== p) };
                                  }
                                  return prev;
                                });
                              }}
                            />
                            <label htmlFor={`p-${p}`} className="skill-label">
                              {p}
                            </label>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Salary Section */}
                    <div style={{ marginBottom: '3rem' }}>
                      <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>Expected Salary (INR)</h3>
                      <input 
                        type="number" 
                        className="custom-input"
                        placeholder="e.g. 800000"
                        value={formData.expected_salary}
                        onChange={(e) => setFormData(prev => ({ ...prev, expected_salary: e.target.value }))}
                        required
                      />
                    </div>

                    {error && <div style={{ color: '#ef4444', marginBottom: '1.5rem', textAlign: 'center', padding: '12px', background: '#fee2e2', borderRadius: '8px' }}>{error}</div>}

                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                      <button type="submit" className="btn-primary" disabled={isLoading} style={{ width: '100%', maxWidth: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
                        {isLoading ? <div className="spinner" style={{ width: '24px', height: '24px', borderWidth: '3px' }}></div> : (
                          <>Discover Path <ChevronRight size={20} /></>
                        )}
                      </button>
                    </div>
                  </form>
                </div>
              )}
            </motion.div>
          )}

          {currentPage === 'result' && result && (
            <motion.div 
              key="result"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="glass-card"
              style={{ padding: '3rem' }}
            >
              <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
                <div style={{ display: 'inline-flex', background: '#ecfdf5', padding: '16px', borderRadius: '50%', marginBottom: '1.5rem' }}>
                  <CheckCircle2 size={40} color="#10b981" />
                </div>
                <h2 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Your Career Match</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>Based on your skills and personality</p>
              </div>

              <div style={{ display: 'grid', gap: '2rem', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
                {/* Best Match */}
                <div style={{ background: 'linear-gradient(135deg, rgba(79,70,229,0.05), rgba(79,70,229,0.15))', padding: '2rem', borderRadius: '20px', border: '1px solid rgba(79,70,229,0.2)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                    <h3 style={{ fontSize: '1.8rem', color: 'var(--primary)' }}>{result.career_match}</h3>
                    <span style={{ background: 'white', padding: '6px 12px', borderRadius: '20px', fontSize: '0.9rem', fontWeight: 'bold', color: 'var(--primary)', boxShadow: '0 2px 5px rgba(0,0,0,0.05)' }}>
                      {result.confidence_best} Match
                    </span>
                  </div>
                  
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ fontSize: '1.1rem', marginBottom: '0.5rem', color: 'var(--text-main)' }}>Top Companies</h4>
                    <p style={{ color: 'var(--text-muted)' }}>{result.top_companies_best.join(', ') || 'Various companies'}</p>
                  </div>

                  {result.missing_skills_best.length > 0 && (
                    <div>
                      <h4 style={{ fontSize: '1.1rem', marginBottom: '0.8rem', color: 'var(--text-main)' }}>Skills to Learn</h4>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                        {result.missing_skills_best.map(skill => (
                          <li key={skill} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid rgba(0,0,0,0.05)' }}>
                            <span style={{ color: 'var(--text-muted)' }}>{skill}</span>
                            {result.youtube_links_best[skill] && (
                              <a href={result.youtube_links_best[skill]} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', textDecoration: 'none', fontSize: '0.9rem', fontWeight: '500' }}>
                                Tutorial ↗
                              </a>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Alt Match */}
                <div style={{ background: 'white', padding: '2rem', borderRadius: '20px', border: '1px solid var(--card-border)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                    <h3 style={{ fontSize: '1.5rem', color: 'var(--text-main)' }}>{result.alt_career}</h3>
                    <span style={{ background: '#f1f5f9', padding: '6px 12px', borderRadius: '20px', fontSize: '0.9rem', fontWeight: 'bold', color: 'var(--text-muted)' }}>
                      {result.confidence_alt} Match
                    </span>
                  </div>
                  
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ fontSize: '1.1rem', marginBottom: '0.5rem', color: 'var(--text-main)' }}>Top Companies</h4>
                    <p style={{ color: 'var(--text-muted)' }}>{result.top_companies_alt.join(', ') || 'Various companies'}</p>
                  </div>

                  {result.missing_skills_alt.length > 0 && (
                    <div>
                      <h4 style={{ fontSize: '1.1rem', marginBottom: '0.8rem', color: 'var(--text-main)' }}>Skills to Learn</h4>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                        {result.missing_skills_alt.map(skill => (
                          <li key={skill} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid rgba(0,0,0,0.05)' }}>
                            <span style={{ color: 'var(--text-muted)' }}>{skill}</span>
                            {result.youtube_links_alt[skill] && (
                              <a href={result.youtube_links_alt[skill]} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', textDecoration: 'none', fontSize: '0.9rem', fontWeight: '500' }}>
                                Tutorial ↗
                              </a>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', marginTop: '3rem' }}>
                <button 
                  className="btn-secondary" 
                  onClick={() => {
                    setFormData({ skills: [], personalities: [], expected_salary: '' });
                    setCurrentPage('home');
                  }}
                >
                  Retake Quiz
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
