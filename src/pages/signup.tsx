import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import { useAuth } from '../context/AuthContext';
import styles from './login.module.css';
import { useLocation } from '@docusaurus/router';

export default function SignupPage(): JSX.Element {
  const { signup, isAuthenticated } = useAuth();
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const redirect = params.get('redirect') || '/';

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isAuthenticated) {
      if (typeof window !== 'undefined') window.location.href = redirect;
    }
  }, [isAuthenticated, redirect]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!name || !email || !password) return setError('All fields are required');
    if (password !== confirm) return setError('Passwords do not match');

    setIsLoading(true);
    try {
      await signup(name, email, password);
    } catch (err: any) {
      setError(err?.message ?? 'Signup failed');
      setIsLoading(false);
    }
  };

  return (
    <Layout title="Signup" description="Create an account">
      <main className={styles.loginContainer}>
        <div className={styles.loginBox}>
          <div className={styles.header}>
            <h1>Create Account</h1>
            <p>Join VedicSkill Academy</p>
          </div>

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.formGroup}>
              <label className={styles.label}>Name</label>
              <input value={name} onChange={(e) => setName(e.target.value)} className={styles.input} />
            </div>
            <div className={styles.formGroup}>
              <label className={styles.label}>Email</label>
              <input value={email} onChange={(e) => setEmail(e.target.value)} className={styles.input} />
            </div>
            <div className={styles.formGroup}>
              <label className={styles.label}>Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className={styles.input} />
            </div>
            <div className={styles.formGroup}>
              <label className={styles.label}>Confirm Password</label>
              <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} className={styles.input} />
            </div>

            {error && (
              <div className={styles.errorMessage}>
                <span className={styles.errorIcon}>⚠️</span>
                {error}
              </div>
            )}

            <button className={styles.submitButton} disabled={isLoading} type="submit">
              {isLoading ? 'Creating account...' : 'Create account'}
            </button>
          </form>
        </div>
      </main>
    </Layout>
  );
}
