import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import { useAuth } from '../context/AuthContext';
import styles from './login.module.css';
import { useLocation } from '@docusaurus/router';

export default function LoginPage(): JSX.Element {
  const { login, isAuthenticated } = useAuth();
  const location = useLocation();

  const params = new URLSearchParams(location.search);
  const redirect = params.get('redirect') || '/';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
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
    setIsLoading(true);
    try {
      await login(email, password);
      // redirect will happen via isAuthenticated effect
    } catch (err: any) {
      setError(err?.message ?? 'Login failed');
      setIsLoading(false);
    }
  };

  return (
    <Layout title="Login" description="Login to access protected documentation">
      <main className={styles.loginContainer}>
        <div className={styles.loginBox}>
          <div className={styles.header}>
            <h1>VedicSkill Academy</h1>
            <p>Protected Documentation Portal</p>
          </div>

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.formGroup}>
              <label htmlFor="email" className={styles.label}>
                Email
              </label>
              <input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={styles.input}
                disabled={isLoading}
                autoFocus
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="password" className={styles.label}>
                Password
              </label>
              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={styles.input}
                disabled={isLoading}
              />
            </div>

            {error && (
              <div className={styles.errorMessage}>
                <span className={styles.errorIcon}>⚠️</span>
                {error}
              </div>
            )}

            <button
              type="submit"
              className={styles.submitButton}
              disabled={isLoading || !email || !password}
            >
              {isLoading ? 'Logging in...' : 'Login'}
            </button>
          </form>

          <div className={styles.footer}>
            <p className={styles.disclaimer}>
              💡 After login you will be redirected back to the page you requested.
            </p>
          </div>
        </div>

        <div className={styles.infoBox}>
          <h2>Access Protected Documentation</h2>
          <p>Log in to access documentation marked as restricted.</p>
        </div>
      </main>
    </Layout>
  );
}
