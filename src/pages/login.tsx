import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import Layout from '@theme/Layout';
import styles from './login.module.css';

/**
 * Login Page
 * 
 * Provides a login form for users to authenticate.
 * After successful login, redirects to the page they originally tried to access,
 * or the home page if no redirect path is set.
 */
export default function LoginPage(): JSX.Element {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const { login, isAuthenticated, getRedirectPath } = useAuth();

  /**
   * If already authenticated, redirect to the intended page
   */
  useEffect(() => {
    if (isAuthenticated) {
      const redirectPath = getRedirectPath();
      if (typeof window !== 'undefined') {
        window.location.href = redirectPath;
      }
    }
  }, [isAuthenticated, getRedirectPath]);

  /**
   * Handle form submission
   */
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await login(username, password);
      // Auth state update will trigger the useEffect above
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
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
              <label htmlFor="username" className={styles.label}>
                Username
              </label>
              <input
                id="username"
                type="text"
                placeholder="Enter your email"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
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
              disabled={isLoading || !username || !password}
            >
              {isLoading ? 'Logging in...' : 'Login'}
            </button>
          </form>

          <div className={styles.footer}>
            <h3>Demo Credentials</h3>
            <div className={styles.credentialsList}>
              <div className={styles.credential}>
                <strong>Admin:</strong>
                <div className={styles.credValue}>
                  Username: <code>admin</code>
                  <br />
                  Password: <code>any value</code>
                </div>
              </div>
              <div className={styles.credential}>
                <strong>User:</strong>
                <div className={styles.credValue}>
                  Username: <code>user</code>
                  <br />
                  Password: <code>any value</code>
                </div>
              </div>
            </div>
            <p className={styles.disclaimer}>
              💡 This is a demo environment. The password field accepts any value.
            </p>
          </div>
        </div>

        <div className={styles.infoBox}>
          <h2>Access Protected Documentation</h2>
          <p>
            Log in to access documentation marked as restricted. Your session will persist
            across browser refreshes.
          </p>
          <ul>
            <li>Secure authentication using localStorage</li>
            <li>Session persistence across page refreshes</li>
            <li>Automatic redirect after login</li>
            <li>Role-based access control ready</li>
          </ul>
        </div>
      </main>
    </Layout>
  );
}
