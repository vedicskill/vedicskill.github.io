import React, { useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useLocation } from '@docusaurus/router';
import styles from './ProtectedContent.module.css';

interface ProtectedContentProps {
  children: React.ReactNode;
  requiresLogin?: boolean;
}

/**
 * ProtectedContent Component
 * 
 * Wraps content that should be protected by authentication.
 * - If requiresLogin is false or not set, renders children normally
 * - If requiresLogin is true and user is not authenticated, shows login prompt
 * - If requiresLogin is true and user is authenticated, renders children
 * 
 * Usage:
 * <ProtectedContent requiresLogin={true}>
 *   <MyProtectedComponent />
 * </ProtectedContent>
 */
export const ProtectedContent: React.FC<ProtectedContentProps> = ({
  children,
  requiresLogin = false,
}) => {
  const { isAuthenticated, isLoading, setRedirectPath } = useAuth();
  const location = useLocation();

  /**
   * Store the current path for redirect after login
   */
  useEffect(() => {
    if (requiresLogin && !isAuthenticated && !isLoading) {
      setRedirectPath(location.pathname);
    }
  }, [requiresLogin, isAuthenticated, isLoading, location.pathname, setRedirectPath]);

  /**
   * Show loading state while auth is initializing
   */
  if (isLoading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Loading...</p>
      </div>
    );
  }

  /**
   * If content doesn't require login, show it
   */
  if (!requiresLogin) {
    return <>{children}</>;
  }

  /**
   * If content requires login and user is authenticated, show it
   */
  if (isAuthenticated) {
    return <>{children}</>;
  }

  /**
   * If content requires login and user is NOT authenticated, show login prompt
   */
  return (
    <div className={styles.loginPrompt}>
      <div className={styles.lockIcon}>🔒</div>
      <h2>Login Required</h2>
      <p>This documentation is restricted and requires authentication to access.</p>
      <p>Please log in with your credentials to continue.</p>
      
      <button
        className={styles.loginButton}
        onClick={() => {
          if (typeof window !== 'undefined') window.location.href = '/login';
        }}
      >
        Go to Login
      </button>

      <div className={styles.helpText}>
        <p>
          <strong>Demo credentials:</strong>
          <br />
          Username: <code>admin</code> or <code>user</code>
          <br />
          Password: any value
        </p>
      </div>
    </div>
  );
};

export default ProtectedContent;
