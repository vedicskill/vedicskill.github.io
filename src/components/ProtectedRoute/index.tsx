import React from 'react';
import { useAuth } from '../../context/AuthContext';
import Link from '@docusaurus/Link';

interface Props {
  children: React.ReactNode;
}

export default function ProtectedRoute({ children }: Props) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) return <div>Loading authentication...</div>;

  if (!isAuthenticated) {
    const redirect = typeof window !== 'undefined' ? window.location.pathname + window.location.search : '/';
    return (
      <div style={{ padding: 24, border: '1px solid #eee', borderRadius: 6 }}>
        <h2>🔒 Members Only Content</h2>
        <p>This documentation requires login.</p>
        <div style={{ display: 'flex', gap: 12 }}>
          <Link href={`/login?redirect=${encodeURIComponent(redirect)}`} className="button button--primary">
            Login
          </Link>
          <Link href={`/signup?redirect=${encodeURIComponent(redirect)}`} className="button button--secondary">
            Signup
          </Link>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
