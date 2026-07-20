import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import Link from '@docusaurus/Link';

export default function AuthButtons() {
  const { isAuthenticated, profile, logout, loading } = useAuth();
  const [open, setOpen] = useState(false);

  if (loading) return null;

  if (!isAuthenticated) {
    return (
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <Link href="/login" className="button button--outline">
          Login
        </Link>
        <Link href="/signup" className="button button--primary">
          Signup
        </Link>
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      <button
        type="button"
        className="button"
        onClick={() => setOpen((s) => !s)}
        aria-haspopup="true"
        aria-expanded={open}
      >
        Hi {profile?.name ?? 'Member'} ▾
      </button>
      {open && (
        <div
          style={{ position: 'absolute', right: 0, top: '100%', background: 'var(--ifm-background-color)', border: '1px solid var(--ifm-color-border)', padding: 8, marginTop: 8, zIndex: 50 }}
        >
          <Link href="/profile" className="dropdown-item">
            Profile
          </Link>
          <button className="dropdown-item" onClick={() => logout()}>
            Logout
          </button>
        </div>
      )}
    </div>
  );
}
