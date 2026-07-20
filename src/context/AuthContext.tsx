import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';

// This AuthContext uses a simple local auth server instead of Supabase.
// The server URL can be configured via `REACT_APP_AUTH_SERVER_URL`.
const getAuthServerUrl = () => {
  const envValue = (globalThis as typeof globalThis & {
    process?: { env?: Record<string, string | undefined> };
  }).process?.env?.REACT_APP_AUTH_SERVER_URL;

  return envValue || 'http://localhost:3000';
};

const AUTH_SERVER = getAuthServerUrl();

type User = {
  id?: string;
  email: string;
};

interface Profile {
  id: string;
  name?: string | null;
  email?: string | null;
  created_at?: string | null;
}

export interface AuthContextType {
  user: User | null;
  profile: Profile | null;
  loading: boolean;
  isAuthenticated: boolean;

  login(email: string, password: string): Promise<void>;
  signup(name: string, email: string, password: string): Promise<void>;
  logout(): Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = (): AuthContextType => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);

  const isAuthenticated = !!user;

  async function fetchProfile(userId: string) {
    try {
      const token = localStorage.getItem('authToken');
      const res = await fetch(`${AUTH_SERVER}/auth/profile`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        setProfile(null);
        return;
      }
      const data = await res.json();
      setProfile(data as Profile | null);
    } catch (err) {
      console.error(err);
      setProfile(null);
    }
  }

  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        const token = localStorage.getItem('authToken');
        if (!token) return;
        const res = await fetch(`${AUTH_SERVER}/auth/session`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) return;
        const { user: currentUser } = await res.json();
        if (!mounted) return;
        setUser(currentUser as User | null);
        if (currentUser?.id) await fetchProfile(currentUser.id);
      } catch (err) {
        console.warn('Failed to restore session', err);
      } finally {
        if (mounted) setLoading(false);
      }
    }

    init();
    // No real-time auth listener for the simple server. Cleanup.
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const login = async (email: string, password: string) => {
    setLoading(true);
    try {
      const res = await fetch(`${AUTH_SERVER}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.message || 'Login failed');
      }
      const { user: u, token } = await res.json();
      localStorage.setItem('authToken', token);
      setUser(u as User | null);
      if (u?.id) await fetchProfile(u.id);
    } finally {
      setLoading(false);
    }
  };

  const signup = async (name: string, email: string, password: string) => {
    setLoading(true);
    try {
      const res = await fetch(`${AUTH_SERVER}/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.message || 'Signup failed');
      }
      const { user: u, token } = await res.json();
      localStorage.setItem('authToken', token);
      if (u) {
        setUser(u as User);
        if (u.id) await fetchProfile(u.id);
      }
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('authToken');
      await fetch(`${AUTH_SERVER}/auth/logout`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {});
      localStorage.removeItem('authToken');
      setUser(null);
      setProfile(null);
    } finally {
      setLoading(false);
    }
  };

  const value = useMemo(() => ({ user, profile, loading, isAuthenticated, login, signup, logout }), [user, profile, loading, isAuthenticated]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthContext;

