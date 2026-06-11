import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

/**
 * User object structure for authentication
 */
export interface User {
  username: string;
  role: string;
}

/**
 * Auth context type
 */
export interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  getRedirectPath: () => string;
  setRedirectPath: (path: string) => void;
}

/**
 * Create the authentication context
 */
const AuthContext = createContext<AuthContextType | undefined>(undefined);

/**
 * Mock authentication data
 * In production, this would validate against a backend API
 * For now, we accept any username/password combination for demo purposes
 */
const MOCK_USERS: Record<string, User> = {
  admin: { username: 'admin', role: 'admin' },
  user: { username: 'user', role: 'user' },
};

/**
 * Local storage keys
 */
const AUTH_STORAGE_KEY = 'vedicskill_auth_user';
const REDIRECT_PATH_KEY = 'vedicskill_redirect_path';

/**
 * AuthProvider component
 * Wraps the entire app and provides authentication state
 */
export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [redirectPath, setRedirectPath] = useState<string>('/');

  /**
   * Initialize auth state from localStorage on mount
   */
  useEffect(() => {
    const initializeAuth = () => {
      try {
        // Restore user from localStorage
        const storedUser = localStorage.getItem(AUTH_STORAGE_KEY);
        if (storedUser) {
          setUser(JSON.parse(storedUser));
        }

        // Restore redirect path
        const storedPath = localStorage.getItem(REDIRECT_PATH_KEY);
        if (storedPath) {
          setRedirectPath(storedPath);
          localStorage.removeItem(REDIRECT_PATH_KEY);
        }
      } catch (error) {
        console.error('Failed to initialize auth state:', error);
      } finally {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, []);

  /**
   * Login function
   * Accepts any username/password for demo
   * In production, validate against backend API
   */
  const login = async (username: string, password: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      try {
        // Simple validation
        if (!username || !password) {
          reject(new Error('Username and password are required'));
          return;
        }

        // Mock authentication - in production, call your backend
        // For demo: accept any username/password
        const userData: User = MOCK_USERS[username] || {
          username: username,
          role: 'user',
        };

        // Store in localStorage
        localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(userData));
        setUser(userData);
        resolve();
      } catch (error) {
        reject(error);
      }
    });
  };

  /**
   * Logout function
   * Clears user data from state and localStorage
   */
  const logout = (): void => {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    setUser(null);
  };

  /**
   * Get redirect path for post-login navigation
   */
  const getRedirectPath = (): string => {
    return redirectPath || '/';
  };

  /**
   * Set redirect path to navigate after login
   */
  const setRedirectPathFn = (path: string): void => {
    setRedirectPath(path);
    localStorage.setItem(REDIRECT_PATH_KEY, path);
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: user !== null,
    isLoading,
    login,
    logout,
    getRedirectPath,
    setRedirectPath: setRedirectPathFn,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

/**
 * useAuth hook
 * Use this hook to access authentication state and methods
 * 
 * Example:
 * const { user, isAuthenticated, login, logout } = useAuth();
 */
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
