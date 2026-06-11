import { useAuth } from '../context/AuthContext';
import { useLocation } from '@docusaurus/router';

/**
 * useProtectedRoute Hook
 * 
 * Determines if the current route requires login and if the user has access.
 * 
 * Returns:
 * - isProtected: boolean - Whether this route is protected
 * - hasAccess: boolean - Whether the user has access to this route
 * - redirectToLogin: () => void - Function to redirect to login page
 * 
 * Example:
 * const { isProtected, hasAccess } = useProtectedRoute();
 * if (isProtected && !hasAccess) {
 *   return <LoginRequired />;
 * }
 */
export const useProtectedRoute = () => {
  const { isAuthenticated, setRedirectPath } = useAuth();
  const location = useLocation();

  const redirectToLogin = () => {
    setRedirectPath(location.pathname);
    // Navigation should be handled by the component using this hook
  };

  return {
    hasAccess: isAuthenticated,
    redirectToLogin,
  };
};

/**
 * useRequiresLogin Hook
 * 
 * Simple hook to check if user is authenticated for login-required pages.
 * 
 * Returns:
 * - isAuthenticated: boolean
 * - user: User | null
 * - login: (username: string, password: string) => Promise<void>
 * - logout: () => void
 * 
 * Example:
 * const { isAuthenticated, login, logout } = useRequiresLogin();
 */
export const useRequiresLogin = () => {
  const { isAuthenticated, user, login, logout } = useAuth();

  return {
    isAuthenticated,
    user,
    login,
    logout,
  };
};

export default useProtectedRoute;
