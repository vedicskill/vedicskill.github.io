import React from 'react';
import Navbar from '@docusaurus/theme-classic/lib/theme/Navbar';
import { useAuth } from '../../context/AuthContext';
import styles from './navbar-auth.module.css';

/**
 * NavbarAuthMenu Component
 * 
 * Displays authentication-aware menu items in the navbar:
 * - If not logged in: Show "Login" button
 * - If logged in: Show user profile and "Logout" button
 */
const NavbarAuthMenu: React.FC = () => {
  const { isAuthenticated, user, logout, isLoading } = useAuth();

  if (isLoading) {
    return null;
  }

  if (!isAuthenticated) {
    return (
      <button
        className={styles.loginBtn}
        onClick={() => {
          if (typeof window !== 'undefined') window.location.href = '/login';
        }}
        title="Login to access protected content"
      >
        Login
      </button>
    );
  }

  return (
    <div className={styles.userMenu}>
      <span className={styles.userGreeting}>
        👤 {user?.username}
        {user?.role && user.role !== 'user' && (
          <span className={styles.roleBadge}>{user.role}</span>
        )}
      </span>
      <button
        className={styles.logoutBtn}
        onClick={() => {
          logout();
          if (typeof window !== 'undefined') window.location.href = '/';
        }}
        title="Logout from your account"
      >
        Logout
      </button>
    </div>
  );
};

/**
 * Navbar Component (Swizzled)
 * 
 * This component swizzles Docusaurus's default Navbar to add
 * authentication controls.
 * 
 * It renders the original Navbar and adds the NavbarAuthMenu component
 * to the right side of the navbar.
 */
export default function NavbarWrapper(props: any): JSX.Element {
  return (
    <>
      <div className={styles.authMenuContainer}>
        <NavbarAuthMenu />
      </div>
      <Navbar {...props} />
    </>
  );
}
