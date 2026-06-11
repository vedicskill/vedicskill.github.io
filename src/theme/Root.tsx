import React, { FC, ReactNode } from 'react';
import { AuthProvider } from '../context/AuthContext';

/**
 * Root Component
 * 
 * This is the top-level wrapper component for the Docusaurus site.
 * It provides the AuthProvider to all child components, making authentication
 * state available throughout the entire application.
 * 
 * This file should be placed in src/theme/Root.tsx and will be automatically
 * picked up by Docusaurus.
 */
const Root: FC<{ children: ReactNode }> = ({ children }) => {
  return <AuthProvider>{children}</AuthProvider>;
};

export default Root;
