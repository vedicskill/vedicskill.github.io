import React from 'react';
import OriginalRoot from '@theme-original/Root';
import { AuthProvider } from '../context/AuthContext';

export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <OriginalRoot>{children}</OriginalRoot>
    </AuthProvider>
  );
}

