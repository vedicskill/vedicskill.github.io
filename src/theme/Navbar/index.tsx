import React from 'react';
import OriginalNavbar from '@theme-original/Navbar';
import AuthButtons from '../../components/AuthButtons';

export default function Navbar(props: any) {
  return (
    <>
      <OriginalNavbar {...props} />
      {/* Auth buttons rendered separately so they can float in the top-right */}
      <div style={{ position: 'fixed', right: 16, top: 12, zIndex: 999 }}>
        <AuthButtons />
      </div>
    </>
  );
}

