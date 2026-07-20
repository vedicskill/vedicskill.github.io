"use client";
import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import { useAuth } from '../context/AuthContext';
import { useHistory } from '@docusaurus/router';

export default function ProfilePage() {
  const { isAuthenticated, profile, loading } = useAuth();
  const history = useHistory();

  useEffect(() => {
    if (!loading && !isAuthenticated) {
      history.push('/login');
    }
  }, [isAuthenticated, loading, history]);

  if (loading) return <Layout title="Profile">Loading...</Layout>;

  return (
    <Layout title="Profile">
      <main style={{ padding: 24 }}>
        <h1>Account Profile</h1>
        <dl>
          <dt>Name</dt>
          <dd>{profile?.name ?? '—'}</dd>
          <dt>Email</dt>
          <dd>{profile?.email ?? '—'}</dd>
          <dt>Account created</dt>
          <dd>{profile?.created_at ? new Date(profile.created_at).toLocaleString() : '—'}</dd>
        </dl>
      </main>
    </Layout>
  );
}
