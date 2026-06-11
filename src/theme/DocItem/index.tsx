import React from 'react';
import OriginalDocItem from '@theme-original/DocItem';
import type { Props } from '@theme-original/DocItem';
import ProtectedRoute from '../../components/ProtectedRoute';

export default function DocItem(props: Props) {
  const requiresLogin = props.frontMatter?.requiresLogin === true;

  if (requiresLogin) {
    return (
      <ProtectedRoute>
        <OriginalDocItem {...props} />
      </ProtectedRoute>
    );
  }

  return <OriginalDocItem {...props} />;
}
